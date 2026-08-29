#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Ticket 00's decisive measurement: MoE's share of a decode step, and how MoE
# time relates to the expert-weight read floor, at several real decode batches.
#
# Design notes that are not obvious:
#
#  * `--profiler-config.torch_profiler_with_stack=false`. The default traces
#    every Python call (785k events in the first attempt), which inflated TPOT
#    from 80 ms to 387 ms. That inflation lands in the NCCL kernels, because they
#    spin until every peer arrives and so absorb all CPU-side jitter. With stacks
#    off, the collective time is far closer to what serving actually pays.
#
#  * One wave per point: NUM_PROMPTS equals the concurrency, so every request is
#    admitted together and the run is a prefill burst followed by a long, clean
#    decode phase. The profile window is taken well inside that phase.
#
#  * `--ignore-eos`, so all requests decode the same length and the batch does
#    not shrink under the profile window.
#
# Usage: run_boundness_profile.sh [out_dir]
set -euo pipefail

# The `vllm` console script puts /usr/local/bin on sys.path, never the current
# directory, so without this it silently runs the *installed* vLLM rather than
# this working tree. A whole accuracy run once produced six empty dumps that way,
# with nothing but an "Unknown vLLM environment variable" warning to show for it.
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
loaded=$(python3 -c "import vllm; print(vllm.__file__)" 2>/dev/null || true)
case "$loaded" in
  "$REPO_ROOT"/*) : ;;
  *) echo "[FATAL] vLLM resolves to '$loaded', not $REPO_ROOT." >&2
     echo "        The run would measure the installed build, not this tree." >&2
     exit 4 ;;
esac
echo "[env] vLLM from $loaded"

OUT_DIR="${1:-$(dirname "$0")/results/boundness}"
MODEL="${MODEL:-/models/preset/Qwen/Qwen3-30B-A3B/v1.0}"
PORT="${PORT:-8130}"
TRACE_ROOT="${TRACE_ROOT:-/tmp/prof-boundness}"
# 3072 covers the 1024-prompt/2048-decode shape exactly. KV holds ~174k tokens
# per rank at util 0.9, so 48 sequences per rank (384 total) is the ceiling.
MAX_MODEL_LEN=3072
PROMPT_LEN=1024
OUTPUT_LEN=2048
CONCURRENCIES="${CONCURRENCIES:-8 64 192 384}"

mkdir -p "$OUT_DIR" "$TRACE_ROOT"
SERVER_LOG="$OUT_DIR/server.log"

LOCK="${TMPDIR:-/tmp}/predictive-baseline.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "another run holds $LOCK" >&2; exit 3; }

PROFILER_CFG=$(cat <<'JSON'
{"profiler":"torch","torch_profiler_dir":"__TRACE_ROOT__",
 "torch_profiler_with_stack":false,"torch_profiler_record_shapes":false,
 "torch_profiler_with_flops":false,"ignore_frontend":true}
JSON
)
PROFILER_CFG="${PROFILER_CFG//__TRACE_ROOT__/$TRACE_ROOT}"

echo "[boundness] starting server -> $SERVER_LOG"
vllm serve "$MODEL" \
  --port "$PORT" \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 \
  --uvicorn-log-level warning \
  --enable-eplb \
  --eplb-config '{"num_redundant_experts":0,"step_interval":1000000000,"window_size":1000,"log_balancedness":true,"log_balancedness_interval":1,"use_async":false}' \
  --profiler-config "$PROFILER_CFG" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true; sleep 5; pkill -9 -f "VLLM::" 2>/dev/null || true' EXIT

echo "[boundness] waiting for readiness (up to 20 min)"
for _ in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "[boundness] ready"; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[boundness] server died" >&2; tail -40 "$SERVER_LOG" >&2; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[boundness] never ready" >&2; exit 1; }

# Shared across runs: these sets are 2 MB each and identical between runners,
# so a per-run copy is pure duplication.
PROMPTS="$(dirname "$0")/results/prompts-code-p$PROMPT_LEN.jsonl"
if [[ ! -s "$PROMPTS" ]]; then
  echo "[boundness] building $PROMPT_LEN-token code prompts"
  python3 "$(dirname "$0")/make_prompts.py" \
    --dataset likaixin/InstructCoder --tokenizer "$MODEL" \
    --target-prompt-len "$PROMPT_LEN" --tolerance 0.10 \
    --num-prompts 512 --out "$PROMPTS" \
    >"$OUT_DIR/prompts.log" 2>&1 || true
  [[ -s "$PROMPTS" ]] || { echo "[boundness] prompt build failed" >&2; exit 1; }
fi

for conc in $CONCURRENCIES; do
  tag="c$conc"
  dest="$TRACE_ROOT/$tag"
  mkdir -p "$dest"
  # Give the decode phase time to establish before the window: the prefill burst
  # scales with the wave size, so the delay does too.
  delay=$(( 20 + conc / 6 ))
  echo "[boundness] $tag: one wave of $conc requests, profile window at t+${delay}s"

  vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --model "$MODEL" --port "$PORT" \
    --dataset-name custom --dataset-path "$PROMPTS" \
    --custom-output-len "$OUTPUT_LEN" --ignore-eos \
    --num-prompts "$conc" --max-concurrency "$conc" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir "$OUT_DIR" --result-filename "bench-$tag.json" \
    >"$OUT_DIR/bench-$tag.log" 2>&1 &
  CLIENT_PID=$!

  sleep "$delay"
  if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
    echo "[boundness] $tag: client already exited before the window; see bench-$tag.log" >&2
  fi
  curl -sf -X POST "http://127.0.0.1:$PORT/start_profile" >/dev/null || \
    echo "[boundness] $tag: start_profile failed" >&2
  sleep 4
  curl -sf -X POST "http://127.0.0.1:$PORT/stop_profile" >/dev/null || \
    echo "[boundness] $tag: stop_profile failed" >&2

  # Traces flush asynchronously and the ranks do not finish together; the first
  # attempt saw a four-minute spread. Wait for eight files rather than a fixed
  # sleep, then move them out so the next point starts from an empty directory.
  for _ in $(seq 1 90); do
    n=$(find "$TRACE_ROOT" -maxdepth 1 -name 'dp*.pt.trace.json*' 2>/dev/null | wc -l)
    [[ "$n" -ge 8 ]] && break
    sleep 5
  done
  n=$(find "$TRACE_ROOT" -maxdepth 1 -name 'dp*.pt.trace.json*' 2>/dev/null | wc -l)
  echo "[boundness] $tag: $n rank traces flushed"
  find "$TRACE_ROOT" -maxdepth 1 -name '*.pt.trace.json*' -exec mv {} "$dest"/ \; 2>/dev/null || true

  echo "[boundness] $tag: draining the wave"
  kill -INT "$CLIENT_PID" 2>/dev/null || true
  wait "$CLIENT_PID" 2>/dev/null || true
  # Let the server finish whatever the client abandoned before the next point,
  # otherwise the next wave's prefill overlaps this one's decode.
  for _ in $(seq 1 60); do
    running=$(curl -sf "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
      | awk '/^vllm:num_requests_running/{s+=$2} END{printf "%d", s+0}')
    [[ "${running:-0}" -eq 0 ]] && break
    sleep 5
  done
  echo "[boundness] $tag: drained (running=${running:-unknown})"
done

echo "[boundness] parsing"
for conc in $CONCURRENCIES; do
  python3 "$(dirname "$0")/parse_profile.py" \
    --trace-dir "$TRACE_ROOT/c$conc" --num-layers 48 \
    --out "$OUT_DIR/attribution-c$conc.json" >/dev/null 2>&1 \
    && echo "[boundness] c$conc parsed" \
    || echo "[boundness] c$conc parse FAILED" >&2
done

echo "[boundness] done; results in $OUT_DIR, traces in $TRACE_ROOT"
