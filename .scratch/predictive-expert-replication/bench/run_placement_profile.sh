#!/usr/bin/env bash
# Two torch profiles of the same prefill workload: placement off and placement on.
#
# For inspecting whether the replica transfers overlap the compute or serialize into
# it. The transfers run on their own CUDA stream, so a timeline shows directly whether
# they hide behind the layers between planning and activation, or extend the step.
#
# Deliberate choices:
#
#  * `python3 -m vllm.entrypoints.openai.api_server`, never `vllm serve`. The console
#    script does not put the tree on `sys.path`, so it loads the installed stock vLLM,
#    which has no predictive code at all and silently ignores the config. That cost six
#    empty runs once.
#  * Concurrency 8 at 1024 tokens is 454 tokens per expert, above the one-block bar, so
#    placement is actually armed. Concurrency 1 is 69.9 and is gated off by design —
#    profiling it would show a placed arm doing nothing.
#  * A short decode and a single wave, so the trace stays small enough to open. The
#    prefill forwards are the ones that place.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
OUT_DIR="${1:-$HERE/results/placement-profile}"
MODEL="${MODEL:-/models/preset/Qwen/Qwen3-30B-A3B/v1.0}"
PORT="${PORT:-8181}"
CONC="${CONC:-8}"
OUT_LEN="${OUT_LEN:-4}"
BUDGETS="${BUDGETS:-0 43}"

export PYTHONPATH="$REPO_ROOT"
RESOLVED=$(python3 -c 'import vllm,sys; sys.stdout.write(vllm.__file__)' 2>/dev/null || true)
case "$RESOLVED" in
  "$REPO_ROOT"/vllm/*) echo "[prof] vLLM from $RESOLVED" ;;
  *) echo "[prof] ABORT: vLLM resolves to '$RESOLVED', not this tree" >&2; exit 1 ;;
esac

mkdir -p "$OUT_DIR"
LOCK="${TMPDIR:-/tmp}/predictive-baseline.lock"
exec 9>"$LOCK"
flock -w 10800 9 || { echo "[prof] lock not acquired" >&2; exit 3; }

PROFILE="$OUT_DIR/cost-profile-local.json"
python3 - "$HERE/results/cost-profile.json" "$PROFILE" "$MODEL" <<'PY'
import json, sys
src, dst, model = sys.argv[1:4]
p = json.load(open(src)); p["fingerprint"]["model"] = model
json.dump(p, open(dst, "w"), indent=2)
PY

PROMPTS="$HERE/results/prompts-code-p1024.jsonl"
[[ -s "$PROMPTS" ]] || { echo "[prof] missing $PROMPTS" >&2; exit 1; }

for budget in $BUDGETS; do
  tag="b$budget"
  TRACE_DIR="$OUT_DIR/trace-$tag"
  rm -rf "$TRACE_DIR"; mkdir -p "$TRACE_DIR"
  LOG="$OUT_DIR/server-$tag.log"
  echo "[prof] budget=$budget: starting server"

  ADDITIONAL=$(python3 -c "
import json,sys
cfg={'enabled': True, 'cost_profile_path': sys.argv[1]}
if int(sys.argv[2]) > 0:
    cfg['max_transfers_per_forward'] = int(sys.argv[2])
print(json.dumps({'predictive_expert_replication': cfg}))" "$PROFILE" "$budget")

  PROFILER_CFG=$(python3 -c "
import json,sys
print(json.dumps({'profiler':'torch','torch_profiler_dir':sys.argv[1],
 'torch_profiler_with_stack':False,'torch_profiler_record_shapes':False,
 'torch_profiler_with_flops':False,'ignore_frontend':True}))" "$TRACE_DIR")

  VLLM_PREDICTIVE_PLACE_PER_FORWARD="$budget" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" \
    --data-parallel-size 8 --enable-expert-parallel \
    --all2all-backend allgather_reducescatter --enforce-eager \
    --max-model-len 3072 --gpu-memory-utilization 0.88 \
    --max-num-seqs 64 --seed 0 --uvicorn-log-level warning \
    --additional-config "$ADDITIONAL" \
    --profiler-config "$PROFILER_CFG" \
    --eplb-config '{"log_balancedness":true,"log_balancedness_interval":1,"step_interval":1000000000,"window_size":1000,"use_async":false}' \
    >"$LOG" 2>&1 &
  PID=$!

  ready=0
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 "$PID" 2>/dev/null || break
    sleep 5
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "[prof] $tag: server never ready" >&2; tail -40 "$LOG" >&2
    kill -9 "$PID" 2>/dev/null; continue
  fi
  echo "[prof] $tag: ready"

  # Warm up outside the profile: the first forwards JIT kernels and run a single-rank
  # step, which is not the steady state and would dominate a short trace.
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --model "$MODEL" --port "$PORT" \
    --dataset-name custom --dataset-path "$PROMPTS" \
    --custom-output-len "$OUT_LEN" --ignore-eos \
    --num-prompts "$CONC" --max-concurrency "$CONC" \
    >"$OUT_DIR/warmup-$tag.log" 2>&1 || echo "[prof] $tag warmup failed" >&2

  curl -sf -X POST "http://127.0.0.1:$PORT/start_profile" >/dev/null \
    || echo "[prof] $tag: start_profile failed" >&2
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --model "$MODEL" --port "$PORT" \
    --dataset-name custom --dataset-path "$PROMPTS" \
    --custom-output-len "$OUT_LEN" --ignore-eos \
    --num-prompts "$CONC" --max-concurrency "$CONC" \
    --percentile-metrics ttft --metric-percentiles 99 \
    >"$OUT_DIR/bench-$tag.log" 2>&1 || echo "[prof] $tag bench failed" >&2
  curl -sf -X POST "http://127.0.0.1:$PORT/stop_profile" >/dev/null \
    || echo "[prof] $tag: stop_profile failed" >&2

  # Traces flush asynchronously after stop_profile returns.
  for _ in $(seq 1 60); do
    n=$(find "$TRACE_DIR" -name '*.pt.trace.json*' 2>/dev/null | wc -l)
    [[ "$n" -ge 8 ]] && break
    sleep 5
  done
  n=$(find "$TRACE_DIR" -name '*.pt.trace.json*' 2>/dev/null | wc -l)
  echo "[prof] $tag: $n rank traces in $TRACE_DIR"
  grep -c "Predictive expert replication: activated" "$LOG" 2>/dev/null \
    | xargs -I{} echo "[prof] $tag: {} placement log lines"

  kill -TERM "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; sleep 5
  for p in $(ps -eo pid,cmd --no-headers | grep "VLLM::" | grep -v grep | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null
  done
  sleep 10
done
echo "[prof] done; traces in $OUT_DIR"
