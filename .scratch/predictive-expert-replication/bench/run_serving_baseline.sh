#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Ticket 00 serving baseline. Starts one server configuration, runs the request
# matrix against it, and captures the server log so per-rank expert load can be
# scraped afterwards.
#
# Two configurations exist deliberately:
#   memory   - no EPLB at all. The honest memory/KV baseline.
#   recording- EPLB enabled with zero redundant experts and an unreachable
#              rearrangement interval, so it records and logs per-rank load
#              without ever changing placement. This costs one extra transfer
#              buffer, which is why it must not be used for the memory numbers.
#
# Usage: run_serving_baseline.sh <memory|recording> [out_dir]
set -euo pipefail

MODE="${1:?usage: $0 <memory|recording> [out_dir]}"
OUT_DIR="${2:-$(dirname "$0")/results}"
MODEL="${MODEL:-/models/preset/Qwen/Qwen3-30B-A3B/v1.0}"
PORT="${PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.90}"
# Per-rank running-sequence cap. vLLM defaults to 128, which is exactly the
# decode batch the boundness sweep needs to exceed, so it must be settable.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"

mkdir -p "$OUT_DIR"
SERVER_LOG="$OUT_DIR/server-$MODE.log"

# Serialize runs. Two benchmark loops sharing one server silently corrupts every
# latency and throughput number, which is how the first session's results were
# lost. Fail loudly instead of measuring nonsense.
LOCK="${TMPDIR:-/tmp}/predictive-baseline.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another baseline run holds $LOCK; refusing to share a server" >&2
  exit 3
fi

COMMON=(
  --port "$PORT"
  --data-parallel-size 8
  --enable-expert-parallel
  --all2all-backend allgather_reducescatter
  --enforce-eager
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
  --uvicorn-log-level warning
)

case "$MODE" in
  memory) EXTRA=() ;;
  recording)
    # step_interval far beyond any run length, so rearrangement never fires.
    EXTRA=(
      --enable-eplb
      --eplb-config
      '{"num_redundant_experts":0,"step_interval":1000000000,"window_size":1000,"log_balancedness":true,"log_balancedness_interval":1,"use_async":false}'
    ) ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

echo "[baseline] starting server ($MODE), log -> $SERVER_LOG"
vllm serve "$MODEL" "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

echo "[baseline] waiting for readiness (up to 20 min)"
for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[baseline] server ready"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[baseline] server exited early; see $SERVER_LOG" >&2
    tail -40 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[baseline] never became ready" >&2; exit 1; }

# Request shapes fixed by the spec: prefill-weighted and decode-weighted.
# Domains chosen because expert routing is content dependent.
#
# --ignore-eos is required, not cosmetic. Without it `--custom-output-len` is only
# an upper bound: requests stop at EOS, decode length varies by domain, and TPOT
# stops being comparable across configurations. A first run produced 1716 tokens
# against a 2048 target for exactly this reason.
run_case() {
  local domain="$1" dataset="$2" prompt_len="$3" output_len="$4" conc="$5"
  local tag="$MODE-$domain-p${prompt_len}d${output_len}-c${conc}"
  # The stock CLI can force decode length but cannot bound an HF dataset's prompt
  # length, so filter to the target band first and serve the result as a custom
  # dataset. Without this the request shape is whatever the dataset happens to be.
  local prompts="$OUT_DIR/prompts-$domain-p$prompt_len.jsonl"
  if [[ ! -s "$prompts" ]]; then
    # Judge by the artifact, not the exit status. The HuggingFace streaming
    # reader can abort during interpreter teardown *after* the prompts are
    # written, which discarded a complete file on the first attempt.
    python3 "$(dirname "$0")/make_prompts.py" \
      --dataset "$dataset" --tokenizer "$MODEL" \
      --target-prompt-len "$prompt_len" --tolerance 0.10 \
      --num-prompts "$MAX_PROMPTS" --out "$prompts" \
      >"$OUT_DIR/prompts-$domain-p$prompt_len.log" 2>&1 || true
    if [[ ! -s "$prompts" ]]; then
      echo "[baseline] $tag SKIPPED: could not build $prompt_len-token prompts"
      return
    fi
  fi
  # Concurrency can never exceed the number of prompts the client holds, so a
  # smaller prompt count silently caps the running batch and every higher
  # concurrency point measures the same thing. This invalidated a whole sweep.
  if (( NUM_PROMPTS < conc )); then
    echo "[baseline] $tag REFUSED: NUM_PROMPTS=$NUM_PROMPTS < concurrency $conc," \
         "so the batch would cap at $NUM_PROMPTS and the point would be a duplicate" >&2
    return
  fi
  echo "[baseline] $tag"
  vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --model "$MODEL" --port "$PORT" \
    --dataset-name custom --dataset-path "$prompts" \
    --custom-output-len "$output_len" --ignore-eos \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$conc" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --save-result --result-dir "$OUT_DIR" --result-filename "bench-$tag.json" \
    >"$OUT_DIR/bench-$tag.log" 2>&1 || echo "[baseline] $tag FAILED (see log)"
}

NUM_PROMPTS="${NUM_PROMPTS:-200}"
MAX_PROMPTS="${MAX_PROMPTS:-1024}"
# Concurrency ceiling comes from KV, not from taste: vLLM reports 181,040 KV
# tokens per rank, and both agreed shapes have a 3072-token context, so 58
# sequences per rank (464 total) is the hard limit. 384 uses about 80% of that,
# leaving room for prefill bursts; higher starts preempting and measures jitter
# rather than steady state. 32 total is only 4 per rank and says nothing that 64
# does not.
CONCURRENCIES="${CONCURRENCIES:-64 192 384}"

# Request shapes as "<prompt tokens>:<decode tokens>". The two defaults are the
# spec's prefill-weighted and decode-weighted shapes. Override for the boundness
# sweep, where short contexts are the only way to reach a decode batch large
# enough to leave the expert-weight-bandwidth-bound regime: KV holds a fixed
# number of tokens, so sequences per rank is capacity divided by context length.
SHAPES="${SHAPES:-1024:2048 2048:1024}"
# Entries may carry a third field to pin one domain: "1024:2048:code".
DOMAINS="${DOMAINS:-text code math}"

dataset_for() {
  case "$1" in
    text) echo "Aeala/ShareGPT_Vicuna_unfiltered" ;;
    code) echo "likaixin/InstructCoder" ;;
    math) echo "AI-MO/NuminaMath-CoT" ;;
    *) echo "unknown domain: $1" >&2; return 1 ;;
  esac
}

for conc in $CONCURRENCIES; do
  for shape in $SHAPES; do
    # "<prompt>:<decode>" runs every domain in DOMAINS.
    # "<prompt>:<decode>:<domain>" pins one, which is how a dataset whose natural
    # length already sits near the target gets paired with that shape instead of
    # being trimmed into it.
    IFS=':' read -r prompt_len output_len shape_domain <<<"$shape"
    for domain in ${shape_domain:-$DOMAINS}; do
      run_case "$domain" "$(dataset_for "$domain")" \
        "$prompt_len" "$output_len" "$conc"
    done
  done
done

echo "[baseline] done; results in $OUT_DIR"
