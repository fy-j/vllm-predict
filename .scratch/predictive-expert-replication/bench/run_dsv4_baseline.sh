#!/usr/bin/env bash
# Baseline profile and expert-load dump for DeepSeek-V4-Flash-FP8.
#
# The one question this answers: **what fraction of a prefill step is expert GEMM on this
# model?** Multiplied by what perfect balance can recover, that is the ceiling on anything
# a placement policy can win, and it is the number that decides whether the feature is
# worth building for this shape. On Qwen3-30B-A3B it measured 10.76%, giving a ceiling of
# about 5.05% of a step against a measured cost of +31.5% TTFT.
#
# Why this model is the interesting one, from the two configs:
#
#   Qwen3-30B-A3B : 128 experts, top-8, 768 x 2048  ->  75.5 MFLOP/token of expert work
#   DSV4-Flash    : 256 experts, top-6, 2048 x 4096 -> 302.0 MFLOP/token, and FP8 halves
#                   the time, so about 2x Qwen's expert GEMM time per token
#
# Prediction's overhead is per source layer and barely changes between the two, so the
# ratio of headroom to overhead should improve by roughly 2x. This run is what replaces
# that "should" with a number.
#
# No predictive config at all: this is a stock server. EPLB is enabled only to make the
# per-layer expert-load dump available, with zero redundant experts so the layout is
# unchanged. If that combination is rejected, set NO_EPLB=1 and the profile still answers
# the ceiling question; only the imbalance report needs the dump.
#
# **The kernel classifier does not know this model.** `parse_profile.classify_kernel`
# matches `fused_moe_kernel` and `grouped_gemm`, and DeepSeek V4 runs DeepGEMM under a
# different name, so an unmodified classifier reports an expert-GEMM share near zero —
# a wrong number, not an error. Read the top kernels out of the trace first and extend
# the classifier before believing any share from this run.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
OUT_DIR="${1:-$HERE/results/h100/dsv4-baseline}"
MODEL="${MODEL:-/mnt/Models/models--sgl-project--DeepSeek-V4-Flash-FP8/snapshots/ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17}"
PORT="${PORT:-8182}"
CONC="${CONC:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
OUT_LEN="${OUT_LEN:-4}"
MAX_LEN="${MAX_LEN:-3072}"
GPU_UTIL="${GPU_UTIL:-0.92}"
NO_EPLB="${NO_EPLB:-0}"
# DeepSeek V4's sparse MLA uses the `fp8_ds_mla` KV layout, which asserts an fp8 KV cache:
# with the default `auto` every worker dies on "only supports fp8 kv-cache, got auto".
KV_DTYPE="${KV_DTYPE:-fp8}"

export PYTHONPATH="$REPO_ROOT"
RESOLVED=$(python3 -c 'import vllm,sys; sys.stdout.write(vllm.__file__)' 2>/dev/null || true)
case "$RESOLVED" in
  "$REPO_ROOT"/vllm/*) echo "[dsv4] vLLM from $RESOLVED" ;;
  *) echo "[dsv4] ABORT: vLLM resolves to '$RESOLVED', not this tree" >&2; exit 1 ;;
esac

PROMPTS="$HERE/results/prompts-ko-dsv4-p1024.jsonl"
[[ -s "$PROMPTS" ]] || { echo "[dsv4] missing $PROMPTS; build it with make_prompts.py using this model's tokenizer" >&2; exit 1; }

mkdir -p "$OUT_DIR"
LOCK="${TMPDIR:-/tmp}/predictive-baseline.lock"
exec 9>"$LOCK"
flock -w 10800 9 || { echo "[dsv4] lock not acquired" >&2; exit 3; }

TRACE_DIR="$OUT_DIR/trace"
rm -rf "$TRACE_DIR"; mkdir -p "$TRACE_DIR"
LOG="$OUT_DIR/server.log"
DUMP="$OUT_DIR/dump.jsonl"
rm -f "$DUMP"

PROFILER_CFG=$(python3 -c "
import json,sys
print(json.dumps({'profiler':'torch','torch_profiler_dir':sys.argv[1],
 'torch_profiler_with_stack':False,'torch_profiler_record_shapes':False,
 'torch_profiler_with_flops':False,'ignore_frontend':True}))" "$TRACE_DIR")

EPLB_ARGS=()
if [[ "$NO_EPLB" != "1" ]]; then
  # Recording only: zero redundant experts leaves the physical layout exactly as a stock
  # server's, and `step_interval` far beyond the run stops the native controller from
  # rearranging anything. All this buys is the per-layer load the dump needs.
  EPLB_ARGS=(
    --enable-eplb
    --eplb-config '{"num_redundant_experts":0,"log_balancedness":true,"log_balancedness_interval":1,"step_interval":1000000000,"window_size":1000,"use_async":false}'
  )
fi

echo "[dsv4] starting server, 274 GiB of weights over 8 ranks"
VLLM_EPLB_DUMP_LOAD_PATH="$DUMP" \
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port "$PORT" \
  --data-parallel-size 8 --enable-expert-parallel \
  --all2all-backend allgather_reducescatter --enforce-eager \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" \
  --kv-cache-dtype "$KV_DTYPE" \
  --max-num-seqs 64 --seed 0 --uvicorn-log-level warning \
  --profiler-config "$PROFILER_CFG" \
  "${EPLB_ARGS[@]}" \
  >"$LOG" 2>&1 &
PID=$!

ready=0
for _ in $(seq 1 480); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 "$PID" 2>/dev/null || break
  sleep 5
done
if [[ "$ready" -ne 1 ]]; then
  echo "[dsv4] server never ready" >&2; tail -60 "$LOG" >&2
  kill -9 "$PID" 2>/dev/null || true
  exit 4
fi
echo "[dsv4] ready"

# Warm up outside the profile: the first forwards JIT kernels, which is not steady state
# and would dominate a short trace.
vllm bench serve --backend openai --endpoint /v1/completions \
  --model "$MODEL" --port "$PORT" \
  --dataset-name custom --dataset-path "$PROMPTS" \
  --custom-output-len "$OUT_LEN" --ignore-eos --skip-chat-template \
  --num-prompts "$CONC" --max-concurrency "$CONC" \
  >"$OUT_DIR/warmup.log" 2>&1 || echo "[dsv4] warmup failed" >&2

curl -sf -X POST "http://127.0.0.1:$PORT/start_profile" >/dev/null || echo "[dsv4] start_profile failed" >&2
vllm bench serve --backend openai --endpoint /v1/completions \
  --model "$MODEL" --port "$PORT" \
  --dataset-name custom --dataset-path "$PROMPTS" \
  --custom-output-len "$OUT_LEN" --ignore-eos --skip-chat-template \
  --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" \
  --percentile-metrics ttft --metric-percentiles 99 \
  >"$OUT_DIR/bench.log" 2>&1 || echo "[dsv4] bench failed" >&2
curl -sf -X POST "http://127.0.0.1:$PORT/stop_profile" >/dev/null || echo "[dsv4] stop_profile failed" >&2

for _ in $(seq 1 90); do
  n=$(find "$TRACE_DIR" -name '*.pt.trace.json*' 2>/dev/null | wc -l)
  [[ "$n" -ge 8 ]] && break
  sleep 5
done
echo "[dsv4] $(find "$TRACE_DIR" -name '*.pt.trace.json*' | wc -l) rank traces"
if ! grep -q "Mean TTFT" "$OUT_DIR/bench.log" 2>/dev/null; then
  echo "[dsv4] FATAL: the benchmark reported no TTFT, so no request was served." >&2
  tail -20 "$OUT_DIR/bench.log" >&2
  MEASURED_NOTHING=1
fi
if ! python3 "$HERE/check_trace_nonempty.py" "$TRACE_DIR"; then
  echo "[dsv4] FATAL: traces carry no step annotation; nothing to attribute." >&2
  MEASURED_NOTHING=1
fi
[[ -s "$DUMP" ]] && echo "[dsv4] $(wc -l < "$DUMP") forwards dumped" || echo "[dsv4] no dump (set NO_EPLB=1 if EPLB was rejected)" >&2

# Bounded teardown, then escalate. A bare `wait` after SIGTERM deadlocked a run once:
# the DP=8 server did not die, the remaining work never started, and nothing said why.
kill -TERM "$PID" 2>/dev/null || true
for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
kill -9 "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
sleep 5
for p in $(ps -eo pid,cmd --no-headers | grep "VLLM::" | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
sleep 10
if [[ "${MEASURED_NOTHING:-0}" == "1" ]]; then
  echo "[dsv4] MEASURED NOTHING - do not read these results" >&2; exit 5
fi
echo "[dsv4] done; $OUT_DIR"
