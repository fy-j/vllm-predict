#!/usr/bin/env bash
# Find the highest concurrency that does not queue, on one server, one arm.
#
# Every TTFT figure in this project was taken at concurrency 8, and that is the wrong point
# in both directions. It is too low: 8 requests over DP=8 is one per rank, and half the ranks
# then spend 93% of a prefill window waiting in a collective for peers that have nothing to
# do. And it already queues: the stock arm's p99 over median is 2.15, and the placed arm's is
# 16.3, so mean TTFT there is mostly time spent behind other requests — which no amount of
# expert balancing can shorten.
#
# What the feature can shorten is one prefill step. So the operating point to measure at is
# the knee: the largest concurrency at which TTFT is still flat, because below it TTFT is the
# step time and above it TTFT is the queue. Theory puts it near 64 here — a 958-token prompt
# means a rank fits 8 of them in one 8192-token forward, times 8 ranks — and at that point
# each expert sees 3832 tokens, far above the block-quantization bar. Theory is not
# measurement, hence this sweep.
#
# One server, several benchmarks against it, so the comparison across concurrencies carries
# no server-restart drift. Read the output as a latency-versus-concurrency curve: flat, then
# a knee, then linear.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
OUT_DIR="${1:-$HERE/results/h100/knee}"
MODEL="${MODEL:-/models/preset/Qwen/Qwen3-30B-A3B/v1.0}"
PORT="${PORT:-8183}"
DOMAIN="${DOMAIN:-ko}"
LEVELS="${LEVELS:-1 4 8 16 32 48 64 96 128}"
# Enough prompts that even the largest level runs several waves, so a single slow first
# request cannot set the mean.
NUM_PROMPTS="${NUM_PROMPTS:-512}"
OUT_LEN="${OUT_LEN:-1}"

export PYTHONPATH="$REPO_ROOT"
RESOLVED=$(python3 -c 'import vllm,sys; sys.stdout.write(vllm.__file__)' 2>/dev/null || true)
case "$RESOLVED" in
  "$REPO_ROOT"/vllm/*) echo "[knee] vLLM from $RESOLVED" ;;
  *) echo "[knee] ABORT: vLLM resolves to '$RESOLVED', not this tree" >&2; exit 1 ;;
esac

PROMPTS="$HERE/results/prompts-$DOMAIN-p1024.jsonl"
[[ -s "$PROMPTS" ]] || { echo "[knee] missing $PROMPTS" >&2; exit 1; }

mkdir -p "$OUT_DIR"
LOCK="${TMPDIR:-/tmp}/predictive-baseline.lock"
exec 9>"$LOCK"
flock -w 10800 9 || { echo "[knee] lock not acquired" >&2; exit 3; }

LOG="$OUT_DIR/server.log"
echo "[knee] starting a stock server; the knee is a property of the baseline"
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port "$PORT" \
  --data-parallel-size 8 --enable-expert-parallel \
  --all2all-backend allgather_reducescatter --enforce-eager \
  --max-model-len 3072 --gpu-memory-utilization 0.88 \
  --max-num-seqs 64 --seed 0 --uvicorn-log-level warning \
  >"$LOG" 2>&1 &
PID=$!

ready=0
for _ in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 "$PID" 2>/dev/null || break
  sleep 5
done
if [[ "$ready" -ne 1 ]]; then
  echo "[knee] server never ready" >&2; tail -40 "$LOG" >&2
  kill -9 "$PID" 2>/dev/null; exit 4
fi
echo "[knee] ready"

# Warm up once, outside every measurement, so the first level does not carry the JIT cost of
# the whole sweep.
vllm bench serve --backend openai --endpoint /v1/completions \
  --model "$MODEL" --port "$PORT" --dataset-name custom --dataset-path "$PROMPTS" \
  --custom-output-len "$OUT_LEN" --ignore-eos --skip-chat-template \
  --num-prompts 32 --max-concurrency 8 >"$OUT_DIR/warmup.log" 2>&1 \
  || echo "[knee] warmup failed" >&2

for conc in $LEVELS; do
  prompts=$NUM_PROMPTS
  # At least four waves at every level, so the mean is not one wave's tail.
  if [[ $((conc * 4)) -gt $prompts ]]; then prompts=$((conc * 4)); fi
  echo "[knee] concurrency $conc, $prompts prompts"
  vllm bench serve --backend openai --endpoint /v1/completions \
    --model "$MODEL" --port "$PORT" --dataset-name custom --dataset-path "$PROMPTS" \
    --custom-output-len "$OUT_LEN" --ignore-eos --skip-chat-template \
    --num-prompts "$prompts" --max-concurrency "$conc" \
    --percentile-metrics ttft --metric-percentiles 99 \
    >"$OUT_DIR/bench-c$conc.log" 2>&1 || echo "[knee] level $conc failed" >&2
  grep -E "Mean TTFT|Median TTFT|P99 TTFT|Request throughput" "$OUT_DIR/bench-c$conc.log" \
    | tr -s ' ' | sed "s/^/  c$conc /" || true
done

kill -TERM "$PID" 2>/dev/null || true
for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
kill -9 "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
sleep 5
for p in $(ps -eo pid,ppid,cmd --no-headers | awk '/VLLM::/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
sleep 10
echo "[knee] done; $OUT_DIR"
