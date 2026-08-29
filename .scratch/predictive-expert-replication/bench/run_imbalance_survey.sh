#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Re-derive every imbalance figure on mixed-domain traffic, at two chunked-prefill
# budgets. Everything quoted so far rests on 6 prefill forwards from a single
# domain, which cannot support a go/no-go decision.
#
# Design notes that are about correctness, not taste:
#
#  * Native EPLB recording, not predictive mode. This measures the *baseline*
#    imbalance the policy would act on, so the feature must not be in the loop.
#    Zero redundant experts and an unreachable rearrange interval mean placement
#    never changes.
#
#  * Two budgets. `max_num_batched_tokens` sets the post-allgather token count the
#    MoE kernel sees, and that is what decides whether imbalance is quantized
#    away. One budget gives one operating point and no curve.
#
#  * Three domains in one server per budget, dumped to separate files. Routing is
#    content dependent, and cross-request placement stability measured within one
#    domain overstates what mixed production traffic would give.
#
#  * Short decode (128) so most forwards are prefill: the prefill regime is what
#    the TTFT question turns on, and a 2048-token decode would bury it under
#    thousands of decode forwards.
#
# Usage: run_imbalance_survey.sh [out_dir]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
loaded=$(python3 -c "import vllm; print(vllm.__file__)" 2>/dev/null || true)
case "$loaded" in
  "$REPO_ROOT"/*) : ;;
  *) echo "[FATAL] vLLM resolves to '$loaded', not $REPO_ROOT." >&2; exit 4 ;;
esac
echo "[env] vLLM from $loaded"
python3 -c "
import sys, vllm.envs as e
sys.exit(0 if 'VLLM_EPLB_DUMP_LOAD_PATH' in e.environment_variables else 1)
" || { echo "[FATAL] this build has no load-dump variable." >&2; exit 4; }

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${1:-$HERE/results/imbalance-survey}"
MODEL="${MODEL:-/models/preset/Qwen/Qwen3-30B-A3B/v1.0}"
PORT="${PORT:-8170}"
BUDGETS="${BUDGETS:-2048 8192}"
NUM_PROMPTS="${NUM_PROMPTS:-120}"
CONC="${CONC:-64}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
PROMPT_LEN=2048
mkdir -p "$OUT_DIR"

LOCK="${TMPDIR:-/tmp}/predictive-baseline.lock"
exec 9>"$LOCK"
flock -w 10800 9 || { echo "[survey] lock not acquired" >&2; exit 3; }

declare -A DATASET=(
  [text]="Aeala/ShareGPT_Vicuna_unfiltered"
  [code]="likaixin/InstructCoder"
  [math]="AI-MO/NuminaMath-CoT"
)
for domain in text code math; do
  prompts="$HERE/results/prompts-$domain-p$PROMPT_LEN.jsonl"
  if [[ ! -s "$prompts" ]]; then
    echo "[survey] building ${PROMPT_LEN}-token $domain prompts"
    python3 "$HERE/make_prompts.py" \
      --dataset "${DATASET[$domain]}" --tokenizer "$MODEL" \
      --target-prompt-len "$PROMPT_LEN" --tolerance 0.10 \
      --num-prompts 256 --out "$prompts" \
      >"$OUT_DIR/prompts-$domain.log" 2>&1 || true
    [[ -s "$prompts" ]] || { echo "[survey] $domain prompts failed" >&2; exit 1; }
  fi
done

for budget in $BUDGETS; do
  # 3072 covers a 2048 prompt plus 128 decode with room to spare, and keeps the
  # KV ceiling comparable between budgets.
  echo "[survey] budget=$budget: starting server"
  LOG="$OUT_DIR/server-b$budget.log"
  DUMP_LIVE="$OUT_DIR/dump-live.jsonl"
  rm -f "$DUMP_LIVE"
  VLLM_EPLB_DUMP_LOAD_PATH="$DUMP_LIVE" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" \
    --data-parallel-size 8 --enable-expert-parallel \
    --all2all-backend allgather_reducescatter --enforce-eager \
    --max-model-len 3072 --gpu-memory-utilization 0.88 \
    --max-num-seqs 128 --max-num-batched-tokens "$budget" \
    --uvicorn-log-level warning \
    --enable-eplb \
    --eplb-config '{"num_redundant_experts":0,"step_interval":1000000000,"window_size":1000,"log_balancedness":true,"log_balancedness_interval":1,"use_async":false}' \
    >"$LOG" 2>&1 &
  PID=$!

  ready=0
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 "$PID" 2>/dev/null || break
    sleep 5
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "[survey] budget=$budget: never ready; skipping" >&2
    tail -30 "$LOG" >&2
    kill -9 "$PID" 2>/dev/null || true; sleep 5
    for p in $(ps -eo pid,cmd --no-headers | grep -E "VLLM::" | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
    sleep 8; continue
  fi
  echo "[survey] budget=$budget: ready"

  for domain in text code math; do
    tag="b$budget-$domain"
    echo "[survey] $tag: $NUM_PROMPTS requests"
    timeout 1800 vllm bench serve \
      --backend openai-chat --endpoint /v1/chat/completions \
      --model "$MODEL" --port "$PORT" \
      --dataset-name custom \
      --dataset-path "$HERE/results/prompts-$domain-p$PROMPT_LEN.jsonl" \
      --custom-output-len "$OUTPUT_LEN" --ignore-eos \
      --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" \
      --percentile-metrics ttft,tpot --metric-percentiles 99 \
      --save-result --result-dir "$OUT_DIR" --result-filename "bench-$tag.json" \
      >"$OUT_DIR/bench-$tag.log" 2>&1 || echo "[survey] $tag bench FAILED" >&2
    # Rotate: the dump path is fixed at launch, so the file separates the domains.
    if [[ -s "$DUMP_LIVE" ]]; then
      mv "$DUMP_LIVE" "$OUT_DIR/dump-$tag.jsonl"
      echo "[survey] $tag: $(wc -l < "$OUT_DIR/dump-$tag.jsonl") forwards dumped"
    else
      echo "[survey] $tag: EMPTY DUMP — aborting, every later point fails the same" >&2
      kill -9 "$PID" 2>/dev/null || true
      exit 5
    fi
  done

  kill -TERM "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true
  sleep 5
  for p in $(ps -eo pid,cmd --no-headers | grep -E "VLLM::" | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
  sleep 10
done

echo "[survey] done; dumps in $OUT_DIR"
