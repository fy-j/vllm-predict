#!/usr/bin/env bash
# Which dataset gives a placement the most to recover, per layer.
#
# Routing is content dependent, so the imbalance a placement can remove depends on the
# workload. The three domains measured so far span 1.28x to 1.58x, and the *least*
# imbalanced was code — the broadest token distribution (identifiers, punctuation,
# English comments). That suggests narrower registers skew more, so the candidates here
# are all narrower than math: a non-Latin script, formal proof syntax, and SQL.
#
# One server per domain rather than one server for all: the dump path is read at
# startup, so rotating it would need a restart anyway, and mixing domains into one dump
# would leave no way to attribute a forward to its dataset.
#
# Prefill-heavy on purpose (short decode, moderate concurrency): the prefill regime is
# the one where imbalance costs anything, and a 128-token decode yields only a handful
# of prefill forwards out of hundreds.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
OUT_DIR="${1:-$HERE/results/domain-search}"
MODEL="${MODEL:-/models/preset/Qwen/Qwen3-30B-A3B/v1.0}"
PORT="${PORT:-8182}"
CONC="${CONC:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
PROMPT_LEN="${PROMPT_LEN:-1024}"

export PYTHONPATH="$REPO_ROOT"
RESOLVED=$(python3 -c 'import vllm,sys; sys.stdout.write(vllm.__file__)' 2>/dev/null || true)
case "$RESOLVED" in
  "$REPO_ROOT"/vllm/*) echo "[search] vLLM from $RESOLVED" ;;
  *) echo "[search] ABORT: vLLM resolves to '$RESOLVED', not this tree" >&2; exit 1 ;;
esac
mkdir -p "$OUT_DIR"

declare -A DATASET=(
  [sql]=b-mc2/sql-create-context
  [zh]=shibing624/alpaca-zh
  [ja]=kunishou/databricks-dolly-15k-ja
  [ko]=beomi/KoAlpaca-v1.1a
  [lean]=internlm/Lean-Workbook
)
# code, math and text already have prompt sets at this length and are the reference
# points the new candidates have to beat.
EXISTING="code math text"
NEW="${NEW:-sql zh ja ko lean}"

for domain in $NEW; do
  prompts="$HERE/results/prompts-$domain-p$PROMPT_LEN.jsonl"
  if [[ ! -s "$prompts" ]]; then
    echo "[search] building $PROMPT_LEN-token $domain prompts from ${DATASET[$domain]}"
    python3 "$HERE/make_prompts.py" \
      --dataset "${DATASET[$domain]}" --tokenizer "$MODEL" \
      --target-prompt-len "$PROMPT_LEN" --tolerance 0.10 \
      --num-prompts 256 --out "$prompts" \
      >"$OUT_DIR/prompts-$domain.log" 2>&1 || true
    [[ -s "$prompts" ]] || { echo "[search] $domain prompts FAILED, skipping" >&2; continue; }
  fi
done

LOCK="${TMPDIR:-/tmp}/predictive-baseline.lock"
exec 9>"$LOCK"
flock -w 10800 9 || { echo "[search] lock not acquired" >&2; exit 3; }

PROFILE="$OUT_DIR/cost-profile-local.json"
python3 - "$HERE/results/cost-profile.json" "$PROFILE" "$MODEL" <<'PY'
import json, sys
src, dst, model = sys.argv[1:4]
p = json.load(open(src)); p["fingerprint"]["model"] = model
json.dump(p, open(dst, "w"), indent=2)
PY

for domain in $EXISTING $NEW; do
  prompts="$HERE/results/prompts-$domain-p$PROMPT_LEN.jsonl"
  [[ -s "$prompts" ]] || { echo "[search] $domain: no prompt set, skipping" >&2; continue; }
  DUMP="$OUT_DIR/dump-$domain.jsonl"
  LOG="$OUT_DIR/server-$domain.log"
  rm -f "$DUMP"
  echo "[search] $domain: starting server"

  VLLM_EPLB_DUMP_LOAD_PATH="$DUMP" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" \
    --data-parallel-size 8 --enable-expert-parallel \
    --all2all-backend allgather_reducescatter --enforce-eager \
    --max-model-len 3072 --gpu-memory-utilization 0.88 \
    --max-num-seqs 64 --seed 0 --uvicorn-log-level warning \
    --additional-config "{\"predictive_expert_replication\": {\"enabled\": true, \"cost_profile_path\": \"$PROFILE\"}}" \
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
    echo "[search] $domain: server never ready" >&2; tail -20 "$LOG" >&2
    kill -9 "$PID" 2>/dev/null; continue
  fi

  timeout 1800 vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --model "$MODEL" --port "$PORT" \
    --dataset-name custom --dataset-path "$prompts" \
    --custom-output-len 1 --ignore-eos \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" \
    >"$OUT_DIR/bench-$domain.log" 2>&1 || echo "[search] $domain bench FAILED" >&2

  [[ -s "$DUMP" ]] && echo "[search] $domain: $(wc -l < "$DUMP") forwards dumped" \
                   || echo "[search] $domain: EMPTY DUMP" >&2

  kill -TERM "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; sleep 5
  for p in $(ps -eo pid,cmd --no-headers | grep "VLLM::" | grep -v grep | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null
  done
  sleep 8
done

echo "[search] ranking"
python3 "$HERE/report_domain_imbalance.py" "$OUT_DIR"/dump-*.jsonl \
  | tee "$OUT_DIR/ranking.txt"
echo "[search] done; results in $OUT_DIR"
