#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# First end-to-end run: plan from measured load, transfer, activate, and measure
# what changed. Two servers with identical settings except the transfer budget, so
# the comparison isolates the placement path.
#
# What this measures and what it does not: on this node the expected payoff is
# about 3.3% of a prefill step, while the interconnect term is 119 ms with a 2.8x
# cross-rank spread — so end-to-end TTFT cannot resolve it. The reported quantity is
# the **per-layer rank imbalance the placement path achieves**, read from the load
# dump, plus correctness (output equivalence, inactive slots idle).
#
# Usage: run_e2e_placement.sh [out_dir]
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
need = {'VLLM_PREDICTIVE_PLACE_PER_FORWARD', 'VLLM_EPLB_DUMP_LOAD_PATH'}
sys.exit(0 if need <= set(e.environment_variables) else 1)
" || { echo "[FATAL] this build lacks the placement or dump variable." >&2; exit 4; }

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${1:-$HERE/results/e2e-placement}"
MODEL="${MODEL:-/models/preset/Qwen/Qwen3-30B-A3B/v1.0}"
PORT="${PORT:-8180}"
BUDGETS="${BUDGETS:-0 43}"
NUM_PROMPTS="${NUM_PROMPTS:-120}"
CONC="${CONC:-64}"
# Decode length. Short values raise the share of forwards that are prefill, which is
# what the prefill figures need: at the default, 120 requests at concurrency 64 yield
# only 8 full prefill forwards, so that column rests on 8 samples.
OUT_LEN="${OUT_LEN:-128}"
mkdir -p "$OUT_DIR"

LOCK="${TMPDIR:-/tmp}/predictive-baseline.lock"
exec 9>"$LOCK"
flock -w 10800 9 || { echo "[e2e] lock not acquired" >&2; exit 3; }

PROFILE="$OUT_DIR/cost-profile-local.json"
python3 - "$HERE/results/cost-profile.json" "$PROFILE" "$MODEL" <<'PY'
import json, sys
src, dst, model = sys.argv[1:4]
p = json.load(open(src)); p["fingerprint"]["model"] = model
json.dump(p, open(dst, "w"), indent=2)
PY

# Which domain's prompts. Routing is content dependent and the domains differ widely
# in how much a placement can recover, measured per layer over 12 prefill forwards
# each: ko 89.6% of excess, zh 77.8%, ja 70.9%, lean 62.1%, code 59.7%, sql 59.5%,
# math 50.9%, text 32.6%. Non-Latin scripts lead by a wide margin — Qwen3's experts
# specialise by language — while narrow formal registers (sql, lean) came out level
# with code, so the intuition that a narrow vocabulary skews more did not hold.
# `results/domain-search/ranking.txt` carries the table.
DOMAIN="${DOMAIN:-code}"
PROMPTS="$HERE/results/prompts-$DOMAIN-p1024.jsonl"
[[ -s "$PROMPTS" ]] || { echo "[e2e] missing $PROMPTS" >&2; exit 1; }
echo "[e2e] domain=$DOMAIN"

for budget in $BUDGETS; do
  tag="b$budget"
  echo "[e2e] budget=$budget: starting server"
  LOG="$OUT_DIR/server-$tag.log"
  DUMP="$OUT_DIR/dump-$tag.jsonl"
  rm -f "$DUMP"
  # The env var only arms the path; the budget is `max_transfers_per_forward`, per the
  # spec. Its default of 4 was sized for decode and caps the ceiling at 1.0% of a
  # prefill step, so the placed arm asks for one per reachable layer.
  ADDITIONAL=$(python3 -c "
import json,sys
cfg={'enabled': True, 'cost_profile_path': sys.argv[1]}
if int(sys.argv[2]) > 0:
    cfg['max_transfers_per_forward'] = int(sys.argv[2])
print(json.dumps({'predictive_expert_replication': cfg}))" "$PROFILE" "$budget")

  VLLM_PREDICTIVE_PLACE_PER_FORWARD="$budget" \
  VLLM_EPLB_DUMP_LOAD_PATH="$DUMP" \
  VLLM_PREDICTIVE_VERIFY_INACTIVE_SLOTS=0 \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" \
    --data-parallel-size 8 --enable-expert-parallel \
    --all2all-backend allgather_reducescatter --enforce-eager \
    --max-model-len 3072 --gpu-memory-utilization 0.88 \
    --max-num-seqs 64 --seed 0 --uvicorn-log-level warning \
    --additional-config "$ADDITIONAL" \
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
    echo "[e2e] budget=$budget: never ready" >&2; tail -40 "$LOG" >&2
    kill -9 "$PID" 2>/dev/null || true; sleep 5
    for p in $(ps -eo pid,cmd --no-headers | grep "VLLM::" | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
    sleep 8; continue
  fi
  echo "[e2e] budget=$budget: ready"

  timeout 1800 vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --model "$MODEL" --port "$PORT" \
    --dataset-name custom --dataset-path "$PROMPTS" \
    --custom-output-len "$OUT_LEN" --ignore-eos \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" \
    --percentile-metrics ttft,tpot --metric-percentiles 99 \
    --save-result --result-dir "$OUT_DIR" --result-filename "bench-$tag.json" \
    >"$OUT_DIR/bench-$tag.log" 2>&1 || echo "[e2e] $tag bench FAILED" >&2

  # This prefix must stay identical to `_PLACEMENT_LINE` in analyse_e2e.py. An
  # earlier version matched "activated .* replicas" against a line that says
  # "replica(s)", so a fully working placed arm reported "no placement log line"
  # and blamed a budget of 0 regardless of the arm. Absence of this line is the
  # only cheap signal that an arm was silently inert, so a wrong pattern here
  # turns the one self-check into a false all-clear.
  placement_lines=$(grep -c "Predictive expert replication: activated" "$LOG" 2>/dev/null || true)
  placement_lines=${placement_lines:-0}
  if [[ "$placement_lines" -gt 0 ]]; then
    echo "[e2e] $tag: $placement_lines placement log lines"
    if [[ "$budget" -eq 0 ]]; then
      echo "[e2e] $tag: UNEXPECTED - budget 0 placed replicas" >&2
    fi
  elif [[ "$budget" -eq 0 ]]; then
    echo "[e2e] $tag: no placement log line (correct for budget 0)"
  else
    echo "[e2e] $tag: NO PLACEMENT LOG LINE at budget $budget - arm was inert" >&2
  fi
  [[ -s "$DUMP" ]] && echo "[e2e] $tag: $(wc -l < "$DUMP") forwards dumped" \
                   || echo "[e2e] $tag: EMPTY DUMP" >&2

  kill -TERM "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; sleep 5
  for p in $(ps -eo pid,cmd --no-headers | grep "VLLM::" | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
  sleep 10
done
echo "[e2e] done; results in $OUT_DIR"
