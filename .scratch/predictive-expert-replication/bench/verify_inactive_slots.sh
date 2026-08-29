#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Ticket 03, final criterion: observe that an inactive replica slot attracts no
# routed tokens.
#
# The check itself lives in EplbState and raises on the forward that violates it,
# because a token routed to an unwritten slot reads uninitialised expert weights
# and yields plausible-looking output rather than an error. This script runs real
# requests with it enabled and reports the log line that says how many slots were
# examined - a silent pass is not evidence.
#
# Usage: verify_inactive_slots.sh [out_dir]
set -euo pipefail

# See run_prediction_accuracy.sh: the `vllm` console script never puts the working
# tree on sys.path, so a bare run would measure the installed build instead.
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
loaded=$(python3 -c "import vllm; print(vllm.__file__)" 2>/dev/null || true)
case "$loaded" in
  "$REPO_ROOT"/*) : ;;
  *) echo "[FATAL] vLLM resolves to '$loaded', not $REPO_ROOT." >&2; exit 4 ;;
esac
echo "[env] vLLM from $loaded"

OUT_DIR="${1:-$(dirname "$0")/results/inactive-slots}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-/models/preset/Qwen/Qwen3-30B-A3B/v1.0}"
PORT="${PORT:-8160}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/server.log"

LOCK="${TMPDIR:-/tmp}/predictive-baseline.lock"
exec 9>"$LOCK"
flock -w 10800 9 || { echo "[verify] lock not acquired" >&2; exit 3; }

PROFILE="$OUT_DIR/cost-profile-local.json"
python3 - "$HERE/results/cost-profile.json" "$PROFILE" "$MODEL" <<'PY'
import json, sys
src, dst, model = sys.argv[1:4]
profile = json.load(open(src))
profile["fingerprint"]["model"] = model
json.dump(profile, open(dst, "w"), indent=2)
PY

PROMPTS="$HERE/results/prompts-code-p1024.jsonl"
[[ -s "$PROMPTS" ]] || { echo "[verify] missing $PROMPTS" >&2; exit 1; }

ADDITIONAL_CFG=$(python3 -c "
import json,sys
print(json.dumps({'predictive_expert_replication': {
  'enabled': True, 'cost_profile_path': sys.argv[1],
}}))" "$PROFILE")

echo "[verify] starting server with the inactive-slot check armed"
# Single API server: `vllm serve` starts one per DP rank and the config round trip
# loses enable_eplb while keeping num_redundant_experts.
VLLM_PREDICTIVE_VERIFY_INACTIVE_SLOTS=1 \
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port "$PORT" \
  --data-parallel-size 8 --enable-expert-parallel \
  --all2all-backend allgather_reducescatter --enforce-eager \
  --max-model-len 3072 --gpu-memory-utilization 0.88 \
  --max-num-seqs 64 --uvicorn-log-level warning \
  --additional-config "$ADDITIONAL_CFG" \
  --eplb-config '{"log_balancedness":true,"log_balancedness_interval":50,"step_interval":1000000000,"window_size":1000,"use_async":false}' \
  >"$LOG" 2>&1 &
PID=$!
trap 'kill -TERM "$PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$PID" 2>/dev/null || { echo "[verify] server died" >&2; tail -40 "$LOG" >&2; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[verify] never ready" >&2; exit 1; }
echo "[verify] ready; sending requests"

# Bounded: a diagnostic that deadlocks the engine leaves the client waiting
# forever, which is how one run silently consumed the node.
timeout 600 vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --model "$MODEL" --port "$PORT" \
  --dataset-name custom --dataset-path "$PROMPTS" \
  --custom-output-len 64 --ignore-eos \
  --num-prompts 32 --max-concurrency 32 \
  --percentile-metrics ttft,tpot --metric-percentiles 99 \
  >"$OUT_DIR/bench.log" 2>&1 || echo "[verify] bench failed or timed out" >&2

kill -TERM "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true
sleep 5

echo
if grep -q "inactive physical slots carry no routed load" "$LOG"; then
  echo "[verify] PASS"
  grep -o "verified [0-9]* inactive physical slots carry no routed load" "$LOG" | head -1
elif grep -q "inactive physical slots received" "$LOG"; then
  echo "[verify] FAIL: tokens reached an inactive slot" >&2
  grep -A2 "inactive physical slots received" "$LOG" | head -6 >&2
  exit 1
elif grep -q "has no inactive slot to observe" "$LOG"; then
  echo "[verify] INCONCLUSIVE: the layout marked no slot inactive" >&2
  exit 1
else
  echo "[verify] INCONCLUSIVE: the check never reported; see $LOG" >&2
  exit 1
fi
