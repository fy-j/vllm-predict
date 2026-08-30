#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Ticket 03's remaining acceptance criterion: with a statically placed replica
# active, generated output must match canonical-only execution.
#
# This is the externally meaningful contract. Source-rank routing sends some
# ranks' tokens to a second physical copy of one logical expert; if the logical
# semantics are preserved, the text must not change. Anything else means routing
# altered the model, which no placement policy is allowed to do.
#
# Runs the same prompts twice against the same build, once without a placement
# and once with one, at temperature 0, and diffs the completions.
set -euo pipefail

# MODE=device runs ticket 06's dynamic path instead of ticket 03's static placement: the
# plan is computed, transferred and published on the device, so what is compared is
# whatever the planner chose on each forward rather than one fixed replica. A static
# placement cannot exercise reversion, so neither replaces the other.
#
# MODE=host is the same dynamic placement through the host-issued transfer, and it exists
# to attribute a difference. Placement regroups each token's expert contributions across
# ranks, so the combine sums them in another order and BF16 output moves a little; a
# replica holding the *wrong bytes* also moves it. The two modes differ only in the
# transport, so running both says which one is happening.
MODE="${MODE:-static}"
OUT_DIR="${1:-$(dirname "$0")/results}"
MODEL="${MODEL:-/models/preset/Qwen/Qwen3-30B-A3B/v1.0}"
PROFILE="${PROFILE:-/tmp/predictive-cost-profile.json}"
PORT="${PORT:-8102}"
DP="${DP:-8}"
PLACEMENT="${PLACEMENT:-20:5}"   # expert 20 is owned by rank 1, so rank 5 is a real cross-rank replica
mkdir -p "$OUT_DIR"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# The venv, because `nvshmem.core` is installed only there and the device path falls back
# to the host one without it - a fallback that leaves this comparison measuring the host
# path while the log says the run was fine. It resolves the working tree either way.
PY="${PY:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3
loaded=$("$PY" -c "import vllm; print(vllm.__file__)" 2>/dev/null || true)
case "$loaded" in
  "$REPO_ROOT"/*) : ;;
  *) echo "[FATAL] vLLM resolves to '$loaded', not $REPO_ROOT." >&2; exit 4 ;;
esac
echo "[verify] mode=$MODE DP=$DP, vLLM from $loaded"

# Placement is suppressed below the block-quantization bar, so in device mode a short
# prompt is correctly ignored and the replica arm would be identical to the canonical one
# for the most boring possible reason. The filler is prepended to both arms, so the
# comparison stays like-for-like.
FILLER=""
if [[ "$MODE" != "static" ]]; then
  FILLER="$(printf 'The routing of tokens to experts is uneven across ranks. %.0s' $(seq 1 300))"
fi

PROMPTS=(
  "${FILLER}Explain in one paragraph why a hash table has amortized constant lookup."
  "${FILLER}Write a Python function that reverses a linked list in place."
  "${FILLER}What is the derivative of x^3 * ln(x)? Show the steps."
  "${FILLER}Summarize the tradeoffs between mutexes and lock-free queues."
)

run_variant() {
  # Separate statements: a single `local` expands its whole argument list before
  # assigning any of it, so a later value cannot reference an earlier one.
  local name="$1"
  local extra_json="$2"
  local log="$OUT_DIR/verify-$name.log"
  local out="$OUT_DIR/verify-$name.txt"
  echo "[verify] starting $name"
  VLLM_PREDICTIVE_PLACE_PER_FORWARD="${PLACE_PER_FORWARD:-0}" \
  VLLM_PREDICTIVE_PLACEMENT_REPORT_EVERY=4 \
  "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" \
    --data-parallel-size "$DP" --enable-expert-parallel \
    --all2all-backend allgather_reducescatter --enforce-eager \
    --max-model-len 4096 --gpu-memory-utilization 0.85 \
    --seed 0 --uvicorn-log-level warning \
    --additional-config "$extra_json" >"$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || { echo "[verify] $name died; see $log" >&2; tail -30 "$log" >&2; exit 1; }
    sleep 5
  done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[verify] $name never ready" >&2; exit 1; }

  : >"$out"
  local lp="$OUT_DIR/verify-$name.logprobs.jsonl"
  : >"$lp"
  for p in "${PROMPTS[@]}"; do
    curl -sf "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d "$(python3 -c "
import json,sys
print(json.dumps({'model':'$MODEL','prompt':sys.argv[1],'max_tokens':96,
                  'temperature':0,'seed':0}))" "$p")" \
      | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'])" >>"$out"
    echo "---" >>"$out"
    # One token with its top-k logprobs. This is the comparison the spec asks
    # for: routing to a second copy of the same weights regroups tokens inside
    # the expert GEMM, which perturbs reduction order, so the distribution must
    # be compared with a tolerance rather than demanded bit-identical.
    curl -sf "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d "$(python3 -c "
import json,sys
print(json.dumps({'model':'$MODEL','prompt':sys.argv[1],'max_tokens':1,
                  'temperature':0,'seed':0,'logprobs':10}))" "$p")" \
      | python3 -c "
import json,sys
d=json.load(sys.stdin)['choices'][0]['logprobs']
print(json.dumps(d['top_logprobs'][0]))" >>"$lp"
  done
  kill -TERM "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  # Workers can outlive the parent; wait for the GPUs to actually come back.
  for _ in $(seq 1 40); do
    [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && break
    sleep 3
  done
  echo "[verify] $name done -> $out"
}

BASE_JSON="{\"predictive_expert_replication\":{\"enabled\":true,\"cost_profile_path\":\"$PROFILE\"}}"
if [[ "$MODE" == "device" ]]; then
  REPL_JSON="{\"predictive_expert_replication\":{\"enabled\":true,\"cost_profile_path\":\"$PROFILE\",\"device_issued_transfer\":true}}"
elif [[ "$MODE" == "host" ]]; then
  REPL_JSON="{\"predictive_expert_replication\":{\"enabled\":true,\"cost_profile_path\":\"$PROFILE\"}}"
else
  REPL_JSON="{\"predictive_expert_replication\":{\"enabled\":true,\"cost_profile_path\":\"$PROFILE\",\"static_replica_placement\":\"$PLACEMENT\"}}"
fi

# CONTROL=1 runs the canonical configuration twice. Comparing generated text over
# many greedy tokens is an extremely sensitive equality test: two separate server
# processes can differ in NCCL reduction order, and greedy decoding amplifies a
# bit-level difference into different words. Without this control a mismatch
# between canonical and replica cannot be attributed to routing at all.
run_variant canonical "$BASE_JSON"
if [[ "${CONTROL:-0}" == "1" ]]; then
  run_variant replica "$BASE_JSON"
  echo "[verify] CONTROL run: both variants were canonical, so any difference below"
  echo "[verify] is run-to-run nondeterminism, not routing."
else
  PLACE_PER_FORWARD=$([[ "$MODE" == "static" ]] && echo 0 || echo 8)
  run_variant replica "$REPL_JSON"
fi

# In device mode nothing is placed unless the planner chose something, so the comparison
# is only meaningful if it did. Identical output from an arm that placed nothing says
# nothing at all, and that is the shape of half the false results in this project.
if [[ "$MODE" != "static" && "${CONTROL:-0}" != "1" ]]; then
  placed=$(grep -oE "activated [0-9]+ (device-issued )?replica" "$OUT_DIR/verify-replica.log" \
    | grep -oE "[0-9]+" | sort -n | tail -1)
  echo
  echo "[verify] replicas the device path actually placed: ${placed:-0}"
  if [[ "${placed:-0}" -lt 1 ]]; then
    echo "[verify] INCONCLUSIVE: the replica arm placed nothing, so it was the canonical" >&2
    echo "[verify] configuration under another name. Check the block bar and the log." >&2
    exit 1
  fi
fi

echo
echo "[verify] normalization lines:"
grep -h "normalized" "$OUT_DIR"/verify-canonical.log "$OUT_DIR"/verify-replica.log | sed 's/^.*] //' || true
echo
echo "[verify] greedy text equality (strict, informational only):"
if diff -q "$OUT_DIR/verify-canonical.txt" "$OUT_DIR/verify-replica.txt" >/dev/null; then
  echo "[verify]   identical"
else
  echo "[verify]   differs; greedy decoding amplifies any reduction-order change,"
  echo "[verify]   so this alone does not decide the criterion"
fi
echo
echo "[verify] first-token distribution within tolerance (the actual criterion):"
python3 "$(dirname "$0")/compare_logprobs.py" \
  "$OUT_DIR/verify-canonical.logprobs.jsonl" \
  "$OUT_DIR/verify-replica.logprobs.jsonl"
