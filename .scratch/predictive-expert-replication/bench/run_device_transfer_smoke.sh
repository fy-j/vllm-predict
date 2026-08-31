#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Ticket 06: does a real server survive the device-issued transfer?
#
# The three device-side pieces each verify against the host path they replace, and the
# transfer moves 112/112 weight tensors byte-identically over every rank pair - and the
# first server that enabled it lost three workers to a segfault in NVSHMEM's proxy
# thread. Isolated checks cannot catch that: they hold the plan in a local variable and
# synchronise straight after launching, where a forward drops it and keeps allocating.
#
# So this is the cheapest thing that exercises the path the way the engine does. Dummy
# weights, because the crash was at the startup EPLB rearrange and 57 GiB of real weights
# would not make it more likely - output equivalence is a separate run with real ones.
#
# Usage: run_device_transfer_smoke.sh [out_dir]
#   DP=<n>      ranks to run, default every visible GPU
#   REAL=1      load real weights instead of dummy
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# `.venv/bin/python`, not system python3: `nvshmem.core` is installed only in the venv,
# and without it the device path falls back to the host one with a warning - which is a
# run that looks healthy and measures the cost it was meant to remove.
PY="$REPO_ROOT/.venv/bin/python"
loaded=$("$PY" -c "import vllm; print(vllm.__file__)" 2>/dev/null || true)
case "$loaded" in
  "$REPO_ROOT"/*) : ;;
  *) echo "[FATAL] vLLM resolves to '$loaded', not $REPO_ROOT." >&2; exit 4 ;;
esac
"$PY" -c "import nvshmem.core" 2>/dev/null || {
  echo "[FATAL] nvshmem.core is missing, so the device path cannot arm." >&2; exit 4; }
echo "[env] vLLM from $loaded"

OUT_DIR="${1:-$(dirname "$0")/results/device-transfer-smoke}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-/models/preset/Qwen/Qwen3-30B-A3B/v1.0}"
PORT="${PORT:-8170}"
DP="${DP:-$(nvidia-smi --list-gpus | wc -l)}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/server.log"

LOCK="${TMPDIR:-/tmp}/predictive-baseline.lock"
exec 9>"$LOCK"
flock -w 10800 9 || { echo "[smoke] lock not acquired" >&2; exit 3; }

# The fingerprint pins the EP size and the device, so a profile measured at EP=8 is
# rejected here by design rather than silently accepted.
PROFILE="$OUT_DIR/cost-profile-local.json"
"$PY" - "$HERE/results/cost-profile.json" "$PROFILE" "$MODEL" "$DP" <<'PY'
import json, subprocess, sys
src, dst, model, ep = sys.argv[1:5]
profile = json.load(open(src))
device = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).splitlines()[0].strip()
profile["fingerprint"].update(model=model, ep_size=int(ep), device_name=device)
json.dump(profile, open(dst, "w"), indent=2)
print(f"[smoke] cost profile: ep_size={ep}, device={device}")
PY

ADDITIONAL=$("$PY" -c "
import json,sys
cfg={'enabled': True, 'cost_profile_path': sys.argv[1],
     'device_issued_transfer': True}
if int(sys.argv[2]) > 0:
    group = int(sys.argv[2])
    cfg['prediction_target_group'] = group
    cfg['prediction_lookahead_layers'] = group
print(json.dumps({'predictive_expert_replication': cfg}))" "$PROFILE" "${PRED_GROUP:-0}")

LOAD=(--load-format dummy)
[[ "${REAL:-0}" == "1" ]] && LOAD=()

echo "[smoke] starting a $DP-rank server with the device-issued transfer armed"
# The placement counter is the only cheap evidence that a replica was placed rather than
# merely planned: "transfer launched for layer" is logged whether or not the plan found
# anything. Every 4 forwards here, against 50 in a benchmark.
VLLM_PREDICTIVE_PLACEMENT_REPORT_EVERY=4 \
VLLM_PREDICTIVE_PLACE_PER_FORWARD=8 \
"$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port "$PORT" \
  --data-parallel-size "$DP" --enable-expert-parallel \
  --all2all-backend allgather_reducescatter --enforce-eager \
  --max-model-len 3072 --gpu-memory-utilization 0.88 \
  --max-num-seqs 64 --seed 0 --uvicorn-log-level warning \
  "${LOAD[@]}" \
  --additional-config "$ADDITIONAL" \
  --eplb-config '{"log_balancedness":true,"log_balancedness_interval":50,"step_interval":1000000000,"window_size":1000,"use_async":false}' \
  >"$LOG" 2>&1 &
PID=$!
trap 'kill -TERM "$PID" 2>/dev/null || true' EXIT

ready=0
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 "$PID" 2>/dev/null || break
  sleep 5
done

status=0
if [[ "$ready" -ne 1 ]]; then
  echo "[smoke] FAIL: never became ready" >&2
  grep -iE "segfault|Segmentation|signal|proxy_progress|Traceback|Error" "$LOG" | tail -20 >&2
  status=1
else
  # Long enough to clear the block-quantization bar, which is what arms placement at all:
  # tokens per expert is `M x topk / logical experts`, so 128 needs more than 2048 tokens
  # summed across DP ranks. A short prompt is correctly suppressed and would leave this
  # test measuring nothing.
  PROMPT="$(printf 'balance the experts across every rank %.0s' $(seq 1 400))"
  echo "[smoke] ready; sending prefill-shaped requests over the block bar"
  for i in $(seq 1 12); do
    curl -sf "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\": \"$MODEL\", \"prompt\": \"$PROMPT\", \"max_tokens\": 4, \"temperature\": 0}" \
      >"$OUT_DIR/request-$i.json" 2>&1 || { echo "[smoke] request $i failed" >&2; status=1; }
  done
fi

kill -TERM "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true
sleep 5

echo
# A green run is not a connected one: five defects on this branch each left the feature
# inert while everything looked healthy, so the evidence is the log lines.
armed=$(grep -c "device-issued transfer active" "$LOG" || true)
launched=$(grep -c "device-issued transfer launched for layer" "$LOG" || true)
fellback=$(grep -c "falling back to the host-issued transfer" "$LOG" || true)
crashed=$(grep -ciE "segmentation fault|proxy_progress|died with .signal" "$LOG" || true)

placed=$(grep -o "activated [0-9]* device-issued replica" "$LOG" | grep -o "[0-9]*" | sort -n | tail -1)
placed="${placed:-0}"

echo "[smoke] device path armed on $armed worker(s), launched on $launched layer(s)"
echo "[smoke] replicas actually placed (device counter): $placed"
echo "[smoke] host-path fallbacks: $fellback   crash signatures: $crashed"
[[ "$armed" -ge 1 ]] || { echo "[smoke] FAIL: the device path never armed" >&2; status=1; }
[[ "$launched" -ge 1 ]] || { echo "[smoke] FAIL: no layer launched a transfer" >&2; status=1; }
[[ "$placed" -ge 1 ]] || {
  echo "[smoke] FAIL: nothing was placed, so the path ran and did nothing" >&2
  echo "        either every forward was suppressed below the block bar, or the" >&2
  echo "        planner refused every candidate; check the prompt length first." >&2
  status=1; }
[[ "$fellback" -eq 0 ]] || { echo "[smoke] FAIL: fell back to the host path" >&2; status=1; }
[[ "$crashed" -eq 0 ]] || { echo "[smoke] FAIL: a worker crashed" >&2; status=1; }
if [[ "$status" -eq 0 ]]; then echo "[smoke] PASS"; else echo "[smoke] see $LOG" >&2; fi
exit "$status"
