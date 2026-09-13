#!/usr/bin/env bash
# Supervise the GPU keep-alive: hold ~10% utilisation, and get out of the way instantly.
#
# The worker is killed, not throttled, whenever anything else touches the GPUs. vLLM
# sizes its KV cache from the free memory it sees at startup and this branch reads TTFT
# at a 0.3% resolution, so a spectator process holding a CUDA context is a measurement
# defect rather than a nuisance.
#
# Detection cannot compare pids we own against pids the driver reports: in a container
# nvidia-smi reports HOST pids, and the first version of this had the worker exclude its
# own `os.getpid()`, which excluded nothing -- it read its own eight contexts as an
# intruder and exited within a second, every time, holding 0% instead of 10%. So the set
# of compute pids present just after the worker settles is learned and treated as ours;
# anything outside that set is someone else.
#
# Stop with:  touch /tmp/keepalive.stop
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_BIN="${PY_BIN:-/mnt/yhong/vllm-predict/.venv/bin/python}"
TARGET="${TARGET_UTIL:-0.10}"
STOP_FILE="${STOP_FILE:-/tmp/keepalive.stop}"
BURST="${BURST_SECONDS:-600}"
LOG="$HERE/keepalive.log"

pids_now() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
             | tr -d ' ' | grep '^[0-9]\+$' | sort -u; }
say() { echo "[sup $(date +%F' '%T)] $*" >>"$LOG"; }

worker=""
cleanup() { [[ -n "$worker" ]] && kill "$worker" 2>/dev/null; wait "$worker" 2>/dev/null; }
trap 'cleanup; say "signalled; exiting"; exit 0' INT TERM

rm -f "$STOP_FILE"
say "starting, target ${TARGET}; stop with: touch $STOP_FILE"

while [[ ! -e "$STOP_FILE" ]]; do
  if [[ -f "$LOG" && $(stat -c%s "$LOG" 2>/dev/null || echo 0) -gt 2000000 ]]; then : >"$LOG"; fi

  if [[ -n "$(pids_now)" ]]; then
    say "GPUs in use by someone else; standing down"
    sleep 60
    continue
  fi

  "$PY_BIN" "$HERE/keepalive_gemm.py" --target-util "$TARGET" --seconds "$BURST" \
    >>"$LOG" 2>&1 &
  worker=$!
  # Let its eight contexts appear, then take that set as ours.
  sleep 25
  ours="$(pids_now)"
  say "worker $worker up; our driver pid(s): $(echo "$ours" | tr '\n' ' ')"

  while kill -0 "$worker" 2>/dev/null; do
    if [[ -e "$STOP_FILE" ]]; then break; fi
    foreign="$(comm -23 <(pids_now) <(echo "$ours"))"
    if [[ -n "$foreign" ]]; then
      say "foreign pid(s) $(echo "$foreign" | tr '\n' ' ') appeared; killing worker"
      break
    fi
    sleep 3
  done
  cleanup
  worker=""
  sleep 5
done

say "stop file seen; exiting"
