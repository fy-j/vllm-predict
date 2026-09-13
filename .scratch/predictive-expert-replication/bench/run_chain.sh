#!/usr/bin/env bash
# Start the sweep when the backlog finishes, without editing the running backlog script.
#
# Editing a bash script that is executing corrupts it -- bash reads by byte offset, so inserted
# lines make the running shell resume mid-token. That already killed one run's guard here. So
# the queue is extended by chaining a separate file rather than by appending to the live one.
#
# The backlog ends by entering a keep-alive that never exits on its own. This waits for the
# drain line, stops the keep-alive, waits for the process to actually go, then starts the sweep.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
LOG="$HERE/results/backlog2.log"
STOP="$HERE/results/backlog.stop"
say() { echo "[chain $(date +%H:%M:%S)] $*"; }

say "waiting for the backlog to drain"
until grep -q "backlog drained" "$LOG" 2>/dev/null; do
  [[ -e "$STOP" ]] && { say "stop file present before drain, exiting"; exit 0; }
  sleep 60
done
say "backlog drained; stopping its keep-alive"
touch "$STOP"
for _ in $(seq 1 60); do pgrep -f "run_backlog.sh" >/dev/null || break; sleep 5; done
pkill -f "keep-alive" 2>/dev/null
sleep 10
rm -f "$STOP"
say "starting the sweep"
exec bash "$HERE/run_sweep8k.sh"
