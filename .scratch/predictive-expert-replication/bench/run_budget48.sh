#!/usr/bin/env bash
# Re-measure the two headline configurations with a budget that cannot bind.
#
# 43 and 86 were the reachable-layer count under the OLD default lookahead of 2. Ticket 07
# moved the default to 1 on 2026-08-30 and reachable layers went 43 -> 44, but the arm
# constants in every driver stayed at 43. Steady state never approaches the cap (2.7-3.2
# activations per forward at cap 1), so no recorded figure is affected there -- but the
# FIRST forward wants all 44 at once and 43 could refuse one, which the 10-forward report
# granularity cannot resolve. 48 and 96 are above 44 and 88 by construction, so the
# question stops existing.
#
# This is also a free natural experiment on the cold start: if the first report window
# grows from 65 activations to ~66, the first forward really was capped at 43.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
say() { echo "[b48 $(date +%H:%M:%S)] $*"; }

say "A: 16k / CONC=8 -- the best measured configuration, budgets 48 and 96"
DOMAIN=ko PROMPT_LEN=16384 MAX_MODEL_LEN=18432 DP=8 NUM_PROMPTS=96 OUT_LEN=1 \
  REPEATS=6 CONC=8 BUDGETS="off 48:device:1 96:device:1:c2" \
  bash "$HERE/run_e2e_placement.sh" "$HERE/results/b48-16k-c8" \
  > "$HERE/results/b48-16k-c8.log" 2>&1 && say "A done" || say "A FAILED rc=$?"

say "B: 8k / CONC=16 -- the headline configuration, budgets 48 and 96"
DOMAIN=ko PROMPT_LEN=8192 MAX_MODEL_LEN=10240 DP=8 NUM_PROMPTS=128 OUT_LEN=1 \
  REPEATS=6 CONC=16 BUDGETS="off 48:device:1 96:device:1:c2" \
  bash "$HERE/run_e2e_placement.sh" "$HERE/results/b48-8k-c16" \
  > "$HERE/results/b48-8k-c16.log" 2>&1 && say "B done" || say "B FAILED rc=$?"

say "both done"
