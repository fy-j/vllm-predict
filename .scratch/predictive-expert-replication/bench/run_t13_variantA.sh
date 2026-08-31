#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Ticket 13, second design, on hardware. The first one was refuted here, not in a test:
# 566 unit tests passed while placement's benefit fell from 26.9% of critical-path excess
# to 3.7%, because four layers' gates on one layer's hidden states select almost the same
# experts. So the question this run answers is the only one that matters — does a window
# of *different* source layers keep the benefit while cutting the collectives?
#
# Arms are interleaved within each pass and carry their own group, because comparing
# across driver invocations charges the difference to whatever the machine did in between
# and that drift has measured 6.1% where the effect is a few points.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/results"
GROUP="${GROUP:-4}"

echo "###### stage 1: does a window of $GROUP serve, and is coverage intact"
env REAL=1 PRED_GROUP="$GROUP" \
  bash "$HERE/run_device_transfer_smoke.sh" "$OUT/t13a-g${GROUP}-smoke"
echo "###### rc=$?"

echo "###### stage 2: excess and TTFT, four interleaved arms, three passes"
env DOMAIN=ko DP=8 REPEATS=3 NUM_PROMPTS=512 CONC=16 OUT_LEN=1 \
  BUDGETS="off 0:device:1 43:device:1 43:device:${GROUP}" \
  bash "$HERE/run_e2e_placement.sh" "$OUT/t13a-g${GROUP}-ruler"
echo "###### rc=$?"
