#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# The two confirmations ticket 11's first result was reported without.
#
# 1. A profile of the probe arm. The isolating run said removing the 44 snapshot
#    AllGathers gives back 11.04 ms of 23.21, but nothing on hardware confirmed the
#    collectives actually went from 236 per window to 192 — only a unit test that no
#    collective is issued. A profile makes it direct.
#
# 2. Prediction accuracy at lookahead 4, which every batching scheme needs and which has
#    never been measured. 1 to 3 are re-run on the same domain in the same session so the
#    curve is self-consistent, and because 1 and 2 have known values on `ko` (recall@1
#    0.770 and 0.713) that this run should reproduce — a curve that fails to is measuring
#    something else.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/results"

echo "###### profile: prediction-only with the AllGather removed"
env DOMAIN=ko DP=8 BUDGETS="0" NUM_PROMPTS=128 CONC=16 OUT_LEN=4 \
  VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER=1 \
  bash "$HERE/run_placement_profile.sh" "$OUT/t11-probe-profile"
echo "###### rc=$?"

echo "###### accuracy curve on ko, lookahead 1 through 4"
env DOMAINS=ko DP=8 LOOKAHEADS="1 2 3 4" \
  bash "$HERE/run_prediction_accuracy.sh" "$OUT/t11-accuracy-L1234"
echo "###### rc=$?"
