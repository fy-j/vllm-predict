#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Ticket 13: the target group raised to 4, and what it buys.
#
# Ticket 11 measured prediction's +13.68% mean TTFT as 11.04 ms of barrier coupling over
# 44 per-layer collectives plus 12.17 ms of launches and compute. At a group of 4 the
# collectives fall to 11 and only every fourth layer predicts, so both halves should move.
# Predicted landing point about +4.8%, which is break-even against a 5.26% ceiling.
#
# Three stages, correctness first: a K=4 smoke, then the collective count, then TTFT.
# One driver so the tree is provably identical across all of them — editing the tree
# mid-run has already invalidated one measurement this session.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/results"
GROUP="${GROUP:-4}"

echo "###### stage 1: does a group of $GROUP serve at all, and does coverage hold"
env REAL=1 PRED_GROUP="$GROUP" \
  bash "$HERE/run_device_transfer_smoke.sh" "$OUT/t13-g${GROUP}-smoke"
echo "###### rc=$?"

echo "###### stage 2: collectives per prefill window, expect 44 -> $((44 / GROUP)) snapshots"
env DOMAIN=ko DP=8 BUDGETS="0" NUM_PROMPTS=128 CONC=16 OUT_LEN=4 PRED_GROUP="$GROUP" \
  bash "$HERE/run_placement_profile.sh" "$OUT/t13-g${GROUP}-profile"
echo "###### rc=$?"

echo "###### stage 3: three arms at the knee, three passes"
env DOMAIN=ko DP=8 REPEATS=3 BUDGETS="off 0 43" NUM_PROMPTS=512 CONC=16 OUT_LEN=1 \
  PRED_GROUP="$GROUP" \
  bash "$HERE/run_e2e_placement.sh" "$OUT/t13-g${GROUP}-ruler"
echo "###### rc=$?"
