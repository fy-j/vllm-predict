#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Does prediction's cost live in its 44 per-layer AllGathers, or in its launches?
#
# Three arms at the knee, three passes each. The first two are the usual stock and
# prediction-only arms; the third is prediction-only with the snapshot AllGather removed
# and everything else — the gate GEMM, the top-k, the counting kernel — left in place.
#
# The probe arm must run at budget 0. Its snapshot is each rank's own counts rather than
# the allgathered one, so ranks would derive different plans; `eplb_state` refuses to arm
# placement alongside it rather than trusting this script.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/results"
COMMON=(DOMAIN=ko DP=8 REPEATS=3 NUM_PROMPTS=512 CONC=16 OUT_LEN=1)

echo "###### arms with the AllGather: stock and prediction-only"
env "${COMMON[@]}" BUDGETS="off 0" bash "$HERE/run_e2e_placement.sh" "$OUT/t08-ag-baseline"
echo "###### rc=$?"

echo "###### arm without the AllGather: prediction-only, collective removed"
env "${COMMON[@]}" BUDGETS="0" VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER=1 \
  bash "$HERE/run_e2e_placement.sh" "$OUT/t08-ag-removed"
echo "###### rc=$?"
