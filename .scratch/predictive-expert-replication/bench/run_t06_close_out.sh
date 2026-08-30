#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Ticket 06's two remaining criteria, on the 8-GPU node, back to back.
#
# Both criteria need 8 GPUs and were left open when this node came back from a restart
# with 2. They are measured in one driver so the GPUs are never idle between them and so
# the tree is provably identical for all three stages — editing the tree while a run was
# in flight is what invalidated the first attempt at this, with a worker dying on an
# environment variable that did not exist when its server started.
#
# Stage order is deliberate: correctness first, because it is the cheapest and because a
# measurement of an incorrect placement is worth nothing.
#
#   1. Does a *dynamically placed* replica hold the bytes of the expert it claims?
#      Real weights are required: with `--load-format dummy` every expert row holds the
#      same values, so a replica pointing at the wrong row would still pass.
#   2. Criterion: at least 24.0% of full-prefill critical-path excess removed, on every
#      reachable layer. Measured at the operating point the 24.0% was recorded at
#      (`ko`, 400 requests, CONC=8, OUT_LEN=1), because that is what "reproduced" means.
#      This point cannot resolve a TTFT effect and no TTFT claim is made from it.
#   3. Criterion: GPU occupancy inside real forward windows, against the recorded 86.8%
#      and 52.9%. At the knee (CONC=16) with the same shape as the host-path profile it
#      is compared against, so the two are diffable with one tool.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/results"
STAGES="${STAGES:-verify excess occupancy}"

run_stage() {
  echo
  echo "############ stage: $1 ############"
  shift
  "$@"
  echo "############ rc=$? ############"
}

for stage in $STAGES; do
  case "$stage" in
    verify)
      run_stage "replica weights, real weights, checked every forward" \
        env REAL=1 VLLM_PREDICTIVE_VERIFY_REPLICA_WEIGHTS=1 \
        bash "$HERE/run_device_transfer_smoke.sh" "$OUT/t06-verify-weights"
      ;;
    excess)
      run_stage "critical-path excess, DP=8, the 24.0% operating point" \
        env DOMAIN=ko DP=8 REPEATS=1 BUDGETS="off 0 43" \
        NUM_PROMPTS=400 CONC=8 OUT_LEN=1 \
        bash "$HERE/run_e2e_placement.sh" "$OUT/t06-dp8-excess"
      ;;
    occupancy)
      run_stage "occupancy at the knee, DP=8, paired with knee16-profile" \
        env DOMAIN=ko DP=8 BUDGETS="off 0 43" \
        NUM_PROMPTS=128 CONC=16 OUT_LEN=4 \
        bash "$HERE/run_placement_profile.sh" "$OUT/t06-knee16-profile"
      ;;
    *) echo "unknown stage $stage" >&2; exit 2 ;;
  esac
done
echo
echo "############ all stages done ############"
