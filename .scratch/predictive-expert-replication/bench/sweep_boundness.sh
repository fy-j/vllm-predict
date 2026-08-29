#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Find where MoE stops being expert-weight-bandwidth bound, which is the one
# measurement that decides whether any placement policy can pay off here.
#
# Load imbalance only turns into time once MoE time exceeds the cost of merely
# reading a rank's local expert weights (measured at 102 us for 144 MiB). That
# needs a large decode batch per rank. KV holds a fixed number of tokens, so
# sequences per rank is capacity divided by context length: at the spec's 3072
# token context only about 59 fit, which is far short. Short contexts are the
# only way to reach the regime at all, so this sweep trades realistic shapes for
# the ability to answer the question.
#
# Reported per point: sequences per rank, p99 TPOT, and the balancedness the
# server logs. TPOT flat across concurrency means overhead dominated; TPOT rising
# with concurrency means the batch finally costs compute.
#
# Usage: sweep_boundness.sh [out_dir]
set -euo pipefail

OUT_DIR="${1:-$(dirname "$0")/results}"
HERE="$(dirname "$0")"

# 256-token context: KV fits roughly 700 sequences per rank, so every point below
# is reachable, unlike the spec's shapes.
export SHAPES="${SHAPES:-128:128}"
export DOMAINS="${DOMAINS:-code}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"

# Three separate caps decide the decode batch, and every one of them has to be
# raised together. The first sweep missed two and measured the same batch three
# times:
#   1. the client's prompt count, since concurrency cannot exceed it
#   2. `max_num_seqs`, whose 128 default is exactly the threshold to cross
#   3. KV capacity, which at a 256-token context allows roughly 700 per rank
export NUM_PROMPTS="${NUM_PROMPTS:-3000}"
export MAX_PROMPTS="${MAX_PROMPTS:-3000}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-384}"

# Total concurrency; divide by the 8 DP ranks for sequences per rank. The probe
# grid puts the regime change between 64 and 128 sequences per rank, so the sweep
# has to straddle 512 and 1024 total.
export CONCURRENCIES="${CONCURRENCIES:-512 1024 2048}"

echo "[sweep] shapes=$SHAPES concurrencies=$CONCURRENCIES"
echo "[sweep] sequences per rank = concurrency / 8; max_num_seqs=$MAX_NUM_SEQS per rank"
echo "[sweep] verify the running batch from throughput x TPOT, not from the concurrency flag"
bash "$HERE/run_serving_baseline.sh" recording "$OUT_DIR"
