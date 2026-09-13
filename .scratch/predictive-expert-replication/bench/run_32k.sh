#!/usr/bin/env bash
# 32k on a dataset that does not concatenate, and the first experiment that moves
# tokens-per-forward.
#
# Two findings drive the shape. (1) Every long-prompt run on this branch left
# `max_num_batched_tokens` at vLLM's default 8192, so "16k" and "8k" both ran at ~8192
# tokens per forward -- the axis CLAUDE.md credits for the 8k->16k improvement never
# moved between those two runs. (2) pg19 documents have a median length of 77.7k tokens,
# so a 32k prompt is one real document trimmed, where `ko` at 32k would be ~910
# concatenated fragments and concatenation inflates expert stability.
#
# A and B differ in the cap alone: same domain, same length, same 256 prompts, same
# concurrency. Only A's request is chunked into 4 forwards and B's is one forward.
# 256 prompts rather than 64 because coverage ramps per forward and B has a quarter as
# many, so a smaller count would measure B's cold start instead of its steady state.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
say() { echo "[32k $(date +%H:%M:%S)] $*"; }

COMMON=(DOMAIN=books PROMPT_LEN=32768 MAX_MODEL_LEN=34816 DP=8
        NUM_PROMPTS=256 OUT_LEN=1 REPEATS=6 CONC=8)

say "A: cap 8192 -- 4 forwards per request, the regime every recorded figure ran in"
env "${COMMON[@]}" MAX_BATCHED_TOKENS=8192 BUDGETS="off 96:device:1:c2" \
  bash "$HERE/run_e2e_placement.sh" "$HERE/results/p32k-cap8k" \
  > "$HERE/results/p32k-cap8k.log" 2>&1 && say "A done" || say "A FAILED rc=$?"

say "B: cap 32768 -- 1 forward per request, tokens-per-forward finally moved"
env "${COMMON[@]}" MAX_BATCHED_TOKENS=32768 BUDGETS="off 96:device:1:c2" \
  bash "$HERE/run_e2e_placement.sh" "$HERE/results/p32k-cap32k" \
  > "$HERE/results/p32k-cap32k.log" 2>&1 && say "B done" || say "B FAILED rc=$?"

say "both done"
