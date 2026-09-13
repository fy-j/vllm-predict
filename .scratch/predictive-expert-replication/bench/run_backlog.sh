#!/usr/bin/env bash
# Keep the node busy with the measurement backlog, in priority order, one at a time.
#
# Why a queue runner rather than a keep-alive loop: the cards must not sit idle or the node
# gets reclaimed, and there is real work queued. Every stage below is an experiment some
# document already says is missing, so the cost of keeping the node warm is paid back in
# results. The keep-alive at the end is the fallback, not the plan.
#
# Serial by necessity: every stage uses all 8 GPUs.
#
# Two hazards this script is built around, both of which have already cost this project a run:
#
#   * Editing a running bash script corrupts it. Bash reads by byte offset, so inserting lines
#     ahead of the execution point makes the running shell resume mid-token. A previous run
#     died in its guard this way. So this script SNAPSHOTS the harness into a private
#     directory and runs the snapshot: later edits to `run_e2e_placement.sh` cannot reach a
#     stage that is already in flight.
#   * A failed stage must not stop the queue. `set -e` is deliberately off; each stage's exit
#     status is logged and the next one starts.
#
# Prompt sets are built FIRST, before any GPU work, because tokenising is CPU-heavy and this
# engine is host-dispatch-bound -- building one during a benchmark perturbs the thing being
# measured.
#
# Usage:  setsid nohup bash run_backlog.sh > results/backlog.log 2>&1 < /dev/null &
# Stop:   touch results/backlog.stop     (checked between stages, and by the keep-alive)

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
MODEL="${MODEL:-/models/preset/Qwen/Qwen3-30B-A3B/v1.0}"
PY="${PY:-/mnt/yhong/vllm-predict/.venv/bin/python}"
STOP="$HERE/results/backlog.stop"
# Sibling of `bench/`, not inside it: the harness derives the repo root as
# `dirname $0/../../..` and asserts vLLM resolves to this tree. A snapshot at any other depth
# makes that guard compute the wrong root and abort every stage.
SNAP="$(cd "$HERE/.." && pwd)/.backlog-snapshot"

say() { echo "[backlog $(date +%H:%M:%S)] $*"; }
stopped() { [[ -e "$STOP" ]] && { say "stop file present, exiting"; return 0; } || return 1; }

# ---------------------------------------------------------------- harness snapshot
rm -rf "$SNAP"; mkdir -p "$SNAP"
cp "$HERE"/*.sh "$HERE"/*.py "$SNAP"/ 2>/dev/null
ln -sfn "$HERE/results" "$SNAP/results"
say "harness snapshotted to $SNAP; edits to $HERE cannot affect stages in flight"

# ---------------------------------------------------------------- prompt sets, CPU only
build_prompts() {
  local domain="$1" len="$2" dataset="$3" n="$4"
  local out="$HERE/results/prompts-$domain-p$len.jsonl"
  if [[ -s "$out" ]]; then say "prompts $domain p$len already present"; return 0; fi
  say "building $domain p$len from $dataset"
  timeout 3600 "$PY" "$HERE/make_prompts.py" --dataset "$dataset" --tokenizer "$MODEL" \
    --target-prompt-len "$len" --tolerance 0.10 --num-prompts "$n" --out "$out" \
    >"$HERE/results/prompts-$domain-p$len.build.log" 2>&1
  [[ -s "$out" ]] && say "built $domain p$len" || say "FAILED to build $domain p$len"
}

KO=beomi/KoAlpaca-v1.1a
build_prompts ko 2048 "$KO" 256
build_prompts ko 4096 "$KO" 192
build_prompts ko 16384 "$KO" 96
# Genuine long documents: 1.6 segments per 8k prompt against KoAlpaca's 161. Every result on
# this branch so far is on concatenated short samples, and routing is content dependent.
GOV=ccdv/govreport-summarization
build_prompts gov 8192 "$GOV" 160
build_prompts gov 16384 "$GOV" 112

# ---------------------------------------------------------------- the queue
# Each entry: label | env assignments | arms
# Priority order, most informative first. Rationale per stage is in the say line.
run_stage() {
  local label="$1" envs="$2" arms="$3" why="$4"
  stopped && return 1
  local out="$HERE/results/$label"
  # Completion is "the guard passed", which the harness itself logs. `arm-spread.json` was
  # the wrong marker: my analysis writes it, the harness does not, so a restart re-ran
  # every finished stage.
  if grep -q "done; results in" "$HERE/results/$label.log" 2>/dev/null; then
    say "$label already complete, skipping"; return 0
  fi
  say "START $label -- $why"
  say "  env: $envs"
  say "  arms: $arms"
  local t0=$SECONDS
  env $envs BUDGETS="$arms" bash "$SNAP/run_e2e_placement.sh" "$out" \
    >"$HERE/results/$label.log" 2>&1
  local rc=$?
  say "END $label rc=$rc after $(( (SECONDS-t0)/60 )) min"
  return 0
}

BASE8K="DOMAIN=ko PROMPT_LEN=8192 MAX_MODEL_LEN=10240 DP=8 NUM_PROMPTS=128 OUT_LEN=1"
GOV8K="DOMAIN=gov PROMPT_LEN=8192 MAX_MODEL_LEN=10240 DP=8 NUM_PROMPTS=128 OUT_LEN=1"
GOV16K="DOMAIN=gov PROMPT_LEN=16384 MAX_MODEL_LEN=18432 DP=8 NUM_PROMPTS=96 OUT_LEN=1"

run_stage gov-8k-c16 \
  "$GOV8K REPEATS=6 CONC=16" \
  "off 43:device:1" \
  "THE ONE THAT CAN MOVE THE CONCLUSION: real long documents (1.6 segments per prompt) instead of 161 concatenated Korean instructions. Routing is content dependent and every result so far is on concatenations."

run_stage t19-16k-cap2 \
  "DOMAIN=ko PROMPT_LEN=16384 MAX_MODEL_LEN=18432 DP=8 NUM_PROMPTS=96 OUT_LEN=1 REPEATS=6 CONC=8" \
  "off 43:device:1 86:device:1:c2" \
  "the two winning levers together: cap 2 gave -3.17% at 8k/c16, cap 1 gave -4.67% at 16k/c8."

run_stage t15-ratchet-8k \
  "$BASE8K REPEATS=3 CONC=16" \
  "off 4:device:1 43:device:1" \
  "ticket 15's last criterion. At 1k there was no steady state to converge to (21% stability), so that run proved nothing; at 93% the coverage ratchet should settle and stay."

run_stage t14-barrier-share-8k \
  "$BASE8K REPEATS=3 CONC=16" \
  "off 0:device:1 0:device:1:noag" \
  "ticket 14's ceiling, recomputed at 8k. Its arithmetic is a share of prediction's cost and that share is amortised over 4.8x the tokens here. The probe needs budget 0: it leaves each rank a different snapshot."

run_stage gov-16k-c8 \
  "$GOV16K REPEATS=6 CONC=8" \
  "off 43:device:1" \
  "the best known configuration, on real documents."

say "backlog drained"

# ---------------------------------------------------------------- fallback keep-alive
# Reached only when the queue is done. This produces NO results -- it exists purely so the
# cards are not idle. Low intensity on purpose: it should hold the allocation, not compete
# with anything a human starts. Dies on the stop file.
say "entering keep-alive; it measures nothing. touch $STOP to end."
STOP="$STOP" "$PY" - <<'PYEOF'
import os, time, pathlib, torch
stop = pathlib.Path(os.environ["STOP"])
n = torch.cuda.device_count()
print(f"[keep-alive] holding {n} device(s); this produces no measurements", flush=True)
# Small and slow on purpose: enough to register as utilisation, little enough that a human
# starting real work on this node is not fighting it for SMs.
a = [torch.randn(1024, 1024, device=f"cuda:{i}", dtype=torch.bfloat16) for i in range(n)]
beat = 0
while not stop.exists():
    for x in a:
        (x @ x).sum()
    torch.cuda.synchronize()
    time.sleep(2.0)
    beat += 1
    if beat % 300 == 0:
        print(f"[keep-alive] {beat} cycles ({beat * 2 // 60} min), still holding", flush=True)
print("[keep-alive] stop file seen, releasing", flush=True)
PYEOF
say "done"
