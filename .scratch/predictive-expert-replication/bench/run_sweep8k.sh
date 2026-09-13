#!/usr/bin/env bash
# How much does the 8k result depend on what the tokens actually are?
#
# Every figure on this branch comes from one dataset, and routing is content dependent, so the
# question this answers is **the range across domains**, not which domain is best. Quoting the
# best domain would be selection bias; quoting the range is the robustness statement.
#
# Two-stage on purpose: screen every domain at 3 passes, which is enough to rank and to see a
# sign, then confirm the extremes at 6. Three passes could not establish a 2.5% effect here --
# that is measured, not assumed (results/t19-8k-c32 gave -0.90% +/- 4.3% where six passes gave
# -1.20% +/- 0.6%). **Do not quote a screening number.**
#
# Datasets are probed for segments-per-prompt when built: `make_prompts.py` concatenates short
# samples to reach the target length, so a dataset of short rows produces a prompt that is
# dozens of unrelated fragments. KoAlpaca at 8k is 161 fragments; govreport is 1.6. Those are
# different workloads wearing the same token count, and the build log records which is which.
#
# Usage:  setsid nohup bash run_sweep8k.sh > results/sweep8k.log 2>&1 < /dev/null &
# Stop:   touch results/backlog.stop

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
MODEL="${MODEL:-/models/preset/Qwen/Qwen3-30B-A3B/v1.0}"
PY="${PY:-/mnt/yhong/vllm-predict/.venv/bin/python}"
STOP="$HERE/results/backlog.stop"
SNAP="$(cd "$HERE/.." && pwd)/.sweep-snapshot"

say() { echo "[sweep $(date +%H:%M:%S)] $*"; }
stopped() { [[ -e "$STOP" ]]; }

rm -rf "$SNAP"; mkdir -p "$SNAP"
cp "$HERE"/*.sh "$HERE"/*.py "$SNAP"/ 2>/dev/null
ln -sfn "$HERE/results" "$SNAP/results"
say "harness snapshotted to $SNAP"

# domain | HF dataset. Chosen for length and for spread of content, not for expected result.
DOMAINS=(
  "arxiv:ccdv/arxiv-summarization"
  "pubmed:ccdv/pubmed-summarization"
  "books:emozilla/pg19-test"
  "chat:Aeala/ShareGPT_Vicuna_unfiltered"
  "wiki:wikimedia/wikipedia"
)
BUILT=()
for entry in "${DOMAINS[@]}"; do
  d="${entry%%:*}"; ds="${entry#*:}"
  out="$HERE/results/prompts-$d-p8192.jsonl"
  if [[ ! -s "$out" ]]; then
    say "building $d from $ds"
    timeout 2700 "$PY" "$HERE/make_prompts.py" --dataset "$ds" --tokenizer "$MODEL" \
      --target-prompt-len 8192 --tolerance 0.10 --num-prompts 160 --out "$out" \
      >"$HERE/results/prompts-$d-p8192.build.log" 2>&1
  fi
  if [[ -s "$out" ]]; then
    seg=$("$PY" -c "
import json,sys
n=[json.loads(l)['prompt'].count(chr(10)*2)+1 for l in open('$out')]
print(f'{sum(n)/len(n):.1f}')" 2>/dev/null || echo "?")
    say "$d ready, $seg segments per prompt"
    BUILT+=("$d")
  else
    say "$d FAILED to build, skipping (see results/prompts-$d-p8192.build.log)"
  fi
done
say "domains available: ${BUILT[*]:-none}"

screen() {
  local d="$1" label="sweep8k-$d"
  stopped && return 1
  if grep -q "done; results in" "$HERE/results/$label.log" 2>/dev/null; then
    say "$label already complete"; return 0
  fi
  say "SCREEN $d (3 passes -- ranking only, not quotable)"
  local t0=$SECONDS
  env DOMAIN="$d" PROMPT_LEN=8192 MAX_MODEL_LEN=10240 DP=8 NUM_PROMPTS=128 OUT_LEN=1 \
      REPEATS=3 CONC=16 BUDGETS="off 43:device:1" \
    bash "$SNAP/run_e2e_placement.sh" "$HERE/results/$label" \
    >"$HERE/results/$label.log" 2>&1
  say "END $label rc=$? after $(( (SECONDS-t0)/60 )) min"
}

for d in "${BUILT[@]}"; do screen "$d"; done

say "screening done; ranking by paired mean TTFT"
"$PY" - <<'PYEOF'
import json, glob, os, statistics as st
rows=[]
for p in sorted(glob.glob("results/sweep8k-*/")):
    d=p.rstrip("/").split("sweep8k-")[-1]
    tt=[]
    for r in (1,2,3):
        a=f"{p}bench-boff-r{r}.json"; b=f"{p}bench-b43-device-g1-r{r}.json"
        if os.path.exists(a) and os.path.exists(b):
            s=json.load(open(a))["mean_ttft_ms"]; f=json.load(open(b))["mean_ttft_ms"]
            tt.append(100*(f-s)/s)
    if tt: rows.append((st.mean(tt), d, len(tt), sum(1 for x in tt if x<0)))
for m,d,n,neg in sorted(rows):
    print(f"  {d:8s} paired mean TTFT {m:+6.2f}%  ({neg}/{n} faster)  SCREENING ONLY")
print("  range across domains is the result; the best single domain is not.")
PYEOF

# Confirm the two extremes at six passes, which is what a quotable number needs.
BEST=$("$PY" -c "
import json,glob,os,statistics as st
rows=[]
for p in sorted(glob.glob('results/sweep8k-*/')):
    d=p.rstrip('/').split('sweep8k-')[-1]; tt=[]
    for r in (1,2,3):
        a=f'{p}bench-boff-r{r}.json'; b=f'{p}bench-b43-device-g1-r{r}.json'
        if os.path.exists(a) and os.path.exists(b):
            s=json.load(open(a))['mean_ttft_ms']; f=json.load(open(b))['mean_ttft_ms']
            tt.append(100*(f-s)/s)
    if tt: rows.append((st.mean(tt), d))
rows.sort()
print(' '.join(d for _, d in ([rows[0], rows[-1]] if len(rows) > 1 else rows)))" 2>/dev/null)
say "confirming extremes at 6 passes: ${BEST:-none}"
for d in $BEST; do
  stopped && break
  label="sweep8k-$d-confirm"
  grep -q "done; results in" "$HERE/results/$label.log" 2>/dev/null && { say "$label done"; continue; }
  say "CONFIRM $d (6 passes)"
  env DOMAIN="$d" PROMPT_LEN=8192 MAX_MODEL_LEN=10240 DP=8 NUM_PROMPTS=128 OUT_LEN=1 \
      REPEATS=6 CONC=16 BUDGETS="off 43:device:1" \
    bash "$SNAP/run_e2e_placement.sh" "$HERE/results/$label" \
    >"$HERE/results/$label.log" 2>&1
  say "END $label rc=$?"
done
say "sweep done"
