#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# First end-to-end run: plan from measured load, transfer, activate, and measure
# what changed. Two servers with identical settings except the transfer budget, so
# the comparison isolates the placement path.
#
# What this measures and what it does not: on this node the expected payoff is
# about 3.3% of a prefill step, while the interconnect term is 119 ms with a 2.8x
# cross-rank spread — so end-to-end TTFT cannot resolve it. The reported quantity is
# the **per-layer rank imbalance the placement path achieves**, read from the load
# dump, plus correctness (output equivalence, inactive slots idle).
#
# Usage: run_e2e_placement.sh [out_dir]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# The venv, not system python3: `nvshmem.core` is installed only there, and without it a
# device-transport arm falls back to the host path with a warning — measuring the very
# synchronisation it was meant to remove, under the label that says it removed it.
PY_BIN="${PY_BIN:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY_BIN" ]] || PY_BIN=python3
loaded=$("$PY_BIN" -c "import vllm; print(vllm.__file__)" 2>/dev/null || true)
case "$loaded" in
  "$REPO_ROOT"/*) : ;;
  *) echo "[FATAL] vLLM resolves to '$loaded', not $REPO_ROOT." >&2; exit 4 ;;
esac
echo "[env] vLLM from $loaded"
"$PY_BIN" -c "
import sys, vllm.envs as e
need = {'VLLM_PREDICTIVE_PLACE_PER_FORWARD', 'VLLM_EPLB_DUMP_LOAD_PATH'}
sys.exit(0 if need <= set(e.environment_variables) else 1)
" || { echo "[FATAL] this build lacks the placement or dump variable." >&2; exit 4; }

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${1:-$HERE/results/e2e-placement}"
MODEL="${MODEL:-/models/preset/Qwen/Qwen3-30B-A3B/v1.0}"
PORT="${PORT:-8180}"
# Three arms by default: `off` is the stock server (ticket 14's missing denominator),
# `0` enables prediction but withholds placement, `43` places. Any two of these answer
# a different question, so quote which pair a number came from.
# An arm is `<budget>[:<transport>[:<group>]]`, transport being `device` or `host` and
# `group` the target group of ticket 12. `43:host` pays the 5.28 ms per-layer host
# synchronisation and `43:device` does not, which is the whole claim of ticket 06.
# Carrying the group **per arm** matters for the same reason arms are interleaved at all:
# comparing a group of 4 against a group of 1 across two driver invocations charges the
# difference with whatever the machine did in between, and that drift has measured 6.1%
# on this baseline where the effect is a few points.
BUDGETS="${BUDGETS:-off 0 43}"
# Data-parallel size. 8 is where every recorded figure for this feature comes from; a
# smaller value runs and is warned about at startup, because EP size sets the per-rank
# expert count and with it how much imbalance there is to recover at all.
DP="${DP:-8}"
# How many times to measure each arm. One pass cannot see a 5% effect: between two runs of
# this script the stock arm alone moved 28% on mean TTFT and 40% on throughput, on identical
# workloads, because the machine was in a better state. Arms are measured in blocks —
# every arm once, then again — so a drift in machine state spreads across all of them
# instead of landing on whichever ran last.
REPEATS="${REPEATS:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-120}"
CONC="${CONC:-64}"
# Decode length. Short values raise the share of forwards that are prefill, which is
# what the prefill figures need: at the default, 120 requests at concurrency 64 yield
# only 8 full prefill forwards, so that column rests on 8 samples.
OUT_LEN="${OUT_LEN:-128}"
mkdir -p "$OUT_DIR"

LOCK="${TMPDIR:-/tmp}/predictive-baseline.lock"
exec 9>"$LOCK"
flock -w 10800 9 || { echo "[e2e] lock not acquired" >&2; exit 3; }

PROFILE="$OUT_DIR/cost-profile-local.json"
"$PY_BIN" - "$HERE/results/cost-profile.json" "$PROFILE" "$MODEL" "$DP" <<'PYEOF'
import json, subprocess, sys
src, dst, model, ep = sys.argv[1:5]
p = json.load(open(src))
device = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).splitlines()[0].strip()
p["fingerprint"].update(model=model, ep_size=int(ep), device_name=device)
json.dump(p, open(dst, "w"), indent=2)
PYEOF

# Which domain's prompts. Routing is content dependent and the domains differ widely
# in how much a placement can recover, measured per layer over 12 prefill forwards
# each: ko 89.6% of excess, zh 77.8%, ja 70.9%, lean 62.1%, code 59.7%, sql 59.5%,
# math 50.9%, text 32.6%. Non-Latin scripts lead by a wide margin — Qwen3's experts
# specialise by language — while narrow formal registers (sql, lean) came out level
# with code, so the intuition that a narrow vocabulary skews more did not hold.
# `results/domain-search/ranking.txt` carries the table.
DOMAIN="${DOMAIN:-code}"
PROMPTS="$HERE/results/prompts-$DOMAIN-p1024.jsonl"
[[ -s "$PROMPTS" ]] || { echo "[e2e] missing $PROMPTS" >&2; exit 1; }
echo "[e2e] domain=$DOMAIN"

for repeat in $(seq 1 "$REPEATS"); do
for arm in $BUDGETS; do
  budget="${arm%%:*}"
  rest="${arm#*:}"
  if [[ "$arm" == *:* ]]; then transport="${rest%%:*}"; else transport="${DEFAULT_TRANSPORT:-device}"; fi
  # The third field is optional; absent means "whatever PRED_GROUP says", which is how
  # every existing recorded run was labelled.
  arm_group="${PRED_GROUP:-0}"
  [[ "$rest" == *:* ]] && arm_group="${rest#*:}"
  label="$budget"
  [[ "$arm" == *:* ]] && label="${budget}-${transport}"
  [[ "$rest" == *:* ]] && label="${label}-g${arm_group}"
  # The tag carries the repeat only when there is more than one, so a single-pass run keeps
  # the filenames every existing reader and every recorded result already expects.
  if [[ "$REPEATS" -gt 1 ]]; then tag="b${label}-r${repeat}"; else tag="b$label"; fi
  echo "[e2e] arm=$arm: starting server (budget=$budget, transport=$transport)"
  LOG="$OUT_DIR/server-$tag.log"
  DUMP="$OUT_DIR/dump-$tag.jsonl"
  rm -f "$DUMP"
  # The env var only arms the path; the budget is `max_transfers_per_forward`, per the
  # spec. Its default of 4 was sized for decode and caps the ceiling at 1.0% of a
  # prefill step, so the placed arm asks for one per reachable layer.
  # Three arms, and the third one is the point of ticket 14. `budget=off` is a
  # genuinely stock server: no predictive config, so no `enable_eplb`, no redundant
  # experts and no replica slots, which is also what prices the 432 MiB per rank and
  # the startup layout normalization. `budget=0` still enables prediction and only
  # withholds placement, so a 0-versus-43 comparison answers "what does placement cost
  # on top of prediction" and never "what does the feature cost" — prediction alone
  # measured 19% of TPOT. Every TTFT number in this branch before 2026-08-29 has that
  # missing denominator.
  FEATURE_ARGS=()
  FEATURE_ENV=()
  if [[ "$budget" == "off" ]]; then
    echo "[e2e] arm=off: stock server, feature fully disabled"
  else
    # The transport decides what the arm measures: the host-issued path pays 5.28 ms of
    # synchronisation per predicted layer and the device-issued one pays none, so the
    # +31.5% mean TTFT figure belongs to `43:host`.
    ADDITIONAL=$("$PY_BIN" -c "
import json,sys
cfg={'enabled': True, 'cost_profile_path': sys.argv[1],
     'device_issued_transfer': sys.argv[3] == 'device'}
if int(sys.argv[2]) > 0:
    cfg['max_transfers_per_forward'] = int(sys.argv[2])
if int(sys.argv[4]) > 0:
    group = int(sys.argv[4])
    cfg['prediction_target_group'] = group
    # The window's collective is issued at its last source, so the distance must cover
    # the window. Equality is the cheapest choice: a longer distance only costs accuracy.
    cfg['prediction_lookahead_layers'] = group
print(json.dumps({'predictive_expert_replication': cfg}))" \
      "$PROFILE" "$budget" "$transport" "$arm_group")
    FEATURE_ARGS=(
      --additional-config "$ADDITIONAL"
      --eplb-config '{"log_balancedness":true,"log_balancedness_interval":1,"step_interval":1000000000,"window_size":1000,"use_async":false}'
    )
    FEATURE_ENV=(
      "VLLM_PREDICTIVE_PLACE_PER_FORWARD=$budget"
      "VLLM_EPLB_DUMP_LOAD_PATH=$DUMP"
      "VLLM_PREDICTIVE_VERIFY_INACTIVE_SLOTS=0"
      # The device path cannot log per activation - the host no longer knows what was
      # placed - so it counts on the device and reports every N forwards. At the default
      # 50 a short run finishes before its first report, and the guard then calls a
      # working arm inert. 10 keeps the one synchronisation it costs out of the way while
      # still producing the line every reader here looks for.
      "VLLM_PREDICTIVE_PLACEMENT_REPORT_EVERY=10"
    )
  fi

  env "${FEATURE_ENV[@]}" \
  "$PY_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" \
    --data-parallel-size "$DP" --enable-expert-parallel \
    --all2all-backend allgather_reducescatter --enforce-eager \
    --max-model-len 3072 --gpu-memory-utilization 0.88 \
    --max-num-seqs 64 --seed 0 --uvicorn-log-level warning \
    "${FEATURE_ARGS[@]}" \
    >"$LOG" 2>&1 &
  PID=$!

  ready=0
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 "$PID" 2>/dev/null || break
    sleep 5
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "[e2e] arm=$arm: never ready" >&2; tail -40 "$LOG" >&2
    kill -9 "$PID" 2>/dev/null || true; sleep 5
    # `pgrep -f` matches the same worker processes the `ps | grep` pipeline did, without
    # the grep-matches-itself hazard the `grep -v grep` was there to dodge.
    for p in $(pgrep -f "VLLM::" || true); do kill -9 "$p" 2>/dev/null; done
    sleep 8; continue
  fi
  echo "[e2e] arm=$arm: ready"

  # `$PY_BIN -m`, not the `vllm` console script: the script runs under system python,
  # where the bench extra's pandas is not installed - and on a pod restart it disappears
  # again. It also leaves the working tree off sys.path, which has silently measured stock
  # vLLM before.
  timeout 1800 "$PY_BIN" -m vllm.entrypoints.cli.main bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --model "$MODEL" --port "$PORT" \
    --dataset-name custom --dataset-path "$PROMPTS" \
    --custom-output-len "$OUT_LEN" --ignore-eos \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" \
    --percentile-metrics ttft,tpot --metric-percentiles 99 \
    --save-result --result-dir "$OUT_DIR" --result-filename "bench-$tag.json" \
    >"$OUT_DIR/bench-$tag.log" 2>&1 || echo "[e2e] $tag bench FAILED" >&2

  # This prefix must stay identical to `_PLACEMENT_LINE` in analyse_e2e.py. An
  # earlier version matched "activated .* replicas" against a line that says
  # "replica(s)", so a fully working placed arm reported "no placement log line"
  # and blamed a budget of 0 regardless of the arm. Absence of this line is the
  # only cheap signal that an arm was silently inert, so a wrong pattern here
  # turns the one self-check into a false all-clear.
  placement_lines=$(grep -c "Predictive expert replication: activated" "$LOG" 2>/dev/null || true)
  placement_lines=${placement_lines:-0}
  # String comparison, not `-eq`: `off` is not a number, and under `set -u` an
  # arithmetic test treats it as a variable name and aborts the run after the arm has
  # already served its whole benchmark.
  if [[ "$budget" == "off" || "$budget" == "0" ]]; then
    expect_placement=0
  else
    expect_placement=1
  fi
  if [[ "$placement_lines" -gt 0 ]]; then
    echo "[e2e] $tag: $placement_lines placement log lines"
    if [[ "$expect_placement" -eq 0 ]]; then
      echo "[e2e] $tag: UNEXPECTED - arm $budget placed replicas" >&2
    fi
  elif [[ "$expect_placement" -eq 0 ]]; then
    echo "[e2e] $tag: no placement log line (correct for arm $budget)"
  else
    echo "[e2e] $tag: NO PLACEMENT LOG LINE at budget $budget - arm was inert" >&2
  fi
  # The stock arm records no expert load by design, so an absent dump there is the
  # expected outcome rather than the "measured nothing" failure it is on the others.
  if [[ "$budget" == "off" ]]; then
    echo "[e2e] $tag: no dump expected (feature disabled)"
  else
    [[ -s "$DUMP" ]] && echo "[e2e] $tag: $(wc -l < "$DUMP") forwards dumped" \
                     || echo "[e2e] $tag: EMPTY DUMP" >&2
  fi

  # Bounded teardown, then escalate. `kill -TERM` followed by a bare `wait` deadlocked
  # a run: the DP=8 server did not die on SIGTERM, `wait` blocked forever, and the two
  # remaining arms never started — 26 minutes of an idle 8-GPU server with the first
  # arm's result already on disk and nothing in the log to say why. Never trust the
  # graceful path to return here; the reap loop below depends on getting past this.
  kill -TERM "$PID" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 2
  done
  kill -9 "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  sleep 5
  for p in $(pgrep -f "VLLM::" || true); do kill -9 "$p" 2>/dev/null; done
  sleep 10
done
done
# One guard over every arm of every repeat, because the per-arm messages above only ever
# printed. Three runs in this project reported success having measured nothing, and each of
# them printed a warning to stderr and carried on.
guard_arms=()
for repeat in $(seq 1 "$REPEATS"); do
  for arm in $BUDGETS; do
    # The same construction the run loop uses. It was `${label}-${arm#*:}`, which turns
    # `43:device:4` into `43-device:4` and sends the guard looking for files that do not
    # exist — a guard that cries wolf on a healthy run is a guard that gets deleted.
    label="${arm%%:*}"
    rest="${arm#*:}"
    if [[ "$arm" == *:* ]]; then label="${label}-${rest%%:*}"; fi
    if [[ "$rest" == *:* ]]; then label="${label}-g${rest#*:}"; fi
    if [[ "$REPEATS" -gt 1 ]]; then guard_arms+=("${label}-r${repeat}"); else guard_arms+=("$label"); fi
  done
done
if ! "$PY_BIN" "$HERE/check_run_measured.py" --results-dir "$OUT_DIR" --arms "${guard_arms[@]}"; then
  echo "[e2e] MEASURED NOTHING - do not read $OUT_DIR" >&2
  exit 5
fi
echo "[e2e] done; results in $OUT_DIR"
