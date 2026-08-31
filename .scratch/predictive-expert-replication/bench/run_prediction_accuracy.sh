#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Ticket 06: how well the cross-layer gate predicts its target layer's load, and
# how that degrades with lookahead distance.
#
# The lookahead is a launch-time setting, so each distance needs its own server.
# Within one server the dump file is rotated between request shapes, because the
# dump path is also fixed at launch.
#
# Decode length is deliberately short (128 tokens). Accuracy is sampled once per
# forward, so 128 decode steps is already a large sample, while a full 2048-token
# decode would write gigabytes: one record holds every source layer's predicted
# and actual counts, about 88 KB per forward. Prompt length and content are the
# parts that drive routing and those are kept at the agreed shapes.
#
# Usage: run_prediction_accuracy.sh [out_dir]
set -euo pipefail

# The `vllm` console script puts /usr/local/bin on sys.path, never the current
# directory, so without this it silently runs the *installed* vLLM rather than
# this working tree. A whole accuracy run once produced six empty dumps that way,
# with nothing but an "Unknown vLLM environment variable" warning to show for it.
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# The venv, not system python3: the bench client needs the bench extra's pandas, which
# lives only there and disappeared with a pod restart, and `vllm serve` would leave the
# working tree off sys.path entirely.
PY_BIN="${PY_BIN:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PY_BIN" ]] || PY_BIN=python3
loaded=$("$PY_BIN" -c "import vllm; print(vllm.__file__)" 2>/dev/null || true)
case "$loaded" in
  "$REPO_ROOT"/*) : ;;
  *) echo "[FATAL] vLLM resolves to '$loaded', not $REPO_ROOT." >&2
     echo "        The run would measure the installed build, not this tree." >&2
     exit 4 ;;
esac
echo "[env] vLLM from $loaded"

# The dump is driven by an environment variable the running build must know
# about; an unrecognised one is a warning, not an error, so check it here.
"$PY_BIN" -c "
import sys, vllm.envs as e
sys.exit(0 if 'VLLM_PREDICTIVE_ACCURACY_DUMP_PATH' in e.environment_variables else 1)
" || { echo "[FATAL] this build does not register the accuracy dump variable." >&2; exit 4; }

OUT_DIR="${1:-$(dirname "$0")/results/accuracy}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-/models/preset/Qwen/Qwen3-30B-A3B/v1.0}"
PORT="${PORT:-8140}"
LOOKAHEADS="${LOOKAHEADS:-1 2 3}"
# Data-parallel size. 8 is where the recorded accuracy figures come from; a smaller value
# runs and is warned about at startup, and for a *lookahead comparison* what matters is
# that both arms see the same prompts on the same node rather than the absolute number.
DP="${DP:-8}"
# Which prompt shapes. Accuracy is a property of the content's routing, so the domain is
# part of the result; `ko` is the highest-imbalance domain measured and needs no rebuild.
DOMAINS="${DOMAINS:-code text}"
SKIP_FIRST="${SKIP_FIRST:-3}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
CONC="${CONC:-32}"

mkdir -p "$OUT_DIR"

# Queue behind any other run on this node rather than sharing a server: two
# benchmark loops against one server corrupts every number.
LOCK="${TMPDIR:-/tmp}/predictive-baseline.lock"
exec 9>"$LOCK"
echo "[accuracy] waiting for the node lock (up to 3h)"
flock -w 10800 9 || { echo "[accuracy] lock not acquired" >&2; exit 3; }
echo "[accuracy] lock acquired"

# The fingerprint is compared for exact equality against the served model
# identity, which here is a local path rather than the HuggingFace id the
# canonical profile records.
PROFILE="$OUT_DIR/cost-profile-local.json"
"$PY_BIN" - "$HERE/results/cost-profile.json" "$PROFILE" "$MODEL" "$DP" <<'PYEOF'
import json, subprocess, sys
src, dst, model, ep = sys.argv[1:5]
profile = json.load(open(src))
device = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
).splitlines()[0].strip()
profile["fingerprint"].update(model=model, ep_size=int(ep), device_name=device)
json.dump(profile, open(dst, "w"), indent=2)
print(f"[accuracy] cost profile for {model} at EP={ep} on {device} -> {dst}")
PYEOF

# Prompts at the two agreed shapes, paired with the domain whose natural length
# already fits: code near 1k, conversational text near 2k.
declare -A DATASET=( [code]="likaixin/InstructCoder" [text]="Aeala/ShareGPT_Vicuna_unfiltered" [ko]="beomi/KoAlpaca-v1.1a" )
declare -A PROMPT_LEN=( [code]=1024 [text]=2048 [ko]=1024 )
for domain in $DOMAINS; do
  # Shared across runs rather than copied per out_dir: 2 MB each, identical.
  prompts="$HERE/results/prompts-$domain-p${PROMPT_LEN[$domain]}.jsonl"
  if [[ ! -s "$prompts" ]]; then
    echo "[accuracy] building ${PROMPT_LEN[$domain]}-token $domain prompts"
    "$PY_BIN" "$HERE/make_prompts.py" \
      --dataset "${DATASET[$domain]}" --tokenizer "$MODEL" \
      --target-prompt-len "${PROMPT_LEN[$domain]}" --tolerance 0.10 \
      --num-prompts 64 --out "$prompts" \
      >"$OUT_DIR/prompts-$domain.log" 2>&1 || true
    [[ -s "$prompts" ]] || { echo "[accuracy] $domain prompt build failed" >&2; exit 1; }
  fi
done

for lookahead in $LOOKAHEADS; do
  DUMP="$OUT_DIR/dump-live.jsonl"
  rm -f "$DUMP"
  echo "[accuracy] lookahead=$lookahead: starting server"
  SERVER_LOG="$OUT_DIR/server-L$lookahead.log"

  # The predictive controller is configured through additional_config, not a
  # dedicated flag.
  ADDITIONAL_CFG=$("$PY_BIN" -c "
import json,sys
cfg={'enabled': True,
     'cost_profile_path': sys.argv[1],
     'prediction_lookahead_layers': int(sys.argv[2]),
     'prediction_skip_first_layers': int(sys.argv[3])}
if int(sys.argv[4]) > 0:
    cfg['prediction_target_group'] = int(sys.argv[4])
print(json.dumps({'predictive_expert_replication': cfg}))" \
    "$PROFILE" "$lookahead" "$SKIP_FIRST" "${PRED_GROUP:-0}")

  # Deliberately the module form, not `vllm serve`. `vllm serve` starts one API
  # server per DP rank, and each rebuilds the config from serialized engine args;
  # the predictive wiring's `num_redundant_experts` survives that round trip but
  # its `enable_eplb` does not, so every rank dies on
  # "num_redundant_experts is set to 8 but EPLB is not enabled". A single API
  # server has no such round trip, and this is the form ticket 03 validated with.
  VLLM_PREDICTIVE_ACCURACY_DUMP_PATH="$DUMP" \
  "$PY_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --data-parallel-size "$DP" \
    --enable-expert-parallel \
    --all2all-backend allgather_reducescatter \
    --enforce-eager \
    --max-model-len 3072 \
    --gpu-memory-utilization 0.88 \
    --max-num-seqs 64 \
    --uvicorn-log-level warning \
    --additional-config "$ADDITIONAL_CFG" \
    --eplb-config '{"log_balancedness":true,"log_balancedness_interval":50,"step_interval":1000000000,"window_size":1000,"use_async":false}' \
    >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  ready=0
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    sleep 5
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "[accuracy] lookahead=$lookahead: server never became ready; skipping" >&2
    tail -30 "$SERVER_LOG" >&2
    kill -TERM "$SERVER_PID" 2>/dev/null || true; sleep 5
    pkill -9 -f "VLLM::" 2>/dev/null || true; sleep 5
    continue
  fi
  echo "[accuracy] lookahead=$lookahead: ready"

  for domain in $DOMAINS; do
    prompt_len="${PROMPT_LEN[$domain]}"
    prompts="$HERE/results/prompts-$domain-p$prompt_len.jsonl"
    tag="L$lookahead-$domain-p$prompt_len"
    echo "[accuracy] $tag: $NUM_PROMPTS requests"
    "$PY_BIN" -m vllm.entrypoints.cli.main bench serve \
      --backend openai-chat --endpoint /v1/chat/completions \
      --model "$MODEL" --port "$PORT" \
      --dataset-name custom --dataset-path "$prompts" \
      --custom-output-len "$OUTPUT_LEN" --ignore-eos \
      --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" \
      --percentile-metrics ttft,tpot --metric-percentiles 99 \
      --save-result --result-dir "$OUT_DIR" --result-filename "bench-$tag.json" \
      >"$OUT_DIR/bench-$tag.log" 2>&1 || echo "[accuracy] $tag bench FAILED" >&2

    # Rotate: the dump path is fixed at launch, so the file is what separates
    # one shape from the next.
    if [[ -s "$DUMP" ]]; then
      mv "$DUMP" "$OUT_DIR/dump-$tag.jsonl"
      echo "[accuracy] $tag: dump $(wc -l < "$OUT_DIR/dump-$tag.jsonl") forwards"
    else
      echo "[accuracy] $tag: EMPTY DUMP - recording did not reach the dump" >&2
      echo "[accuracy] aborting: every later point would fail the same way" >&2
      kill -TERM "$SERVER_PID" 2>/dev/null || true; sleep 5
      pkill -9 -f "VLLM::" 2>/dev/null || true
      exit 5
    fi
  done

  kill -TERM "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  sleep 5
  pkill -9 -f "VLLM::" 2>/dev/null || true
  sleep 10
done

echo "[accuracy] scoring"
for dump in "$OUT_DIR"/dump-L*.jsonl; do
  [[ -s "$dump" ]] || continue
  label="$(basename "$dump" .jsonl | sed 's/^dump-//')"
  "$PY_BIN" "$HERE/prediction_accuracy.py" --dump "$dump" --ks 1 2 4 8 \
    --label "$label" --out "$OUT_DIR/accuracy-$label.json" >/dev/null 2>&1 \
    && echo "[accuracy] $label scored" \
    || echo "[accuracy] $label SCORING FAILED" >&2
done

echo "[accuracy] done; results in $OUT_DIR"
