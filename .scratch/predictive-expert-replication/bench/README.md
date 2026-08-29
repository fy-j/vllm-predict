# Ticket 00 baseline harness

Measures whether Predictive expert replication has anything to win on this node,
before any replication code is written.

```bash
# 1. Hardware, transfer cost, per-layer budget, KV ceiling. No server needed.
python3 .scratch/predictive-expert-replication/bench/probe_hardware.py

# 2. Serving baseline. Two configurations, deliberately separate:
#    memory    - no EPLB. The honest memory/KV baseline.
#    recording - EPLB with zero redundant experts and an unreachable rearrange
#                interval, so it logs per-rank load without changing placement.
#                Costs one extra transfer buffer, so it must not supply the
#                memory numbers.
bash .scratch/predictive-expert-replication/bench/run_serving_baseline.sh recording
bash .scratch/predictive-expert-replication/bench/run_serving_baseline.sh memory

# 3. Aggregate into the serving report (ticket 09) and seed a cost profile.
python3 .scratch/predictive-expert-replication/bench/report.py --tokens-per-rank 64

# 4. Ticket 00's decisive measurement: what a decode step actually spends time on,
#    at several real batches. Windows are pure-decode only and the step count comes
#    from the runner's own annotation, because getting either wrong silently halves
#    or doubles every figure.
bash .scratch/predictive-expert-replication/bench/run_boundness_profile.sh
python3 .scratch/predictive-expert-replication/bench/boundness_table.py

# 5. Ticket 06: prediction accuracy against lookahead. One server per lookahead.
bash .scratch/predictive-expert-replication/bench/run_prediction_accuracy.sh
python3 .scratch/predictive-expert-replication/bench/accuracy_report.py

# 6. Ticket 03: observe that an inactive replica slot attracts no routed tokens.
bash .scratch/predictive-expert-replication/bench/verify_inactive_slots.sh
```

Every session's numbers go in `RESULTS.md`, appended, never overwritten.

## Dependencies

`vllm bench serve` needs vLLM's `bench` extra (defined in `setup.py`):
`pandas matplotlib seaborn datasets scipy plotly`.

Install the packages directly, **not** as `vllm[bench]`:

```bash
uv pip install --system pandas matplotlib seaborn datasets scipy plotly
```

Resolving `vllm[bench]` would try to satisfy `vllm` itself and can replace the
installed build with one from PyPI, which is exactly the build these benchmarks
are measuring. Installing the extra's packages by name avoids touching vLLM.

## Datasets

Two paths exist and they do not accept the same datasets.

Through `make_prompts.py` (what the harness uses, because it is the only way to
fix prompt length) all three work, verified at both target lengths:

| Domain | Dataset | Notes |
| --- | --- | --- |
| text | `Aeala/ShareGPT_Vicuna_unfiltered` | large; real multi-turn conversation |
| code | `likaixin/InstructCoder` | `instruction` + `input` joined, so the prompt contains real code |
| math | `AI-MO/NuminaMath-CoT` | large; first download is slow |

Through vLLM's own `--dataset-name hf` path, `Aeala/ShareGPT_Vicuna_unfiltered`
fails against `datasets` 5.x (`TypeError: string indices must be integers`).
`make_prompts.py` is unaffected because it reads the dataset itself rather than
going through vLLM's dataset classes.

`philschmid/mt-bench` also works but has only 80 samples, so it cannot supply
many 2k-token prompts; prefer ShareGPT for the prefill-weighted shape.

Never use `--dataset-name random` for the headroom number: it generates prompts
by sweeping the vocabulary, which routes almost uniformly across experts and so
understates the imbalance being measured. It is a control only.

`make_prompts.py` builds fixed-length prompts. It exists because
`vllm bench serve` can force an HF dataset's decode length but not its prompt
length, so the two agreed request shapes cannot be produced from a stock HF run.
The harness calls it automatically and serves the result as a custom dataset.

Runs take an exclusive lock. Two benchmark loops sharing one server silently
corrupts every latency number, so a second run refuses to start rather than
measure nonsense.

## Which vLLM a run measures

`vllm serve` is a console script, so `sys.path[0]` is `/usr/local/bin` and the
working directory is **never** on the path: it loads the installed build, not this
tree, and that build does not contain the feature at all. It does not fail either,
because `additional_config` ignores unknown keys. Every runner here exports
`PYTHONPATH` and aborts unless vLLM resolves inside the tree; keep that guard in
anything new. Prefer `python3 -m vllm.entrypoints.openai.api_server`, which is
also required for predictive mode — `vllm serve` starts one API server per DP rank
and the config round trip loses `enable_eplb` while keeping
`num_redundant_experts`.

## Tests

The pure logic in each tool is unit-tested; the runners are shell and are not.

```bash
python3 -m pytest .scratch/predictive-expert-replication/bench/
```
