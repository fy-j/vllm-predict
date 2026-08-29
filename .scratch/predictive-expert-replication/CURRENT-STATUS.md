# Current implementation status — vLLM CUDA PoC

Read this before changing code. `AGENTS.md` remains binding.

**Moving to another node?** Start with
[`HANDOFF-2026-08-29-H100.md`](HANDOFF-2026-08-29-H100.md), written for a session with
no prior context.

**Just want the conclusions?** Read "Current baseline" and "What today's profiling
settled" below, in that order. [`HANDOFF-2026-08-24.md`](HANDOFF-2026-08-24.md) holds
the earlier decode answer and is **superseded on the prefill question**: the target
moved to TTFT on 2026-08-25 and its verdict predates every measurement since. Method
and caveats stay in `bench/RESULTS.md`.

## Scope

Implement only the vLLM CUDA tickets in `issues/00` through `issues/09`.  Do not implement an Ascend port in this repository or broaden the approved model/runtime scope. `reference/ascend-prior-art.md` is read-only timing prior art; the spec overrides it.

## Current baseline

- Branch: `feature/predictive-expert-replication`
- Original vLLM base: `10704541aaf72567fe9d6229b3e3d84d37f2ddba`
- Tickets `01` `02` `03` `06` `10` implemented and closed. `00` answered for both
  regimes. `11` `12` open. `04` `05` `07` `08` `09` unstarted.
- The feature runs end to end and is **correct**: replicas are transferred, published
  where routing reads, routed to, and reverted. It removes about **15% of prefill
  critical-path excess** against an oracle of 35%.
- It is **net negative on this node**: mean TTFT +13% to +29% across four
  configurations, TPOT +7%.
- **The cost is host synchronisation, not the transfers.** See the section below
  before touching anything.

## What today's profiling settled (2026-08-26)

Read this before optimising. Four earlier rounds of design chased the wrong cost.

**The replica transfers are not the bottleneck.** On a two-arm torch profile
(`bench/results/placement-profile/`, `bench/run_placement_profile.sh`), the P2P
transfers are **5.9 ms, 0.32% of GPU time**, 20 `ncclDevKernel_SendRecv` calls on
their own stream. They are genuinely asynchronous and genuinely overlapped. Every
claim that "the transfer cannot be hidden" was wrong, including the argument for
raising the lookahead to widen the overlap window.

**The cost is the host waiting for device data.** Measured inside the real forward
windows: GPU occupancy **86.8% (baseline) versus 52.9% (placed)**, gaps over 0.5 ms
totalling **3.0 ms versus 30.2 ms**, and one `cudaEventSynchronize` of **11.02 ms**.
The CPU runs ahead of the GPU by roughly that much; `plan_and_launch` throws that
run-ahead away every predicted layer, and the engine becomes launch-bound.

**It is structural, not a stream-placement mistake.** `ncclSend`'s peer is a
`ctypes.c_int` consumed when the host enqueues (`pynccl_wrapper.py`), so the host must
know which expert goes to which rank, so it must read a device tensor, so it must wait.
Moving the planner to the GPU does not change this on its own. The escape is
device-initiated one-sided put (NVSHMEM), which needs NVLink this node does not have.
`bench/probe_nvshmem.py` is the 8-GPU validation to run on H200 first.

**The prediction snapshot AllGather is 4 KB and costs 5 us to 11181 us.** p50 493 us.
Two thousand-fold spread on a fixed payload: it is a barrier waiting for the slowest
rank, so what it measures is arrival skew, not communication. Faster interconnect
shrinks the floor, which was never the cost. It runs on its own stream already.

**The baseline arm is not a clean baseline.** Both arms pass
`predictive_expert_replication.enabled = True`; only the budget differs. So every
TTFT and TPOT comparison in this branch answers "what does placement cost on top of
prediction", not "what does the feature cost". Prediction alone measured **19% of
TPOT** at concurrency 1, where the placement gate rejected every forward. A third arm
with the feature fully disabled is needed before quoting a total cost.

**Whole-profile totals are contaminated by idle dummy batches.** In a 12.95 s trace
the annotated forward windows cover only the last **0.54 s**; the rest is
`execute_dummy_batch` keeping DP ranks in lockstep while the benchmark client starts
up. Aggregate operator tables mix that in. Restrict every measurement to the
`execute_context` annotations.

## Four defects fixed today, each of which alone made the feature inert or worse

Any of these would also have wasted a run on new hardware.

1. **Published into buffers routing does not read.** `_apply_eplb_mapping` prefers
   `source_local_physical_map`; the activation path wrote `logical_to_physical_map`.
   Replicas were transferred, described correctly, and never routed to: physical
   per-rank load equalled canonical ownership to 0.00%, and 131 replicas per forward
   removed 0.6% where the oracle removes 35.1%.
2. **The transfer budget was spent per layer.** `max_transfers_per_forward` caps a
   forward; each layer received the full budget and 43 layers spent it 43 times.
3. **Nothing reverted.** `reconcile`, `revert_replicas` and `active_replicas` were
   written for the whole-layout path and sat uncalled. Replicas accumulated to 47-76
   per forward against a budget of 43.
4. **The budget concentrated on the earliest layers.** Each layer planned with the
   whole remaining budget, so 6-7 layers took ~5 replicas each and removed 5.0%.
   Coverage is what drives benefit: 22 layers removes 16.9%.

A fifth, found and fixed earlier the same day: the snapshot copy's event was recorded
on the predictive stream while the copy was enqueued on the current one, so the wait
was vacuous, ranks planned from partly-filled buffers, and the engine deadlocked with
no error.

## Environment (important)

The installed `vllm` at `/usr/local/lib/python3.12/dist-packages/vllm` is **not** this repository: 869 of 2272 Python files differ from the base commit, and it does not contain `vllm/distributed/eplb/predictive.py` at all.

**Which copy loads depends on how the process starts, and the difference is not intuitive:**

| Invocation | Loads | Why |
| --- | --- | --- |
| `python3 -c ...`, `python3 -m vllm...`, `pytest` | this repository | the working directory is `sys.path[0]` |
| `vllm serve ...` | the **installed** build | it is a console script, so `sys.path[0]` is `/usr/local/bin`; the working directory is never on the path |

A `vllm serve` run therefore silently measures stock vLLM. It does not fail: `additional_config` is a free-form dict, so `predictive_expert_replication` is accepted and ignored, and the only hint is a `WARNING: Unknown vLLM environment variable` line. One prediction-accuracy run produced six empty dumps this way.

Every harness script must therefore `export PYTHONPATH="$REPO_ROOT"` **and assert the loaded module resolves inside the tree**, as `run_prediction_accuracy.sh`, `run_boundness_profile.sh`, and `verify_inactive_slots.sh` now do. Prefer `python3 -m vllm.entrypoints.openai.api_server` over `vllm serve`: it puts the tree on the path anyway, and predictive mode *requires* it, because `vllm serve` starts one API server per DP rank and the config round trip keeps `num_redundant_experts` while losing `enable_eplb`, so every rank dies on "num_redundant_experts is set to 8 but EPLB is not enabled".

To run the repo's own code without a full rebuild, the installed compiled extensions are symlinked into the source tree (`*.so` is gitignored, so the repository stays clean):

```bash
for f in /usr/local/lib/python3.12/dist-packages/vllm/*.so; do ln -sfn "$f" "vllm/$(basename $f)"; done
for f in /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/*.so; do ln -sfn "$f" "vllm/vllm_flash_attn/$(basename $f)"; done
```

Packages installed on this machine for this work, all system-wide:

| Package(s) | Why |
| --- | --- |
| `tblib` | `tests/conftest.py` imports it; without it every pytest run fails at collection |
| `ruff`, `pre-commit`, `mypy` | reinstalled 2026-08-24 (`pip install ruff pre-commit mypy`). They had gone missing, and work done while they were absent checked only the 88-character limit by hand; that work has since been linted. |
| `datasets` (pulls `pandas`, `pyarrow`, `xxhash`) | `vllm bench serve` dataset loading |
| `matplotlib`, `seaborn`, `scipy`, `plotly` | the rest of vLLM's `bench` extra (`setup.py`) |

Install the `bench` extra's packages **by name**, never as `vllm[bench]`: resolving
that form can replace the installed build these benchmarks measure.

Symlinks into the source tree so the repo's code can run, all gitignored or listed
in `.git/info/exclude` so the repository stays clean:

- `vllm/*.so` and `vllm/vllm_flash_attn/*.so`
- `vllm/third_party/{deep_gemm,fmha_sm100,tml_fa4,triton_kernels}` and
  `flashmla/flash_mla_interface.py`, absent from this tree; without them warmup
  dies with `No module named vllm.third_party.flashmla.flash_mla_interface`

Verified working: config validation, router kernels (`topk_softmax`), single-GPU
MoE layer tests, and a full 8-GPU predictive-mode server.

**This is adequate for unit work only.** Ticket #05 serving benchmarks and any claim about real overlap or end-to-end output require a genuine build of this tree (`VLLM_USE_PRECOMPILED=1 uv pip install -e .`), because the symlinked kernels come from a different commit.

## Model runner: V1 is the target

`Qwen3MoeForCausalLM` is **not** in `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` (`vllm/config/vllm.py`), so the PoC runs on the **V1** runner (`vllm/v1/worker/gpu_model_runner.py`), not `vllm/v1/worker/gpu/model_runner.py`.

Predictive behavior therefore lives in **`EplbState`**, which both runners funnel through (`add_model`, `step`, `prepare_forward`). Do not add predictive branches to either runner's EPLB glue; extend `EplbState` so both paths stay identical.

## Ticket 03: complete, output equivalence verified on hardware

Source-rank routing is implemented and its last criterion is met. A startup
assertion compares position-weighted checksums of every physical copy of each
logical expert across the EP group and confirmed **48 of 48 layers byte-identical**
with a replica placed. Because the copies are provably identical, routing to either
is arithmetically equivalent and any output difference can only be summation order.
argmax was unchanged and 96 greedy tokens matched.

Two lessons are recorded in `bench/RESULTS.md` rather than only here. A control run
was needed: text equality first failed, and the obvious cross-process
nondeterminism explanation was wrong, since two canonical runs are byte-identical.
And the logprob tolerance used was invented rather than calibrated, so it could not
decide anything; the weight-equality assertion is what settles it, and it now
reports how many pairs it compared so a vacuous pass cannot be mistaken for a real
one.

## Tickets 01 and 02: validated on hardware

Confirmed on the 8-GPU node with this repository's code (`--load-format dummy`,
so no 57 GiB read; details and the exact command in `bench/RESULTS.md`):

- Config validation and cost-profile fingerprint accepted on a real worker.
- Startup normalization ran before readiness in **218 ms**, installing 16
  canonical + 1 inactive rows per rank.
- **Native EPLB placement stayed suppressed**: exactly one rearrangement in the
  whole run, and it is the profile reservation. This was the defect that made the
  original ticket 01 inert, so it is the criterion that most needed proving.
- Forwards complete with prediction active across 43 source layers on 8 ranks,
  0 failed requests at concurrency 12, no invariant violation.

Two environment fixes were needed first, neither caused by the feature: the repo
tree lacks four vendored `vllm/third_party` payloads (symlinked, listed in
`.git/info/exclude`), and the cost-profile fingerprint must pin the served model
*path* rather than the HuggingFace name.

Not established here: overlap timing and output equivalence. Dummy weights cannot
show either; both belong to ticket 05 with real weights.

## Implemented and unit-validated

### Ticket #01

- `additional_config.predictive_expert_replication` parses into `ParallelConfig.predictive_expert_replication_config`, disabled by default.
- Enabling it provisions Expert replication infrastructure (`enable_eplb`, `num_redundant_experts = ep_size * replica_slots_per_rank`, `use_async=False`, `communicator="pynccl"`) and rejects Native EPLB.
- Because the mutation happens after `ParallelConfig` validation, the EPLB preconditions that validator enforces are re-checked in `VllmConfig._validate_predictive_runtime_scope` (CUDA platform, TP/PCP/PP/DP, expert parallelism, DBO, token backend, no speculative decoding).
- The cost profile must be readable JSON with a full `fingerprint` (model, dtype, ep_size, num_logical_experts, device_name) and four positive cost values. `validate_fingerprint` rejects a profile measured on another runtime; the config validator checks model/dtype/EP, leaving `device_name` for the worker.
- `EplbState.add_model` installs the fixed `16 canonical + 1 inactive (-1)` layout, zeroes the inactive rows, republishes the maps, and accumulates `startup_normalization_ms`.
- `EplbState.step` returns before Native EPLB's load window and periodic rearrangement in predictive mode, but still performs the `is_profile` rearrangement so transfer buffers stay reserved for ticket #03.

### Ticket #02

- `FusedMoERouter.select_logical_experts` is a new read-only routing seam: `_compute_routing` only, so it applies no EPLB mapping, records no expert load, and writes no routing-replay state. `BaseRouter` implements it, covering every router in the tree.
- `CrossLayerLoadPredictor` evaluates the target gate and target router on source-local pre-dispatch hidden states, counting **logical** experts (`num_logical_experts`, not the physical `num_experts`) with a sync-free `scatter_add_`.
- Padding is excluded using the EPLB `num_unpadded_tokens_tensors` device scalar, matching what the existing EPLB load-recording kernel already does. `_dummy_run` now publishes a zero count, so a dummy forward predicts zero load and still joins every collective.
- The AllGather starts after dispatch, overlaps the local expert GEMM, and completes before combine. The gather buffer is flat because `ProcessGroupGloo` rejects a pre-shaped `[ep_size, num_logical]` output that NCCL accepts.
- `bind_adjacent_moe_prediction_targets` is the cross-layer gate registry: it binds each MoE to the next, leaves the final MoE unbound, and rejects a topology with a dense-layer gap.

## Spec revision (this session)

`spec.md` was revised after measuring the target node. The changes that alter existing contracts:

- **Prediction lookahead is configurable** (`prediction_lookahead_layers`, default 2). A lookahead of 1 cannot hide a 175 us expert transfer behind a 25-31 us Attention block on PCIe. Plan ownership is now "the layer `lookahead` positions back", and the invariant is "at most one *inbound* pending plan per layer" — several plans are legitimately in flight across the model at once.
- **Leading layers predict nothing** (`prediction_skip_first_layers`, default 3), matching the measured unreliability of early-layer cross-layer prediction.
- **Overlap now means "concurrent AND not the new bottleneck"**, bounded by `max_concurrent_transfer_bytes` (default one expert) rather than a per-forward count.
- **Cost profile keys changed**: `attention_overlap_window_us` -> `attention_window_us`; `transfer_bandwidth_bytes_per_us` -> `usable_transfer_bandwidth_bytes_per_us`, which must be measured under load, not idle.
- **`hot_load_ratio` is only a candidate pre-filter** (default raised 1.0 -> 1.5). The policy's positive-benefit test is the single source of truth for hotness.
- **Enforced minimum residency is derived**, `max(config, ceil(exposed transfer / per-step gain))`.
- **Staging reuses `expert_buffer`'s leading row** (0 new bytes). `EplbState.rearrange` now raises in predictive mode unless `is_profile`, which is the enforced precondition of that reuse.
- **Ticket 00 is new and gates 03/04/05.** Do not implement further replication until its headroom measurement and proceed/stop recommendation exist.
- **Ticket 03 was split into three.** It carried eight criteria spanning routing, P2P, stream events, a map-commit protocol and distributed tests, which does not fit one context window. Now: `03` source-rank routing (verifiable with a statically placed replica, no transfer machinery, and **not** gated by `00` because it is the spec's correctness contract), `07` safe weight transfer into an inactive slot (byte equality and event ordering only, nothing routes to the slot, gated by `00`), and `08` activation end to end (joins the two). `03` and `07` are independent and can run in parallel. Ticket `04` now depends on `08` rather than the old `03`.
- **Ticket 06 is new**: the prediction-accuracy study was split out of ticket 02, because it needs a real multi-GPU model run rather than CPU code, it decides the lookahead default, and ticket 04's cost model consumes it. It is blocked only by 02, so it can run alongside 00. Filename numbers no longer imply order; each ticket's `Blocked by` field does, and `CLAUDE.md` carries the graph.
- **Workload matrix is fixed**: two request shapes (2k/1k prefill-weighted, 1k/2k decode-weighted) over text, code, and math. Decode length forced; prompt length filtered from real samples. Never use vLLM's synthetic `random` dataset for the headroom number — it sweeps the vocabulary and routes almost uniformly, which understates the imbalance being measured. It is a control only.
- **Decode concurrency is capped by KV capacity on 32 GB cards** to roughly 50-83 sequences per rank, putting the MoE-time to weight-read-floor ratio at about 1.4-1.8. A comfortably compute-bound decode regime is not reachable, so that ratio is a reported condition of every result rather than a configurable precondition.
- **No numeric success threshold is fixed in advance.** The goal is lower p99 TPOT at equal RPS (equivalently higher sustainable RPS at fixed SLO). Results are reported as the share of measured imbalance headroom recovered, because an absolute percentage cannot separate a weak policy from an operating point with little to recover.

## Ticket set restructured

`00` was carrying both the feasibility decision and the serving report, and it
gates `07`/`04`/`05`, so implementation was waiting on reporting that has no
bearing on the decision. Split: `00` now asks only whether imbalance converts into
time, and new `09` holds the serving baseline and gates only `05`. `07` was written
for one transfer per layer and now covers up to `max_replicas_per_layer`.

Tickets `01` and `02` are marked done and `03` is 5 of 6; their criteria had been
left unticked, which read as though none of the work existed.

## Settled: the placement policy touches several distinct experts per layer

Measured per-layer load concentration (`bench/RESULTS.md`, fifth session) shows
one expert per layer **cannot** equalize a layer, at any fan-out. Within a layer the
peak rank must shed 39.6% (median) while its hottest expert carries only 33.6%, and
spreading one expert of share `s` over `K` ranks sheds at most `s x K/(K+1) < s`.
The top three experts carry 68.5%, so touching two is enough.

Chunk granularity is not the constraint: 8 source-rank chunks give 256 achievable
split points. The constraint is the spec's "one `ReplicaPlan` per layer".

The user chose fanning one expert out to several ranks, but that choice was made
against a wrongly aggregated version of this data. The measurement now favours
replicating several different experts instead. **Unresolved.**

## ~~Ticket 00 feasibility checkpoint: UNRESOLVED~~ (superseded; answered for both regimes)

An earlier negative verdict was **withdrawn**. The sweep meant to reach a decode
batch large enough to leave the expert-weight-bandwidth-bound regime never got
there: deriving the running batch from `throughput x TPOT` shows 12, 24, 46, 46,
46 sequences per rank, flat from concurrency 512 on, because `--num-prompts 400`
caps requests in flight regardless of `--max-concurrency`. It explored 12 to 46
per rank, and the probe grid puts the transition near 128.

Three caps decide the decode batch and all three must be raised together: the
client's prompt count, `max_num_seqs` (default **128 per rank**, exactly the
threshold to cross), and KV capacity (roughly 700 per rank at a 256-token
context, so not binding). The harness now refuses a point whose concurrency
exceeds the prompt count, and exposes `max_num_seqs`.

What still holds from that run: p99 TPOT is 79.7 to 84.6 ms across every
configuration measured so far, two shapes, two domains, contexts from 256 to 3072
tokens, and batches from 8 to 46 per rank. Measured rank imbalance is 50.8%
median. Whether that imbalance converts into time at a *larger* batch is the open
question.

## ~~Ticket 00 feasibility checkpoint: negative~~ (withdrawn; see above)

Measured with real weights and this repository's code. Full numbers and the exact
commands are in `bench/RESULTS.md`, third session.

Across the whole reachable decode range (16 to 64 sequences per rank; beyond that
requests queue rather than run) **p99 TPOT is flat at 79.7 to 84.6 ms, +6%**, while
measured rank imbalance is **50.8% median**. At 64 sequences per rank MoE accounts
for about 5.2 ms of an 84.6 ms TPOT, so **6.2%**. Perfectly balancing every layer
would save at most 3.1% of TPOT, and one replica per layer captures a fraction of
that. The imbalance is real; it does not convert into time here.

**Tickets 07, 08, 04 and 05 should not proceed on this configuration.** Ticket 03
is unaffected: source-rank routing is the correctness contract, not the optimization.

This is a conclusion about this node, not the policy. Eager execution is
spec-mandated and eager launch overhead is the likely occupant of the other 78 ms;
under CUDA graphs MoE's share would rise, though whether the predictive stream and
event machinery is graph-capturable is itself unsettled. The device also has no
tuned MoE configuration, which *overstates* MoE's share.

Note the terminology: this is the **feasibility checkpoint**, a project decision.
It says nothing about prediction accuracy, which is ticket 06 and still unmeasured.
"gate" is reserved for the MoE routing gate.

## Ticket 00 first results (measured, partial)

Harness lives in `.scratch/predictive-expert-replication/bench/`; **every session's raw numbers are in `bench/RESULTS.md`**, appended per session. Measured on 8xRTX 5090 with `/models/preset/Qwen/Qwen3-30B-A3B/v1.0`, served by the **installed** vLLM 0.27.1, not a build of this tree. Re-measure with the predictive build before ticket 05 compares against it.

**Headroom is real and large.** EPLB balancedness over 3.5k logged steps gives median headroom (1 - avg/max summed per layer) of **44%**, p95 52%, stable across load levels (39-45%). That is far more than sampling noise, so Qwen3-30B-A3B has genuine routing skew. This is the encouraging half.

**But it does not convert into time at reachable concurrency.** Measured MoE-time to expert-weight-read-floor ratio is **1.02 at 8 tokens per rank** (concurrency 64 over DP=8). Reading a rank's 144 MiB of local experts costs 102 us, and MoE at that load costs 102 us: the layer is entirely weight-streaming bound, so a 44% token imbalance costs almost nothing. The probe grid only reaches ratio 1.87 at 128 tokens per rank, and KV capacity caps decode at roughly 85 per rank.

**Lookahead 2 is confirmed as the right call.** One expert is 9.00 MiB and moves in 176-177 us, uniform across all 56 pairs at ~54 GB/s (PCIe Gen5 x16; the RTX 5090 has no NVLink). Lookahead 1 hides only 15% of that. Lookahead 2 gives a 155 us window and hides **87.5%**, leaving 22 us exposed. *(Corrected 2026-08-24: the window is 162.7 us and hides 92%; see `bench/RESULTS.md`.)*

**A larger structural problem surfaced.** Measured p99 TPOT is ~82 ms at concurrency 64, while 48 layers of attention plus MoE account for only ~6.5 ms of it. Eager execution, which the spec mandates, leaves per-layer launch overhead dominating TPOT at the concurrency KV capacity allows. Any MoE-level gain is therefore diluted by roughly an order of magnitude before it reaches TPOT. This bounds the achievable TPOT improvement well below the imbalance figure and needs confirming with a profile.

**Two harness defects invalidated the serving latency numbers** and are now fixed: prompt length was never controlled (actual prompts were 87-167 tokens while filenames claimed 1024/2048, so neither agreed request shape was tested), and two benchmark loops shared one server. `make_prompts.py` now builds fixed-length prompts from real text, and runs take an exclusive lock. The imbalance figure is unaffected, being a per-step routing property rather than a latency measurement.

~~**Verdict so far: do not start ticket 03.**~~ **SUPERSEDED 2026-08-24**: ticket 03 is complete (7 of 7) and was never gated by the feasibility question — source-rank routing is the spec's correctness contract, not an optimization. Everything this paragraph lists as still-to-measure has since been measured; see `bench/RESULTS.md` 2026-08-24. The original text follows for the record. Not because the policy is wrong, but because the tested operating point cannot reward any placement policy. Still to measure before the gate is settled: concurrency high enough to leave the weight-bound regime, per-domain and per-request-shape breakdown, PCIe utilization under load to replace the idle bandwidth currently in the cost profile, and a profile attributing TPOT so the MoE share is known rather than inferred.

Caveats that could change the verdict: the boundness ratio comes from a `bmm` microbenchmark rather than vLLM's fused MoE kernel; and no tuned MoE config exists for this device, so the served kernel is untuned.

## Test status

`tests/distributed/test_predictive_expert_replication.py` — 56 passing:

```bash
python3 -m pytest tests/distributed/test_predictive_expert_replication.py -q
```

Covers configuration and scope rejection, cost-profile schema and fingerprint mismatch, the fixed layout and its logical maps, the gate registry (including gap rejection), and — on GPU — that prediction counts logical experts, does **not** apply the EPLB physical mapping, does **not** record actual load, excludes padding, and predicts zero for a dummy forward. A 4-rank gloo test asserts every rank receives a byte-identical snapshot with correct source-rank provenance.

No regressions: `test_eplb_algo.py`, `test_eplb_utils.py` (21 passing) and `tests/kernels/moe/test_moe_layer.py` (291 passing). `test_moe_layer[False-allgather_reducescatter-1-4-False]` fails **identically on the unmodified base commit** — pre-existing, confirmed by stash-and-rerun.

`pre-commit run` passes on every changed file, including `mypy-3.12` at CI's hook stage.

## Not yet validated

1. **The feature's total cost.** Every measurement compares placement against
   prediction, because both arms enable prediction. A third arm with
   `predictive_expert_replication.enabled = False` is needed.
2. **Where the residual TTFT goes.** Inside the real windows, the transfers (5.9 ms),
   the extra syncs (+10.1 ms) and the exposed prediction AllGather (5.0 ms) account
   for about 21 ms of a 70 ms gap. The rest is the GPU idle gaps, but the two arms'
   windows are not strictly paired (4 versus 6, different first-window names), so that
   comparison is not sound either. A profile with matched window shapes is needed.
3. **Why 15% of prefill excess and not the oracle's 35%.** Believed to be the online
   planner's causality — it cannot see later layers' relative value — but unverified.
   Offline, planning from predicted load reaches 33.2%, so it is not accuracy.
4. **Lookahead 4 and 5.** Ticket 10 measured 1, 2 and 3. If the H200 probe shows a
   9 MiB put near 13 us, raising the lookahead becomes unnecessary and this stays
   unmeasured on purpose.
5. **Cross-forward residency.** Ticket 00 measured -20.0%; this branch measures decode
   with resident replicas at **+3.0%**. The two disagree and the conditions differ.
   The current code keeps replicas resident across gated forwards on the strength of
   the direct measurement.
6. **Under-load interconnect bandwidth.** The cost profile still carries the idle
   figure. Ticket `09` owns it.

## Not implemented

- **Device-side planning and device-initiated transfer.** The design is settled and
  priced; see the profiling section above and `bench/probe_nvshmem.py`. Needs NVLink.
- **Ticket 12**: VMM aliasing to cut the replica slots' 432 MiB per rank to ~45 MiB.
  Deferred, with the alternative (slots on only the worst K layers) ruled out by
  measurement — it is a straight line, ~0.65 points per layer, no knee.
- **Tickets 04, 05, 07, 08, 09, 11.**
- A cross-layer batched prediction snapshot, which is the other way to cut the 43
  collectives per forward. Not designed.

## Required reading order

1. `AGENTS.md`
2. `spec.md`
3. This status document — "Current baseline", then "What today's profiling settled".
   Read them before any optimisation: four rounds of design chased a cost that
   measured 0.32%.
4. The ticket being picked up, from `issues/`
5. `glossary.md`
6. `reference/ascend-prior-art.md` (read-only; the spec overrides it)
