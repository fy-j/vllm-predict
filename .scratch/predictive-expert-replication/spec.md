# Predictive expert replication for MoE inference

Status: ready-for-agent  
Labels: ready-for-agent  
Scope: initial CUDA PoC in the separately pinned vLLM fork; this document also fixes the contract for the later MindSpeed-RL Ascend port.

## Problem Statement

A hot logical expert makes its canonical owner the slowest EP rank. Native EPLB can create redundant physical experts, but it decides placement from a historical actual-load window and periodically changes placement. That delay can leave bursty or phase-changing workloads imbalanced for several forwards.

The serving system needs to predict the next MoE layer's load early enough to create one useful expert replica while the next layer's Attention runs. It must preserve canonical ownership and output correctness, avoid adding a global synchronization barrier to every replica change, and make its performance claims comparable with both no-replica serving and Native EPLB.

## Solution

Add **Predictive expert replication**: a predictive controller that uses the current layer's cross-layer gate to estimate the next layer's logical-expert loads for every valid routed token. It deterministically selects at most one replica placement per layer, copies that logical expert's canonical weights to one inactive replica slot during the next layer's Attention, and enables source-rank routing to the replica only after the target slot is ready.

The controller reuses **Expert replication infrastructure**—physical slots, logical/physical maps, weight-transfer buffers, and the distinct EPLB communicator—but does not run Native EPLB's historical placement controller. The initial serving implementation is for Qwen3-30B-A3B BF16 on the pinned latest-vLLM fork. The validated lifecycle and policy contract will subsequently be ported to MindSpeed-RL for Ascend, beginning with its All-to-All path.

## User Stories

1. As a serving operator, I want predictive expert replication disabled by default, so that existing deployments retain their current behavior.
2. As a serving operator, I want enabling predictive expert replication to provision the required replica infrastructure without enabling Native EPLB's controller, so that the two placement policies cannot race.
3. As a serving operator, I want an invalid or missing cost profile to fail startup when predictive replication is enabled, so that a benchmark never silently runs without the feature.
4. As a model operator, I want each logical expert's canonical owner to remain fixed, so that replicas improve execution placement without changing model ownership.
5. As a model operator, I want exactly one inactive replica slot per EP rank in the initial PoC, so that memory cost and comparison with Native EPLB are controlled.
6. As a model operator, I want the initial physical layout to contain the rank's canonical experts plus one inactive replica slot, so that every rank can receive one predictive replica.
7. As a serving operator, I want model initialization to finish layout normalization before readiness is reported, so that request TTFT and TPOT exclude startup rearrangement.
8. As an inference request, I want routing to preserve logical expert semantics whether a replica exists or not, so that predictive placement never changes generated output except for normal floating-point execution tolerance.
9. As a source rank, I want all my tokens routed to a given logical expert to use one chosen physical copy, so that the first implementation has deterministic source-rank routing rather than token-level routing.
10. As a distributed worker, I want every rank to derive the same placement plan from the same gathered prediction snapshot, so that no plan broadcast or controller leader is required.
11. As a serving operator, I want predicted-load communication to overlap current-layer expert compute and expert-weight transfer to overlap next-layer Attention, so that replication improves load balance without unnecessarily increasing TPOT.
12. As a serving operator, I want the target rank to wait for the received weights before using a new replica, so that a forward cannot execute against partially copied weights.
13. As a serving operator, I want non-participating ranks to proceed normally to the next EP collective, so that replica activation does not add an explicit global activation barrier.
14. As a model operator, I want a replica to remain resident long enough to amortize its transfer, so that transient prediction changes do not cause copy thrashing.
15. As a model operator, I want an expired replica to be reclaimed only after repeated absence of positive predicted benefit, so that one noisy observation does not unnecessarily change routing.
16. As a serving operator, I want dummy or padding-only forwards to preserve collective ordering while leaving placement unchanged, so that profile and uneven-batch execution remain correct.
17. As a serving operator, I want prediction error to affect only placement performance and never output correctness, so that no online fallback controller is needed in the first release.
18. As a developer, I want plan ownership and forward versions enforced as invariants, so that a plan from one forward cannot be committed by another forward.
19. As a benchmark owner, I want the same replica-slot budget for Native EPLB and Predictive expert replication, so that their TPOT and RPS comparison is fair.
20. As a benchmark owner, I want compute-limited and memory-capacity results reported separately, so that throughput gains are not confused with a reduced KV-cache capacity.
21. As an Ascend operator, I want the later port to preserve the same placement, lifecycle, failure, and benchmark contract, so that CUDA and Ascend results are comparable despite different token communication backends.

## Implementation Decisions

1. **Feature boundary and configuration.** The public feature lives in vLLM `additional_config` under `predictive_expert_replication` and defaults to disabled. The initial configuration contains `enabled`, `cost_profile_path`, `replica_slots_per_rank`, `hot_stable_steps`, `min_residency_steps`, and `hot_load_ratio`. The PoC accepts one replica slot per rank, `hot_stable_steps=2`, and `min_residency_steps=4`; unsupported values fail validation rather than suggesting functionality not implemented.

2. **Controller exclusivity.** Expert replication infrastructure is enabled when either Native EPLB or Predictive expert replication is enabled. Enabling both controllers together is a configuration error. Predictive mode creates physical slots, maps, transfer buffers, and the separate EPLB NCCL communicator, but suppresses Native EPLB's historical-load window, periodic placement, and asynchronous rebalance worker. Actual-load recording may remain available for metrics.

3. **Initial Qwen3 physical layout and routing maps.** For 128 logical experts and EP size eight, each rank starts serving with 16 canonical physical rows and one inactive replica row. The implementation keeps distinct logical, global physical, and local physical expert counts. Logical router IDs remain intact for prediction and metrics. Before token dispatch, each source rank applies its source-local physical map to derive the physical routing IDs; this is what lets the same logical expert use canonical execution for one source rank and a replica for another. An inactive physical row is represented by an explicit inactive mapping and has zero routed-token count.

4. **Startup normalization.** The initial PoC may reuse native EPLB loading and then synchronously normalize weights and maps before the server reports ready. It rearranges canonical rows into the fixed layout, clears inactive rows, and records `startup_normalization_ms`. This is not request-path work. Directly loading only canonical rows is a later optimization.

5. **Cross-layer gate binding and prediction input.** The initial Qwen3-30B-A3B scope has a sparse MoE in every decoder layer. After the ordered decoder layers are constructed, each current sparse MoE binds to the immediately following sparse MoE's gate module; the final MoE has no target and creates no plan. The binding retains the target gate module, never a snapshot of its weight tensor, so normal weight loading and router initialization remain authoritative. Prediction is phase-agnostic: on source-local current-MoE hidden states before token dispatch, the target gate and target router's own selection semantics produce predicted logical-expert IDs and counts for all valid routed tokens. Dummy and padding tokens are excluded. Ranks with no valid tokens contribute zero counts and still participate in the predicted-count AllGather. A pure dummy forward produces no placement or lifecycle update.

6. **Planning API and deterministic policy.** A narrow policy seam accepts a global predicted-load snapshot and returns either one `ReplicaPlan` or no plan. The first policy is `GreedySourceRankPolicy`. All ranks run it locally with deterministic tie-breaking by logical-expert and rank IDs. It chooses the expert, canonical/replica target split, and source-rank partition with the largest positive predicted reduction in peak rank time after exposed, amortized transfer cost. It sorts whole source-rank chunks by predicted load and assigns each to the lighter canonical or replica target. It never token-splits a source-rank chunk.

7. **Static cost profile.** Planner inputs are offline microbenchmarks for per-token expert compute, next-layer Attention overlap window, and rank-pair transfer latency/bandwidth. The profile is validated against model shape, dtype, EP size, and hardware/topology fingerprint at startup. No online calibration or EMA is used in the decision path. The one-time P2P transfer cost is compared with at least `min_residency_steps` of predicted compute gain. An invalid profile fails startup when the feature is enabled.

8. **Communication ownership and order.** EP's existing communicator continues token dispatch and combine. The separate EPLB NCCL communicator performs predicted-count AllGather and expert-weight P2P, using one ordered predictive communication stream. Cross-layer gate evaluation produces source-rank-local counts before dispatch; the predicted-count AllGather starts only after current-layer dispatch and overlaps that layer's local expert GEMM. It completes before the current layer's combine. A pinned CPU snapshot lets the deterministic planner finish before the next layer's Attention begins. Only the canonical owner sends and only the replica target receives weights in the MVP; topology-aware alternate senders are deferred.

9. **Attention overlap and no extra global fence.** The planned P2P is launched on the predictive stream during next-layer Attention. Source and target synchronize the communication only where their own dependency requires it. Other ranks may reach the following EP collective earlier and naturally wait there for participating ranks. There is no additional EPLB-group all-reduce or barrier for replica activation.

10. **Safe replica replacement.** A target receives P2P data into a shared staging workspace. Before a staging-to-slot copy overwrites a slot, the predictive stream waits for the event recorded when that slot was last read by a MoE kernel. It then copies into the slot and records `replica_weight_ready`. The next-layer default stream waits for this event; only then does it promote `pending_plan` to `active_plan` and expose the new map to routing. Maps are never mutated concurrently with router reads.

11. **Lifecycle.** Each layer holds an active plan, at most one pending plan, plan version, and ownership metadata. An empty slot may activate immediately. Replacing an occupied slot requires two consecutive hot observations and satisfaction of the four-step minimum residency. After residency, a replica becomes inactive only after two consecutive observations show no positive benefit. A plan includes its producing layer and forward identity. The model increments a monotonic forward identity; a target accepts only the pending plan produced by its preceding layer for that same forward, then clears pending state. Forward completion asserts that no pending plan remains. With eager execution and `virtual_engine == 0`, a version mismatch is an invariant violation and fail-fast protection, not a fallback path.

12. **Slow and failure behavior.** If Attention does not fully hide P2P, the target waits for `replica_weight_ready`, records exposed wait, activates the plan, and completes the forward. The implementation does not cancel a launched transfer, reroute back to canonical, or silently discard the plan. A true NCCL, HCCL, or copy failure is fail-fast. Prediction error never triggers immediate fallback or online correction; it is measured for later analysis only.

13. **Initial runtime scope.** The CUDA PoC targets one node with eight RTX 5090 GPUs, CUDA 12.8 or newer, BF16 Qwen3-30B-A3B, eager execution, TP=1, DP=8, EP=8, and the `allgather_reducescatter` token backend. DBO is disabled. The initial Ascend port targets BF16/unquantized Qwen3-30B-A3B, eager execution, DBO disabled, and `virtual_engine == 0`; it starts with the available All-to-All path. MC2 decode support is deferred until an environment with at least 16 ranks is available.

14. **Observability.** Normal serving keeps only low-overhead counters. Benchmark mode may collect detailed timing for predicted-count AllGather, planning, P2P, Attention window, exposed wait, hidden ratio, replica switches, replica-routed tokens, and predicted versus actual peak load. It also records prediction accuracy, including hot-expert overlap and count/load error, to validate the high cross-layer similarity reported for deep layers.

15. **Benchmark contract.** Every reported configuration uses the same model, request mix, hardware, and slot budget. The three mandatory baselines are no EPLB/no replica, Native EPLB with the same redundant-slot count, and Predictive expert replication. Compute-limited measurements report fixed-RPS TPOT, TTFT, end-to-end latency, EP load distribution, and maximum sustainable RPS under a p99 TPOT SLO. Memory-capacity measurements separately report KV blocks, peak memory, maximum concurrency, and the OOM boundary. Startup normalization is separately reported and excluded from serving latency.

## Testing Decisions

1. **Primary integration seam.** Test through the existing configured MoE model-forward seam, because it observes the externally meaningful contract: identical logical routing semantics and output versus canonical-only execution, correctly activated replicas, and valid distributed forward completion. Do not test private stream calls as the primary proof of correctness.

2. **Planner and lifecycle unit tests.** Use CPU-only tests for deterministic `ReplicaPlan` selection, source-rank partitioning, tie-breaking, positive-benefit rejection, cost-profile validation, residency/hotness transitions, inactive-slot reuse, dummy-forward behavior, and forward-version ownership. Test the Qwen3 cross-layer-gate registry binds each sparse MoE to the immediately following sparse MoE gate, leaves the final MoE unbound, and rejects an unsupported non-adjacent topology. These tests assert plans and observable lifecycle states, not internal implementation order.

3. **Existing EPLB state and rearrangement prior art.** Extend the repository's existing EPLB algorithm, execution, event, fused-MoE-layer, routing, and GPU-model-runner EPLB test suites rather than creating parallel test infrastructure. In particular, test the existing map and weight-rearrangement seam with inactive physical rows, canonical-only normalized layout, and weight equality after transfer.

4. **CUDA distributed correctness.** On the eight-GPU RTX 5090 target, verify P2P replica transfer and source-rank routing across ranks; compare logits or generated outputs to no-replica reference within BF16 tolerance; verify all ranks complete repeated replica activation, replacement, and reclamation without collective deadlock. Exercise a transfer that outlasts Attention and verify the forward waits and records exposed time.

5. **Negative configuration and failure tests.** Verify disabled mode preserves existing behavior; Native EPLB plus Predictive mode is rejected; invalid/missing/mismatched cost profiles prevent readiness; unsupported initial scope is rejected; and a deliberately injected communication failure follows the fail-fast path. Verify startup normalization completes before readiness and never becomes first-request TTFT work.

6. **Ascend tests.** Add BF16-tolerant integration tests for the same map, lifecycle, and output contract. They must auto-skip when `torch_npu` or the required distributed environment is absent. Actual HCCL overlap, zero-count grouped-MatMul behavior, and MC2 tests run only on an NPU server.

7. **Performance validation.** Before interpreting serving results, capture `nvidia-smi topo -m`, CUDA P2P bandwidth/latency results, and NCCL test results. A topology without usable P2P is reported as an environment limitation rather than evidence for or against the policy. Run all three baselines and both benchmark families using the metric contract above.

## Out of Scope

- Token-level routing, more than one replica slot per rank, or multiple replicas per layer.
- Moving canonical ownership or full expert migration.
- Native EPLB and Predictive expert replication operating concurrently.
- Online calibration, EMA-driven decisions, prediction-error fallback, or automatic controller switching.
- Lazy replica tensor allocation, a distinct expert-GEMM stream, topology-aware alternate P2P sources, and direct canonical-only checkpoint loading.
- Quantized or mixed-quantization expert transfer support.
- DBO, nonzero virtual engines, pipeline parallel scheduling, and broader model-family support in the first PoC.
- Ascend MC2 decode implementation or validation before a 16-rank-capable NPU environment is available.

## Further Notes

- The initial CUDA work belongs in the separately pinned latest-vLLM fork, not in MindSpeed-RL. MindSpeed-RL remains the home for the later Ascend adaptation and its tests.
- One BF16 Qwen3-30B-A3B expert is approximately 9 MiB. One static slot on each of 48 MoE layers is approximately 432 MiB per rank; this is why memory-capacity results are mandatory and distinct from compute-limited results.
- The paper's strong deep-layer cross-layer gate similarity is a hypothesis to validate with prediction metrics and serving outcomes, not a correctness assumption.
- The startup-normalization choice intentionally prioritizes a fast, low-risk PoC. It may temporarily load redundant rows that serving does not retain; that cost belongs to startup accounting only.
- No repository ADR conflicts were found for this work.
