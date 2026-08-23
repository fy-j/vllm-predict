# Current implementation status — vLLM CUDA PoC

Read this before changing code. `AGENTS.md` remains binding.

## Scope

Implement only the vLLM CUDA tickets in `issues/01` through `issues/05`. Do not implement an Ascend port in this repository or broaden the approved model/runtime scope.

## Current baseline

- Branch: `feature/predictive-expert-replication`
- Current implementation commit: `dad123223` (`ticket01/02 implement eazy`)
- Original vLLM base: `10704541aaf72567fe9d6229b3e3d84d37f2ddba`
- Tickets #01 and #02 are partially implemented in the same commit. Neither is validated or complete.

## Implemented but unvalidated

### Ticket #01

- `additional_config.predictive_expert_replication` is parsed and represented in parallel configuration.
- Enabled predictive mode validates the initial one-slot, Qwen3-30B-A3B BF16, eager TP/DP/EP, DBO, backend, and readable JSON cost-profile constraints.
- Predictive mode provisions existing EPLB infrastructure, disables Native EPLB control activity, and requests one redundant slot per rank.
- Startup normalization uses existing EPLB rearrangement to install `16 canonical + 1 inactive (-1)` physical rows per rank and records normalization duration.
- CPU-oriented tests cover configuration and fixed-map calculations.

### Ticket #02

- Qwen3 model construction binds adjacent sparse-MoE runners.
- The prediction skeleton evaluates the target gate, selects logical experts through the target router, builds local counts before dispatch, starts EPLB-group AllGather after dispatch, and waits before combine.
- The gathered tensor is currently stored on the current MoE runner only; it is not connected to a planner.

## Must fix or verify before starting ticket #03

1. Exclude dummy and padding tokens from prediction. A dummy-only forward must contribute zero counts and leave lifecycle state unchanged.
2. Replace or intentionally encapsulate the router private `_select_experts` use; verify Qwen3 selection semantics, required inputs, dtype, and routing variants against the actual target router.
3. Add focused tests for adjacent gate binding, final-layer behavior, source-rank provenance, valid-token masking, and equal Global predicted-load snapshots across ranks.
4. Validate asynchronous EPLB-group AllGather ordering and overlap on a real CUDA/NCCL setup. Do not assume the current asynchronous collective choice is correct.
5. Run ticket #01/#02 tests using the project-required `uv`/`.venv/bin/python`, then run formatting/type checks required by `AGENTS.md`.
6. Review configuration mutation order and actual worker/group initialization in a real worker. The current code derives EPLB settings after initial parallel configuration construction.

## Not implemented

- Source-local physical-map rewrite.
- `ReplicaPlan`, deterministic planner, static cost-model consumption, or lifecycle.
- Canonical-owner-to-target P2P, staging workspace, stream events, map commit, or transfer failure path.
- CUDA distributed correctness, P2P/NCCL preflight, model-output equivalence, and serving benchmarks.

## Required reading order

1. `AGENTS.md`
2. `spec.md`
3. `issues/01-vllm-predictive-infrastructure-and-layout.md`
4. `issues/02-vllm-cross-layer-prediction.md`
5. This status document
6. `glossary.md`

Then inspect commit `dad123223`; do not treat its tests or comments as proof of completion.
