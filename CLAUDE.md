@AGENTS.md

## Predictive expert replication CUDA PoC

Before working on this feature, read in order:

1. `.scratch/predictive-expert-replication/spec.md`
2. `.scratch/predictive-expert-replication/issues/01-vllm-predictive-infrastructure-and-layout.md`
3. `.scratch/predictive-expert-replication/issues/02-vllm-cross-layer-prediction.md`
4. `.scratch/predictive-expert-replication/CURRENT-STATUS.md`
5. `.scratch/predictive-expert-replication/glossary.md`
6. `.scratch/predictive-expert-replication/reference/ascend-prior-art.md` (read-only timing prior art; the spec overrides it)

Implement only vLLM CUDA tickets `01`–`05` in this repository. Tickets `01` and `02` are partial and unvalidated; repair and validate them before starting ticket `03`. Do not implement vLLM-Ascend here.
