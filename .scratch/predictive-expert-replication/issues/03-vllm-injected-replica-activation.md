# 03 — vLLM injected ReplicaPlan activation and source-rank routing

**What to build:** Demonstrate an externally injected one-replica `ReplicaPlan` end to end: canonical expert weights arrive in a target inactive slot during Attention, only selected source ranks route to that physical replica, and the following MoE remains output-equivalent to canonical execution.

**Blocked by:** 01 — vLLM predictive infrastructure and fixed layout.

**Status:** ready-for-agent

- [ ] A source-local physical map rewrites routing IDs before dispatch while preserving logical IDs for prediction and metrics.
- [ ] Canonical owner-to-target P2P uses the EPLB communicator and staging workspace; target-slot overwrite waits for the preceding MoE use event.
- [ ] The target commits the pending map only after `replica_weight_ready`; no explicit global activation barrier is introduced.
- [ ] A transfer that exceeds Attention causes the necessary forward wait rather than cancellation or canonical fallback.
- [ ] Distributed tests verify map activation, inactive-slot safety, no collective deadlock, and BF16-tolerant output equivalence.
