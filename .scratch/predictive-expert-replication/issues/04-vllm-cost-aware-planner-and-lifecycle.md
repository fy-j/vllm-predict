# 04 — vLLM cost-aware planner and replica lifecycle

**What to build:** Connect real Global predicted-load snapshots to deterministic one-replica placement, transfer, routing, replacement, and reclamation, using a validated offline cost profile.

**Blocked by:** 02 — vLLM cross-layer prediction and global predicted-load snapshot; 03 — vLLM injected ReplicaPlan activation and source-rank routing.

**Status:** ready-for-agent

- [ ] Every rank deterministically produces the same `ReplicaPlan` or no plan from the same snapshot, using `GreedySourceRankPolicy` and stable tie-breaking.
- [ ] The planner requires a cost profile matching model, dtype, EP size, and topology; missing or invalid profiles fail startup.
- [ ] Positive predicted benefit, transfer amortization, hot stability, minimum residency, replacement, and reclamation follow the approved lifecycle contract.
- [ ] Forward/version ownership prevents stale plan commits; dummy forwards and prediction errors do not trigger online fallback.
- [ ] CPU tests cover planner outcomes and lifecycle transitions; distributed tests prove plans drive the injected activation path correctly.
