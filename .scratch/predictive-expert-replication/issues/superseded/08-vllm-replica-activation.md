# 08 — vLLM replica activation end to end

**What to build:** Join routing and transfer. Given an externally injected `ReplicaPlan`, the weights move while the layers between the predicting layer and the target layer execute, the target starts routing to the replica only once those weights are known to be there, and the model's output is unchanged. When the transfer outruns its window the forward waits and says so, rather than cancelling or quietly falling back.

This is the ticket that proves the lifecycle's safety properties, so it is the last one that can be written before the planner in ticket 04 has anything trustworthy to drive.

**Blocked by:** 03 — vLLM source-rank routing with a statically placed replica; 07 — vLLM safe expert-weight transfer into an inactive slot; 10 — Prediction accuracy in the prefill regime, which gates the prefill build decision the way 00 gates the decode one.

**Status:** ready-for-agent, but **partly implemented already** — the transfer,
activation and publish path runs end to end on this branch. See `CURRENT-STATUS.md`'s
2026-08-29 code audit, findings 1, 3 and 4, before rebuilding any of it: the overlap
window is not the one this ticket assumes, no barrier-free activation criterion has
been verified against a decode forward, and the exposed-wait accounting does not exist.

- [ ] An injected `ReplicaPlan` completes end to end: transfer launched at the predicting layer, weights ready by the target layer, and the target's source ranks routing to the replica on that same forward.
- [ ] The pending source-local physical map is committed only after the weights are ready. Routing maps are never mutated while a router is reading them, and a forward can never execute against a partially copied expert.
- [ ] No additional global barrier is introduced for activation. Ranks not participating in a transfer proceed to the next EP collective and wait there naturally.
- [ ] A transfer that outlasts its overlap window causes the target to wait, records the exposed time, activates the plan, and completes the forward. It is not cancelled, not rerouted back to canonical, and not silently discarded.
- [ ] Output equivalence across repeated activation: generated output matches canonical-only execution within BF16 tolerance over many forwards, not just one.
- [ ] Repeated activation, replacement, and reclamation across all ranks complete without collective deadlock.
- [ ] Distributed tests cover the commit ordering, the exposed-wait path with a deliberately slowed transfer, output equivalence, and deadlock freedom under repetition.
