# 03 — vLLM source-rank routing with a statically placed replica

**What to build:** Make a logical expert able to have a second physical copy that only some source ranks use. Every token a given source rank sends to that logical expert goes to the same physical copy, chosen per source rank rather than per token, and generated output is unchanged from canonical-only execution.

This is verifiable with no transfer machinery at all: a statically configured replica placement is applied during startup normalization, which copies the canonical weights into the target's inactive slot before the server reports ready. Routing can then be exercised end to end while the question of how weights get there is left to ticket 07.

**Why this is not blocked by 00:** source-rank routing is the spec's correctness contract, not an optimization. The predictive path currently inherits a per-token hashed replica choice, which is token-level routing and the opposite of what the spec requires. That has to be corrected whether or not the measured headroom ever justifies a transfer.

**Blocked by:** 01 — vLLM predictive infrastructure and fixed layout.

**Status:** DONE — 7 of 7, re-verified on the 8-GPU node 2026-08-24 after the first pass was withdrawn as potentially vacuous. The check now requires non-zero recorded traffic before it will report a pass, so an all-zero load table (the default, since `log_balancedness` is off) can no longer look like 384 idle slots.

- [x] A source-local physical map turns logical routing IDs into physical routing IDs before token dispatch, while logical IDs remain intact for prediction and metrics.
- [x] All tokens from one source rank to one logical expert use one physical copy. The same logical expert may be served from its canonical owner for one source rank and from a replica for another, and no source rank's chunk is ever split across copies.
- [x] The per-token hashed replica choice no longer decides placement on the predictive path. The existing behaviour stays the default for every other caller of the shared routing path, including the other model families that use it, so their routing is untouched.
- [x] A statically configured replica placement can be installed during startup normalization, populating the target slot with the canonical weights, so this ticket is testable without any weight transfer.
- [x] Output equivalence: with a statically placed replica active, the output
      distribution matches canonical-only execution. Verified by a startup
      assertion that every physical copy of a logical expert is byte-identical
      (48 of 48 layers), plus unchanged argmax and identical greedy text. Compare
      distributions with a tolerance, never many greedy tokens for equality:
      routing to a second copy of the same weights regroups tokens inside the
      expert GEMM and perturbs reduction order, so greedy text can diverge while
      the model is unchanged. A control run confirmed two runs of the *same*
      configuration are byte-identical, so a text difference comes from the
      regrouping rather than process nondeterminism.
- [x] An inactive replica slot attracts zero routed tokens. Verified under real requests by `verify_inactive_slots.sh`: 384 inactive physical slots (48 layers x 8 ranks) carried no routed load across 32 completed requests. The check raises on the offending forward rather than reporting afterwards, because an inactive slot's weights are never written and a token routed there reads uninitialised memory and yields plausible-looking output. It also refuses a layout that marks no slot inactive, so it cannot pass vacuously.
      load during a real forward: the rows still marked inactive must be exactly
      zero, and both rows of a replicated expert must be non-zero, which also
      confirms at runtime that source ranks are really split across the copies
      rather than only structurally unable to split.
- [x] Tests cover the map derivation, the per-source-rank assignment including the deterministic tie-break, that a chunk is never split, and that disabled mode and other model families keep the previous behaviour.
