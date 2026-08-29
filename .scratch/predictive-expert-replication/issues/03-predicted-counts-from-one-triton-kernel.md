# 03: Predicted counts from one Triton kernel

**What to build:** The predicted per-logical-expert count vector, produced by one Triton
kernel instead of about twelve elementwise operations, with the existing path retained as
the reference it is tested against.

**This is the ticket the arithmetic turns on.** Prediction's own GPU work is small — about
2.2 ms per prefill window — but it adds roughly 700 kernel launches, and the token
collectives then grow **+39.9 ms** with the same kernel count and the same byte volume. That
is pure waiting: the launches push each rank's arrival at the next collective apart, and the
amplification is about 18x. The cost is **linear in the number of predicting layers**, 7.00 ms
per source layer at 11 layers against 7.50 ms at 43, so cutting per-layer launches cuts cost
proportionally.

Measured per source layer, prediction adds 17.2 kernels. About 11 to 12 are the elementwise
tail that builds the count vector — padding mask, range check, dtype casts, multiply, clamp,
scatter-add. The remainder are the gate GEMM, the router's top-k, the snapshot AllGather and
EPLB's own load recording. So this ticket removes roughly **64% of the added launches**, and
at linear scaling that extrapolates the prediction arm's cost from **+7.6% to about +2.7%**
mean TTFT against a measured ceiling of 5.05%. The extrapolation is what this ticket
converts into a measurement.

**It does not absorb the top-k.** That kernel is vLLM's own and is shared with real routing;
reusing it is cheaper than maintaining a second implementation that has to agree with it, and
it accounts for only about 8% of the added GPU time.

**Blocked by:** 01 — Harness fails loudly when a run measures nothing. Its measurement
claims are only worth making once a run that measured nothing cannot report success.

**Status:** DONE 2026-08-29 — launches per source layer 17.2 -> 7.1, extra collective waiting 322.4 -> 266.5 ms. The 59% launch cut recovered 17% of the cost, which separated the per-layer cost into 2.21 ms that scales with launches and 5.28 ms that does not; the fixed part is the host synchronisation, so ticket 06 carries the rest.

- [x] The kernel consumes the target gate's logits, the source-local hidden states' token
      count and the EPLB unpadded-token device scalar, and emits the int32 per-logical-expert
      count vector. It counts **logical** experts, applies no logical-to-physical mapping and
      records no actual load, exactly as the path it replaces.
- [x] The previous implementation is retained as a named reference, not deleted, and the
      kernel is tested by **equality against it** over randomised hidden states, token counts,
      padding boundaries, out-of-range expert ids and an empty batch. The failure mode here is
      a wrong count rather than a crash, and three of this project's five historical defects
      were of that shape, so a differential test is the acceptance criterion rather than a
      convenience.
- [x] Padding is still excluded, and a dummy forward still predicts zero while joining every
      collective.
- [x] Nothing in the kernel path synchronises with the host or branches a collective on
      per-rank state.
- [x] Kernels per source layer measured before and after, from a profile restricted to the
      `execute_context_*` annotations. The expectation to test is 17.2 falling to about 5 to 6.
- [x] The three-arm TTFT re-measured — and the method could not resolve it. The stock arm
      got 28% faster on mean TTFT and 40% on throughput between the two runs, on identical
      workloads, so the baseline moved eight times further than the 5.05% ceiling. Within-run
      ratios rose (prediction +7.6% -> +24.2%), which is consistent with a fixed overhead
      against a faster baseline rather than with fusion failing; the same-run profile is what
      showed fusion working. Ticket 08's method needs repeats or interleaved arms before any
      TTFT verdict. Original wording: The prediction arm's cost against a stock server is the
      number this ticket exists to move; report it beside the 5.05% ceiling and beside the
      +7.6% it started from, whatever the result.
