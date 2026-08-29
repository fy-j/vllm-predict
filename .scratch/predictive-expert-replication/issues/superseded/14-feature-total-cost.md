# 14 — The feature's total cost, measured against a disabled arm

**What to build:** No feature code. One missing benchmark arm, and the arithmetic that
follows from it.

Every TTFT and TPOT number this branch has produced compares **placement against
prediction**, because both arms of every runner pass
`predictive_expert_replication.enabled = True` and differ only in
`max_transfers_per_forward`. So the feature's own cost has never been measured. What is
known: prediction alone measured **19% of TPOT** at concurrency 1, where the placement
gate rejected every forward and nothing was placed.

The question this answers is not a tuning question. Prediction costs 43 extra gate
matmuls and 43 AllGathers per prefill forward, of which 14.8% stays exposed. If that
exposed cost is the same order as the corrected ceiling — 1.37% to 1.73% of a prefill
step — then no planner, no budget and no interconnect can make the feature net
positive, and the project is decided independently of hardware. That is cheap to find
out and it has never been checked.

**Blocked by:** None. Run it before ticket 05, and before spending an H100 run on
end-to-end TTFT.

**Blocks:** 05, and any claim of the form "the feature costs X".

**Status:** **ANSWERED, negative, 2026-08-29.** Prediction alone costs **+7.6% mean
TTFT** against a perfect-balance ceiling of **5.05% of a prefill step** — 1.51x the
ceiling before a single replica moves. The whole feature costs **+31.5%**, which is 6.2x
the ceiling and 26x the 1.21% it actually delivered. Unlike the 5090 verdict this is a
statement about the **mechanism**: ticket 13 removes the host sync and therefore most of
placement's share, but not the 43 gate matmuls and 43 AllGathers per forward that
prediction itself costs. The ceiling stays below the floor on the fastest interconnect
NVIDIA ships, so the result transfers to the Ascend port. See `bench/RESULTS.md`,
2026-08-29.

- [x] A third arm with `predictive_expert_replication.enabled = False` in
      `run_e2e_placement.sh`, so the sweep is disabled / prediction-only / placed. The
      disabled arm must be a genuinely stock configuration: no `enable_eplb`, no
      redundant experts, no replica slots, so it also prices the 432 MiB per rank and
      the layout normalization.
- [x] Mean and p99 TTFT for all three arms, on `ko` at the configuration
      `HANDOFF-2026-08-29-H100.md` section 5 step 2 uses, so the existing
      `237.49 -> 263.68 ms` pair extends rather than being replaced.
- [x] **Prediction's exposed cost as a share of a prefill step**, from a profile
      restricted to the `execute_context_*` annotations. Report it beside the
      1.37%-1.73% ceiling. This comparison is the deliverable; the TTFT deltas are
      supporting evidence.
- [ ] TPOT for all three arms, to confirm or correct the 19%-of-TPOT figure with the
      decode gate in place. Prediction is gated off in decode by
      `_prediction_is_worth_it`, so the current cost should be near zero and a
      measurement that is not near zero means the gate is not firing.
- [x] A recommendation. If exposed prediction cost is at or above the ceiling, say so
      plainly and stop: that is a conclusion about the mechanism, not about this node's
      interconnect, and it transfers to the Ascend port.

**Two traps this ticket exists inside.** Restrict every profile measurement to the
`execute_context_*` annotations — a 12.95 s trace held 0.54 s of real forwards. And
report the per-layer critical-path imbalance, never the aggregated kind. Both are
described in `HANDOFF-2026-08-29-H100.md` section 3.
