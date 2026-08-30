# 13: Four targets per collective, and what it actually buys

**What to build:** The group size raised to 4 by default, and the measurement that says whether
prediction's cost fell the way ticket 11 predicted it would.

Ticket 11 measured the two halves of prediction's +13.68% mean TTFT: **11.04 ms of 23.21 is the
44 per-layer snapshot barriers** and 12.17 ms is its launches and compute. At K=4 the barriers
go 44 -> 11 and only every fourth layer predicts, so both halves should fall. The predicted
landing point is about **+4.8%**, against a ceiling of 5.26% of a prefill window and a placement
return of about 4% — which is break-even, not positive, and saying so plainly is part of this
ticket rather than a disappointment to be worked around.

**Blocked by:** 12 — One layer predicts a group of targets. The mechanism has to be known
bit-identical at K=1 before its behaviour at K=4 means anything.

**Status:** ready-for-agent

- [ ] Group size defaults to 4, and the configuration is rejected rather than silently reduced
      where 4 cannot be honoured.
- [ ] **Collectives per prefill window measured, not assumed**: 236 -> 203 expected, from 44
      snapshot AllGathers down to 11, with the 192 token collectives untouched. Ticket 11's
      probe profile is the method and `bench/window_occupancy.py` the tool. A count that does
      not move means the grouping did not take effect, which is the failure mode a TTFT number
      alone would hide.
- [ ] Three arms at the knee — stock, prediction only, placing — three interleaved passes,
      reported against the disabled arm. The baseline's own spread has been 0.3% to 1.5% at this
      operating point, so state it and judge the effect against it.
- [ ] Prediction's cost decomposed the same way ticket 11 did it, so the two are comparable:
      how much of the remaining overhead is barriers and how much is launches. The launch half
      is what ticket 09 attacks and this ticket cannot.
- [ ] Excess removed re-measured on hardware against the **26.5-27.0%** three repeats gave at
      K=1, with the reachable-layer count and `connected: true` beside it. The expected loss is
      about 1.2 points, from ticket 11's measured 1..4 accuracy curve.
- [ ] Accuracy reported at the group's actual distances rather than at one lookahead, and
      `prediction_skip_first_layers` re-decided from the per-target-layer curve: ticket 11 found
      leading layers below tolerance at `[6, 7]` for distance 3 and `[7]` for distance 4, where
      distance 1 has only `[4]`.
- [ ] The result stated against the ceiling, and against ticket 08's stop gate. If prediction
      does not come under about 2%, the feature does not turn positive on this mechanism and the
      report says which lever is left rather than proposing another round of tuning.
