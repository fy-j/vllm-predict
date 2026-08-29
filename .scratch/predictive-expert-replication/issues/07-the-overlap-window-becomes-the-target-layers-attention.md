# 07: The overlap window becomes the target layer's Attention

**What to build:** The shape the operator asked for from the start — predict the next layer at
this one, move the weights during the next layer's Attention, and have that layer's MoE find
the replica already live.

The launch moves from the head of the following layer's MoE to the **tail of the predicting
layer's MoE**, after the snapshot completes; the wait stays at the head of the target layer's
MoE. With `prediction_lookahead_layers = 1` the span between them is exactly the target
layer's Attention.

It is bandwidth-feasible and that is measured, not assumed: one layer's Attention projections
are 37.5 us at 512 tokens per rank and more at prefill chunk sizes, against a 33.0 us
one-sided put. **The order matters and this ticket must come last of the four**, because with
a host planner the launch and the wait are adjacent statements and the window is zero — which
is why ticket 02 rejects `lookahead = 1` until now.

**Blocked by:** 06 — No host synchronisation on the per-forward path.

**Status:** ready-for-agent

- [ ] The launch is issued at the predicting layer's MoE tail and the wait at the target
      layer's MoE head, with at least one Attention block of compute between them.
- [ ] `prediction_lookahead_layers = 1` is accepted and becomes the default, and ticket 02's
      rejection is lifted with its reason updated rather than deleted.
- [ ] A profile shows the transfer concurrent with the target layer's Attention, and reports
      what share of it stays exposed.
- [ ] Prediction accuracy at lookahead 1 is reported. It should improve, since 1 was always
      the accurate distance and 2 existed only to buy the host a layer of latency to hide in.
- [ ] Plan ownership still holds: a target accepts only the plan produced one position before
      it for that same forward, forward completion asserts no pending plan remains, and a
      version mismatch is an invariant violation rather than a fallback.
- [ ] The baseline from ticket 06 is held or improved: at least 24.0% of full-prefill excess
      removed, all 43 reachable layers active.
