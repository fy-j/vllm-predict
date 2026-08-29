# 09: Can a CUDA graph capture the prediction path?

**What to build:** An answer, not a feature. Exploratory, and deliberately off the mainline.

Prediction's cost is launch-driven: about 2.2 ms of GPU work per prefill window turns into
**+39.9 ms** of collective waiting, because roughly 700 extra launches desynchronise the DP
ranks and the collectives absorb the skew. Graph capture attacks that directly and generically,
and it is plausibly worth more than every other optimisation in this ticket set combined.

Two reasons it is not a mainline dependency. It contradicts the eager execution the current
scope mandates. And whether the predictive stream, its events and a plan that varies per
forward can be captured at all is **unverified** — this branch has already paid for building on
unverified premises more than once.

**Blocked by:** None (can start immediately). It can run in parallel with anything.

**Status:** ready-for-agent

- [ ] Whether a graph can capture a region containing the predictive stream and its events at
      all, with the failure mode recorded if not.
- [ ] Whether a plan that changes every forward can live inside a captured graph, or whether it
      forces a replay-with-updated-inputs shape, and what that costs.
- [ ] The launch count and the collective waiting inside real forward windows, captured against
      eager, so the size of the prize is a number.
- [ ] A recommendation with its evidence: whether this should replace the fusion work of ticket
      03, complement it, or be dropped. If it would replace it, say so plainly — ticket 03 is
      most of what would become unnecessary.
