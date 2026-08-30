# 09: Can a CUDA graph capture the prediction path?

**What to build:** An answer, not a feature. **On the mainline as of 2026-08-30, and now the
largest single lever in the project** — this ticket used to say "exploratory, and deliberately
off the mainline", and the measurement below is why that changed.

**The prize is measured, not argued.** Ticket 11 split prediction's +13.68% mean TTFT in two by
removing only its collectives: **11.04 ms of 23.21 is the 44 per-layer barriers, and 12.17 ms
(52%) is its launches and compute.** In the window profile the same term is 8.3 ms per prefill
window at *unchanged* collective count. Tickets 12, 13 and 14 attack the barrier half; **nothing
in the ticket set attacks the launch half except this one.** Even a perfect grouping leaves
prediction near +4.8% against a 5.26% ceiling, which is break-even; the launch half is what
would make the feature positive.

An earlier version of this ticket priced the prize as "about 2.2 ms of GPU work turns into
+39.9 ms of collective waiting, roughly 700 extra launches". That framing is superseded:
prediction adds **no net compute** — per-rank compute-only occupancy is uniform in both arms and
the absolute compute time is unchanged — so there is no small-GPU-cost-amplified-hugely story.
The launch term is a direct 8.3 ms of window that survives when every prediction collective is
removed.

It stays an *answer* ticket rather than an implementation one for two unchanged reasons. It
contradicts the eager execution the current scope mandates, so adopting it is a scope decision
the operator makes. And whether the predictive stream, its events and a plan that varies per
forward can be captured at all is **unverified** — this branch has already paid for building on
unverified premises more than once.

**Blocked by:** None (can start immediately). It can run in parallel with anything, and it
should start before 08 rather than after: 08's verdict would otherwise be measured against a
cost this ticket may show is removable.

**Status:** ready-for-agent

- [ ] Whether a graph can capture a region containing the predictive stream and its events at
      all, with the failure mode recorded if not.
- [ ] Whether a plan that changes every forward can live inside a captured graph, or whether it
      forces a replay-with-updated-inputs shape, and what that costs.
- [ ] The launch count and the collective waiting inside real forward windows, captured against
      eager, so the size of the prize is a number. Compare against the term this ticket is aimed
      at: **8.3 ms of prefill window, 12.17 ms of mean TTFT**, measured by removing prediction's
      collectives and finding that much left over. A graph that does not move that term has not
      found the prize, whatever it does to the launch count.
- [ ] A recommendation with its evidence: whether this should replace the fusion work of ticket
      03, complement it, or be dropped. If it would replace it, say so plainly — ticket 03 is
      most of what would become unnecessary.
- [ ] **What it would take to adopt, and what it costs elsewhere**, since the answer feeds a
      scope decision rather than a merge. Eager execution is mandated by `spec.md`'s runtime
      scope and by the operator's own comparison basis; every figure this project has recorded is
      eager. State whether graph capture invalidates them, or only adds an arm.
