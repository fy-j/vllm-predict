# 04: The plan is computed on the device

**What to build:** A deterministic device-side reduction that turns the global predicted-load
snapshot into the layer's replica placement plan, replacing the host planner on the serving
path.

At `max_replicas_per_layer = 1` the plan is a single argmax: the hottest expert on the layer's
peak rank, placed on the lightest rank not already holding a replica for that layer, subject
to the positive-benefit test and the minimum-move floor. The data is 128 integers per layer,
so this is a small reduction and not a kernel-engineering problem — the host planner exists as
its oracle and stays in the tree for exactly that reason.

**This ticket is a correctness slice and makes no performance claim.** The host
synchronisation cannot go until the transfer is device-initiated too: `ncclSend`'s peer is a
host integer consumed at enqueue, so with the current transport the host would read the plan
back and the same wait would reappear. Ticket 06 is where the sync disappears.

**Blocked by:** 01 — Harness fails loudly when a run measures nothing.

**Status:** ready-for-agent

- [ ] The device plan is **bit-identical** to the retained host planner at
      `max_replicas_per_layer = 1`, over randomised snapshots that include ties, all-zero
      load, loads where relocating the peak would not lower it, and loads below the
      minimum-move floor. Determinism is a correctness property here, not a nicety: a plan
      that differs by rank pairs a sender with no receiver and hangs the engine.
- [ ] Integer end to end. The snapshot stays integer through the reduction, because a device
      argmax over floats is order-dependent and two ranks reducing the same values in a
      different block order can select different experts.
- [ ] Ties break on the lower logical-expert id, then the lower rank id, and a test asserts
      the tie-break rather than the happy path.
- [ ] Candidates come only from the layer's peak rank, and a placement that would not lower
      the peak is not emitted.
- [ ] Every rank derives the plan locally from the same snapshot; no broadcast, no leader.
- [ ] The plan is readable as a device tensor by the transfer and publish paths, so that
      ticket 06 can consume it without a host round trip.
