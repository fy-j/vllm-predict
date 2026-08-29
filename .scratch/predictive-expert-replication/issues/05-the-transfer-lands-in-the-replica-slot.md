# 05: The transfer lands in the replica slot

**What to build:** The chosen expert's weights arrive in the target rank's replica row,
byte-identical to the source, moved without the host knowing the plan. Nothing routes to the
row yet — this ticket ends where routing begins, which is what makes it verifiable on its own.

The route is a symmetric **staging workspace** and a local copy, and it is chosen from
measurement: a one-sided put of a 9.00 MiB expert takes **33.0 us** at 285 GB/s, and the local
copy into the replica row takes **6.3 us** at the measured 3.00 TB/s HBM bandwidth — 39.3 us
together, 0.06% of a 67.87 ms prefill step. Two alternatives were probed and must not be
re-tried blindly: registering the model's own weight tensors is **not available on this
stack**, because registration needs a dynamic VMM heap and the VMM heap needs a user
allocation handle PyTorch does not create, and the two settings exclude each other. PyTorch
symmetric memory **does** work and is the better shape if the staging copy ever matters, but it
needs a memory-pool context around expert-weight creation inside the MoE and quantisation
paths, where the staged route touches the weight allocation not at all.

**Blocked by:** 04 — The plan is computed on the device. The transfer reads the plan, so the
plan has to be on the device first or the host round trip returns.

**Status:** ready-for-agent

- [ ] The replica row is byte-identical to the canonical source row after the transfer, over
      repeated transfers and across every rank pair.
- [ ] **One staging workspace, shared by all layers**, because at most one expert is in flight.
      A transfer aimed at a later layer must not overwrite a replica a nearer layer still
      needs, and a test covers that ordering rather than trusting it.
- [ ] Ordered by a plain CUDA event on the consumer's stream. No fence and no flag polling:
      polling is per-rank timing, which is the divergence class that deadlocked this branch
      twice, and a stream wait was verified sufficient.
- [ ] The staging-to-slot copy waits for the event recorded when that slot was last read by a
      MoE kernel, so a copy cannot overwrite weights a kernel is still reading.
- [ ] Bytes in flight respect `max_concurrent_transfer_bytes` from ticket 02.
- [ ] A transfer deliberately slowed past its window causes the target to wait, record the
      exposed time, and complete — not cancel, not reroute to canonical, not silently discard.
- [ ] Distributed test on 8 ranks. A true communication or copy failure is fail-fast.
