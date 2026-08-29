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

**Status:** done, 7/7 (2026-08-30)

- [x] The replica row is byte-identical to the canonical source row after the transfer, over
      repeated transfers and across every rank pair.
- [x] **One staging workspace, shared by all layers**, because at most one expert is in flight.
      A transfer aimed at a later layer must not overwrite a replica a nearer layer still
      needs, and a test covers that ordering rather than trusting it.
- [x] Ordered by a plain CUDA event on the consumer's stream, plus — **correcting this
      ticket** — a stream-ordered NVSHMEM barrier for arrival. A CUDA event is *not*
      sufficient and was never verified to be: `probe_nvshmem.py`'s check had a
      `dist.barrier()` inside the region it was checking, so the barrier did the work.
      Measured directly with `--control-no-barrier`: **61/112 weight tensors match**
      without it, 112/112 with. The barrier costs 13.9 us of the 53.7 us span, needs no
      host, and is collective — so it stays clear of the per-rank timing that deadlocked
      this branch twice, and it replaces a collective (`pynccl execute`) that was already
      called once per layer on every rank.
- [x] The staging-to-slot copy waits for the event recorded when that slot was last read by a
      MoE kernel, so a copy cannot overwrite weights a kernel is still reading.
- [x] Bytes in flight respect `max_concurrent_transfer_bytes` from ticket 02. The engine
      chunks a layer's placements to the cap and **serialises rather than refusing**, and
      it chunks from the whole plan's bytes rather than this rank's, so every rank issues
      the same number of barriers. Reading the config value into the engine lands with
      ticket 06's wiring, which is where the engine first meets a real forward.
- [x] A transfer deliberately slowed past its window causes the target to wait, record the
      exposed time, and complete — not cancel, not reroute to canonical, not silently discard.
- [x] Distributed test on 8 ranks: `bench/probe_replica_transfer.py`, which drives the
      production classes over all 56 ordered rank pairs. Fail-fast, and that took a fix of
      its own — the first version printed `FAIL` and exited 0.

**Measured, 8x H100, 2026-08-30.** One 9.00 MiB expert, put + barrier + staging copy:
**p50 53.7 us** (min 49.3, max 68.5). All 43 layers would be 2.3 ms against the 5.28 ms
*per layer* of host synchronisation this path removes. The slowed arm exposes 0.456 ms
against 0.013 ms unslowed, so the wait is real and measured rather than asserted.

**Two teardown defects, both of which look like transfer bugs and are not.** NVSHMEM keeps
its own reference count beside Python's, so dropping the last reference to the symmetric
buffer leaves it tracked: finalize then reports every buffer leaked and segfaults every
rank *after* the results have printed. `free_tensor` before `finalize` is required. And a
hypothesis that the free's collectivity was racing cost time before the real cause was
found — an edit to `close()` that had silently failed to apply, so `free_tensor` was never
being called at all.
