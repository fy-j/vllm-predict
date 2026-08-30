# 14: The snapshot reduction leaves NCCL, and stops being a barrier

**What to build:** The predicted-load snapshot as a device-side reduction over NVSHMEM instead
of an NCCL collective, so it stops coupling the ranks at all.

The planner needs only the **sum over source ranks** of the `[num_logical]` counts —
`record_prediction` already reduces the snapshot with `sum(dim=0)`, so the per-source-rank
breakdown the AllGather delivers has never been read. That is 512 bytes. NVSHMEM is already
initialised on every worker for the weight transfer, and a put-with-flag reduction needs no
group-wide alignment: a rank reads its peers' contributions as they arrive, and a per-forward
sequence number makes "this forward's values" checkable without a barrier.

**Why this is sequenced after batching rather than instead of it.** Batching is pure scheduling
against a mechanism ticket 12 proves inert; this is a new kernel with a new correctness
argument, and its value is only whatever barrier cost ticket 13 leaves behind. If 13 lands the
remaining 11 barriers at a cost that no longer matters, this ticket should be dropped rather
than built, and dropping it is an acceptable outcome.

**Blocked by:** 13 — Four targets per collective. Its measurement is what prices this one.

**Status:** ready-for-agent

- [ ] The reduction produces a `[num_logical]` sum that is **bit-identical to the AllGather
      path's**, over randomised per-rank counts including empty and all-zero inputs. Integer
      throughout, for the same reason the planner is: every rank must derive the same plan, and
      a float reduction whose order differs by rank is how two ranks pick different experts.
- [ ] Reads only this forward's values. A per-forward sequence number written with the payload,
      checked on read, and a value from another forward is an invariant violation rather than
      something to average in. Ticket 06's plan-ownership rule is the precedent.
- [ ] **No group-wide barrier and no host read**, asserted by measurement rather than by grep:
      `set_sync_debug_mode("error")` clean, and the host returns in microseconds with 100 ms
      queued on the compute stream, which is the shape ticket 06's test used.
- [ ] Every rank agrees on whether this path is in use, through the existing
      `agree_across_ranks`, and a group that falls back closes what it opened. A rank deciding
      this alone is the deadlock that helper exists to prevent, and the fallback path must not
      land on a configuration the validator forbids — which it did once already.
- [ ] Measured against ticket 13's numbers at the same operating point: collectives per prefill
      window, window wall-clock, and three-arm TTFT. The claim to test is that the remaining
      barrier cost goes to zero, not that the reduction is faster.
