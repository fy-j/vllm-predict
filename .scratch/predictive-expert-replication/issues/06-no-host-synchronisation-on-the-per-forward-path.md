# 06: No host synchronisation on the per-forward path

**What to build:** The replica becomes live and tokens route to it, with the host never
learning the plan. This is the ticket where the measured cost centre disappears.

The host currently stalls once per predicted layer, and it is structural rather than a stream
placement mistake: the planner runs on the host, so it must read a device tensor. Measured
inside real forward windows, GPU occupancy falls from **86.8% to 52.9%**, gaps over 0.5 ms
total **3.0 ms against 30.2 ms**, and one `cudaEventSynchronize` runs **11.02 ms**. With the
plan already on the device from ticket 04 and the transfer device-initiated from ticket 05,
publishing is the last host involvement — and routing already consumes the maps it needs as
device tensors, so a device scatter suffices.

**Blocked by:** 05 — The transfer lands in the replica slot.

**Status:** ready-for-agent

- [ ] A device scatter publishes the **source-local** map pair. Writing the global pair
      instead transfers the replica, describes it correctly, and publishes it where nothing
      reads: a measured run then activated 131 replicas per forward and removed 0.6% of
      prefill excess against an oracle's 35.1%. The routing path prefers the source-local pair
      whenever it is set, and under this feature it always is.
- [ ] The published set is a layer's **complete** desired set, so an empty set reverts what the
      previous forward left. Reversion is a map edit with no transfer, and carrying an unwanted
      replica is not neutral: it goes on shedding half of an expert that may no longer be hot
      onto a rank that may now be the peak.
- [ ] Placement suppression is decided per forward from a value every rank agrees on **before
      anything is recorded**. Deciding it after a snapshot exists is what previously made every
      decode and every dummy forward publish an empty set on all 48 layers, which reverted
      everything and defeated transfer reuse under any mixed traffic.
- [ ] **No `synchronize()` anywhere on the per-forward path**, asserted by a source-level test
      the way event polling is already forbidden. This is the ticket's headline criterion.
- [ ] The existing baseline is reproduced, not merely equalled in spirit: at least **24.0% of
      full-prefill critical-path excess removed**, activation on **all 43 reachable layers**,
      and physical per-rank load diverging from canonical ownership by a non-zero amount.
- [ ] Generated output matches a no-replica reference within BF16 tolerance over many forwards,
      and repeated activation, replacement and reclamation complete on every rank without
      deadlock.
- [ ] GPU occupancy inside real forward windows measured, and reported against the baseline's
      86.8% and the placed arm's 52.9%.
