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

**Blocked by:** 05 — The transfer lands in the replica slot. Done.

**Status:** in progress — precondition answered 2026-08-30, wiring not started

**The precondition is answered and it changes the shape of this ticket.** Ticket 05's
transport cannot reach this ticket's headline criterion, because a **host-issued** put takes
its peer, its source pointer and its byte count as host integers consumed at enqueue — the
same constraint that put the 5.28 ms there through `ncclSend`. So the transfer must be issued
from inside a kernel, and `bench/probe_device_put.py` proves that works here: 8/8 ranks
receive a payload aimed by a plan the host never read, at **p50 47.6 us against 53.7 us
host-issued**, so the device-issued route costs nothing in time. The masking alternative —
issue every put the plan might have chosen — is ruled out at about 230 us per layer, 10 ms
across 43 layers against a ceiling near 5%.

Two consequences for the work here. `plan_one_layer_on_device` currently returns through the
host (`int()`, `float()`, `bool()` on device tensors), so it must be made fully tensorised
even though ticket 04 is closed; the tensor it already returns is the right contract. And the
put kernel needs a small build-and-register path in the worker, whose seven toolchain traps
are written down in `bench/RESULTS.md`, 2026-08-30 — each one fails with a message naming the
wrong cause.

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
