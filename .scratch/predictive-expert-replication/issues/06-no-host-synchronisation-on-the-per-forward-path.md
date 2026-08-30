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

**Status:** implemented and running end to end (2026-08-30). The segfault is fixed and its
cause was the plan's ownership, not the transport. Two criteria stay open and both need 8
GPUs: the 24.0%/43-layer reproduction and the occupancy measurement. This node now has 2.

**The precondition is answered and it changes the shape of this ticket.** Ticket 05's
transport cannot reach this ticket's headline criterion, because a **host-issued** put takes
its peer, its source pointer and its byte count as host integers consumed at enqueue — the
same constraint that put the 5.28 ms there through `ncclSend`. So the transfer must be issued
from inside a kernel, and `bench/probe_device_put.py` proves that works here: 8/8 ranks
receive a payload aimed by a plan the host never read, at **p50 47.6 us against 53.7 us
host-issued**, so the device-issued route costs nothing in time. The masking alternative —
issue every put the plan might have chosen — is ruled out at about 230 us per layer, 10 ms
across 43 layers against a ceiling near 5%.

Two consequences for the work here. **The first is done:** `plan_one_layer_on_device` returned
through the host (`int()`, `float()` and `bool()` on device tensors, each a synchronisation) and
is now fully tensorised, with bit-identity to the host planner still holding over the randomised
sweep on CPU and CUDA. It is guarded by `set_sync_debug_mode("error")` rather than by grepping
the source, because a synchronisation arrives through a dozen spellings and an indexing
expression does not look like one. Second, still open: the put kernel needs a build-and-register
path in the worker, whose seven toolchain traps are written down in `bench/RESULTS.md`,
2026-08-30 — each one fails with a message naming the wrong cause.

**A third consequence, found 2026-08-30 while scoping the wiring, and not implied by the
criteria as written.** Once the plan is a device tensor the host no longer knows what was
placed, and three pieces of bookkeeping currently derive from knowing: the residency set that
makes an already-resident replica cost no transfer, the per-forward transfer budget, and the
layout edit that reverts a row. All three are Python objects built from `list[Placement]`. Left
as they are, each one reads the plan back and puts the synchronisation straight back — which is
the failure mode most likely to produce a run that looks correct and measures nothing. They
have to become device tensors: a `[num_layers, ep_size]` resident-expert table, a device
counter for the budget, and a scatter for the layout. The residency table is the one that
matters for benefit, not just for cost, because transfer reuse across forwards is what lets
coverage ratchet up to all 43 layers.

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
- [x] **No `synchronize()` anywhere on the per-forward path** — asserted by *measurement*
      rather than by the source-level test this asked for. A grep cannot see the spellings
      that matter: `set_sync_debug_mode("error")` passes on the first `torch.stack` in a
      process, which blocks the host for 50-100 ms while its kernel loads. The test queues
      100 ms on the compute stream, runs all three phases, and asserts the host came back
      in microseconds; a single `int(plan[0])` inserted deliberately makes it fail.
- [ ] The existing baseline is reproduced, not merely equalled in spirit: at least **24.0% of
      full-prefill critical-path excess removed**, activation on **all 43 reachable layers**,
      and physical per-rank load diverging from canonical ownership by a non-zero amount.
- [x] Generated output matches a no-replica reference within BF16 tolerance over many
      forwards, and repeated activation, replacement and reclamation complete on every rank
      without deadlock. Done at DP=2; see the equivalence note below for what the logprob
      tolerance does and does not decide.
- [ ] GPU occupancy inside real forward windows measured, and reported against the baseline's
      86.8% and the placed arm's 52.9%.

## Where this stands, 2026-08-30 night

Done, each verified against the host path it replaces:

- [x] A device scatter publishes the **source-local** map pair. Checked against
      `apply_replica_maps` plus the layout edit over a 200-plan random sequence and the four
      transitions that actually break it: a slot changing hands, a placement repeating, a
      forward placing nothing, and a replica whose row sits below its own canonical row.
- [x] The published set is a layer's complete desired state, so `found == 0` reverts.
- [x] Placement suppression is decided per forward from `num_tokens_across_dp_cpu`, before
      anything is recorded.
- [x] No host read in the plan or the publish, asserted with `set_sync_debug_mode("error")`
      rather than by grepping the source — a synchronisation has a dozen spellings and an
      indexing expression does not look like one.

Also done, beyond what the criteria asked: residency and the transfer budget moved to the
device. They had to. Once the plan stops reaching the host, reconstructing "is this replica
already resident" by reading it back restores the synchronisation, and residency is what makes
transfer reuse free and coverage ratchet up across forwards.

## The segfault, 2026-08-30 afternoon: it was the plan's ownership

- [x] `put_expert` no longer crashes a server, and the transport was never at fault.
      `plan_and_launch` built the `[4]` plan as a temporary on the compute stream and
      launched two kernels that read it on the **predictive** stream. The temporary died
      when the method returned, PyTorch's allocator handed its block to the next
      allocation on the compute stream — measured, the 5th — and the kernels, still queued
      behind a 40 us transfer, read whatever the forward had put there. `pe` was then not
      a rank. Nothing in `vllm/distributed/eplb/` called `record_stream`.

      Demonstrated rather than argued, in three steps: a pure-torch consumer on a second
      stream read the poison; the production classes did the same, moving expert 11 where
      the plan said 3 (`bench/probe_plan_lifetime.py`); and a plan whose recycled block
      named PE 12345 of 2 killed rank 0 with **SIGSEGV**, which is the server's crash.

      Fixed by giving each layer a plan row the coordinator owns for its lifetime, so
      there is no allocation on the path and nothing to recycle.

- [x] Verified end to end on **2 GPUs** (this node lost its other 6): both workers arm,
      all 43 reachable layers per rank launch a device-issued transfer, the device counter
      reports 8 replicas actually placed, requests are served, and no worker dies.
      `bench/run_device_transfer_smoke.sh`.

- [x] Output equivalence, at DP=2 with real weights: greedy text over 4 prompts x 96
      tokens is **identical** to a no-replica reference while placement is demonstrably
      active. The first-token top-10 logprobs move by 0.25 against the harness's invented
      0.05 tolerance — but the **host-issued** path moves them 0.34 on the same test, and
      two canonical runs move them 0.000, so this is what dynamic placement does to the
      cross-rank combine order and not something the device transport introduced.

Still open, and both need 8 GPUs:

- [ ] At least 24.0% of prefill critical-path excess removed, on all 43 reachable layers.
- [ ] GPU occupancy inside real forward windows, against the baseline's 86.8% and the
      placed arm's 52.9%.

One residual risk worth naming: nothing in a real server checks that a *dynamically*
placed replica holds the bytes of the expert it claims. `verify_replica_weight_equality`
runs at startup, where nothing is placed yet, and says so — it reported "0 pairs,
vacuous". The evidence that it does is indirect: byte equality over every rank pair in the
probe, and 384 greedy tokens unchanged with placement active, which a wrong row would very
likely have broken. A runtime checksum check is the cheap way to make it direct.
