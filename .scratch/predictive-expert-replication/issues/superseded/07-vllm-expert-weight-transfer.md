# 07 — vLLM safe expert-weight transfer into an inactive slot

**What to build:** Move a logical expert's canonical weights into another rank's inactive replica slot, correctly and without disturbing the forward that is running. A layer may place up to `max_replicas_per_layer` distinct experts, so the machinery must carry several transfers per layer, not one. Success is that the target slot holds the right bytes and that nothing raced: no MoE kernel was still reading the slot when it was overwritten, and the interconnect the token collectives depend on was not saturated.

Nothing routes to the slot in this ticket. That keeps it independent of ticket 03: this one only has to prove the bytes arrive safely, verified by byte equality and event ordering rather than by model output.

**Blocked by:** 00 — Feasibility checkpoint; 01 — vLLM predictive infrastructure and fixed layout; 10 — Prediction accuracy in the prefill regime, which gates the prefill build decision the way 00 gates the decode one.

**Status:** ready-for-agent, **revised 2026-08-25**: the per-forward transfer count
this must sustain is about 48 for prefill, not 4. Measured cost at 48: 432 MiB per
forward against the 6 GiB of activations the collectives already move (+7%), and
8.5 ms serialized against a ~200 ms prefill forward (4%). Both affordable, but the
byte accounting and the serialized-span report must be sized for 48.

**Partly implemented already**, by `transfer_replicas`. Read `CURRENT-STATUS.md`'s
2026-08-29 code audit findings 3 and 4 first: `max_concurrent_transfer_bytes` is never
read, so the byte bound this ticket owns does not exist yet, and the "transfer only the
difference" reuse that sizes the per-forward count is suspected to be defeated on any
mixed traffic.

- [ ] The canonical owner sends and only the replica target receives, over the EPLB communicator on the predictive stream, into a shared staging workspace sized for one expert rather than one per layer.
- [ ] The staging workspace reuses the leading row of the already-allocated expert transfer buffer, adding no memory. Its precondition, that native rearrangement never runs on the request path, is **already enforced and tested** — do not re-implement it, but do assert the reuse still depends on it.
- [ ] Before a staging-to-slot copy overwrites a slot, the transfer waits for the event recorded when that slot was last read by a MoE kernel, and the target slot's contents are byte-identical to the canonical owner's afterwards.
- [ ] Expert-weight bytes in flight are tracked. The cap is enforced **before launch**: a placement that would exceed `max_concurrent_transfer_bytes` is deferred, never started. Exceeding it is then a fail-fast invariant violation, because it can only mean the deferral logic is broken — not a silent queue, and **not** a refusal of the second of a layer's K placements, which the serialization item below requires to succeed. Completion is detected without synchronizing with the host, since a per-forward host sync is the cost this whole path is written to avoid.
- [ ] While one transfer is in flight, a second placement is deferred rather than launched concurrently, and deferrals are counted in benchmark mode **in two separate buckets**. Serialization deferrals are expected: with `max_replicas_per_layer` at 2 and the byte cap at one expert, K-1 per layer per forward are the designed behaviour, so a single pooled counter cannot distinguish normal operation from starvation and would be useless for the diagnosis it exists for. The bucket that matters counts placements deferred so long that the transfer did not complete before its target layer ran — those are the ones the byte cap actually cost, and they are what shows `max_transfers_per_forward` never taking effect.
- [ ] The interconnect share the transfer consumes is reported alongside what token dispatch and combine already consume, so "overlapped" can be shown to mean the transfer did not become that span's bottleneck.
- [ ] A layer's placements are carried as a set, not one at a time: up to `max_replicas_per_layer` distinct experts, each from its own canonical owner to its own distinct target rank. Targets are necessarily distinct because a rank holds one replica slot per layer.
- [ ] Serialized transfers do not overwrite the staging workspace while it is still being read. The workspace holds one expert, so transfer `i+1`'s receive must wait for transfer `i`'s staging-to-slot copy to complete, not merely for `i`'s receive. The only ordering stated elsewhere is the wait on the *target slot*'s last MoE read, which does not cover this, and a literal implementation of the items above would corrupt the second transfer.
- [ ] `max_concurrent_transfer_bytes` serializes those transfers rather than forbidding them. At its default of one expert, a layer's K placements move one after another, so the exposed cost the lifecycle must amortize is K transfers, not one. Report the serialized span so the planner's cost model can be checked against it.
- [ ] The byte-equality assertion already added for ticket 03 covers a real transfer too: after the transfer completes, every physical copy of each replicated logical expert must be byte-identical. Reuse it rather than writing a second check, and confirm it reports a non-zero comparison count so it cannot pass vacuously.
- [ ] Distributed tests verify byte equality after transfer, inactive-slot safety, the in-flight byte bound, deferral behaviour, and that a genuine communication failure follows the fail-fast path rather than being swallowed.
