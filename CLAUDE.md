@AGENTS.md

## Predictive expert replication CUDA PoC

Before working on this feature, read in order:

1. `.scratch/predictive-expert-replication/spec.md`
2. `.scratch/predictive-expert-replication/CURRENT-STATUS.md` — start here for what is done, what is measured, and what is blocked
3. The ticket you are picking up, from `.scratch/predictive-expert-replication/issues/`
4. `.scratch/predictive-expert-replication/glossary.md`
5. `.scratch/predictive-expert-replication/reference/ascend-prior-art.md` (read-only timing prior art; the spec overrides it)

The spec was **rewritten on 2026-08-29** around Device-planned placement, and the ticket
set replaced. Work `issues/01`–`10`; the previous `00`–`14` are in `issues/superseded/`
with a note on why, because their measurements are still cited by the spec. Do not
implement vLLM-Ascend here.

Ticket order is set by each ticket's `Blocked by` field, not by its filename number:

```
01 harness fails on empty runs   blocks 03, 04     no blockers; every later ticket measures
02 config and accounting         terminal          no blockers; removes the lookahead=1 trap
03 fused Triton counting kernel  blocks 08         (needs 01)  DONE 2026-08-29
04 device-side plan              blocks 05         (needs 01)
05 transfer lands in the slot    blocks 06         (needs 04)
06 no host sync on the path      blocks 07         (needs 05)  ← the one that decides it
07 window = target's Attention   blocks 08         (needs 06)
08 three-arm verdict + stop gate blocks 10         (needs 03, 07)
09 CUDA graph feasibility        terminal          no blockers; exploratory, off the mainline
10 DeepSeek-V4-Flash             terminal          (needs 08)
```

Edges are listed rather than drawn: an ASCII diagram of this silently misaligned its
edges into neighbouring labels once, and each ticket's `Blocked by` field is
authoritative anyway.

**This node has 2 H100s, not 8** (since the 2026-08-30 pod restart). `06` is implemented and
runs end to end there — the segfault that had `device_issued_transfer` off was the plan tensor
being freed while the transfer kernels still had it queued, and it is fixed with a regression
test. The runtime scope check no longer pins DP=8; it warns instead, because **no measured
figure for this feature survives a change of EP size**. `06`'s two remaining criteria — 24.0%
of prefill excess on 43 layers, and occupancy against 86.8%/52.9% — need the 8-GPU node back.

**`06` is the ticket the arithmetic turns on, not `03`.** Measured on 2026-08-29: of the
per-source-layer cost, 2.21 ms scales with launch count and 5.28 ms does not, and the
fixed part is essentially the host synchronisation. `03` cut launches 59% and recovered
17% of the cost, which is its whole share. Do not re-derive the ordering from the earlier
note in this file that projected +2.7% from fusion alone; that projection is corrected in
`CURRENT-STATUS.md`.

Each ticket's own `Status` line is authoritative; do not restate it here, because
a copy goes stale and this file is read first.

**Moving to different hardware?** Read
`.scratch/predictive-expert-replication/HANDOFF-2026-08-29-H100.md` first. It carries
the verdict, the corrected ceiling, the two measurement traps, and the order to run
things in.

**`00` is answered, and the answer differs by regime.** Read its two
recommendation sections, not just the first.

**Decode: negative** (2026-08-24). Perfect expert balance is worth 0.08% to 0.15%
of a decode step, because the MoE kernel rounds each expert up to `BLOCK_SIZE_M`
and at every concurrency KV can serve, tokens per expert stays below it — so the
imbalance costs nothing to begin with.

**Prefill: conditionally positive** (2026-08-25), and this is now the target: the
operator's goal moved to TTFT. At 1024 tokens per expert the kernel spends 8 blocks
per expert, quantization is open, and the per-layer imbalance is a real 1.2801x to
1.5821x across three domains. The ceiling is 3.3% of a prefill step here and 7.8% on
an NVLink machine.

**Built and measured end to end (2026-08-26). It works and it does not pay here.**
The feature removes about **15% of prefill critical-path excess** against an oracle
of 35%, confirmed on 50 prefill forwards. It costs **+13% to +29% mean TTFT** across
four configurations. The ceiling on this node is 3.3%, so the cost exceeds the best
possible benefit by roughly five times — a decode-shaped conclusion arrived at for
prefill, and not something tuning closes.

**VERDICT (2026-08-29, 8x H100 SXM): negative, and about the mechanism rather than the
interconnect.** Ticket 14's third arm finally measured the feature against a stock
server: prediction and its infrastructure cost **+7.6% mean TTFT** against a
perfect-balance ceiling of
**5.05% of a prefill step**, so the mechanism that creates the transfer window spends
1.51x what perfect expert balance could ever return, before a replica moves. The whole
feature costs **+31.5%**. Ticket 13's device-side transfer removes the host sync and most
of placement's share, but not prediction's 43 gate matmuls and 43 AllGathers per forward,
so the ceiling stays under the floor on the fastest interconnect NVIDIA ships — and the
conclusion transfers to the Ascend port. Batching those AllGathers is **not** the escape:
traced here they are 9.3 us each, 0.40 ms per forward, 4% of the window they hide in. The
cost is prediction's ~645 kernel launches per forward, and +7.6% is an upper bound because
arm `0` also records expert load. The mechanics are sound and improved this
session (coverage 22 -> all 43 layers, recovered excess 15-17% -> 24.0%); the arithmetic
does not close. Read `CURRENT-STATUS.md`'s verdict section before planning work.

**A 2026-08-29 code audit found six disagreements between the code and these
documents.** Read `CURRENT-STATUS.md`'s "Code audit, 2026-08-29" before touching the
feature or believing a number in this file. In particular: the overlap window is one
whole MoE layer wider than the design describes and `lookahead=1` silently has **no**
window at all; the online planner is per layer, which is arithmetically the 15%-versus-
33% gap; decode and dummy forwards are suspected to revert every replica, which would
defeat the transfer reuse; and `max_concurrent_transfer_bytes` is never read.

**Before optimising anything, read `CURRENT-STATUS.md`'s "What today's profiling
settled".** Four rounds of design chased the transfers, which measure **0.32% of GPU
time**. The real cost is the host waiting for device data: GPU occupancy drops from
86.8% to 52.9%, one `cudaEventSynchronize` runs 11.02 ms, and the engine becomes
launch-bound. It is structural — `ncclSend`'s peer is a host `c_int` consumed at
enqueue, so the host must read the plan — and the escape is device-initiated
one-sided put, which needs NVLink. `bench/probe_nvshmem.py` is the 8-GPU probe to run
on H200 first; a 9 MiB put near 13 us would make several open questions moot.

**Two measurement traps this branch fell into, both still live.** The `budget=0`
baseline arm enables prediction, so every TTFT number compares placement *against
prediction*, not against nothing — prediction alone measured 19% of TPOT. And a
profile's aggregate operator table is dominated by idle `execute_dummy_batch` work:
in a 12.95 s trace the real forwards cover 0.54 s. Restrict to the `execute_context`
annotations.

`10` is **answered** (2026-08-26): prefill prediction is good enough. Planning from
predicted load and scoring on actual removes 33.2% against an oracle of 36.8%, and
accuracy on the peak rank's own experts — the only ones a placement can choose — is
0.90. The ticket's own claim that the realistic figure is the oracle bound "times the
accuracy" is wrong and corrected in it: a misled placement moves load onto the true
peak, so the quantity can go negative and a product cannot.

**Cross-forward residency now has conflicting evidence.** `00` measured -20.0%; the
2026-08-26 runs measure decode with resident replicas at **+3.0%**. The code follows
the direct measurement and keeps them resident across gated forwards. Re-check before
relying on either number.

`12` records the replica slots' 432 MiB per rank and defers cutting it; the obvious
alternative is ruled out by measurement.

`bench/RESULTS.md` carries the evidence and the errors corrected along the way, and
there were many: five defects in this path were each individually enough to make the
feature inert or harmful. Do not assume a green run means a connected one — check the
activation log line and that physical per-rank load diverges from canonical ownership.

`09` still has no blockers, and it is **not** true that it gates only `05`: it is
the sole owner of the under-load interconnect bandwidth that `04` requires. Its
wider purpose is live again now that the target is TTFT — `05` compares against its
numbers — and one of its own items is already answered: raising
`max_num_batched_tokens` makes mean TTFT 35% to 46% *worse*, so it is not the fix
for the queueing it looked like.

`06` needed only `02` and is done. `03` is done, 7/7.
