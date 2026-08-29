@AGENTS.md

## Predictive expert replication CUDA PoC

Before working on this feature, read in order:

1. `.scratch/predictive-expert-replication/spec.md`
2. `.scratch/predictive-expert-replication/CURRENT-STATUS.md` — start here for what is done, what is measured, and what is blocked
3. The ticket you are picking up, from `.scratch/predictive-expert-replication/issues/`
4. `.scratch/predictive-expert-replication/glossary.md`
5. `.scratch/predictive-expert-replication/reference/ascend-prior-art.md` (read-only timing prior art; the spec overrides it)

Implement only vLLM CUDA tickets `00`–`09` in this repository. Do not implement vLLM-Ascend here.

Ticket order is set by each ticket's `Blocked by` field, not by its filename number:

```
01 infra + layout          blocks 02, 03
02 prediction              blocks 06, 08     (needs 01)
03 source-rank routing     blocks 08         (needs 01)
00 feasibility checkpoint  blocks 07, 04, 05
06 accuracy study          blocks 04         (needs 02)
07 weight transfer         blocks 08         (needs 00)
08 activation              blocks 04         (needs 02, 03, 07)
09 serving baseline        blocks 05, and 04 for the under-load bandwidth item
10 prefill accuracy        blocks 07, 04, 08, 11 (needs 02)
11 quota routing eval     terminal          (needs 10) — evaluation only, may close 03's contract
12 replica slot memory    terminal          deferred; sized and priced, not on the critical path
13 device-side plan       terminal          needs NVLink + probe_nvshmem.py; removes the host sync
04 planner + lifecycle     blocks 05         (needs 00, 02, 06, 08, 09, 10)
05 validation + benchmark  terminal          (needs 04, 09)
```

Edges are listed rather than drawn: an ASCII diagram of this silently misaligned
its `06` and `04` edges into neighbouring labels, and each ticket's `Blocked by`
field is authoritative anyway.

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
