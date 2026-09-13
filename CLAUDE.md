@AGENTS.md

## Predictive expert replication CUDA PoC

> **Every negative verdict below was measured at ~877 tokens per forward on concatenated
> prompts, and at long prompts the sign reverses.** Same code and node, six interleaved passes,
> the whole feature against a stock server. The best measured point is **16k prompts, CONC=8,
> two replica slots: -6.17% mean TTFT, +7.03% total token throughput, 6/6 paired**; at 8k /
> CONC=16 with one slot it is -2.50% / +2.80%, 6/6; at 1k it is unreadable.
>
> **Two axes decide it, and a third qualifies every number.** (1) **Tokens per forward** — the
> cost is per forward (44 barriers, 44 launches, neither scaling with tokens) and the benefit is
> per token, so nothing here means anything without it stated. (2) **Queueing share** — the
> feature shortens compute, so the gain falls as queueing grows: -3.36% at CONC=8, -2.50% at 16,
> -1.20% at 32, and it **turns positive again at CONC=4** (+0.67%, 5/6 worse), which refutes the
> monotone prediction this file recorded on 2026-09-07. (3) **The domain.** `ko` is 161
> concatenated Korean instructions and concatenation inflates expert stability; `gov` is real
> long documents, and at 8k it has 72.4% stability against ko's 93.3% and is **+1.12%, 3/6 —
> unreadable and positive**. It takes 16k for gov to turn (-2.70%). **"8k is net positive" is
> true of concatenated prompts only.**
>
> **Expert stability is a precondition, not the explanation.** 2026-09-07 said one variable
> explained the reversal; that is corrected. Stability is 20.1% at 1k but already 88.5% at 2k,
> while 2k is still **+2.29%, 0/6** — it saturates well below where the sign flips. See
> `RESULTS.md` 2026-09-08.

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
02 config and accounting         terminal          3 of 4; the byte bound is still unread
03 fused Triton counting kernel  blocks 08         (needs 01)  DONE 2026-08-29
04 device-side plan              blocks 05         (needs 01)  DONE 2026-08-29
05 transfer lands in the slot    blocks 06         (needs 04)  DONE 2026-08-30
06 no host sync on the path      blocks 07         (needs 05)  DONE 2026-08-30, 17/17 at DP=8
07 window = target's Attention   blocks 08         (needs 06)  DONE 2026-08-30
08 three-arm verdict + stop gate blocks 10         BLOCKED; must be a curve over tokens/forward
09 CUDA graph feasibility        blocks 08         no blockers; ← the largest single lever
10 DeepSeek-V4-Flash             terminal          (needs 08)
11 prediction's collectives      blocks 12,13,14   ANSWERED 2026-08-30; diagnosis only
12 group of targets per source   blocks 13         DONE, then REFUTED on hardware
13 window of sources, 1 gather   blocks 08, 14     (needs 12) works: 78% of benefit kept
14 snapshot reduction off NCCL   terminal          CLOSED, DO NOT BUILD: nothing left at 8k
15 spent budget must not revert  terminal          DONE 2026-09-08, 5/5 at 8k on default budget
16 block bar dtype + teardown    terminal          no blockers; two review findings
17 spec says what was measured   blocks 08         no blockers; docs only
18 prediction fuses to 1 kernel  terminal          no blockers; ceiling 9.67 ms, launch half
19 more than one replica/layer   terminal          DONE 2026-09-08; cap2 pays at 8k and 16k
```

**`19` is the only lever on what placement *returns*, and it is measured on hardware now.** The
replica count is a placement-side parameter: prediction runs the same 44 gate GEMMs whatever it
is set to, and placement is the half that is negative cost. **Two slots pay at long prompts and
lose at short ones**: 8k/CONC=16 gives -3.17% against one slot's -2.51%, 16k/CONC=8 gives -6.17%
against -5.38%, all four 6/6 paired — but at 1k it costs +5.6%, because the transfer rate doubles
and at 20% expert stability that churn is unaffordable. `replica_slots_per_rank` still defaults
to 1; raising it is a long-prompt decision, and **raising it past 2 needs the budget raised first**
— see below. Offline replay predicted excess removed of 32.2% ->
46.1% for one slot -> two, a ratio of 1.43, and hardware gave 26.7% -> 37.7%, a ratio of 1.41 —
**the replay's scaling is confirmed**, which had never been checked. Its figures for four and
eight slots (59.2%, 70.2%) remain unrun, and `max_transfers_per_forward` defaults to 43 while
four slots over 44 layers want 176. The docstring figure of **68.1% at four slots is withdrawn**
— it is unreproducible on two dumps and is roughly what *eight* return.

**`11` is answered and it re-shaped the rest.** `06` showed placement is *negative cost* and
prediction is the whole overhead; `11` then split prediction's +13.68% by measurement:
**11.04 ms of 23.21 is its 44 per-layer barriers, 12.17 ms is its launches and compute.** So:

- `12` and `13` attack the barrier half (one source layer predicts K targets, one collective per
  group). Worth about 9 points of TTFT for about 0.14, but it lands at break-even, not positive.
  **All of this is a 1k accounting.** At 8k prediction alone costs **+0.24%** against stock and
  removing its AllGather leaves **+1.30%** — its whole cost amortises away, which is why `14` is
  closed rather than built and why the barrier/launch split no longer drives the work.
- `09` was called the only ticket attacking "the launch half, the larger one at 52%".
  **That framing is withdrawn (2026-09-06).** The 52% was a *residual* — everything left after
  removing the collective, labelled launches without being measured. An 8-rank attribution
  measures it: of the 8.14 ms per forward prediction adds, **host dispatch is 8.4%, device
  compute 1.9%, blocking event-sync 7.8% and gap 81.9%**. The old figure came from one rank
  (`sorted(glob)[0]`, dp0) and from counting only `cuda_runtime`, which excludes Triton's
  `cuLaunchKernelEx`. Read `bench/attribute_prediction_ops.py`'s own output, not this label.
  `09` is further reduced by the dispatcher: `num_tokens > max_cudagraph_capture_size` returns
  `CUDAGraphMode.NONE`, that cap is **512** on H100, and every prefill forward here is ~1024
  tokens — so **no captured graph is ever replayed on the path TTFT is made of**.
- `08` is **blocked**. Writing the verdict now would measure a cost `13` and `09` are removing,
  and its stop gate as worded fires on the sum while placement is returning 70-80% of the
  ceiling — see the amendment in the ticket.
- `15` is **done (2026-09-08), 5 of 5**. At the **default**
  `max_transfers_per_forward=4` a spent budget used to revert still-valid resident replicas,
  collapsing coverage from 44 layers to 4 on a traffic shift. The publish row now carries the
  *resident* placement when the planned one is unaffordable — in both fused kernels, the tensor
  oracle and the host coordinator. **Narrow on purpose: `found == 0` still reverts**, because
  that is the planner judging load rather than a budget accident. It affected no recorded
  measurement, all of which ran at budget 43. Its last criterion, the hardware run, closed at 8k:
  the **default budget of 4** gives -1.76% mean TTFT, 3/3, against budget 43's -2.13%. The 1k run
  could not have shown this — at 20% expert stability there is no steady state for the coverage
  ratchet to settle into — which is why the criterion stayed open. It matters more since `19`: at
  four slots a layer charges 4, so exhaustion goes from occasional to every forward, and each
  revert-and-re-transfer cycle is churn, which costs +27% mean TTFT at 4x.

**The transfer budget: `43` is stale, and the rate it bounds is 3 per forward.** The arms on this
branch all say `43`/`86`; the model has **48** layers and **44** are reachable
(`range(skip_first_layers=3, 48 - lookahead=1)`). 43 was the reachable count under the *old*
default lookahead of 2, and ticket 07 changed that on 2026-08-30 without the arm constants
following. It is nearly harmless because **the budget charges what moves, not what is held**:
`keep = found * resident * same` and only the rest is charged, so at 93-95% expert stability just
3 layers of 44 change their mind per forward. Measured steady state is **2.7-3.2 activations per
forward at cap 1** and 17 at cap 2, against caps of 43 and 86 — an order of magnitude of slack.
**The first forward looked like an exception and is not**: raising the budget to 48 leaves the
first report window at 61 activations against 43's 65, so forward 1 was never capped — coverage
ramps because the planner's `found` test does not pass for all 44 layers at once. **The
off-by-one has never affected a measurement.** Re-measured at 48 and 96 on 2026-09-08: at 8k the
conclusion is unchanged within intervals; **at 16k the two runs disagree by about 2 points on
arms that differ only in a budget that never binds**, so the 16k effect size does not reproduce
across runs and is being re-measured inside one interleave. At four slots the slack is gone — 44 layers x 4 wants 176 against 86 — and the budget
would stop bounding churn and start bounding coverage, which is exactly what `15` exists to
prevent.

Edges are listed rather than drawn: an ASCII diagram of this silently misaligned its
edges into neighbouring labels once, and each ticket's `Blocked by` field is
authoritative anyway.

**Placement costs about 3% of mean TTFT at DP=2, down from +32%** (2026-08-30). Two Triton
kernels in `distributed/eplb/fused_placement.py` take a placed layer from **132 kernel
launches to 2** — the device-side plan, publish and bookkeeping were tiny elementwise ops, and
in an eager engine each was a host dispatch. Placement's host overhead per layer is now zero
and the excess it removes is unchanged. Both kernels are asserted bit-identical to the tensor
versions they replace, which are retained as the oracles.

**`prediction_lookahead_layers` now defaults to 1, and it predicts better** — measured, not
assumed: `peak_hit_rate` 0.7702 against lookahead 2's 0.7132 and 27% less total-variation
error, on the layers both cover. Recall@1 is the planner's own metric because the per-layer
cap is 1. Lookahead 2 existed only to buy the host planner a layer of latency, and it cost 8%
of that accuracy to do it. **The launch moved too** (ticket 07, 2026-08-30): the transfer is issued at the predicting layer's MoE **tail**, so the overlap
window is the target layer's Attention and nothing more. Which site launches is the
coordinator's choice, not the runner's — the host planner synchronises inside
`plan_and_launch` and so keeps the old site, where a lookahead of 1 has no window at all and
configuration validation still rejects it. Reachable layers went 43 -> 44.

**This node has 8 H100s again** (NV18 all-pairs), after a spell with 2 following the
2026-08-30 pod restart. Check `nvidia-smi` rather than trusting a document: the count has
changed twice. The runtime scope check no longer pins DP=8; it warns instead, because **no
measured figure for this feature survives a change of EP size** — and that warning was
vindicated on 2026-08-30 night by the most optimistic figure on the branch.

**`06` is DONE, 17 of 17, measured at DP=EP=8** (2026-08-30 night). Its two outstanding
criteria: **26.2%** of full-prefill critical-path excess removed on **44 of 44** reachable
layers, against the 24.0% asked for; and occupancy measured with `bench/window_occupancy.py`,
which the tree lacked. The headline is carried directly rather than through occupancy, because
raw occupancy counts a rank parked inside `ncclDevKernel` as busy: **placement performs the
same number of host event synchronisations as a stock server**, 1081 against 1072, where the
host path performed 7293, and `cudaStreamSynchronize` went 38672 -> 8. Gaps over 0.5 ms in the
placing arm are 3.52 ms against stock's own 3.86 ms and the host path's 5.70 ms — the
86.8%/52.9% reference is a **5090** measurement and cross-hardware, so the same-node comparison
is the one to quote. `06`'s own named residual risk is
closed too: `VLLM_PREDICTIVE_VERIFY_REPLICA_WEIGHTS` verifies on real weights that a
*dynamically* placed replica holds the bytes it claims.

**The two halves of this feature now have opposite verdicts, measured at DP=8 on a 0.3% ruler**
(the knee: `ko`, 512 requests, CONC=16, three interleaved passes — the tightest baseline this
project has had):

```
stock            167.53 ms  —          excess removed, full prefill: 27.0% / 26.5% / 26.7%
prediction only  190.19 ms  +13.5%     across three repeats, 0.5-point spread
placing          183.24 ms   +9.4%     p99 flat across all three arms
```

**Placing is 6.95 ms faster than prediction only** — the arms differ in nothing but the
transfer budget, so placement's own contribution is *negative cost*. Against a ceiling of
5.26% of a prefill window (11.16% expert-GEMM share x 47.09% recoverable, both measured
today) placement captures roughly 70-80% of it. **Prediction alone costs 2.6x that entire
ceiling.** So the feature is net negative because of its *enabling* half, not its acting half.

**And "prediction is nearly free" is a DP=2 statement only.** DP=2 recorded -0.2%; at DP=8 the
same code costs +13.5%. The mechanism is in the profile: token collectives per prefill window
grow **47.3 -> 82.4 ms** while the expert GEMM does not move. That is the recorded
desynchronisation signature at 8 ranks, which 2 ranks have too little arrival skew to show.
`08` owns the verdict and it is now unambiguously about prediction's 44 gate GEMMs and 44
AllGathers per forward, not about placement.

**The snapshot AllGather: the recorded reason for closing the batching question was wrong,
and the correction matters more than the number.** Traced at DP=8 over 8 ranks and 93
prefill windows:

- Its p50 is **545 us**, not 9.3 us, and its aggregate residency is **18.24 ms per prefill
  window**. The recorded 9.3 us was **dp6's** kernel time, and dp6 is the rank that arrives
  **last** — the one rank that never waits. The other seven see 130 to 748 us. A barrier's
  cost is never its own duration on the pace-setting rank.
- **1.6%** of that residency overlaps real compute, though `start_snapshot`'s docstring
  intends it to hide behind the local expert GEMM. **90.4%** is co-resident with the token
  collectives and 9.6% overlaps nothing.
- Prediction adds **no** net compute: per-rank compute-only occupancy is uniform in both arms
  and the absolute compute time is unchanged (25.8 -> 23.5 ms). The window grows 18 ms and
  **all of it is waiting.**
- **Isolated by measurement, and the answer is half and half.**
  `VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER` keeps prediction's compute and removes only the
  collective. Three arms, three passes: stock 169.68 ms, prediction 192.89 ms (+13.68%),
  prediction without the AllGather **181.85 ms (+7.17%)**. So of the 23.21 ms prediction adds,
  **11.04 ms is the 44 barriers (47.6%) and 12.17 ms is its launches and compute.** A profile
  confirms the mechanism directly — collectives per window 236 -> 192 with the token
  collectives untouched, window 106.9 -> 97.2 against stock's 88.9, so 0.22 ms per barrier
  plus 8.3 ms of launch overhead at unchanged collective count. **My "0.46 ms per collective,
  payload-independent" hypothesis is disproved by the same profile**: at the same 192
  collectives, stock is 88.9 ms and the probe 97.2 ms.
- **Batching works, but only from *different* source layers — and the first design proved it
  the expensive way.** Having one source predict `K` targets was built, passed 566 unit tests,
  served, and removed **3.7%** of critical-path excess where one target per source removes
  **26.8%**. Four layers' gates on one layer's hidden states select almost the same experts:
  within a group the predicted distributions differ by an L1 of **16 to 36** out of 7896 while
  the targets' actual loads differ by **5412 to 7660**. So three of every four targets were
  planned from a distribution that was not theirs. Ticket 11's accuracy curve could not predict
  this — it measures *distance*, one source per target, and the quantity this design rested on
  was never measured because nothing had asked for it. The second design — a **window** of `K`
  consecutive sources, each predicting its own target, the last one issuing the single
  AllGather — keeps **20.8%** of excess (78%) and puts the whole feature at **+1.0%** mean TTFT,
  inside the baseline's own 4.3% spread. It needs `lookahead >= group`, and it made the recorded
  staging-buffer race likely enough that the buffer is now indexed per window position rather
  than by layer parity. Default is still `prediction_target_group = 1`.
- **Accuracy is not the obstacle it looks like, and lookahead 4 is now measured.** On `ko` at
  DP=8, replaying predicted placements against actual load: excess removed **34.9 / 34.4 /
  33.5 / 32.1%** at lookahead 1 / 2 / 3 / 4, with 0 forwards made worse at any distance,
  while `count_error` more than doubles (0.058 -> 0.133). Peak-rank recall@1 stays ~0.95
  throughout, which is why the delivered benefit barely moves: the planner only picks from
  there. So **lookahead 4 costs 2.8 points of 34.9** — but read that with the bullet above:
  the curve measures one source per target and cannot price a design that shares a source.
  Measured on hardware, a window of 4 costs **6.0 points** of 26.8, about 2 of which are the
  three trailing layers a lookahead of 4 gives up.

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

**VERDICT (2026-08-29, 8x H100 SXM): negative — but at ~877 tokens per forward only, and
superseded for the long-prompt regime by 2026-09-07 above.** Ticket 14's third arm finally measured the feature against a stock
server: prediction and its infrastructure cost **+7.6% mean TTFT** against a
perfect-balance ceiling of
**5.05% of a prefill step**, so the mechanism that creates the transfer window spends
1.51x what perfect expert balance could ever return, before a replica moves. The whole
feature costs **+31.5%**. Ticket 13's device-side transfer removes the host sync and most
of placement's share, but not prediction's 43 gate matmuls and 43 AllGathers per forward,
so the ceiling stays under the floor on the fastest interconnect NVIDIA ships — and the
conclusion transfers to the Ascend port. ~~Batching those AllGathers is **not** the escape:
traced here they are 9.3 us each, 0.40 ms per forward, 4% of the window they hide in.~~
**That is withdrawn — see the AllGather section below. The 9.3 us was one rank's kernel
time, and it was the rank that arrives last, so it is the one rank that cannot see the
cost.** The
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

**`18` is implemented and measured (2026-09-06).** Prediction's gate, selection and count are
one Triton kernel; launches per source layer go 2.7 -> 1. Four arms, three interleaved passes at
the knee: stock 183.78, prediction unfused 212.56, prediction fused 205.90, placing fused 194.53.
Paired within each pass it is faster **3 of 3** — +12.99, +4.92, +1.94 ms — **median 2.6% of
stock TTFT**. The sign is solid; the magnitude is not, because this run's stock arm drifted 5.0%,
wider than the 3.1% mean gap between the two arms, so only the pairing carries it. **Do not quote
the 3.6% mean.** It beat its own first-order dispatch ceiling of 0.9-1.5%, which is evidence that
launch count drives part of the 82% gap through rank skew — the fused arm's spread is 0.9%
against the unfused arm's 5.2%. Selection is held bit-identical to the reference (fp32 forces
IEEE over TF32; bf16 rounds the accumulator back to bf16 before selecting; ties break low), and
anything unreproducible falls back on static, rank-identical properties.

`09` still has no blockers, and it is **not** true that it gates only `05`: it is
the sole owner of the under-load interconnect bandwidth that `04` requires. Its
wider purpose is live again now that the target is TTFT — `05` compares against its
numbers — and one of its own items is already answered: raising
`max_num_batched_tokens` makes mean TTFT 35% to 46% *worse*, so it is not the fix
for the queueing it looked like.

`06` needed only `02` and is done. `03` is done, 7/7.
