# Current implementation status — vLLM CUDA PoC

Read this before changing code. `AGENTS.md` remains binding.

**Moving to another node?** Start with
[`HANDOFF-2026-08-29-H100.md`](HANDOFF-2026-08-29-H100.md), written for a session with
no prior context.

**Just want the conclusions?** Read "Current baseline" and "What today's profiling
settled" below, in that order. [`HANDOFF-2026-08-24.md`](HANDOFF-2026-08-24.md) holds
the earlier decode answer and is **superseded on the prefill question**: the target
moved to TTFT on 2026-08-25 and its verdict predates every measurement since. Method
and caveats stay in `bench/RESULTS.md`.

## Scope

Implement only the vLLM CUDA tickets in `issues/00` through `issues/09`.  Do not implement an Ascend port in this repository or broaden the approved model/runtime scope. `reference/ascend-prior-art.md` is read-only timing prior art; the spec overrides it.

## Current baseline

- Branch: `feature/predictive-expert-replication`
- Original vLLM base: `10704541aaf72567fe9d6229b3e3d84d37f2ddba`
- Tickets `01` `03` `04` `05` `06` `07` `13` `14` `15` `18` `19` are **done or closed**;
  `02` is 3 of 5, with the byte bound (`max_concurrent_transfer_bytes`) still declared and
  never read. `08` was **unblocked on 2026-09-08** — the curve it was waiting for exists —
  and is the ticket that decides the project; `09` is ready-for-agent, terminal and off the
  mainline; `10` waits on `08`. `00` answered for both regimes. (This line previously mixed
  the superseded `00`-`14` numbering into the current set, and said `07` was unstarted while
  it was done.)
- The feature runs end to end and is **correct**: replicas are transferred, published
  where routing reads, routed to, and reverted. It removes **26%** of prefill
  critical-path excess at one replica slot and **38%** at two, stable across every
  prompt length measured.
- **The sign depends on the regime, and a bare verdict is not one.** Against a stock
  server, six interleaved passes, paired (2026-09-08):

  ```text
  1k  / c16 / ko     unreadable        2k  / c16 / ko    +2.29%  0/6 (worse)
  8k  / c16 / ko     -2.50%  6/6       8k  / c8  / ko    -3.36%  6/6
  16k / c8  / ko     -4.67%  6/6       16k / c8  / cap2  -6.17%  6/6  <- best measured
  8k  / c4  / ko     +0.67%  1/6       8k  / c16 / gov   +1.12%  3/6  <- real documents
  ```

  Three things move it: tokens per forward, queueing share (concurrency), and the domain.
  **`ko` is 161 concatenated instructions per prompt and concatenation inflates expert
  stability** (93.3% against `gov`'s 72.4% at 8k), so the 8k result does not transfer to
  real long documents — `gov` needs 16k to turn, and turns to -2.70%.
- The old headline here, "net negative on this node: mean TTFT +13% to +29%", was a
  ~877-tokens-per-forward measurement and is **superseded**, not contradicted.
- **The cost is host synchronisation, not the transfers**, and at long prompts it
  amortises: prediction alone costs +13.5% at 1k and **+0.24%** at 8k. See the section
  below before touching anything.

## Ticket 13, 2026-08-31: batching the snapshot works, but only from different sources

Two designs were built for one collective per `K` layers. **The first was refuted on hardware
and the second holds**, and the difference is the transferable part.

**First design: one source predicts `K` targets.** It evaluated `K` target gates on one layer's
hidden states — one concatenated GEMM, one counting kernel, one collective, and only every
`K`-th layer predicting, so it attacked both halves of prediction's cost. 566 unit tests passed.
On 8 GPUs it collapsed:

| | full prefill excess removed |
| --- | --- |
| window 1, before the refactor | 26.5 / 26.7 / 27.0% |
| window 1, after the refactor | **26.9 / 26.8%** |
| first design at 4 | **3.7 / 4.6%** |

Everything looked healthy: `connected: true`, 0 failed, 352 launches, coverage intact at 44 of
44 layers. Only the benefit was gone. Ruled out in order: the refactor (window 1 after it
reproduces window 1 before it to 0.1 points, so ticket 12's inertness holds on *benefit*, not
only on launch counts), the row-to-target mapping (a differential test through the real runner
loop), the collective's layout (a 4-rank gloo test: rank-major, target-minor, byte-identical),
the binder, the fused gate projection (`block_norms` match the per-gate logits element for
element), and instantaneous accuracy (distance 4 delivers 32.1% offline).

**The cause is a property of the model, not a defect.** Four different logit vectors whose
top-k selections are nearly the same:

    within a group, the four predicted distributions differ by an L1 of 16, 24, 36
    across groups, two sources' predictions differ by an L1 of 5412 to 7660
    (of 7896 assignments)

while the four target layers' *actual* loads are completely different. Applying four layers'
gates to one layer's hidden states ranks experts almost identically — magnitudes differ by 20%
but top-k only uses the ranking. So one target per group was planned from its own distribution
and three were not: 13 of 44 layers improved, 4 got worse, the aggregated imbalance moved
*more* than at window 1 (-0.0155 against -0.0134) while the critical path moved a seventh as
much. Load was being shed, just not off each layer's own peak.

**The methodological finding.** Ticket 11's accuracy curve measures *distance* with one source
per target. The first design depended on "the same source distinguishes different targets", and
that quantity is **0.2%**. It was never measured because nothing had asked for it. An offline
accuracy curve is not a substitute for the quantity a design actually rests on.

**Second design: a window of `K` consecutive sources, each predicting its own target.** Each
writes one row of a shared buffer and the window's last source issues the single AllGather.
Measured in the same interleaved passes:

| arm | full prefill excess | mean TTFT | vs stock |
| --- | --- | --- | --- |
| stock | — | 180.30 ms | — |
| prediction only, window 1 | — | 203.03 ms | +12.6% |
| placing, window 1 | 26.8 / 26.7 / 26.8% | 187.25 ms | +3.9% |
| **placing, window 4** | **20.8 / 20.7 / 20.8%** | **182.15 ms** | **+1.0%** |

**78% of the benefit survives**, against the first design's 14%. Placement's own contribution is
**-20.9 ms** (203.03 -> 182.15) where the first design's was zero. The feature's total cost is
**+1.0%**, inside the baseline's own 4.3% spread — the first configuration this project has had
that a run cannot distinguish from stock. That is not a claim of break-even: it is a claim that
this measurement cannot resolve the difference, and resolving it needs the tighter baseline this
operating point has reached before (0.3%).

Two things are open. The excess loss is **6.0 points** where the distance curve predicts 2.8;
about 2 of those are coverage, since a lookahead of 4 leaves 41 reachable targets rather than 44
(the smoke confirms it at 328 launches, not 352), and roughly 1 point is unexplained. And the
default is still `prediction_target_group = 1`.

**The staging hazard this design would have made likely is removed rather than tolerated.** The
recorded hazard — `barrier_all` orders arrival, not one rank's next put against another's drain
— was survivable only because consecutive same-buffer layers were a layer of compute apart. A
window removes that margin. Verified first with the runtime replica-weight check (real weights,
window 4, 60 forwards, no mismatch on any of them), and then made impossible: the staging buffer
is indexed by `target % staging_buffers` with one buffer per window position, at
`group x 9.00 MiB` per rank — 36 MiB at a window of 4 against the 432 MiB the replica slots
already hold.

## Ticket 11, 2026-08-30 night: prediction's cost is half barriers and half launches

Read this with the `06` section below; `11` is new and it is the mainline. `06` established
that placement is *negative cost* and prediction is the entire overhead, and this splits that
overhead in two by measurement rather than by argument.

`VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER` keeps prediction's compute — gate GEMM, top-k,
counting kernel — and removes only the 44 per-layer snapshot collectives. Knee, three
interleaved passes, and the control reproduced the earlier run to within 0.2 points:

| arm | median mean TTFT | vs stock |
| --- | --- | --- |
| stock | 169.68 ms | — |
| prediction, AllGather kept | 192.89 ms | **+13.68%** |
| prediction, AllGather removed | **181.85 ms** | **+7.17%** |

    prediction adds                        +23.21 ms
    removing the 44 AllGathers gives back  -11.04 ms   = 47.6%
    what is left                           +12.17 ms   = launches and compute

**A profile confirms it directly rather than by inference**, which matters because inference
from one rank is how the previous answer went wrong. Collectives per prefill window go
**236 -> 192** with the token collectives untouched at 192, and the window goes
**106.9 -> 97.2 ms** against stock's 88.9. So the window decomposes as stock, plus **8.3 ms**
of prediction launch and compute at unchanged collective count, plus **9.7 ms** for 44
barriers — **0.22 ms per barrier** for a 512-byte payload. The window split (54% barriers) and
the TTFT split (47.6%) agree.

**And it disproves the hypothesis this section was written around.** "The window's length
tracks the number of collectives, 0.463 against 0.453 ms each, payload-independent" is wrong:
at the same 192 collectives stock is 88.9 ms and the probe is 97.2 ms. Barriers are one
additive term of three, not the whole model.

**Lookahead 4 is measured, and the accuracy cost of batching is small.** On `ko` at DP=8,
replaying predicted-chosen placements against actual load in the prefill regime:

| lookahead | recall@1 | peak-rank recall@1 | count error | **excess removed** | oracle | made worse |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.9015 | 0.9621 | 0.0581 | **34.9%** | 36.4% | 0 |
| 2 | 0.8682 | 0.9457 | 0.0856 | **34.4%** | 37.0% | 0 |
| 3 | 0.8730 | 0.9524 | 0.1089 | **33.5%** | 37.7% | 0 |
| 4 | 0.8862 | 0.9512 | 0.1333 | **32.1%** | 37.6% | 0 |

Ticket 10's pattern reproduces on a second domain and extends to 4: `count_error` more than
doubles while the delivered benefit falls **2.8 points**, because peak-rank recall@1 holds at
~0.95 at every distance and the planner only ever picks from there. Three full-prefill forwards
per lookahead, so the excess column is a small sample with a monotone trend.

**What this makes worth building, and what it does not reach.** A K=4 grouping — one source
layer evaluating K target gates on its own hidden states, the K gate GEMMs concatenated into
one, one collective per group — costs about 1.2 points of excess (0.14 points of TTFT) and buys
about 9 points. Roughly 70 to 1. But it lands prediction near +4.8% against a ceiling of 5.26%
and placement's return of ~4%, which is **break-even, not positive**. The launch half is 52%
of the cost and neither batching nor a device-side reduction touches it — **ticket `09`, CUDA
graph capture, is now the largest single lever in the project**, held back by the spec's
eager-execution mandate rather than by any evidence.

## Ticket 06 closed on 8 GPUs, 2026-08-30 night: placement is no longer the cost

**This node has 8 H100s again** — 8x H100 80GB HBM3, NV18 between all pairs — after a spell
with 2. Every claim below is DP=EP=8. Ticket `06`'s two outstanding criteria are measured and
it is **17 of 17**.

**Criterion: at least 24.0% of full-prefill critical-path excess removed, on every reachable
layer.** Measured **26.2%**, at the operating point the 24.0% was recorded at (`ko`, 400
requests, CONC=8, OUT_LEN=1):

| band | forwards | critical path | excess removed |
| --- | --- | --- | --- |
| full prefill | 50 / 50 | 1.8901 -> 1.6571 | **26.2%** |
| partial prefill | 50 / 50 | 1.8927 -> 1.7239 | 18.9% |

Coverage is **44 of 44** reachable layers, targets 4 to 47 with no gap, 352 launches = 44 x 8
ranks. `connected: true`, 400/400 requests on both arms, 0 failed. 50 full-prefill forwards per
arm against the 15/13 the original figure rested on.

**Criterion: GPU occupancy inside real forward windows.** Stock **86.7%**, which reproduces the
86.8% reference almost exactly; prediction only 89.9%; placing **41.8%**, which is *below* the
52.9% reference rather than above it. **That number does not mean what it looks like.** Per
rank, occupancy tracks time spent inside `ncclDevKernel`: the ranks at 21-29% are the ones with
11-12 ms of collective time and the ones at 88% have 58 ms. A rank blocked in a collective
waiting for a peer is busy by kernel-time accounting and idle in fact, so raw occupancy rewards
arriving early. The same lesson as "a collective's duration is mostly a measurement of what the
other rank was doing", now applied to occupancy.

What host starvation actually looks like is long gaps, and those say the opposite: **gaps over
0.5 ms are 3.52 ms in the placing arm** against stock's own 3.86 ms on the same node and the
same day. So the placing arm leaves *fewer* long holes than a stock server.

**Read the 86.8% / 52.9% / 30.2 ms reference as cross-hardware, because it is.** Those figures
were measured on the **5090** node on 2026-08-26; the H100 arrived on 08-29. The criterion names
them, so they are reported, but the sound comparison is host path against device path on *this*
node with *this* tool, and that one is: placing occupancy 69.7% -> 41.8% while its gaps over
0.5 ms go 5.70 ms -> 3.52 ms. Occupancy fell and idleness fell with it, which is only
contradictory if occupancy measures what its name suggests.

**The ticket's headline, measured directly instead of through occupancy.** Per profile, 8 ranks:

| | `cudaEventSynchronize` | `cudaStreamSynchronize` |
| --- | --- | --- |
| host path, placing | **7293** calls, 8417 ms, p50 73.2 us | **38672** calls, 285 ms |
| device path, placing | **1081** calls, 1643 ms, p50 5.7 us | **8** calls, 0.07 ms |
| device path, stock | 1072 calls, 1820 ms, p50 5.6 us | 8 calls |

**Placement performs the same number of host event synchronisations as a stock server.** That
is "no host synchronisation on the per-forward path" as a measurement against an external
reference rather than against itself.

**And the marginal cost matches.** Placing costs **+4.5%** mean TTFT over prediction only
(176.02 against 168.44 ms), where the host path cost +22.1% over it on 2026-08-29 (242.79
against 198.81). The placing arm's p99 is **245 ms against stock's 356**.

**But prediction is not free at 8 ranks, and that is the live surprise.** Against a stock
server at the same operating point, prediction only measured **+19.1%** (168.44 against
141.41 ms) where DP=2 had recorded **-0.2%**. The mechanism is in the same profile — token
collectives per prefill window grow **47.3 ms -> 82.4 ms** while the expert GEMM does not move
(9.79 -> 8.32 ms). That is the recorded desynchronisation signature, +39.9 ms before, at 8
ranks; 2 ranks have almost no arrival skew for extra launches to amplify. `CLAUDE.md`'s warning
that **no measured figure for this feature survives a change of EP size** held, and it held for
the most optimistic figure on the branch.

So the cost centre has moved for the third time: host synchronisation -> launch dispatch ->
**prediction's own per-layer collectives and launches**, which is the one thing neither `03`
nor `06` nor `07` touches. Ticket `08` owns the verdict.

### Re-measured at the knee with a 0.3% ruler: the two halves have opposite verdicts

CONC=8 cannot resolve a 5% effect, so the three arms were re-run at the knee (`ko`, 512
requests, CONC=16, OUT_LEN=1, three interleaved passes). **The stock arm's own spread across
passes is 0.3%** — the tightest ruler this project has had, against 4.9% earlier in August:

| arm | n | mean TTFT | p99 | req/s | own spread | vs stock |
| --- | --- | --- | --- | --- | --- | --- |
| stock | 3 | 167.53 ms | 307.47 | 94.07 | **0.3%** | — |
| prediction only | 3 | 190.19 ms | 303.92 | 83.14 | 4.1% | **+13.5%** |
| placing | 3 | **183.24 ms** | 307.98 | 86.08 | 3.1% | **+9.4%** |

**Placing is 6.95 ms *faster* than prediction only.** The arms differ in nothing but the
transfer budget, so that is placement's own contribution and its sign is negative cost:
placement gives back 4.1 of the 13.5 points prediction spends. p99 is flat across all three
arms and throughput moves with it. Excess removed in the same run: **27.0%, 26.5%, 26.7%**
across the three repeats, a 0.5-point spread, `connected: true` and 0 failed everywhere.

    expert GEMM share of a stock prefill window     11.16%   wall-clock, measured today
    recoverable share of MoE time at cp 1.8901      47.09%
    -> perfect balance is worth                      5.26%   of a prefill window = 4.68 ms
    placement returned                               6.95 ms of mean TTFT, so roughly
                                                             3.5 ms per window: 70-80%
                                                             of the ceiling
    prediction alone                               +13.5%    = 2.6x the whole ceiling
    prediction + placement                          +9.4%

**Placement works and captures most of the headroom that exists. Prediction costs 2.6x
everything perfect balance could ever return.** The feature is net negative because of its
enabling half, not its acting half — which is a different conclusion from the recorded verdict,
and a more actionable one. The soft link in that chain is "roughly two prefill windows per
TTFT", inferred from 88.9 ms against 167.53 ms rather than measured; the excess and TTFT
figures either side of it are direct.

**Two new pieces of tooling, both of which the tree lacked.**

`bench/window_occupancy.py` is the occupancy measurement `06` asked for. The recorded 86.8% /
52.9% figures came from ad-hoc analysis that was never saved, so there was nothing to re-run.
It restricts to the runner's own `execute_context` prefill windows, takes busy time as the
**union** of kernel intervals rather than their sum — summing overlapping streams is what once
reported the expert GEMM at 10.76% of a step where wall-clock puts it at 14.29% — and carries
`occupancy_ex_nccl` for the work-versus-waiting split above. 22 unit tests, each pinning a
mistake this project has already made.

`VLLM_PREDICTIVE_VERIFY_REPLICA_WEIGHTS` closes `06`'s own named residual risk: nothing in a
real server checked that a *dynamically* placed replica holds the bytes of the expert it
claims, because `verify_replica_weight_equality` runs at startup and reported "0 pairs,
vacuous". It now runs every forward when armed, and on 8 GPUs with **real weights** — required,
since dummy weights make every expert row identical and a wrong row would pass — it verified
**4 placed pairs** with no mismatch.

**A process failure worth recording, because it invalidated a whole run.** The first attempt
at the excess measurement was ruined by **editing the tree while the run was in flight**: the
placing arm's server started between two edits and died on an environment variable that did not
exist yet, failing 392 of 400 requests. The guard caught it and called the arm inert, which is
what the guard is for. Every stage of the successful run went through one driver
(`bench/run_t06_close_out.sh`) so the tree is provably identical across all three.

**One recorded claim is now doubtful.** "Balancing shortens the expert GEMM from 10.01 ms to
6.88 ms, 67% of the ceiling — the first direct sight of this feature working" does not survive
the per-rank view. Within a single arm the expert GEMM per 1000 prefill tokens spans 4696 to
10134 us across 8 ranks, a 2.2x spread, which is **wider than the difference between arms**. By
`report_arm_spread.py`'s own rule — if an arm's own spread exceeds the difference between arms,
the run says nothing — that comparison is not resolved by this profile. The 26.2% figure above
is measured on token counts and is unaffected.

## Code review, 2026-08-30 night: one fixed, six recorded

A `high`-effort review of the whole branch against upstream (23 commits plus the working tree)
found seven issues; all 363 of the branch's own tests pass, so every one of them is in a path
those tests do not cover. **One was introduced this session and is fixed. Six are pre-existing
and are recorded here rather than fixed, because they are outside ticket `06`'s scope and the
first of them deserves its own change with its own test.**

**Fixed: the new per-forward replica-weight check allocated a float64 copy of every expert
tensor.** `weight.reshape(local_rows, -1).to(torch.float64)` plus `flat * position` is two
transients of 8 bytes per element — 428 MiB each for a measured `[17, 3145728]` expert — and
splitting the check out of startup put it on every forward, on a device whose free memory the KV
cache has taken. It accumulates a row at a time now, so the peak is one row, and the arithmetic
is unchanged, which the equality tests pin. The environment variable also now says in as many
words that it must not be enabled during a measurement run, and why it is unconditional rather
than sampled every N forwards: the check contains an all_gather, and a per-rank interval counter
next to a collective is this branch's most expensive class of bug.

**Open, and the one that would bite an operator on the default configuration: an exhausted
transfer budget reverts still-valid resident replicas instead of keeping them.**
`fused_placement.py`'s `publish_ptr[0] = max(keep, affordable)`. When a layer holds resident
`A@T`, this forward's plan names a different `B@T'`, and the budget is already spent, then
`keep=0`, `needs=1`, `affordable=0`, so the publish takes the **revert** branch and clears the
layer — even though `A@T`'s weights are still in its row and cost nothing to keep. At the default
`max_transfers_per_forward=4` over 44 reachable layers, a traffic shift re-places 4 layers and
**reverts the other 40**, collapsing coverage and needing ~10 forwards to ratchet back. That
contradicts the "coverage is free once resident, and the budget bounds churn" property the same
kernel's docstring claims. The host `PlacementCoordinator` has the same shape at
`remaining <= 0`. **It did not affect this session's numbers**, which ran at budget 43 with 44
reachable layers, where the budget is not exhausted in steady state and the excess figure was
stable to 0.5 points across three repeats. Fixing it needs the publish row to be able to carry
the *resident* placement rather than only the new one, so it is a real change and not a flag.

**Open, five more.**

*`resolve_moe_block_size_m` is never told the quant dtype or block shape*, so on a block-quantised
MoE it resolves 128 where `get_default_config` would return 64, and the tuned-file lookup is
keyed on dtype too. That constant is both the suppression bar and the planner's minimum move, so
every prefill forward between 64 and 128 tokens per expert is suppressed and every placement
shedding 64-127 tokens is refused — the exact failure the function's docstring says it exists to
prevent. `moe_config.quant_dtype` and `.block_shape` are available on the layer. This was already
recorded as open; the review adds the mechanism and the size.

*`max_replicas_per_layer`'s bound is hardcoded to 8 ranks* (`> 7` rejected, "with 8 EP ranks the
maximum is 7") while DP is now accepted from 2 up. At EP=2 a cap of 2 to 7 passes validation and
is silently reduced to 1 by `best_move`'s own constraints, with no diagnostic — on the knob the
branch's thesis says sets coverage. The bound should come from the EP size.

*The NVSHMEM fallback can hang on the cleanup side.* `self._one_sided` is assigned only once
`OneSidedExpertTransfer.__init__` returns, so a rank whose `nvshmem.init` succeeded but whose
symmetric staging allocation then failed skips `close()`, while every other rank blocks forever
in the collective `nvshmem.finalize()`. `agree_across_ranks` was added to turn exactly this
asymmetry into a clean fallback, and this reintroduces it after teardown.

*`static_replica_placement` is invisible to the device residency table*, so the first device
placement on that layer writes the replica row without reverting the static expert's maps, and
half the ranks then route that logical expert to a row holding another expert's weights with
nothing raising. Debug knobs only, but independently settable and documented as validation aids
meant to be used together. (This was already recorded; the review adds the routing consequence.)

*`device_transfer.py` builds a fresh `cuda.core` stream wrapper on every `transfer()`*, ~44 per
forward, around the same underlying handle — per-layer host work and object churn on the one path
this ticket exists to make host-free. Cache it for the coordinator's single predictive stream.

## The fused placement kernels, 2026-08-30 night: placement now costs about 3%

Two Triton kernels replace the **132 launches** a placed layer cost — 46 to plan, 19 to charge
the budget, 67 to publish — and the profile's per-layer host figure went with them:

| placing arm, per MoE layer | before | after |
| --- | --- | --- |
| `moe_forward` host time | 2.20 ms | **1.32 ms** |
| the same, prediction-only arm | 1.25 ms | 1.33 ms |
| kernels per placed layer | 132 | **2** |
| GPU idle in a prefill window | 77.9% | 48.4% |

**Placement's host overhead per layer is now zero.** End to end at DP=2, four repeats each:

| arm | before | after |
| --- | --- | --- |
| stock | 257.8 ms | 260.0 ms |
| prediction only | 264.6 ms (+2.6%) | 259.6 ms (**-0.2%**) |
| placing | 340.3 ms (**+32.0%**) | 266.6 ms (**about +3%**) |

The placing arm's p99 is now the lowest of the three, 374.8-378.1 ms against stock's
379-712 ms, and its repeat spread fell from 11.8% to 4.4%. Read the second run's stock median
with care: its first repeat is 317.8 ms against 258-261 for the rest, so the paired per-repeat
deltas — +2.1%, +2.1%, +6.4% — are the honest read.

**The placement itself is unchanged**, 34.5-36.5% of full-prefill excess removed against
34.8-36.6% before, `connected: true` on all four repeats. Both kernels are asserted
bit-identical to the tensor implementations they replace, over randomised snapshots, a 200-plan
publish sequence and the real expert geometries. That test caught the one bug that mattered:
the kernels work in doubled integer units so half an expert is exact, and the shed amount is
the *undoubled* load — using the doubled one refused 40 of 200 placements the tensor planner
accepts, each time producing a plausible all-zero plan.

**A correction to this session's own advice.** The arrival barrier measured 4.40 ms per forward
and I proposed replacing it with pairwise put-with-signal. It now measures **0.23 ms** with
nothing about it changed: it was exposing arrival skew, and the skew was the launch storm.
Pairwise arrival is worth 0.23 ms and belongs well down the list. A collective's duration is
mostly a measurement of what the other rank was doing — the same lesson as the 5090's
2000x-spread AllGather.

## Code review, 2026-08-30 night: six more fixed, five open

A second review of the branch found 14 issues. Six are fixed with tests; five are recorded
below because they are real and not yet done, and three of the six are worth naming because
of *how* they hid.

**Fixed, and it would have failed CI.** Twelve files carried a merged SPDX header
(`# SPDX-License-Identifier: Apache-2.0 SPDX-FileCopyrightText: ... to` / `# the vLLM
project`), which `check-spdx-header` rejects. It came from a reflow in `d760c2d33`, and the
`pre-commit` hook's own auto-fix then *prepended* the canonical two lines rather than
repairing them, so six files ended up with both. Restored everywhere and verified against
the hook.

**Fixed, my own new bug.** `lookahead_pair` keyed its comparison by lookahead alone, so with
the accuracy runner's two domains per lookahead the last report silently won — the reviewer
reproduced code at 0.9 and text at 0.1 reporting 0.1 for both arms and "not better at 1",
printed as the answer to ticket 07. It is per domain now, compares the two shortest
lookaheads and names any it ignored, and never raises: with the runner's own default
`LOOKAHEADS="1 2 3"` it used to raise into two callers' `except` and the deliverable table
vanished without a word. Three new tests.

**Fixed, and unreachable today, which is how it would have survived.** The device path's
publish did not pass `slot`, so it always wrote row `per_rank_experts + 0` while
`drain_expert` wrote `replica_row_of(canonical_per_rank, slot)`. At any slot but 0 routing
would point at an unwritten row with the weights one row over, silently.
`replica_slots_per_rank` is validated to 1, so nothing could reach it — yet.

Also fixed: the host coordinator never ordered the predictive stream behind the compute
stream, so its drain could overwrite a replica row the previous forward's MoE was still
reading — the same hazard the device path had, and worse since `activate` became a stream
wait and the CPU runs further ahead. `VLLM_PREDICTIVE_PLACEMENT_REPORT_EVERY=0` raised
`ZeroDivisionError` on the first forward, where 0 is the natural spelling for "never". An
empty snapshot crashed the fused planner on `tl.arange(0, 0)` where the tensor version
places nothing. And both fused kernels now reject a non-unit innermost stride rather than
reading the wrong elements, since they index directly and every equality test builds
contiguous tensors.

**Fixed after the review: the device path is now agreed across ranks.** It was decided per
rank inside a `try/except` whose handler fell back to the host path, and the setup contains
collectives — `broadcast_object_list` for NVSHMEM's unique id, then `nvshmem.init` across
every PE. So a rank that raised *before* the broadcast left every other rank waiting in it
forever, and one that raised *after* it left the others aiming one-sided puts at a peer that
had fallen back, with the weights and the routing maps no longer describing the same thing
and nothing raising.

`agree_across_ranks` now reduces each rank's own answer with `MIN`, twice: once after the
purely local preparation and before any collective runs, and once after the collective half,
so a late failure takes the whole group to the host path instead of half of it. **A rank that
failed still reaches the agreement** — that is the point; skipping it is the deadlock. A
group that falls back closes whatever it opened, so a symmetric heap does not survive into
interpreter exit. Four unit tests drive the helper with an injected all-reduce, including the
one that matters: a single dissenting rank takes everyone to the host path.

**Also fixed: NVSHMEM is released at shutdown, and every forward opens the coordinator.**

`EplbState.close()` closes the registered kernels and then the transport, and
`GPUModelRunner.shutdown` calls it before dropping the model. It is idempotent and never
raises: shutdown paths get called twice, a double free there is a segfault, and an exception
on the way out loses results already produced. Both `close` methods had said in their own
docstrings that they were required rather than tidy — leaving the symmetric heap alive into
interpreter exit segfaults every rank *after* every result has printed — and neither had a
caller outside the probes while the device path became the default.

And `note_forward_token_load` is now called on every forward's first MoE layer, `None`
included. It used to be called only when the DP token count was available, while
`_prediction_is_worth_it` returns True in exactly the case where it is not — so such a
forward predicted and recorded, but suppression kept the previous forward's answer and the
forward id did not advance, turning off both of ticket 07's invariants. `None` means unknown
and does not suppress, which keeps the two gates in agreement; their disagreement is what
made every decode forward revert all 48 layers.

**Open, and both now the smaller half of the list.**

*`resolve_moe_block_size_m` passes no `dtype` or `block_shape`*, so on a quantised MoE the
resolved `BLOCK_SIZE_M` can be one the kernel never uses — and it logs "resolved" either
way, which is the guess this function exists to remove. Harmless on BF16 Qwen, which is the
approved scope, and a trap for DSV4's FP8 in ticket 10.

*`replica_transfer.py`'s single staging buffer has the ordering hazard the device path fixed*
with parity-alternated buffers, and `WeightPointers.build` does not bound-check
`replica_row` against the row count. Both are reachable only from probes and tests today,
since `ReplicaTransferEngine` has no production caller.

## Code review, 2026-08-30 evening: three fixed, four open

A review of the branch's unpushed work found seven issues. Three were fixed on the spot and
have tests; four are recorded here because they are real and not yet done.

**Fixed. The compute-to-predictive stream barrier was vacuous.** `barrier.record()` sat
*inside* `with torch.cuda.stream(self.stream)`, so it recorded on the predictive stream and
`self.stream.wait_event(barrier)` waited on its own work. Two things needed that ordering: the
plan row is written on the compute stream and read by the kernels on the predictive one, so
the put could read it before the write landed — the same "`pe` is not a rank" segfault
arriving through *visibility* rather than through lifetime, which the plan-buffer fix did not
address; and `drain_expert` overwrites a replica row the previous forward's MoE may still be
reading. **This project has shipped this exact mistake once before**, on the snapshot copy,
where it made ranks plan from partly-filled buffers and deadlocked the engine with no error.
Now recorded on the compute stream explicitly, and pinned by a test that writes a value on the
compute stream behind a 100 ms delay and asserts the transfer reads it: before the fix it read
0 where 7 had been written.

**Fixed. The NVSHMEM fallback landed on the configuration the validator forbids.** With
`prediction_lookahead_layers=1` and `device_issued_transfer=True` — both defaults now — a
worker that finds no NVSHMEM fell back to the host coordinator, whose launch is a layer later
and whose wait is the next statement: zero window plus a host synchronisation, reached with
only a warning. It now raises, naming the three ways out.

**Fixed. `max_replicas_per_layer > 1` was accepted and ignored on the device path.**
`plan_one_layer_on_device` returns one placement whatever the cap says, while the host path
honours it, so a device run compared against a host run would differ for a reason nothing
logged. Rejected in configuration validation instead.

**Open, and a correctness hazard: nothing orders one layer's put against the previous
layer's drain on the shared staging buffer.** The sequence per layer is put, barrier, drain.
After the barrier releases, rank X proceeds toward layer `L+1`'s put while rank Y is still
running layer `L`'s drain, and the symmetric staging buffer is one expert wide and shared by
every layer. `barrier_all` orders arrival, not X's *next* put against Y's local drain. At EP=2
the target is always the other rank, so every consecutive placed pair is a candidate. The
practical margin is large — a whole layer of compute plus about 1 ms of host work separates
the barrier from the next put, against a 1.1 us drain — which is why 112/112 byte checks and
384 greedy tokens have not caught it. Not ordered is still not safe. The fix is double-buffered
staging, two experts wide, alternating by layer parity; a second barrier after the drain would
also work and costs another collective per layer, which the cost measurements above argue
against. `replica_transfer.py`'s docstring already promises this ordering and its
`_last_drain` is recorded but never used.

**Open: neither NVSHMEM object is ever closed.** `OneSidedExpertTransfer.close()` and
`DeviceExpertTransfer.close()` both document that skipping them segfaults at interpreter exit
— "after the work has finished and every result has been printed, which looks like a transfer
bug and is not one" — and nothing in `vllm/` calls either. `EplbState` has no shutdown hook.
Now that the device path is the default, every worker that builds it leaks the symmetric heap
and the registered kernel library into shutdown. Also: if `DeviceExpertTransfer` construction
raises, `_one_sided` is already initialised and stays that way while the run falls back.

**Open: `static_replica_placement` and the device residency table disagree.** The startup
layout writes the static expert into every layer's replica slot, but `LayerResidency` starts
at `(-1, -1)`, so the first placement on that `(layer, rank)` computes no revert and leaves
the static expert's logical map pointing at a row now holding another expert. Debug knob only
— but its entire purpose is validating routing.

**Open by choice: a leftover pending plan raises on the *next* forward.** The check lives in
`note_forward_token_load`, so a forward that never reaches its target layer is reported one
forward later, against a healthy forward. Ticket 07 asks for an invariant violation rather
than a fallback, so it stays a `RuntimeError`; the message names the forward and the layers,
which is what makes it readable. Worth revisiting if it ever fires in a way that hides a
different cause.

## Ticket 07, 2026-08-30 evening: the window moved, and the cost centre is now named

The launch is at the predicting layer's MoE **tail**, so the overlap window is the target
layer's Attention; `prediction_lookahead_layers` defaults to **1** and reachable layers went
43 -> 44. Which site launches is the coordinator's decision, because the host planner
synchronises inside `plan_and_launch` and cannot move — it keeps the old site and still
rejects a lookahead of 1, where it genuinely has no window.

**At four repeats the stock baseline is finally usable** — 1.6% spread against 56% at two —
and it says:

| arm | median mean TTFT | vs stock |
| --- | --- | --- |
| stock | 257.8 ms | — |
| prediction only | 264.6 ms | **+2.6%** |
| placing, device transfer | 340.3 ms | **+32.0%** |

Prediction is nearly free now (+7.6% at DP=8 before). Placement's +29% is the whole gap, and
a profile says what it is:

    put_expert     p50 1.2 us    total 1.42 ms      the transfer is free
    drain_expert   p50 1.1 us    total 0.53 ms
    barrier        p50 5.0 us    p90 1151 us    max 5898 us    total 57.23 ms

`nvshmemx_barrier_all_on_stream` — the arrival mechanism — costs **29x the transfer it
guards**, 4.40 ms per prefill window, and **87% of it overlaps no compute**. Ticket 05
measured the same barrier at 13.9 us idle. It is not the barrier's own cost: a barrier cannot
complete until the peer arrives, so it turns rank arrival skew into blocking time once per
placed layer, 44 times a forward.

**But the barrier is not the dominant cost, and the CPU side of the same traces says what is.**
Per MoE layer, placement adds **53 kernel launches and 0.95 ms of host time** — `moe_forward`
goes 1.25 ms to 2.20 ms of host time per layer, launches 30.1 to 83.4 — and about 48 of those
53 are tiny elementwise and reduce kernels from `plan_one_layer_on_device`,
`publish_plan_on_device` and the device-side residency and budget bookkeeping. Over 44 placed
layers that is roughly **42 ms of host work per forward**, the right order for placement's
+76 ms. The engine is eager, so every one of them is a host dispatch: 2365 launches per prefill
window against 826, and the placed window carries 35 ms of kernel time in 159 ms.

**This is ticket 03's defect in a new place**, and the same fix applies: fuse the plan, the
publish and the bookkeeping into one kernel. The plan is an argmax over 128 integers and the
publish is a handful of scatter writes. Making the plan device-side removed 5.28 ms per layer
of host *synchronisation* and put 0.95 ms per layer of host *dispatch* back — the right trade
at DP=8 where the sync dominated, a bad one here.

Pairwise arrival is still worth doing; it buys 4.40 ms per forward, not the 42. Neither is
done.

## Ticket 06, 2026-08-30 afternoon: the crash is fixed and the device path runs end to end

**The segfault was the plan's ownership, not the transport.** `plan_and_launch` built the
`[4]` plan as a temporary on the compute stream and launched two kernels that read it on the
predictive stream; the temporary died when the method returned, the allocator handed its
block to the next allocation on the compute stream — the 5th, measured — and the kernels,
still queued behind a 40 us transfer, read whatever the forward had put there. `pe` was then
not a rank. Nothing in `vllm/distributed/eplb/` called `record_stream`. Each layer now has a
plan row the coordinator owns, so there is no allocation on the path and nothing to recycle.

Demonstrated in three steps rather than argued: a pure-torch consumer on a second stream read
the poison; the production classes moved expert 11 where the plan said 3; and a recycled block
naming PE 12345 of 2 killed a rank with **SIGSEGV**, which is the server's crash on demand.
`device_issued_transfer` now **defaults to True**.

**Running end to end, DP=EP=2** (this node came back from its restart with 2 H100s, not 8):
both workers arm, all 43 reachable layers per rank launch a device-issued transfer, the device
counter reports 8 replicas placed, requests are served, no worker dies. With real weights,
greedy output over 4 prompts x 96 tokens is **identical** to a no-replica reference while
placement is active — and the host-issued path moves the first-token logprobs *more* than the
device one (0.34 versus 0.25, against 0.000 for two canonical runs), so the transport
introduces nothing placement does not.

**Both of these criteria are now measured on 8 GPUs — see the close-out section at the top of
this document.** What follows was written when the node had 2.

~~**Two criteria stay open and both need 8 GPUs:**~~ the 24.0%-of-excess reproduction on 43
layers, and the occupancy measurement against 86.8% / 52.9%. No TTFT number here is
comparable to the DP=8 ones; the config validator now says so at startup.

**First TTFT numbers for the device transport, four arms at DP=2** (Korean prompts, 120
requests at concurrency 16, prefill-weighted, each arm twice). The device path's p99 is
**42% and 27% below the host path's** with no overlap between the arms' ranges — which is
what removing a per-layer host synchronisation should look like, since an 11.02 ms
`cudaEventSynchronize` lands in the tail rather than the mean. The mean difference is 2.4%
and sits inside the host arm's own 9.6% repeat spread, so **there is no mean claim in this
run**. Both transports recover the same share of full-prefill excess, 34.8-36.6% against
36.1-36.2%, which is the check that the device path places what the planner chose.

**The stock arm moved 56% between two identical repeats** (406.36 against 260.31 ms mean
TTFT), so nothing in that run can be compared against stock. Previously recorded at 28%.
And placement still costs +38% mean TTFT against prediction alone here — expected, because
two ranks give a critical-path imbalance of 1.17 against eight ranks' 1.885, so there is
much less to win for the same 43 transfers. It says nothing new about the verdict, which is
a DP=8 question.

Also fixed on the way: `BLOCK_SIZE_M` was never actually resolved from the kernel — every
server logged the fallback because EPLB's `expert_weights` are flattened `[rows, numel]` views
and the unit test's fake supplied three-dimensional ones. And `torch.stack` blocks the host
50-101 ms on its **first** call in a process while its kernel loads, which is a measurement
trap rather than a defect, and it made the first version of the lifetime probe pass vacuously.
Details, including a correction of my own first reading of that, in `bench/RESULTS.md`.

Stale claim corrected: this document said `pre-commit` passes including `mypy-3.12`. Two mypy
errors were live on the base commit (`_one_sided` typed as `object`, and two unannotated test
lists); both are fixed now, and `ruff`, `pre-commit` and `mypy` had to be reinstalled after
the pod restart — for the second time, so expect it again.

## ~~Ticket 06, 2026-08-30 night: three pieces done, one crash open~~ (superseded, above)

**Done and committed.** The plan is fully tensorised (bit-identical to the host planner,
`set_sync_debug_mode("error")` clean). Publishing is a device scatter, checked against
`apply_replica_maps` over a 200-plan random sequence. The transfer is two kernels plus a
stream-ordered barrier: **112/112 weight tensors byte-identical over all 56 ordered rank
pairs, p50 36.7-40.0 us against 53.7 us host-issued**. Residency and the transfer budget moved
to the device with them, which they had to — "already resident" is what makes a transfer free.

**Open at the time — answered on 2026-08-30 afternoon; see the section above.** Wired into a
real 8-rank server it initialises on every worker, reaches all 48 layers, and then segfaults
inside NVSHMEM's proxy thread at the startup EPLB rearrange.
Bisected with four server runs to `put_expert` alone: skipping the transfer is healthy, the
barrier alone is healthy, barrier plus drain is healthy, full is not. It does not reproduce in
a standalone script that pipelines 48 transfers per round with NCCL work and no
synchronisation, so the trigger is something the server supplies — most likely the model's own
weight tensors as the put source. **Next step: print the plan and the resolved source address
from inside the kernel for one layer.** Two hypotheses were reasoned through tonight and both
were wrong; the third attempt should measure.

`device_issued_transfer` defaulted to False because of this, and **defaults to True again
since the cause was found**. The host path on the same
commit still comes up healthy and serves, so nothing regressed — but no TTFT number tonight is
a device-path number, and the +31.5% figure still stands as the only measured end-to-end cost.

Debug switches `VLLM_PREDICTIVE_SKIP_DEVICE_TRANSFER` and
`VLLM_PREDICTIVE_DEVICE_TRANSFER_STAGE` are kept; they are what made the bisection possible.
See `bench/RESULTS.md`, 2026-08-30.

## Ticket 06's precondition answered, 2026-08-30: the put can be issued from a kernel

A **host-issued** put cannot be aimed by a device-resident plan — its peer, source pointer and
byte count are host integers consumed at enqueue, which is the same constraint `ncclSend` has
and the reason the 5.28 ms per layer is there. So ticket 06 needs the put issued from inside a
kernel, and that now works on this machine: 8/8 ranks receive a payload aimed by a plan the
host never reads, at **p50 47.6 us against 53.7 us host-issued**. Not a trade — the same time
with the host removed. The masking alternative is ruled out at ~230 us per layer.

The toolchain took seven corrections, six of which fail with a message naming the wrong cause;
they are listed in `bench/RESULTS.md`, 2026-08-30, and anyone touching this should read them
before debugging. Also recorded there: a check of mine that read "1/8 ranks" through a kernel
rewrite while the transfer was correct all along, because `all_reduce` with SUM over a **bool**
tensor saturates back to bool.

Still to do for `06`: make `plan_one_layer_on_device` return without host reads (it currently
uses `int()`, `float()` and `bool()` on device tensors, so ticket 04's "on the device" is
arithmetic-only), build and register the put kernel inside the worker, and publish the
source-local map pair with a device scatter.

## Ticket 05 done, 2026-08-30: the transfer is off the host, and one recorded claim was false

`05` is complete, 7/7, measured on 8x H100 with `bench/probe_replica_transfer.py` driving the
production classes over all 56 ordered rank pairs: **112/112 weight tensors byte-identical**,
one expert's put + barrier + staging copy at **p50 53.7 us**, and a deliberately slowed
transfer exposing 0.456 ms against 0.013 ms unslowed. All 43 layers of transfer come to
2.3 ms against the **5.28 ms per layer** of host synchronisation the device-planned path
removes.

**Read this before touching the ordering.** The ticket said a plain CUDA event was enough for
a consumer to know a peer's put had landed, and said it had been verified. It had not:
`probe_nvshmem.py` put a `dist.barrier()` inside the region it was checking. Removing the
barrier and changing nothing else leaves **51 of 112 tensors wrong**. Arrival now uses
`nvshmemx_barrier_all_on_stream` — 13.9 us, no host, collective, which is safe here precisely
because the plan is rank-identical, and which replaces the pynccl `execute()` that was
already collective per layer.

Also: NVSHMEM's own reference count means dropping the last Python reference to a symmetric
buffer is not freeing it. `free_tensor` before `finalize`, or every rank segfaults after the
results have already printed. And the thing that cost the most time was not any of this — it
was an edit that silently failed to apply while I reasoned about the resulting behaviour as
if it had. Verify the edit landed first.

New modules: `expert_staging` (the flat layout both directions must agree on),
`nvshmem_transfer` (transport plus barrier plus teardown), `replica_transfer` (the three
orderings and the byte cap). 18 new unit tests, driven through a fake fabric with a real
`threading.Barrier` so a receiver cannot read before its sender wrote.

Next is `06`: wiring this into the coordinator in place of pynccl, which is where
`max_concurrent_transfer_bytes` gets its config value and where the host synchronisation
actually leaves the forward.

## Where the work stands, 2026-08-29 evening

The spec was rewritten around **Device-planned placement** and the ticket set replaced; the
old tickets are in `issues/superseded/` with a note on why, because their measurements are
still cited. The new set is `issues/01`-`10`.

**Delivered tonight.** Ticket 03: the predicted-count path is one Triton kernel instead of
about twelve elementwise operations, tested by equality against the retained reference over
200 randomised cases. Ticket 02 in part: `prediction_lookahead_layers=1` is now rejected
(its overlap window is currently zero, which nothing caught before), and
`max_transfers_per_forward` charges for transfers after reconciliation rather than for
planned placements, so it bounds what its name says. 214 predictive tests pass, lint clean.

**And it corrected the projection this ticket set was ordered on.** Launches per source layer
fell 17.2 to 7.1, a 59% cut, but the extra collective waiting fell only 17%. Separating the
two components — fusion moves launches only, where the earlier layer-count test moved
everything together — gives **2.21 ms per layer that scales with launches (30%) and 5.28 ms
that does not (70%)**, and the fixed part is essentially the host synchronisation, since the
snapshot AllGather is under 2% of it. So the prediction arm extrapolates to about **+6.3%**
mean TTFT rather than the +2.7% previously projected here.

**Ticket 06 therefore matters more than ticket 03.** Removing the host synchronisation
attacks 70% of prediction's added cost; the kernel attacked 30% and has delivered its share.
Against the 5.05% ceiling, +6.3% is still above it and only the device-side plan can bring it
below. Details in `bench/RESULTS.md`, 2026-08-29 (ticket 03).

## VERDICT, 2026-08-29 (H100 SXM): negative, and this time about the mechanism

Read this before planning any further work. `bench/RESULTS.md`, 2026-08-29, carries the
evidence.

Ticket 14's missing third arm was measured, and it settles the project:

| arm | mean TTFT | vs stock |
| --- | --- | --- |
| feature fully disabled | 184.70 ms | — |
| prediction on, placement withheld | 198.81 ms | **+7.6%** (upper bound: also records load, 17-row layout) |
| placing, all 43 reachable layers | 242.79 ms | **+31.5%** |

    perfect balance ceiling                     5.05% of a prefill step
    what 24.0% of the excess actually delivered 1.21% of a prefill step
    prediction alone                           +7.6% mean TTFT  = 1.51x the ceiling
    the whole feature                         +31.5% mean TTFT  = 6.2x the ceiling

**Prediction and its infrastructure cost more than perfect expert balance could ever
return**, before a single replica moves. That is the finding, and it is different in kind from the 5090's:
that one was about the interconnect, this one is about the mechanism. Ticket 13 removes
the host synchronisation and with it most of placement's +22.1%, but it does not touch
prediction's +7.6% — that is 43 extra gate matmuls and 43 extra AllGathers per forward,
one per source layer, and a device-side plan leaves every one of them. So the ceiling
sits below the floor on the fastest interconnect NVIDIA ships, and the conclusion
transfers to the Ascend port rather than being a fact about this node.

The mechanics all work, and worked better than before: `max_replicas_per_layer=1` took
coverage from 22 layers to **all 43 reachable ones** and lifted recovered prefill excess
from 15-17% to **24.0%**, about 69% of the offline oracle. The feature is not broken. It
is correctly built and the arithmetic does not close.

**Two refinements, both from measurement after the verdict was written.**

*~~The batched cross-layer snapshot is not the escape~~, and it was the one I named.* The 43
prediction AllGathers were traced on this node: own stream, **9.3 us p50, 0.40 ms per
forward, 4% of the expert-GEMM window they hide in, and a 5% spread between p50 and max**.
The 5090's two-thousand-fold spread on the same 4 KiB payload — the arrival skew that made
43 barriers look expensive — does not exist here. Collapsing them to one saves 0.40 ms
against a 7.6% cost. ~~Closed.~~

**WITHDRAWN 2026-08-30 night. This paragraph is wrong, and the way it is wrong is the
lesson.** It reads one rank's trace — `dp0`, 8 windows — and prices a *barrier* by the
kernel time of a single participant. Re-traced over all 8 ranks and 93 prefill windows, the
same collective's p50 by rank is **748, 628, 595, 545, 131, 588, 9.4, 425 us**. The 9.4 us
rank is `dp6`, and `dp6` is the one that arrives **last**: it never waits, so it is the only
rank that cannot see the cost. Aggregate residency is **18.24 ms per prefill window**, of
which **1.6%** overlaps real compute — the overlap `start_snapshot`'s docstring intends does
not happen — and 90.4% is co-resident with the token collectives. The "no arrival skew here"
conclusion came from the same single-rank view; the skew is exactly what the other seven
ranks are measuring. Batching is back in scope. See the AllGather section near the top.

*The cost is prediction's compute, and `+7.6%` is an upper bound rather than prediction
alone.* Arm `0` also enables EPLB actual-load recording and carries the 17-row layout, so
it is not a clean isolate. What prediction does contribute is launches: about 15 kernels
per source layer — gate GEMM, top-k, and a dozen elementwise ops in
`predict_local_counts` — which is roughly **645 of the ~1910 kernels in a prefill
window**, each also a host-side Python dispatch, in an eager engine.

So the remaining question is narrow and cheap: one profiled `off` arm, diffed against the
existing `budget=0` trace on kernel count and busy time. It separates prediction's compute
from the recording and the layout, and decides whether a fused counting kernel is worth
writing. It would have to bring the cost under about 2% to leave room under the 5.05%
ceiling, and it cannot touch the gate GEMM or the top-k, which are what predicting *means*.

Do not start `04`, `05`, `07`, `08` or the rest of `13` without an explicit decision that
accounts for this.

## H100 SXM, 2026-08-29: the ceiling tripled, and ticket 13 is unblocked

First measurements on the new node. Full numbers in `bench/RESULTS.md`, 2026-08-29.

- **8x H100 80GB HBM3, NV18 between all pairs** — SXM with a full NVSwitch fabric, so
  handoff step 0 is answered on the favourable side.
- **A genuine build of this tree**, `VLLM_USE_PRECOMPILED=1 uv pip install -e .` on
  torch 2.13.0+cu130, first try. Ticket 05's precondition is met here for the first time;
  the 5090 node only ever had symlinked kernels from another commit.
- **`probe_nvshmem.py` passes 5 of 5**, 9.00 MiB put at **33.0 us** against PCIe's 289.
  A plain CUDA event orders the consumer, so no fence and no flag polling. **Ticket 13
  is unblocked.** Note that a 33 us put is not faster than the 40 us NCCL P2P copy — the
  entire value is that the host never reads the plan.
- **One Attention block now hides an expert transfer.** 37.5 us of Attention projections
  at 512 tokens per rank against a 40.6 us transfer, where PCIe hid 18%. The operator's
  intended pipeline shape is feasible on bandwidth; what still blocks it is hook
  position and the host sync (audit finding 1 below).
- **Expert GEMM is 10.76% of attributed prefill GPU time**, against 3.67% on the 5090.
  So the ceiling is roughly **5% of a prefill step**, up about 3x from 1.37%-1.73%.
  Baseline mean TTFT 303.61 ms.
- **Collectives are 77% and went *up* from 66%.** Not bandwidth: at concurrency 8 over
  DP=8 the ranks are unevenly loaded and most of it is waiting for the slowest. dp0
  attributed 24.3 ms against dp7's 136.2 ms. So 10.76% is a **floor** on MoE's share,
  and an evenly loaded point would raise it. A concurrency sweep would pin it.
- **The KV constraint that closed decode is no longer binding at short contexts.**
  643,873 KV tokens per rank puts 5024 concurrent decode tokens in reach at 1024
  context, clearing the one-block bar by 2.45x, where the 5090 reached 464. This does
  not reopen decode on its own — see `RESULTS.md` for the three things that gate it,
  one of which is audit finding 5.

## Code audit, 2026-08-29: where the implementation and the documents disagree

Read this alongside the profiling section below. Nothing here is a new measurement;
it is the code read against `spec.md` and the tickets. Two findings are runtime
claims that have **not** been confirmed by a test and say so. Each finding ends with
the decision it opens, which is the operator's.

### 1. The intended overlap shape is not what runs, and the hooks cannot express it

The design as stated by the operator: predict layer `i + 1`'s load at layer `i`,
complete the transfer and the routing-map update during layer `i + 1`'s **Attention**,
so layer `i + 1`'s MoE executes with the replica already live.

What runs: both `plan_and_launch` and `activate_and_publish` sit at the **head of the
MoE module's forward** (`moe_runner.py:952` and `:957`), as adjacent statements. There
is no hook at a decoder-layer boundary — `qwen3_moe.py` only binds prediction targets.
So:

| lookahead | launch point | wait point | window |
| --- | --- | --- | --- |
| 2 (default) | head of layer `L+1`'s MoE | head of layer `L+2`'s MoE | `L+1`'s whole MoE **plus** `L+2`'s Attention |
| 1 | head of layer `L+1`'s MoE | next statement, same layer | **zero** |

The wait point is where the intended design wants it — immediately after that layer's
Attention. The **launch** point is the problem: it is also a MoE head, so the window is
one full extra MoE layer wider than intended at `lookahead=2`, and empty at
`lookahead=1`. `prediction_lookahead_layers=1` passes config validation and silently
exposes the entire transfer.

Why it was built this way: the planner runs on the host, the snapshot is not complete
until after the predicting layer's dispatch and AllGather, and synchronising on the
host copy inside the predicting layer stalls that layer. Splitting `record_prediction`
from `plan_and_launch` across a layer boundary buys the async copy a layer of compute
to land in, at the cost of one layer of extra distance. That is a deliberate trade and
it is not written down anywhere; the spec's lookahead discussion is about prediction
accuracy versus window size and does not mention it.

To get the intended shape the launch must move to the **tail** of layer `L`'s MoE,
after `finish_snapshot`, leaving layer `L+1`'s Attention as the window. That puts the
host sync immediately after the copy is issued, so it needs either a device-side plan
with a one-sided put (ticket 13, needs NVLink) or an accepted exposed sync.

**Also note the regime shift.** `prediction_lookahead_layers=2`'s justification — one
layer's Attention is 23 to 31 us at up to 128 tokens per rank and 98 us at 512, against
a 175 us transfer — was measured in the decode regime. In prefill at 2k+ tokens per
rank one Attention is several hundred microseconds, so a single Attention block may
already hide a PCIe transfer and `lookahead=1` may be viable *if* the hooks move.
Unmeasured, and it would halve the prediction distance, which ticket 10 measured as the
accurate one anyway.

**Decision open:** keep the two-layer split, or move the launch to the predicting
layer's tail and pay the sync.

### 2. The online planner is per layer, which accounts for 15-17% versus the oracle's 33-35%

`plan_and_launch` calls `plan_replicas(host.unsqueeze(0), ...)`
(`predictive_coordinator.py:233`) — a **single row**, one target layer. `plan_replicas`'
cross-layer ranking loop therefore never has a second candidate to compare. The
spec's section 6 contract, "one list of candidate placements ranked across all layers",
exists only in the offline `bench/imbalance.py:plan_moves`, which is a different
implementation and is where the oracle figure comes from.

This is forced by causality, not an oversight: when layer `L+1` plans for layer `L+2`,
no later layer's prediction exists yet. It cannot be fixed by ranking harder.

What it costs is **allocation order**. Layers are visited in increasing index order and
each takes `min(remaining, max_replicas_per_layer)`, so the budget is spent
first-come-first-served by layer index. At the measured `max_transfers_per_forward=43`
and `max_replicas_per_layer=2`:

    targets 5..25 take 2 each = 42, target 26 takes 1  ->  budget exhausted
    targets 27..47 get nothing, on every forward

**22 covered layers, always the lowest-indexed 22.** That is exactly the "coverage
steady at 22 layers" and "22 layers removes 16.9%" already recorded below — it is
`43 / 2`, not a property of the workload. Offline at the same budget and the same
per-layer cap, global ranking reaches **31.8 layers and 34.9%**, because it gives most
layers one and only the worst layers two.

So `Not yet validated` item 3 is answered: the gap is coverage, not accuracy (ticket 10
already ruled accuracy out at 33.2%) and not only "the online planner's causality" in
the abstract. `spec.md` section 6's claim that `max_replicas_per_layer` is "a per-layer
safety cap and not the allocation target" is **false of the implementation**: per-layer
greedy always fills to the cap.

Three fixes, none needing new mechanism, in increasing cost:

1. `max_replicas_per_layer=1` with budget 43 — 43 reachable layers, one each, full
   coverage.
2. budget 86 with the cap at 2 — full coverage at two each. Spec section 6's own
   measurement puts uniform per-layer spending at 1.245 critical path against global
   ranking's 1.2258 at 86 placements, so uniform is about 1.6% behind the oracle.
3. Allocate the per-layer share from the **previous** forward's per-layer excess, and
   choose experts from the current prediction. This is not the cross-forward residency
   that measured -20.0%: that moved placements, this moves only a budget share, which
   is a far more stable statistic.

Raising the budget does not raise steady-state bytes, because `_spent += len(plan)`
counts placements *before* `reconcile` (see finding 4) — but that only holds once
finding 3 is fixed.

**None of this changes the verdict.** Against a corrected ceiling of 1.37%-1.73% of a
prefill step, going from 15% to 33% of the excess moves the saving from 0.29% to about
0.6% of a step, against a measured cost of +11% TTFT. It closes an open question and it
is the only honest baseline for an H100 comparison. It does not make the feature pay on
PCIe.

**Decision open:** which fix, and whether to verify it offline first (all three are
scoreable from existing dumps with `plan_moves`, no GPU needed).

### 3. CONFIRMED and FIXED: decode and dummy forwards reverted every layer's replicas

Reproduced 2026-08-29 by driving the coordinator with the runner's real decode sequence
— a prefill forward places, then a forward that never calls `record_prediction`:

    prefill published: [(layer 2, 2 placements)]
    decode published:  [(0, 0), (1, 0), (2, 0), (3, 0)]   <- empty set = revert

Fixed by `PlacementCoordinator.note_forward_token_load`, called from the runner at the
forward's first MoE layer with `num_tokens_across_dp_cpu`-derived tokens per expert, so
suppression is decided from a value every rank agrees on *before* anything is recorded.
`_gated` is renamed `_suppressed`, per the glossary rule that reserves "gate" for the MoE
routing gate. The regression test is
`test_a_decode_forward_that_never_predicts_leaves_placement_alone`.

Note why the existing test missed it: `test_a_gated_forward_leaves_what_prefill_placed_alone`
hands the coordinator a *thin prediction* on the decode forward, which the runner cannot
produce — it skips prediction on the same bar, so nothing is recorded at all. The test
encoded a sequence that does not occur.

The code trace, for the record:

`plan_and_launch` returns at `if self._recorded is None` **before** it can ever set
`_gated` (`predictive_coordinator.py:192` versus `:219`). On a decode forward
`_recorded` is never set, because `moe_runner.py:962`'s `_prediction_is_worth_it()`
skips prediction on the same `BLOCK_SIZE_M` bar, so `record_prediction` never runs. The
two gates always agree, and that agreement is what makes `_gated` unreachable.

`_gated` therefore keeps the `False` any prefill forward left it at, so
`activate_and_publish` proceeds, `activate` finds no pending entry, and
`publish(layer, [])` reverts that layer — an empty desired set is the revert trigger by
design. On all 48 layers, on every decode forward, and on every `execute_dummy_batch`
(the placement branch has no `is_dummy` guard).

If it holds, then: the comment "Leave this layer exactly as the last prefill forward
left it" describes behaviour that does not happen; "transfer only the difference" fails
under any mixed traffic, because `_active` is empty again by the next prefill forward,
so all 43 experts and 387 MiB move again; and the **+3.0% decode-with-resident-replicas
measurement was not measuring resident replicas**, which weakens one side of the
cross-forward residency contradiction in `Not yet validated` item 5.

Cheapest confirmation: a unit test that calls `activate_and_publish` with nothing
recorded and asserts no revert. On hardware, count `ncclDevKernel_SendRecv` per prefill
forward — full budget every forward means no reuse.

**Decision open:** fix by setting `_gated` from the token count before the
`_recorded is None` return, or by making the runner skip the placement branch on the
same condition it skips prediction. The second is narrower but leaves two places
holding the same bar.

### 4. Two bandwidth knobs do not do what their names and the spec say

`max_transfers_per_forward` is incremented as `_spent += len(plan)` — **planned
placements**, counted before `reconcile` removes the ones already resident. So it caps
coverage, not transfers, and the documents that call it "the binding constraint" on
bandwidth are describing something else.

`max_concurrent_transfer_bytes` is **never read**. `grep` over `vllm/` finds it only in
`config/parallel.py`. User story 15 and spec section 10 make it the binding constraint
on bytes in flight — the reason a per-forward count was rejected — and it is not
implemented. Nothing bounds expert-weight bytes in flight today.

**Decision open:** implement the byte bound, or strike user story 15 and section 10's
claim and state that the count is the only bound.

### 5. `_MOE_BLOCK_SIZE_M = 128` is hardcoded and is both gates

`eplb_state.py:78`. The same constant is the decode gate (`min_tokens_per_expert`, and
the runner's `prediction_min_tokens_per_expert`) and the planner's `min_tokens` floor.
This node has no tuned `E=128,N=768` config and the H200 one uses 128 only at
M >= 1024, so on other hardware or other M the real block size may differ. Too small
and the decode gate reopens a regime that is settled negative; too large and the
planner rejects placements that would have paid.

**Decision open:** read the selected `BLOCK_SIZE_M` from the kernel config at startup
and carry it in the cost profile's fingerprint, or leave it pinned and assert the
device matches.

### 6. Ticket state misrepresents what is built

`04`, `07` and `08` are marked unstarted, while a planner, a lifecycle
(`reconcile`/revert/re-plan every forward), the weight transfer and the activation all
run end to end. What is genuinely absent from them is the residency and hotness state
machine, any consumer of the cost profile's four cost values, and the byte bound of
finding 4. A reader following `Blocked by` will rebuild work that exists; a reader
following `CLAUDE.md` will read that it works end to end. Both are in the tree.

There is also no terminal ticket for the outcome the evidence currently points at.
`05` is "validation and benchmark" and assumes a result worth validating.

**Decision open:** re-status `04`/`07`/`08` as implemented in reduced form with the
remainder listed, and add a terminal "report the negative result" ticket.

### The gap that has no owner

The feature's total cost is still unmeasured, because both arms of every runner enable
prediction (see below). Prediction alone measured **19% of TPOT**. Its prefill cost is
43 extra gate matmuls and 43 AllGathers per forward, of which 14.8% stays exposed —
plausibly the same order as the entire 1.37%-1.73% ceiling, which would mean no planner
and no interconnect can make the feature net positive here. This is the one measurement
that can settle the project independently of hardware, it costs one extra arm, and no
ticket owned it. Now `14`.

## What today's profiling settled (2026-08-26)

Read this before optimising. Four earlier rounds of design chased the wrong cost.

**The replica transfers are not the bottleneck.** On a two-arm torch profile
(`bench/results/placement-profile/`, `bench/run_placement_profile.sh`), the P2P
transfers are **5.9 ms, 0.32% of GPU time**, 20 `ncclDevKernel_SendRecv` calls on
their own stream. They are genuinely asynchronous and genuinely overlapped. Every
claim that "the transfer cannot be hidden" was wrong, including the argument for
raising the lookahead to widen the overlap window.

**The cost is the host waiting for device data.** Measured inside the real forward
windows: GPU occupancy **86.8% (baseline) versus 52.9% (placed)**, gaps over 0.5 ms
totalling **3.0 ms versus 30.2 ms**, and one `cudaEventSynchronize` of **11.02 ms**.
The CPU runs ahead of the GPU by roughly that much; `plan_and_launch` throws that
run-ahead away every predicted layer, and the engine becomes launch-bound.

**It is structural, not a stream-placement mistake.** `ncclSend`'s peer is a
`ctypes.c_int` consumed when the host enqueues (`pynccl_wrapper.py`), so the host must
know which expert goes to which rank, so it must read a device tensor, so it must wait.
Moving the planner to the GPU does not change this on its own. The escape is
device-initiated one-sided put (NVSHMEM), which needs NVLink this node does not have.
`bench/probe_nvshmem.py` is the 8-GPU validation to run on H200 first.

**The prediction snapshot AllGather is 4 KB and costs 5 us to 11181 us.** p50 493 us.
Two thousand-fold spread on a fixed payload: it is a barrier waiting for the slowest
rank, so what it measures is arrival skew, not communication. Faster interconnect
shrinks the floor, which was never the cost. It runs on its own stream already.

**The baseline arm is not a clean baseline.** Both arms pass
`predictive_expert_replication.enabled = True`; only the budget differs. So every
TTFT and TPOT comparison in this branch answers "what does placement cost on top of
prediction", not "what does the feature cost". Prediction alone measured **19% of
TPOT** at concurrency 1, where the placement gate rejected every forward. A third arm
with the feature fully disabled is needed before quoting a total cost.

**Whole-profile totals are contaminated by idle dummy batches.** In a 12.95 s trace
the annotated forward windows cover only the last **0.54 s**; the rest is
`execute_dummy_batch` keeping DP ranks in lockstep while the benchmark client starts
up. Aggregate operator tables mix that in. Restrict every measurement to the
`execute_context` annotations.

## Four defects fixed today, each of which alone made the feature inert or worse

Any of these would also have wasted a run on new hardware.

1. **Published into buffers routing does not read.** `_apply_eplb_mapping` prefers
   `source_local_physical_map`; the activation path wrote `logical_to_physical_map`.
   Replicas were transferred, described correctly, and never routed to: physical
   per-rank load equalled canonical ownership to 0.00%, and 131 replicas per forward
   removed 0.6% where the oracle removes 35.1%.
2. **The transfer budget was spent per layer.** `max_transfers_per_forward` caps a
   forward; each layer received the full budget and 43 layers spent it 43 times.
3. **Nothing reverted.** `reconcile`, `revert_replicas` and `active_replicas` were
   written for the whole-layout path and sat uncalled. Replicas accumulated to 47-76
   per forward against a budget of 43.
4. **The budget concentrated on the earliest layers.** Each layer planned with the
   whole remaining budget, so 6-7 layers took ~5 replicas each and removed 5.0%.
   Coverage is what drives benefit: 22 layers removes 16.9%.

A fifth, found and fixed earlier the same day: the snapshot copy's event was recorded
on the predictive stream while the copy was enqueued on the current one, so the wait
was vacuous, ranks planned from partly-filled buffers, and the engine deadlocked with
no error.

## Environment (important)

The installed `vllm` at `/usr/local/lib/python3.12/dist-packages/vllm` is **not** this repository: 869 of 2272 Python files differ from the base commit, and it does not contain `vllm/distributed/eplb/predictive.py` at all.

**Which copy loads depends on how the process starts, and the difference is not intuitive:**

| Invocation | Loads | Why |
| --- | --- | --- |
| `python3 -c ...`, `python3 -m vllm...`, `pytest` | this repository | the working directory is `sys.path[0]` |
| `vllm serve ...` | the **installed** build | it is a console script, so `sys.path[0]` is `/usr/local/bin`; the working directory is never on the path |

A `vllm serve` run therefore silently measures stock vLLM. It does not fail: `additional_config` is a free-form dict, so `predictive_expert_replication` is accepted and ignored, and the only hint is a `WARNING: Unknown vLLM environment variable` line. One prediction-accuracy run produced six empty dumps this way.

Every harness script must therefore `export PYTHONPATH="$REPO_ROOT"` **and assert the loaded module resolves inside the tree**, as `run_prediction_accuracy.sh`, `run_boundness_profile.sh`, and `verify_inactive_slots.sh` now do. Prefer `python3 -m vllm.entrypoints.openai.api_server` over `vllm serve`: it puts the tree on the path anyway, and predictive mode *requires* it, because `vllm serve` starts one API server per DP rank and the config round trip keeps `num_redundant_experts` while losing `enable_eplb`, so every rank dies on "num_redundant_experts is set to 8 but EPLB is not enabled".

To run the repo's own code without a full rebuild, the installed compiled extensions are symlinked into the source tree (`*.so` is gitignored, so the repository stays clean):

```bash
for f in /usr/local/lib/python3.12/dist-packages/vllm/*.so; do ln -sfn "$f" "vllm/$(basename $f)"; done
for f in /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/*.so; do ln -sfn "$f" "vllm/vllm_flash_attn/$(basename $f)"; done
```

Packages installed on this machine for this work, all system-wide:

| Package(s) | Why |
| --- | --- |
| `tblib` | `tests/conftest.py` imports it; without it every pytest run fails at collection |
| `ruff`, `pre-commit`, `mypy` | reinstalled 2026-08-24 (`pip install ruff pre-commit mypy`). They had gone missing, and work done while they were absent checked only the 88-character limit by hand; that work has since been linted. |
| `datasets` (pulls `pandas`, `pyarrow`, `xxhash`) | `vllm bench serve` dataset loading |
| `matplotlib`, `seaborn`, `scipy`, `plotly` | the rest of vLLM's `bench` extra (`setup.py`) |

Install the `bench` extra's packages **by name**, never as `vllm[bench]`: resolving
that form can replace the installed build these benchmarks measure.

Symlinks into the source tree so the repo's code can run, all gitignored or listed
in `.git/info/exclude` so the repository stays clean:

- `vllm/*.so` and `vllm/vllm_flash_attn/*.so`
- `vllm/third_party/{deep_gemm,fmha_sm100,tml_fa4,triton_kernels}` and
  `flashmla/flash_mla_interface.py`, absent from this tree; without them warmup
  dies with `No module named vllm.third_party.flashmla.flash_mla_interface`

Verified working: config validation, router kernels (`topk_softmax`), single-GPU
MoE layer tests, and a full 8-GPU predictive-mode server.

**This is adequate for unit work only.** Ticket #05 serving benchmarks and any claim about real overlap or end-to-end output require a genuine build of this tree (`VLLM_USE_PRECOMPILED=1 uv pip install -e .`), because the symlinked kernels come from a different commit.

## Model runner: V1 is the target

`Qwen3MoeForCausalLM` is **not** in `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` (`vllm/config/vllm.py`), so the PoC runs on the **V1** runner (`vllm/v1/worker/gpu_model_runner.py`), not `vllm/v1/worker/gpu/model_runner.py`.

Predictive behavior therefore lives in **`EplbState`**, which both runners funnel through (`add_model`, `step`, `prepare_forward`). Do not add predictive branches to either runner's EPLB glue; extend `EplbState` so both paths stay identical.

## Ticket 03: complete, output equivalence verified on hardware

Source-rank routing is implemented and its last criterion is met. A startup
assertion compares position-weighted checksums of every physical copy of each
logical expert across the EP group and confirmed **48 of 48 layers byte-identical**
with a replica placed. Because the copies are provably identical, routing to either
is arithmetically equivalent and any output difference can only be summation order.
argmax was unchanged and 96 greedy tokens matched.

Two lessons are recorded in `bench/RESULTS.md` rather than only here. A control run
was needed: text equality first failed, and the obvious cross-process
nondeterminism explanation was wrong, since two canonical runs are byte-identical.
And the logprob tolerance used was invented rather than calibrated, so it could not
decide anything; the weight-equality assertion is what settles it, and it now
reports how many pairs it compared so a vacuous pass cannot be mistaken for a real
one.

## Tickets 01 and 02: validated on hardware

Confirmed on the 8-GPU node with this repository's code (`--load-format dummy`,
so no 57 GiB read; details and the exact command in `bench/RESULTS.md`):

- Config validation and cost-profile fingerprint accepted on a real worker.
- Startup normalization ran before readiness in **218 ms**, installing 16
  canonical + 1 inactive rows per rank.
- **Native EPLB placement stayed suppressed**: exactly one rearrangement in the
  whole run, and it is the profile reservation. This was the defect that made the
  original ticket 01 inert, so it is the criterion that most needed proving.
- Forwards complete with prediction active across 43 source layers on 8 ranks,
  0 failed requests at concurrency 12, no invariant violation.

Two environment fixes were needed first, neither caused by the feature: the repo
tree lacks four vendored `vllm/third_party` payloads (symlinked, listed in
`.git/info/exclude`), and the cost-profile fingerprint must pin the served model
*path* rather than the HuggingFace name.

Not established here: overlap timing and output equivalence. Dummy weights cannot
show either; both belong to ticket 05 with real weights.

## Implemented and unit-validated

### Ticket #01

- `additional_config.predictive_expert_replication` parses into `ParallelConfig.predictive_expert_replication_config`, disabled by default.
- Enabling it provisions Expert replication infrastructure (`enable_eplb`, `num_redundant_experts = ep_size * replica_slots_per_rank`, `use_async=False`, `communicator="pynccl"`) and rejects Native EPLB.
- Because the mutation happens after `ParallelConfig` validation, the EPLB preconditions that validator enforces are re-checked in `VllmConfig._validate_predictive_runtime_scope` (CUDA platform, TP/PCP/PP/DP, expert parallelism, DBO, token backend, no speculative decoding).
- The cost profile must be readable JSON with a full `fingerprint` (model, dtype, ep_size, num_logical_experts, device_name) and four positive cost values. `validate_fingerprint` rejects a profile measured on another runtime; the config validator checks model/dtype/EP, leaving `device_name` for the worker.
- `EplbState.add_model` installs the fixed `16 canonical + 1 inactive (-1)` layout, zeroes the inactive rows, republishes the maps, and accumulates `startup_normalization_ms`.
- `EplbState.step` returns before Native EPLB's load window and periodic rearrangement in predictive mode, but still performs the `is_profile` rearrangement so transfer buffers stay reserved for ticket #03.

### Ticket #02

- `FusedMoERouter.select_logical_experts` is a new read-only routing seam: `_compute_routing` only, so it applies no EPLB mapping, records no expert load, and writes no routing-replay state. `BaseRouter` implements it, covering every router in the tree.
- `CrossLayerLoadPredictor` evaluates the target gate and target router on source-local pre-dispatch hidden states, counting **logical** experts (`num_logical_experts`, not the physical `num_experts`) with a sync-free `scatter_add_`.
- Padding is excluded using the EPLB `num_unpadded_tokens_tensors` device scalar, matching what the existing EPLB load-recording kernel already does. `_dummy_run` now publishes a zero count, so a dummy forward predicts zero load and still joins every collective.
- The AllGather starts after dispatch, overlaps the local expert GEMM, and completes before combine. The gather buffer is flat because `ProcessGroupGloo` rejects a pre-shaped `[ep_size, num_logical]` output that NCCL accepts.
- `bind_adjacent_moe_prediction_targets` is the cross-layer gate registry: it binds each MoE to the next, leaves the final MoE unbound, and rejects a topology with a dense-layer gap.

## Spec revision (this session)

`spec.md` was revised after measuring the target node. The changes that alter existing contracts:

- **Prediction lookahead is configurable** (`prediction_lookahead_layers`, default 2). A lookahead of 1 cannot hide a 175 us expert transfer behind a 25-31 us Attention block on PCIe. Plan ownership is now "the layer `lookahead` positions back", and the invariant is "at most one *inbound* pending plan per layer" — several plans are legitimately in flight across the model at once.
- **Leading layers predict nothing** (`prediction_skip_first_layers`, default 3), matching the measured unreliability of early-layer cross-layer prediction.
- **Overlap now means "concurrent AND not the new bottleneck"**, bounded by `max_concurrent_transfer_bytes` (default one expert) rather than a per-forward count.
- **Cost profile keys changed**: `attention_overlap_window_us` -> `attention_window_us`; `transfer_bandwidth_bytes_per_us` -> `usable_transfer_bandwidth_bytes_per_us`, which must be measured under load, not idle.
- **`hot_load_ratio` is only a candidate pre-filter** (default raised 1.0 -> 1.5). The policy's positive-benefit test is the single source of truth for hotness.
- **Enforced minimum residency is derived**, `max(config, ceil(exposed transfer / per-step gain))`.
- **Staging reuses `expert_buffer`'s leading row** (0 new bytes). `EplbState.rearrange` now raises in predictive mode unless `is_profile`, which is the enforced precondition of that reuse.
- **Ticket 00 is new and gates 03/04/05.** Do not implement further replication until its headroom measurement and proceed/stop recommendation exist.
- **Ticket 03 was split into three.** It carried eight criteria spanning routing, P2P, stream events, a map-commit protocol and distributed tests, which does not fit one context window. Now: `03` source-rank routing (verifiable with a statically placed replica, no transfer machinery, and **not** gated by `00` because it is the spec's correctness contract), `07` safe weight transfer into an inactive slot (byte equality and event ordering only, nothing routes to the slot, gated by `00`), and `08` activation end to end (joins the two). `03` and `07` are independent and can run in parallel. Ticket `04` now depends on `08` rather than the old `03`.
- **Ticket 06 is new**: the prediction-accuracy study was split out of ticket 02, because it needs a real multi-GPU model run rather than CPU code, it decides the lookahead default, and ticket 04's cost model consumes it. It is blocked only by 02, so it can run alongside 00. Filename numbers no longer imply order; each ticket's `Blocked by` field does, and `CLAUDE.md` carries the graph.
- **Workload matrix is fixed**: two request shapes (2k/1k prefill-weighted, 1k/2k decode-weighted) over text, code, and math. Decode length forced; prompt length filtered from real samples. Never use vLLM's synthetic `random` dataset for the headroom number — it sweeps the vocabulary and routes almost uniformly, which understates the imbalance being measured. It is a control only.
- **Decode concurrency is capped by KV capacity on 32 GB cards** to roughly 50-83 sequences per rank, putting the MoE-time to weight-read-floor ratio at about 1.4-1.8. A comfortably compute-bound decode regime is not reachable, so that ratio is a reported condition of every result rather than a configurable precondition.
- **No numeric success threshold is fixed in advance.** The goal is lower p99 TPOT at equal RPS (equivalently higher sustainable RPS at fixed SLO). Results are reported as the share of measured imbalance headroom recovered, because an absolute percentage cannot separate a weak policy from an operating point with little to recover.

## Ticket set restructured

`00` was carrying both the feasibility decision and the serving report, and it
gates `07`/`04`/`05`, so implementation was waiting on reporting that has no
bearing on the decision. Split: `00` now asks only whether imbalance converts into
time, and new `09` holds the serving baseline and gates only `05`. `07` was written
for one transfer per layer and now covers up to `max_replicas_per_layer`.

Tickets `01` and `02` are marked done and `03` is 5 of 6; their criteria had been
left unticked, which read as though none of the work existed.

## Settled: the placement policy touches several distinct experts per layer

Measured per-layer load concentration (`bench/RESULTS.md`, fifth session) shows
one expert per layer **cannot** equalize a layer, at any fan-out. Within a layer the
peak rank must shed 39.6% (median) while its hottest expert carries only 33.6%, and
spreading one expert of share `s` over `K` ranks sheds at most `s x K/(K+1) < s`.
The top three experts carry 68.5%, so touching two is enough.

Chunk granularity is not the constraint: 8 source-rank chunks give 256 achievable
split points. The constraint is the spec's "one `ReplicaPlan` per layer".

The user chose fanning one expert out to several ranks, but that choice was made
against a wrongly aggregated version of this data. The measurement now favours
replicating several different experts instead. **Unresolved.**

## ~~Ticket 00 feasibility checkpoint: UNRESOLVED~~ (superseded; answered for both regimes)

An earlier negative verdict was **withdrawn**. The sweep meant to reach a decode
batch large enough to leave the expert-weight-bandwidth-bound regime never got
there: deriving the running batch from `throughput x TPOT` shows 12, 24, 46, 46,
46 sequences per rank, flat from concurrency 512 on, because `--num-prompts 400`
caps requests in flight regardless of `--max-concurrency`. It explored 12 to 46
per rank, and the probe grid puts the transition near 128.

Three caps decide the decode batch and all three must be raised together: the
client's prompt count, `max_num_seqs` (default **128 per rank**, exactly the
threshold to cross), and KV capacity (roughly 700 per rank at a 256-token
context, so not binding). The harness now refuses a point whose concurrency
exceeds the prompt count, and exposes `max_num_seqs`.

What still holds from that run: p99 TPOT is 79.7 to 84.6 ms across every
configuration measured so far, two shapes, two domains, contexts from 256 to 3072
tokens, and batches from 8 to 46 per rank. Measured rank imbalance is 50.8%
median. Whether that imbalance converts into time at a *larger* batch is the open
question.

## ~~Ticket 00 feasibility checkpoint: negative~~ (withdrawn; see above)

Measured with real weights and this repository's code. Full numbers and the exact
commands are in `bench/RESULTS.md`, third session.

Across the whole reachable decode range (16 to 64 sequences per rank; beyond that
requests queue rather than run) **p99 TPOT is flat at 79.7 to 84.6 ms, +6%**, while
measured rank imbalance is **50.8% median**. At 64 sequences per rank MoE accounts
for about 5.2 ms of an 84.6 ms TPOT, so **6.2%**. Perfectly balancing every layer
would save at most 3.1% of TPOT, and one replica per layer captures a fraction of
that. The imbalance is real; it does not convert into time here.

**Tickets 07, 08, 04 and 05 should not proceed on this configuration.** Ticket 03
is unaffected: source-rank routing is the correctness contract, not the optimization.

This is a conclusion about this node, not the policy. Eager execution is
spec-mandated and eager launch overhead is the likely occupant of the other 78 ms;
under CUDA graphs MoE's share would rise, though whether the predictive stream and
event machinery is graph-capturable is itself unsettled. The device also has no
tuned MoE configuration, which *overstates* MoE's share.

Note the terminology: this is the **feasibility checkpoint**, a project decision.
It says nothing about prediction accuracy, which is ticket 06 and still unmeasured.
"gate" is reserved for the MoE routing gate.

## Ticket 00 first results (measured, partial)

Harness lives in `.scratch/predictive-expert-replication/bench/`; **every session's raw numbers are in `bench/RESULTS.md`**, appended per session. Measured on 8xRTX 5090 with `/models/preset/Qwen/Qwen3-30B-A3B/v1.0`, served by the **installed** vLLM 0.27.1, not a build of this tree. Re-measure with the predictive build before ticket 05 compares against it.

**Headroom is real and large.** EPLB balancedness over 3.5k logged steps gives median headroom (1 - avg/max summed per layer) of **44%**, p95 52%, stable across load levels (39-45%). That is far more than sampling noise, so Qwen3-30B-A3B has genuine routing skew. This is the encouraging half.

**But it does not convert into time at reachable concurrency.** Measured MoE-time to expert-weight-read-floor ratio is **1.02 at 8 tokens per rank** (concurrency 64 over DP=8). Reading a rank's 144 MiB of local experts costs 102 us, and MoE at that load costs 102 us: the layer is entirely weight-streaming bound, so a 44% token imbalance costs almost nothing. The probe grid only reaches ratio 1.87 at 128 tokens per rank, and KV capacity caps decode at roughly 85 per rank.

**Lookahead 2 is confirmed as the right call.** One expert is 9.00 MiB and moves in 176-177 us, uniform across all 56 pairs at ~54 GB/s (PCIe Gen5 x16; the RTX 5090 has no NVLink). Lookahead 1 hides only 15% of that. Lookahead 2 gives a 155 us window and hides **87.5%**, leaving 22 us exposed. *(Corrected 2026-08-24: the window is 162.7 us and hides 92%; see `bench/RESULTS.md`.)*

**A larger structural problem surfaced.** Measured p99 TPOT is ~82 ms at concurrency 64, while 48 layers of attention plus MoE account for only ~6.5 ms of it. Eager execution, which the spec mandates, leaves per-layer launch overhead dominating TPOT at the concurrency KV capacity allows. Any MoE-level gain is therefore diluted by roughly an order of magnitude before it reaches TPOT. This bounds the achievable TPOT improvement well below the imbalance figure and needs confirming with a profile.

**Two harness defects invalidated the serving latency numbers** and are now fixed: prompt length was never controlled (actual prompts were 87-167 tokens while filenames claimed 1024/2048, so neither agreed request shape was tested), and two benchmark loops shared one server. `make_prompts.py` now builds fixed-length prompts from real text, and runs take an exclusive lock. The imbalance figure is unaffected, being a per-step routing property rather than a latency measurement.

~~**Verdict so far: do not start ticket 03.**~~ **SUPERSEDED 2026-08-24**: ticket 03 is complete (7 of 7) and was never gated by the feasibility question — source-rank routing is the spec's correctness contract, not an optimization. Everything this paragraph lists as still-to-measure has since been measured; see `bench/RESULTS.md` 2026-08-24. The original text follows for the record. Not because the policy is wrong, but because the tested operating point cannot reward any placement policy. Still to measure before the gate is settled: concurrency high enough to leave the weight-bound regime, per-domain and per-request-shape breakdown, PCIe utilization under load to replace the idle bandwidth currently in the cost profile, and a profile attributing TPOT so the MoE share is known rather than inferred.

Caveats that could change the verdict: the boundness ratio comes from a `bmm` microbenchmark rather than vLLM's fused MoE kernel; and no tuned MoE config exists for this device, so the served kernel is untuned.

## Test status

`tests/distributed/test_predictive_expert_replication.py` — 56 passing:

```bash
python3 -m pytest tests/distributed/test_predictive_expert_replication.py -q
```

Covers configuration and scope rejection, cost-profile schema and fingerprint mismatch, the fixed layout and its logical maps, the gate registry (including gap rejection), and — on GPU — that prediction counts logical experts, does **not** apply the EPLB physical mapping, does **not** record actual load, excludes padding, and predicts zero for a dummy forward. A 4-rank gloo test asserts every rank receives a byte-identical snapshot with correct source-rank provenance.

No regressions: `test_eplb_algo.py`, `test_eplb_utils.py` (21 passing) and `tests/kernels/moe/test_moe_layer.py` (291 passing). `test_moe_layer[False-allgather_reducescatter-1-4-False]` fails **identically on the unmodified base commit** — pre-existing, confirmed by stash-and-rerun.

`pre-commit run` passes on every changed file, including `mypy-3.12` at CI's hook stage.

## Not yet validated

1. **The feature's total cost.** Every measurement compares placement against
   prediction, because both arms enable prediction. A third arm with
   `predictive_expert_replication.enabled = False` is needed.
2. **Where the residual TTFT goes.** Inside the real windows, the transfers (5.9 ms),
   the extra syncs (+10.1 ms) and the exposed prediction AllGather (5.0 ms) account
   for about 21 ms of a 70 ms gap. The rest is the GPU idle gaps, but the two arms'
   windows are not strictly paired (4 versus 6, different first-window names), so that
   comparison is not sound either. A profile with matched window shapes is needed.
3. **Why 15% of prefill excess and not the oracle's 35%.** Believed to be the online
   planner's causality — it cannot see later layers' relative value — but unverified.
   Offline, planning from predicted load reaches 33.2%, so it is not accuracy.
4. **Lookahead 4 and 5.** Ticket 10 measured 1, 2 and 3. If the H200 probe shows a
   9 MiB put near 13 us, raising the lookahead becomes unnecessary and this stays
   unmeasured on purpose.
5. **Cross-forward residency.** Ticket 00 measured -20.0%; this branch measures decode
   with resident replicas at **+3.0%**. The two disagree and the conditions differ.
   The current code keeps replicas resident across gated forwards on the strength of
   the direct measurement.
6. **Under-load interconnect bandwidth.** The cost profile still carries the idle
   figure. Ticket `09` owns it.

## Not implemented

- **Device-side planning and device-initiated transfer.** The design is settled and
  priced; see the profiling section above and `bench/probe_nvshmem.py`. Needs NVLink.
- **Ticket 12**: VMM aliasing to cut the replica slots' 432 MiB per rank to ~45 MiB.
  Deferred, with the alternative (slots on only the worst K layers) ruled out by
  measurement — it is a straight line, ~0.65 points per layer, no knee.
- **Tickets 04, 05, 07, 08, 09, 11.**
- A cross-layer batched prediction snapshot, which is the other way to cut the 43
  collectives per forward. Not designed.

## Required reading order

1. `AGENTS.md`
2. `spec.md`
3. This status document — "Current baseline", then "What today's profiling settled".
   Read them before any optimisation: four rounds of design chased a cost that
   measured 0.32%.
4. The ticket being picked up, from `issues/`
5. `glossary.md`
6. `reference/ascend-prior-art.md` (read-only; the spec overrides it)
