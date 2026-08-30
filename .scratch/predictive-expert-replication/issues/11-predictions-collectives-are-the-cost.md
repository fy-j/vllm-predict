# 11: Prediction's per-layer collective is the cost, and batching is back in scope

**What to build:** Prediction cheap enough to leave room under the ceiling. Ticket `06` proved
placement is no longer where the cost is — placing is 6.95 ms *faster* than prediction only, so
placement's own contribution is negative cost — while prediction alone costs **+13.5%** mean
TTFT against a stock server at DP=8, which is 2.6x the entire perfect-balance ceiling. This
ticket is the one that decides whether the arithmetic can close, and `08`'s verdict should not
be written before it reports.

**The reason this ticket exists is a withdrawn measurement.** `spec.md` listed batching the
per-layer snapshot AllGathers as out of scope, on "9.3 us each, 0.40 ms per forward, 4% of the
window they hide in, no arrival skew here to remove". That figure was **one rank's** kernel
time, and the rank was `dp6` — the one that arrives **last**, so the only participant that never
waits. Over 8 ranks the same collective's p50 is 748, 628, 595, 545, 131, 588, **9.4**, 425 us.
A barrier's cost is the coupling it imposes on the group, not its duration on the pace-setter.
The two quantities now have separate names in `glossary.md`: **Collective residency** and
**Barrier coupling cost**.

**Blocked by:** None. `06` is done and supplies the measurements below.

**Status:** **ANSWERED 2026-08-30 night. This is a diagnosis ticket and its three measurements
are done.** The answer is "both, about half each": of the 23.21 ms prediction adds, **11.04 ms
(47.6%) is the 44 per-layer barriers and 12.17 ms is its launches and compute**. Neither
candidate mechanism accounts for the cost alone.

**The fixes this diagnosis implies are now their own tickets**, because they are three separate
context windows of work and leaving them here made one ticket mean both "the diagnosis is done"
and "prediction is optimised":

- **12** — one layer predicts a group of targets, landing at K=1 where it is provably inert.
- **13** — K=4 and the measurement of what it buys.
- **14** — the snapshot reduction leaves NCCL, if 13 leaves a barrier cost worth removing.

The two criteria below that describe those shapes are kept for their reasoning and are **not**
this ticket's remaining work; they are stated as acceptance criteria in 12, 13 and 14.

## The isolating measurement, 2026-08-30 night: half barriers, half launches

Knee operating point, `ko`, 512 requests, CONC=16, three interleaved passes per arm. The
control reproduced the earlier run to within 0.2 points (+13.68% against +13.53% an hour
earlier, on a stock arm whose own spread is 1.5%), so the ruler is trustworthy.

| arm | r1 | r2 | r3 | median | vs stock |
| --- | --- | --- | --- | --- | --- |
| stock | 168.53 | 170.99 | 169.68 | 169.68 ms | — |
| prediction, AllGather kept | 192.89 | 190.02 | 199.82 | 192.89 ms | **+13.68%** |
| prediction, **AllGather removed** | 181.85 | 180.54 | 184.55 | 181.85 ms | **+7.17%** |

    prediction adds                        +23.21 ms
    removing the 44 AllGathers gives back  -11.04 ms   = 47.6% of it
    what is left                           +12.17 ms   = +7.17%

**So the barrier coupling is real and it is worth 11.04 ms, which is 0.25 ms per barrier — not
the 0.46 ms the two-point window arithmetic suggested.** That earlier figure divided a whole
window by its collective count and so charged the barriers for waiting the window already had.
The remaining 12.17 ms is prediction's launches and its compute, which no batching scheme
removes on its own.

### What that does to the batching estimate

The K=4 "one source, K targets" shape, re-costed against the measurement rather than the
hypothesis:

    barriers 44 -> 11, at the measured 0.25 ms          -8.3 ms
    launches 176 -> 77 (top-k stays at 44), pro rata    -6.8 ms
    prediction 23.21 ms ->                             ~8.1 ms  = about +4.8%

With placement returning about 4%, that puts the feature near **break-even, not positive**.
Reaching positive needs the launch half too, and the two candidates for it are this ticket's
device-side reduction and **ticket `09`, CUDA graph capture — which the launch half being 52%
of the cost promotes from "exploratory, off the mainline" to the largest single lever in the
project.** It is blocked by the spec's eager-execution mandate, not by evidence.

### The trade, now both sides measured rather than one

    cost of K=4 grouping   about 1.2 points of the 34.9% excess removed, interpolated
                           from the measured 1..4 curve, so about 0.14 points of the
                           ~4% mean TTFT that placement returns
    benefit                prediction +13.68% -> about +4.8%, so about 9 points

**Roughly 70 to 1 in favour**, which is why the grouping is worth building even though it
cannot by itself make the feature positive.

## What is measured, before any change

At DP=8, `ko`, the knee (CONC=16), over 8 ranks and 93 prefill windows:

| | stock | prediction only |
| --- | --- | --- |
| prefill window wall-clock | 88.9 ms | 106.9 ms |
| collectives per window | 192 | 192 + 44 = 236 |
| **ms per collective** | **0.463** | **0.453** |
| compute-only occupancy (nccl excluded) | 20.4-32.6% | 19.9-25.6% |
| absolute compute per window | 25.8 ms | 23.5 ms |
| snapshot AllGather residency | — | 18.24 ms/window, p50 545 us |
| of which overlaps real compute | — | **1.6%** |
| of which overlaps token collectives | — | 90.4% |
| of which overlaps nothing | — | 9.6% (1.74 ms/window) |

Two things follow and neither was known before. **Prediction adds no net compute** — the window
grows 18 ms and all of it is waiting. And **the intended overlap does not happen**:
`start_snapshot`'s docstring says the collective hides behind the local expert GEMM, and 1.6% of
it does, because a 545 us barrier cannot hide in a ~190 us GEMM.

~~The working hypothesis is that the window's length tracks the **number** of collectives rather
than their payload — 0.463 against 0.453 ms each, for payloads differing by three orders of
magnitude.~~ **Disproved by the probe profile below**: at the same 192 collectives, stock is
88.9 ms and the probe arm is 97.2 ms. Dividing a whole window by its collective count charges
the barriers for waiting the window already contained. The `ms per collective` row above is kept
because it is what motivated the isolating measurement, not because it means anything.

## Criteria

- [x] **The isolating measurement, first, because it decides whether the rest is worth
      building.** Done: 47.6% barriers, 52.4% launches and compute. Details above. `VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER` holds prediction's compute fixed —
      gate GEMM, top-k, counting kernel all still run — and removes only the collective. Three
      arms at the knee, three interleaved passes, against a baseline whose own spread is 0.3%.
      The probe must refuse to run with placement armed: its snapshot is each rank's
      own counts, so a plan from it is per-rank, and a per-rank plan pairs a put with a peer
      expecting nothing. Built with tests; rejection included.

- [x] **A profile confirming the probe removed what it claims to.** Done, and it completes the
      decomposition. Per prefill window, median over 8 ranks:

      | arm | collectives | of which snapshot AllGather | window |
      | --- | --- | --- | --- |
      | prediction, kept | 236 | 44 | 106.9 ms |
      | prediction, removed | **192** | **0** | **97.2 ms** |
      | stock | 192 | 0 | 88.9 ms |

      The probe removed exactly the 44 and left the 192 token collectives untouched. So:
      **stock 88.9 ms, plus 8.3 ms for prediction's launches and compute at unchanged
      collective count, plus 9.7 ms for the 44 barriers = 0.22 ms each.** The window split
      (54% barriers) and the TTFT split (47.6%) agree, from two independent measurements.

      It also **disproves my own "0.46 ms per collective, payload-independent" hypothesis**:
      at the same 192 collectives, stock is 88.9 ms and the probe is 97.2 ms. The window is not
      barrier-count-bound; barriers are one additive term of three.

- [x] **Lookahead 4 measured rather than extrapolated.** Done on `ko` at DP=8, all four
      distances in one session so the curve is self-consistent, scored by replaying
      predicted-chosen placements against actual load (`prefill_accuracy.py`, prefill regime
      only):

      | L | recall@1 | peak-rank recall@1 | peak hit | count error | **excess removed** | oracle | made worse |
      | --- | --- | --- | --- | --- | --- | --- | --- |
      | 1 | 0.9015 | 0.9621 | 0.9242 | 0.0581 | **34.9%** | 36.4% | 0 |
      | 2 | 0.8682 | 0.9457 | 0.8992 | 0.0856 | **34.4%** | 37.0% | 0 |
      | 3 | 0.8730 | 0.9524 | 0.8889 | 0.1089 | **33.5%** | 37.7% | 0 |
      | 4 | 0.8862 | 0.9512 | 0.7561 | 0.1333 | **32.1%** | 37.6% | 0 |

      **Lookahead 4 costs 2.8 points of the 34.9%, 8% relative**, while `count_error` more than
      doubles. Ticket 10's pattern reproduces on a second domain and extends to 4: the pooled
      metrics degrade steadily and the delivered benefit barely does, because peak-rank
      recall@1 sits at ~0.95 at every distance and the planner only picks from there. No
      forward was made worse at any distance.

      Two limits. The excess column rests on **3 full-prefill forwards per lookahead** — the
      same order as ticket 10's 4 — though the trend is monotone and the baseline is stable at
      1.859 to 1.873. And `accuracy_report.py`'s pooled recall@1 for the same runs reads 0.697
      to 0.522, which is a *different restriction*, not a contradiction: pooling includes
      forwards below the prefill bar, where no placement happens.

      Per target layer, leading layers below tolerance: L=1 `[4]`, L=2 none, L=3 `[6, 7]`,
      L=4 `[7]`. So `prediction_skip_first_layers` needs re-deciding at higher distance, and it
      costs a little coverage at the front of the model.

**Batching — now ticket 12 (mechanism) and 13 (K=4).** Two shapes, and the second is
      better:

      *Uniform lookahead K.* Sources `L..L+K-1` predict targets `L+K..L+2K-1`, one AllGather at
      the last source's tail. Every prediction is at distance K.

      *One source, K targets (preferred).* Layer `L` evaluates the gates of `L+1..L+K` on its
      own hidden states — same input, so the K gate GEMMs concatenate into **one** GEMM, which
      the runner already does for a different purpose in `_maybe_fuse_gate_weights` — and one
      AllGather carries all K count vectors. Distances are 1..K, averaging `(1+K)/2`. It
      dominates the uniform shape: same worst-case distance, better average, one target still
      at distance 1, and only every K-th layer predicts at all, so prediction's compute and its
      launches fall by K as well. Coverage does not drop: groups starting at
      `prediction_skip_first_layers` still cover every target from there to the last layer.

      At K=4 that is 44 barriers -> 11. If the per-collective figure holds, ~33 x 0.46 = 15 ms
      of the 22.66 ms prediction adds, against an accuracy cost of about 1 point of the 26-27%
      excess removed.

**A device-side reduction instead of an NCCL collective — now ticket 14.** The
      planner needs only the **sum over source ranks** of the `[128]` counts —
      `record_prediction` already does `predicted.sum(dim=0)`, so the per-source-rank breakdown
      the AllGather delivers is never read — which is 512 bytes. NVSHMEM is already initialised
      for the weight transfer. A device-side put-and-flag reduction needs no group-wide
      barrier: a rank reads its peers' values when they arrive, and a per-forward sequence
      number makes "the right forward's values" checkable without aligning the group. Sequenced
      last because it is a new kernel with a new correctness argument, where batching is pure
      scheduling.

**Reporting against the ceiling — a criterion of 13, and of 08's stop gate.** The ceiling is 5.26% of
      a prefill window, measured today (11.16% expert-GEMM share x 47.09% recoverable). The
      target is prediction under about 2% so that placement's ~4% return leaves the feature net
      positive. State plainly if it is not reached; `08`'s stop gate applies to this ticket's
      output.

## What this ticket must not do

Reopen cross-forward residency by accident. Batching all 44 layers into one collective would
put the snapshot at the end of the forward and make every plan apply to the *next* one, which
is cross-forward placement — measured at -20.0% by ticket `00` and +3.0% on this branch, two
numbers that disagree and neither of which has been re-derived. K must stay small enough that
plans are produced and consumed within one forward, which is also `06`'s invariant and is
enforced by a `RuntimeError`.
