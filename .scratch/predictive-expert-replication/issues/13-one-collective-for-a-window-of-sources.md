# 13: One collective for a window of source layers

**What to build:** The snapshot collective batched across a **window of consecutive source
layers**, each still predicting its own target from its own hidden states. Ticket 11 measured
the 44 per-layer collectives at 0.22 ms of barrier coupling each, 11.04 ms of the 23.21 ms
prediction adds to mean TTFT. A window of `K` takes them to `ceil(44 / K)`.

Each source in the window writes one row of a shared count buffer and the window's **last**
source issues a single AllGather. Every target keeps a prediction made from its own source's
hidden states, at a uniform distance of `prediction_lookahead_layers`.

**The distance must cover the window** (`lookahead >= group`): the collective is issued at
the window's last source, so a shorter distance would have the window's first target already
behind it. Enforced in the binder and, so a bad configuration stops at startup rather than at
model build, in configuration validation.

**Blocked by:** 12 — One layer predicts a group of targets. Its mechanism is what this
replaces, and its K=1 equality is what makes the replacement checkable.

**Status:** implemented and measured 2026-08-31, 563 tests, lint and CI mypy clean. **It
works: 20.8% of full-prefill excess removed at a window of 4 against 26.8% at a window of 1**,
three repeats, 0.1-point spread, measured in the same interleaved passes. That is 78% of the
benefit kept where the refuted first design kept 14%.

**The default is still `prediction_target_group = 1`.** What is left before moving it is in the
open criteria: the 6.0-point excess loss is more than twice what the distance curve predicted
and only half of it is explained.

## The first design, built and refuted

The first attempt had **one** source layer predict `K` targets, evaluating `K` target gates on
its own hidden states — one concatenated GEMM, one counting kernel, one collective, and only
every `K`-th layer predicting, so it attacked both halves of prediction's cost. It was built,
it served, and it was refuted on hardware:

| | full prefill excess removed |
| --- | --- |
| group 1, before the refactor | 26.5 / 26.7 / 27.0% |
| group 1, after the refactor | **26.9 / 26.8%** |
| group 4, first design | **3.7 / 4.6%** |

`connected: true`, 0 failed requests, 352 launches, coverage intact at 44 of 44 layers. The
path was live and placing; the *benefit* was gone.

**What was ruled out, in order.** The refactor — group 1 after it reproduces group 1 before it
to within 0.1 points, so ticket 12's inertness holds on benefit and not only on launch counts.
The row-to-target mapping — a differential test drives the real runner loop and asserts each
target is planned from its own snapshot row. The collective's layout — a 4-rank gloo test
proves rank-major, target-minor, byte-identical across ranks. The binder — 11 sources, 44
distinct targets, no duplicates. The fused gate projection — `block_norms` match the per-gate
logits element for element (5853.137 / 5284.649 / 4802.325 / 4841.524). Instantaneous
accuracy — distance 4 delivers 32.1% of excess in the offline replay.

**What it actually was, and it is not a defect.** Four different logit vectors, but their
top-k selections are nearly the same:

    within a group, the four predicted distributions differ by an L1 of 16, 24, 36
    across groups, two sources' predictions differ by an L1 of 5412 to 7660
    (out of 7896 assignments)

while the four target layers' *actual* loads are completely different. Applying four layers'
gates to one layer's hidden states ranks experts almost identically — the magnitudes differ by
20% but top-k only uses the ranking. So three of every four targets were planned from a
distribution that was not theirs, one per group was right, and 13 of 44 layers improved while
4 got worse. The aggregated imbalance moved *more* than at group 1 (-0.0155 against -0.0134)
while the critical path moved a seventh as much: load was being shed, just not off each
layer's own peak.

**The methodological finding, which is the transferable part.** Ticket 11's accuracy curve
measures *distance* with one source per target, and cannot predict this design: what the first
design needed was "the same source distinguishes different targets", and that quantity is
**0.2%**. It was never measured because nothing had asked for it. An offline accuracy curve is
not a substitute for the quantity a design actually depends on.

## Criteria

- [x] Every source predicts its own target, at a uniform distance. Asserted for group 1, 2 and
      4: one target per source, every reachable target covered exactly once, and every layer in
      the source range still a source — which is what keeps predictions distinguishable.
- [x] One collective per window, issued by the window's last source only. Asserted through
      `issues_at`, with the window objects' identity checked so two windows cannot share one.
- [x] A short final window is allowed. 41 sources in windows of 4 give ten full windows and one
      of a single source; rejecting the remainder would cost coverage, and unlike the first
      design the remainder is a *window* rather than an unpredicted target layer.
- [x] `lookahead >= group`, rejected in the binder and at startup, with the reason stated.
- [x] The cost probe follows the collective onto the window, so
      `VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER` still isolates barriers from launches.
- [x] **The excess figure on hardware, against a `43@g1` arm interleaved in the same passes.**

      | arm | full prefill | critical path |
      | --- | --- | --- |
      | `43@g1` | 26.8 / 26.7 / 26.8% | 1.8913 -> 1.6521 |
      | `43@g4` | **20.8 / 20.7 / 20.8%** | 1.8913 -> 1.7059 |

      Three repeats, a 0.1-point spread, `connected: true` and 0 failed on every arm, 33
      full-prefill forwards per arm on both sides. **The design holds**: 78% of the benefit
      survives where the first design kept 14%, and the difference is exactly that predictions
      now come from different source layers.

      The stated pass line was "within about a point", and 6.0 points is not that. It is
      recorded as a pass on the *design* and an open question on the *cost*, below.
- [ ] Collectives per prefill window measured at 44 -> 11, with the 192 token collectives
      untouched, and the window wall-clock beside it.
- [ ] Three-arm TTFT at the knee, reported against the disabled arm, with the baseline's own
      spread stated. Prediction's remaining cost decomposed the way ticket 11 did it, so the
      barrier half and the launch half stay separable.
- [ ] **Why the excess loss is 6.0 points where the distance curve predicts 2.8.** Partly
      coverage: a lookahead of 4 makes the reachable targets 41 rather than 44, and the smoke
      confirms it — 328 launches, not 352. Three layers of 44 is about 2 points at the measured
      ~0.65 points per layer, which leaves roughly 1 point unexplained. Measure it rather than
      reason about it: the accuracy runner can score a window of 4 directly now that the group
      is plumbed through it.
- [ ] The result against the ceiling and ticket 08's stop gate. Prediction under about 2% is
      what leaves the feature net positive; say plainly if it is not reached.

## What this ticket does not fix

The launch half. The first design cut predicting layers from 44 to 11 and this one does not —
every layer still predicts, so only ticket 11's barrier half (47.6%) is in reach. The launch
half is 52% and belongs to ticket 09.

### The staging hazard this design would have made likely, and its removal

`CURRENT-STATUS.md` carried an open correctness hazard: `barrier_all` orders arrival, not one
rank's next put against another rank's drain, so two layers sharing a staging buffer can
collide. Two buffers alternating by layer parity were enough only because consecutive
same-buffer layers were a whole layer of compute plus about 1 ms of host work apart, against a
1.1 us drain. **A window removes exactly that margin**: its transfers are all launched at its
last source, microseconds apart with no compute between them, and at a window of 4 the first
and third targets share parity.

Checked first, with the runtime replica-weight verification: real weights, window of 4, 60
forwards, 32 replicas placed — `verified 4 dynamically placed replica (layer, expert) pairs
hold their canonical weights` and **no checksum mismatch on any forward**. The check runs every
forward and raises on the one that finds a mismatch, so the absence covers all 60, not just the
one that logged.

**That is "not observed", not "cannot happen", and a race is exactly the thing a short run
cannot clear.** So it is removed rather than relied on: the staging buffer is now indexed by
`target % staging_buffers` with one buffer per window position, so no two transfers launched
together can share one, while buffers a whole window apart are still separated by that
window's compute. Two remains the floor, which is the configuration that shipped. The cost is
`group x 9.00 MiB` of symmetric memory per rank — 36 MiB at a window of 4, against the 432 MiB
the replica slots already hold. The alternative, a second barrier after each drain, costs a
collective per layer, which is what this ticket exists to remove.
