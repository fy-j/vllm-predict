# 19: More than one replica per layer

**What to build:** A layer places `replica_slots_per_rank` replicas instead of one, planned,
transferred and published as a set under a single barrier.

**Why this ticket exists, and why it is not more of the same.** Every other open ticket attacks
what prediction *costs*. This one is the only lever on what placement *returns*, and the two are
independent: the replica count is a placement-side parameter, and prediction runs the same 44
gate GEMMs and the same collectives whatever it is set to. That matters because the two halves
of this feature have opposite signs — placement is negative cost (`06`, and the `13` runs since),
prediction is the whole overhead. Adding replicas buys benefit **without first making prediction
cheaper**, which nothing else on the board does.

**The measurement that motivates it.** Replaying two real 48-layer dumps offline through the
planner, fed each layer's *actual* logical load, so this is placement's ceiling and not what
prediction delivers into it:

| slots                        | 1         | 2     | 3     | 4         | 6     | 8         |
| ---------------------------- | --------- | ----- | ----- | --------- | ----- | --------- |
| critical-path excess removed | **32.2%** | 46.1% | 53.6% | **59.2%** | 66.0% | **70.2%** |

Canonical critical-path imbalance is 1.8925; one slot takes it to 1.6047 and eight to 1.2661.
The curve is concave and has no knee: the second slot is worth 13.9 points, the eighth 4.2.

**Correction to a number this ticket's own code carried.** `plan_layer_replicas_on_device`'s
docstring said four slots remove **68.1%**. Two dumps put four at **59.2%** and **58.9%**;
68-70% is what *eight* slots return. The figure had no script and appears in no results file, so
it is withdrawn rather than reconciled. The 32.1% it paired with is right (32.2% here), which is
how it read as sound.

## What was built

* `plan_layer_replicas_on_device(load, ep_size, min_tokens, cap)` — the tensor oracle,
  `plan_one_layer_on_device` generalised: `cap` greedy steps, each planned against the load the
  previous one left behind.
* `_plan_replicas_kernel` / `plan_replicas_fused`, `_publish_replicas_kernel` /
  `publish_replicas_fused` — one launch each for the whole set, asserted bit-identical to the
  oracles. The cap is read from `residency`'s leading dimension rather than passed, so the three
  tables cannot disagree about how many slots a layer has.
* `put_expert` / `drain_expert` take a slot dimension **in the grid**, not in a loop of launches.
  A launch per slot needs a barrier per slot; at the 0.22 ms `11` measured, four slots over 44
  layers would add 132 barriers and spend more than the replicas return.
* `LayerResidency.empty_slots(device, slots)` — `[slots, 2]`. Kept separate from `empty`, whose
  `[2]` the single-slot oracle indexes directly and would read a slot count as an expert id.
* The coordinator runs **one** path at every cap. There is no `cap == 1` branch, because
  `plan_replicas_fused` at one slot is asserted bit-identical to `plan_and_charge_fused`; a
  second path would be a second thing to keep correct for no measured difference.
* `replica_slots_per_rank` is no longer pinned to 1. It is rejected on the host path, which plans
  one replica per layer and would place only the first — silently, since the activation count,
  the dump and the log line would all still look like a working multi-replica arm.

## Three properties that are load-bearing and none of them obvious

* **At most one replica per logical expert, per layer.** The unconstrained greedy re-picks the
  same expert in 61% of layers, which needs a third copy of it and therefore a wider
  `logical_to_physical_map` and a `logical_replica_count` above 2. Forbidding it costs 2.82% of
  the benefit — the peak rank's second-hottest expert is nearly as good a shed — and leaves both
  structures untouched. This is what keeps the map two wide at any cap.
* **Rows are stored in expert order, and the budget is charged in gain order.** Deliberately
  different. A slot must keep its physical row across forwards, so an unchanged set arriving in a
  different order does not read as a wholesale change and re-transfer; but a layer that cannot
  afford its whole set must keep the replicas worth most. Residency is matched **as a set**, not
  slot by slot, for the same reason.
* **Churn is a first-order cost here.** Four times the transfer rate at unchanged traffic
  measured **+27% mean TTFT** (`14`, incidentally). That is the number that makes the ordering
  rules above worth their complexity, and it is why this ticket cannot be evaluated on excess
  removed alone.

## Open, and the first is the one that decides the ticket

* **Nothing has run on hardware.** Every figure above is an offline replay against dumps. The
  transfer path at `num_slots > 1` has never been exercised on 8 GPUs, and this branch's record
  is that a path which passes its unit tests and has never served is not yet known to work.
* **The budget interacts, and badly.** At four slots a layer charges 4, so 44 layers want 176
  transfers per forward against a `max_transfers_per_forward` default of 43. Either the budget
  rises with the cap or coverage collapses to the first 10 layers. This also makes `15`'s defect
  — a spent budget reverting still-valid resident replicas — far likelier to fire, and `15` is
  unfixed.
* **The memory is real:** one physical expert row per rank per layer per slot (432 MiB per rank
  at one slot on this model), plus one staging expert per slot per window position.
* **Whether prediction's accuracy survives the wider ask.** The offline replay is oracle-fed. The
  planner's own metric is recall@1 because the cap was 1; at cap 4 the quantity that matters is
  recall@4 on the peak rank, which has not been scored.

**Blocked by:** nothing to run the first measurement. `15` should land before any default above 1.

**Measured 2026-09-07, and the answer differs by regime.**

At **1k prompts** two slots deliver exactly the extra balance the offline replay predicted and
lose TTFT anyway:

```text
1k / CONC=16   excess removed   activations/forward   vs stock (3 passes)
cap 1                 26.7%                    5.8    +0.0%  (unreadable)
cap 2                 37.7%                   13.0    +5.6%  (worse, 2/3)
```

The replay predicted 32.2% -> 46.1% for one slot -> two, a ratio of 1.43; hardware gives
26.7% -> 37.7%, **1.41**. So the offline replay's scaling is confirmed — the first time this
branch has checked it against hardware — and the extra balance is real. It is the transfer rate
that kills it: 2.24x, at an expert stability of 20.1%, so almost every transfer is spent on a
replica that is stale before it is used.

**The control is closed.** `86:device:1` (one slot, doubled budget) is indistinguishable from
`43:device:1` at both prompt lengths, so cap 2's loss is the slot count and not the budget it
needed. That control was owed: the first cap-2 run changed slots and budget together, and the
argument that the budget could not matter was an argument, not a measurement.

**Cap 1 at 8k prompts is net positive** — `-2.50%` mean TTFT and `+2.80%` throughput against
stock, 6/6 paired — which is the first positive end-to-end result on this branch. See
`RESULTS.md` 2026-09-07 and the expert-stability entry in the glossary.

**Cap 2 at 8k is unmeasured, and it is the obvious next run.** Churn, the mechanism that
defeated it, is down 5.5x (4.74 -> 0.86 transfers per 1000 tokens). It may well be positive
there; that is a run, not an inference, and this ticket should not be closed on the 1k result.

**Status: DONE (2026-09-08). Two slots pay at long prompts and lose at short ones.** 19 unit
tests over the planner, the two kernels and their oracles; 261 across the feature's suites.

```text
                       cap 1        cap 2      both paired
1k  / CONC=16          +0.0%        +5.6%      cap 2 worse, 2/3
8k  / CONC=16          -2.51%       -3.17%     6/6 and 6/6
16k / CONC=8           -5.38%       -6.17%     6/6 and 6/6
```

The sign turns on churn, not on balance: cap 2 delivers the extra balance at every length
(excess removed 26.7% -> 37.7% at 1k, a ratio of 1.41 against the offline replay's predicted
1.43 — **the replay's scaling is confirmed on hardware**, which had never been checked), but it
also doubles the transfer rate, and at 20% expert stability that costs more than the balance
returns. At 93% it does not.

**The default stays 1.** Raising it is a decision about the deployment's prompt length, not a
strict improvement, and this branch has no measurement of four or eight slots.

**Past two slots the budget has to move first.** Measured activation rates are 2.7-3.2 per
forward at cap 1 and 17 at cap 2, against caps of 43 and 86 — an order of magnitude of slack at
one slot and a factor of five at two. At four slots, 44 layers x 4 wants **176**, and the budget
would stop bounding churn and start bounding coverage, which is the failure `15` exists to
prevent. Note also that 43 and 86 are stale constants: 44 layers are reachable at the current
lookahead of 1, and 43 was the count under the old lookahead of 2. The headline configurations
were re-measured at 48 and 96 on 2026-09-08.
