# 15: A spent budget must not revert a resident replica

**What to build:** A forward that has run out of transfer budget leaves the replicas it already
has in place, instead of clearing them.

Found by code review, 2026-08-30. `publish_ptr[0] = max(keep, affordable)` in
`fused_placement.py`: when a layer holds resident `A@T`, this forward's plan names a different
`B@T'`, and the budget is already spent, then `keep=0`, `needs=1`, `affordable=0`, so the publish
takes the **revert** branch and clears the layer. The layer ends holding no replica even though
`A@T`'s weights are still in its row and keeping them costs nothing. The host
`PlacementCoordinator` has the same shape at `remaining <= 0`.

**Why it matters at the default configuration.** `max_transfers_per_forward` defaults to **4**
over 44 reachable layers. Once traffic shifts so most layers' best expert changes, one forward
re-places 4 layers and **reverts the other 40**, then needs about ten forwards to ratchet
coverage back. That directly contradicts the property the same kernel's docstring claims —
"charged for what moves, so in a steady state the honest charge is zero and the budget bounds
churn rather than coverage".

It did **not** affect any measurement recorded so far: every run used budget 43 against 44
reachable layers, where the budget is not exhausted in steady state, and the excess figure was
stable to 0.5 points across three repeats.

**Blocked by:** None (can start immediately). It is independent of the grouping work.

**Status: DONE, 5 of 5 (2026-09-08).** Each behavioural test was checked to **fail against the
unfixed code**, which is how the first draft of the coverage test was caught proving nothing.

The last criterion was the hardware run, and it closed at 8k rather than at 1k for a reason the
1k attempt could not have supplied: at 20% expert stability there is no steady state for the
coverage ratchet to settle into, so a run there tests nothing about a fix whose whole subject is
holding a *still-valid* replica. At 8k (93% stability, 3 interleaved passes) the **default**
`max_transfers_per_forward=4` gives **-1.76% +/- 1.88 mean TTFT, 3/3 faster**, against budget
43's -2.13% +/- 0.55. The default budget is now within 0.4 points of the unconstrained one,
which is the outcome this fix predicted: coverage ratchets up and stays.

- [x] With the budget spent, a layer holding a still-valid replica keeps it. The publish row now
      carries the **resident** placement -- `hold = found & ~keep & ~affordable & resident`
      selects `(res_expert, res_target)` into the publish table while the transfer table still
      names the refused one and is charged zero. `moved_x2` goes to 0 on that row: publish never
      reads it, and a row claiming a shed that will not happen is a lie waiting for a debugger.
      Deliberately narrow: **`found == 0` still reverts.** That is the planner saying no
      placement helps this layer, which is a judgement about load rather than a budget accident,
      and it cannot be self-fulfilling because the snapshot counts *logical* experts and so is
      unaffected by what is resident.
- [x] Stated and tested. **Across layers the budget is first-come-first-served by layer index**,
      because layers are visited in increasing order and each plans from its own row -- no layer
      can see a later one's prediction to rank against, so this is a consequence and not a
      choice. **Within a layer it is gain order**, which is a choice: a layer that cannot afford
      every slot keeps the replicas worth most.
      `test_the_earliest_layer_wins_when_the_budget_runs_out_mid_forward` pins the first against
      a load where the later layers are hotter, so a gain-ranked scheme would fail it.
- [x] `PlacementCoordinator._keep_what_is_resident` does the same at `remaining <= 0`, and
      **registers the held plan in `_pending`** rather than only returning it: `activate_and_publish`
      publishes whatever `activate` pops, so an unregistered layer publishes the empty set --
      which is the revert this is meant to prevent. The tensor oracle inside
      `test_fused_placement.py` learned the same rule; without that the three fused-versus-tensor
      equality tests fail, which is the check working.
- [x] `test_a_traffic_shift_does_not_collapse_coverage_to_the_budget`, and the first version of
      it was **wrong in a way worth recording**: it held each layer's hot expert fixed, so every
      layer took the `keep` branch, nothing was ever charged, no budget pressure arose, and it
      passed against the unfixed kernel. The sawtooth needs the plan to *differ* from what is
      resident while the budget is gone -- a domain switch, not a steady state. Rewritten to warm
      coverage up under one traffic pattern and then move every layer's best expert at once.
      Both fused-versus-tensor equality tests still pass, so the affordable case is unchanged.
- [ ] **Open.** Measured on hardware at the default `max_transfers_per_forward=4`: active
      replicas per forward over a run long enough to ratchet, showing coverage climbing and
      holding rather than sawtoothing. Nothing here has run on 8 GPUs.

**Why this matters more since `19`.** Multi-replica placement multiplies the pressure that makes
this defect fire: at four slots a layer charges 4, so 44 layers want 176 transfers per forward
against a budget of 43, and exhaustion goes from "occasional, on a traffic shift" to "every
forward". The churn a revert-and-re-transfer cycle creates is not second-order either -- four
times the transfer rate at unchanged traffic measured **+27% mean TTFT** (`14`, incidentally). So
`19` above a cap of 1 would have measured this defect rather than the replicas.
