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

**Status:** ready-for-agent

- [ ] With the budget spent, a layer holding a still-valid replica keeps it: the maps still
      describe it, the residency still names it, and no transfer is charged. The publish row has
      to be able to carry the *resident* placement rather than only the newly planned one, so
      this is a change to what the plan says and not only to a flag.
- [ ] The layer that could not afford its new placement is left in a state the next forward can
      plan from, and the one after it in the same forward is not silently favoured — state which
      layer wins when the budget runs out mid-forward, and test it.
- [ ] The host coordinator behaves the same way at `remaining <= 0`, since it is retained as the
      oracle the fused kernels are tested against and an oracle that disagrees is worse than
      none.
- [ ] A test at a small budget with more hot layers than budget: coverage reaches the budget on
      the first forward and **does not fall** on the next, where today it collapses. The
      existing fused-versus-tensor equality tests keep passing, which is what says the fix did
      not change the affordable case.
- [ ] Measured on hardware at the default `max_transfers_per_forward=4`: active replicas per
      forward over a run long enough to ratchet, showing coverage climbing and holding rather
      than sawtoothing.
