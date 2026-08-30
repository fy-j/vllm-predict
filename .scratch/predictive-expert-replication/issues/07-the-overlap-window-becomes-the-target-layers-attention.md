# 07: The overlap window becomes the target layer's Attention

**What to build:** The shape the operator asked for from the start — predict the next layer at
this one, move the weights during the next layer's Attention, and have that layer's MoE find
the replica already live.

The launch moves from the head of the following layer's MoE to the **tail of the predicting
layer's MoE**, after the snapshot completes; the wait stays at the head of the target layer's
MoE. With `prediction_lookahead_layers = 1` the span between them is exactly the target
layer's Attention.

It is bandwidth-feasible and that is measured, not assumed: one layer's Attention projections
are 37.5 us at 512 tokens per rank and more at prefill chunk sizes, against a 33.0 us
one-sided put. **The order matters and this ticket must come last of the four**, because with
a host planner the launch and the wait are adjacent statements and the window is zero — which
is why ticket 02 rejects `lookahead = 1` until now.

**Blocked by:** 06 — No host synchronisation on the per-forward path.

**Status:** implemented 2026-08-30, measured at DP=2. Two criteria need a profile and an
accuracy run and are listed at the bottom with what they need.

- [x] The launch is issued at the predicting layer's MoE tail and the wait at the target
      layer's MoE head, with at least one Attention block of compute between them.
      `_placement_after_snapshot` launches, `_placement_before_routing` waits, and which of
      them launches is decided by the coordinator rather than by the runner: the host
      planner synchronises on its snapshot copy inside `plan_and_launch`, so at the tail it
      stalls the layer that issued the copy. It therefore declares
      `launch_at_predicting_layer_tail = False` and keeps the old site; the device
      coordinator declares True. Asserted through the runner's own methods, so a future
      change to the forward body cannot silently move the window.
- [x] `prediction_lookahead_layers = 1` is accepted and is the default. Ticket 02's
      rejection is kept for the host-issued path, where the launch cannot move and the
      window really is zero, with its reason rewritten rather than deleted.
- [x] Plan ownership still holds, and it is enforced in two places rather than asserted in
      prose. Plans are keyed by target layer, so a layer can only be handed a plan aimed at
      it; each carries the forward that produced it, and consuming one from another forward
      raises. Opening a forward with a plan still pending raises too — every plan is
      produced and consumed within one forward, so a leftover means a target layer never
      waited for its transfer. Both are `RuntimeError`, not a fallback: a stale plan names
      an expert chosen from another forward's load, so applying it sheds load onto a rank
      that may since have become the peak.
- [x] A profile shows what is concurrent and what is exposed, and the answer is not the one
      this ticket expected. **The transfer is free: `put_expert` p50 1.2 us and
      `drain_expert` p50 1.1 us, 1.95 ms across a whole profile.** What is exposed is the
      *arrival barrier*: `nvshmemx_barrier_all_on_stream` at p50 5.0 us but p90 1151 us and
      max 5898 us, **57.23 ms in total, 4.40 ms per prefill window**, of which only 1.5%
      overlaps Attention and 11.0% the expert GEMM. A barrier cannot complete until the peer
      reaches it, so it converts rank arrival skew into blocking time once per placed layer —
      44 times a forward — and the compute stream waits on the transfer event that waits on
      it. Ticket 05 measured the same barrier at 13.9 us in isolation.

      So this ticket's premise holds for the thing it was about: the window is long enough and
      the transfer is hidden. Two costs sit next to it, and the CPU side of the same traces
      ranks them. The barrier is worth **4.40 ms per forward** and pairwise arrival would
      remove it — NVSHMEM put-with-signal, with only the target rank waiting, which it can
      decide on the device because the plan is rank-identical. The larger one is **53 extra
      kernel launches and 0.95 ms of host time per layer**, about 42 ms per forward, from the
      tiny elementwise and reduce kernels of `plan_one_layer_on_device`,
      `publish_plan_on_device` and the device-side bookkeeping. That is ticket 03's defect in
      a new place and takes the same fix. Details and the per-kernel table in
      `bench/RESULTS.md`.
- [ ] Prediction accuracy at lookahead 1 is reported against lookahead 2. **Needs an
      accuracy run** (`run_prediction_accuracy.sh`), which wants the GPUs to itself.
- [x] The baseline from ticket 06 is held or improved. At DP=2 — this node's 8 GPUs became 2
      on 2026-08-30 — the reachable layer count went from 43 to **44**, since a lookahead of
      1 leaves one fewer trailing layer unbound, and all of them launch a transfer. The
      excess figure is in `bench/RESULTS.md`; at DP=8 the 24.0% comparison itself needs the
      8-GPU node back.
