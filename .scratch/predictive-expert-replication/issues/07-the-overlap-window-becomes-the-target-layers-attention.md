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

**Status:** DONE 2026-08-30, all criteria measured at DP=2. The launch moved, `lookahead=1`
is the default and predicts 8% better on the planner's own metric, and the profile found the
cost centre the ticket did not expect — 132 kernels per placed layer, since fused to 2. What
remains for the *feature* is a DP=8 re-measurement, which is ticket 08's, not this one's.

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
      decide on the device because the plan is rank-identical. The larger one was **53 extra
      kernel launches and 0.95 ms of host time per layer**, about 42 ms per forward, from the
      tiny elementwise and reduce kernels of the plan, the publish and the device-side
      bookkeeping — ticket 03's defect in a new place. **Fixed the same way**: two Triton
      kernels in `fused_placement.py` take a placed layer from 132 launches to 2, placement's
      host overhead per layer to zero, and its mean TTFT from +32.0% to about +3%, with the
      excess it removes unchanged at 34.5-36.5%. The barrier then measured 0.23 ms rather
      than 4.40 — it had been exposing the launch storm's arrival skew rather than costing
      anything itself. Details and the per-kernel tables in `bench/RESULTS.md`.
- [x] Prediction accuracy at lookahead 1 is reported against lookahead 2, and it is better
      on every metric. Korean prompts at 1024 tokens, DP=2, 1584 and 1548 samples:
      **`peak_hit_rate` (= recall@1) 0.7702 against 0.7132**, recall@2 0.8166 against
      0.7629, and `count_error` **0.0925 against 0.1269** — 8% better on the figure the
      planner depends on and 27% less total-variation error. Recall@1 is that figure
      because `max_replicas_per_layer` defaults to 1: the planner picks one expert, so what
      it needs is that the one it would choose is the one that actually ran hottest.
      `accuracy_report.py`'s `PLANNER_K` said 2 and is corrected, since it was the report's
      headline and the cap changed under it.

      Compared on the 43 target layers both arms cover, because lookahead 1 reaches target
      layer 4 and lookahead 2 does not, and early layers are what a skip decision is about.
      It barely matters here — 0.767 common against 0.770 pooled — which is itself worth
      knowing: the improvement is not an artifact of layer coverage. `lookahead_pair()`
      does the restriction, with tests for the two cases worth pinning.

      So lookahead 2 cost 8% of the planner's operative accuracy to buy the host planner a
      layer of latency, and the device path does not need it.
- [x] The baseline from ticket 06 is held or improved. At DP=2 — this node's 8 GPUs became 2
      on 2026-08-30 — the reachable layer count went from 43 to **44**, since a lookahead of
      1 leaves one fewer trailing layer unbound, and all of them launch a transfer. The
      excess figure is in `bench/RESULTS.md`; at DP=8 the 24.0% comparison itself needs the
      8-GPU node back.
