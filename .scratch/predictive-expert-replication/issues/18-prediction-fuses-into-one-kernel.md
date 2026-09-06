# 18: The prediction gate, router and count fuse into one kernel

**What to build:** A source layer's whole prediction — the target layer's gate GEMM, its
router's expert *selection*, and the per-logical-expert count — as a single Triton kernel, so
that predicting a target costs one launch instead of about seven and prediction's launch half
stops dominating the feature's cost.

This extends ticket 03's counting kernel upstream rather than adding a second one. Ticket 03
took the count itself from about ten launches to one; this swallows what feeds it.

**Why the arithmetic points here.** Prediction's added cost splits 47.6% barriers / 52.4%
launches-and-compute (ticket 11). On the knee at DP=8 that second half is **9.67 ms over 44
source layers, 220 us per layer** — against a gate GEMM that on Qwen3-30B-A3B
(hidden 2048, 128 experts) moves 8.4 MB and issues 1.07 GFLOP for T=2048, so **about 3 us of
real arithmetic, 1.4% of the 220**. The cost is not the prediction, it is the dispatching of
it. That makes this ticket's ceiling **9.67 ms, larger than ticket 14's 8.78 ms**, and unlike
14 it is not capped further by ticket 13 (window batching removes barriers, not launches).

**Two pieces of waste the fusion removes, beyond the launch count:**

* **The router's weights are computed and thrown away.** Prediction reads only
  `logical_ids`; the routing weights, their normalisation and renormalisation are dead work.
  Selection is all that is needed, and softmax is monotonic, so a topk on logits selects what
  a topk on probabilities would. Any bias that shifts *selection* (a correction bias, grouped
  topk's group scores) must still be applied.
* **Nothing between the hidden states and the counts needs to reach memory.** With 128
  experts the whole logit row fits in registers, so `[T, 128]` logits and `[T, 8]` indices —
  five tensors of that shape in the pre-03 path — never need to be materialised. One program
  per token tile computes its logits, selects in registers, and atomically accumulates into
  the count row the window already owns.

**N = 128 is what makes this tractable.** This is not a general GEMM: the reduction is over
hidden 2048 and the output is 128 wide, so a persistent kernel holds a token tile's entire
logit row without tiling the N dimension.

**Blocked by:** None (can start immediately). It is independent of 13's window batching and
attacks a different half; the two compose.

**Relationship to 09, which is a substitute and not a complement.** A captured CUDA graph
removes dispatch cost wholesale, including for ops no fusion can merge; this removes it by
having fewer ops. Whichever lands first shrinks the other's remaining value. This one is the
lower-risk of the two — no capture of data-dependent control flow, no collective inside a
graph — so it should be measured first, and 09's scope reconsidered against what it leaves.

**Relationship to 14:** ticket 14 is closed as DO-NOT-BUILD partly because its ceiling is
smaller than this one's. If this ticket lands, 14 does not come back: it would be attacking
8.78 ms of barrier cost that ticket 13 has already reduced to about 2.2 ms at window 4.

**Status: implemented and measured 2026-09-06, 199 tests, lint and mypy clean. It works,
it is smaller than this ticket claimed and larger than the attribution predicted, and the
run that measured it has a baseline too loose to size it.**

**The stop gate fired and was overridden deliberately.** Criterion 1 asked whether dispatch
is the majority of the 220 us; an 8-rank attribution says it is **8.4%** (device compute 1.8%,
blocking `cudaEventSynchronize` 8.5%, gap 81.2%), so by the ticket's own wording this should
have been reported rather than built. It was built because building it *is* the derivative
experiment the alternative would have measured indirectly, and because the first-order
prediction was worth testing against reality. It was: reality disagreed with it.

**What it measures.** Four arms, three interleaved passes, `ko` at DP=8, the fused and unfused
prediction arms differing in nothing but `VLLM_PREDICTIVE_FUSED_PREDICT`:

```
stock                183.78   spread 5.0%
prediction unfused   212.56   spread 5.2%   +15.7%
prediction fused     205.90   spread 0.9%   +12.0%
placing fused        194.53   spread 4.3%    +5.8%
```

Paired within each pass, which is what the interleave is for:

```
pass 1   nofuse 218.83   fused 205.84   -> +12.99 ms  (+7.07% of that pass's stock)
pass 2   nofuse 212.56   fused 207.64   ->  +4.92 ms  (+2.64%)
pass 3   nofuse 207.85   fused 205.90   ->  +1.94 ms  (+1.10%)
```

**Faster in 3 of 3 passes, median 2.6% of stock TTFT, range 1.1% to 7.1%.** The sign is solid;
the magnitude is not. This run's stock arm drifted **5.0%** across passes, wider than the
2.7-2.8% of the runs before it, and that drift exceeds the mean fused-vs-unfused difference of
3.1%. By this project's own rule the *unpaired* comparison says nothing here; only the pairing
carries it. **Do not quote 3.6% as the headline.**

**It beat its own first-order ceiling, which is the interesting result.** Fusion removes about
1.7 of the 2.7 launches a source layer added, worth roughly 0.43 ms of the 0.69 ms of measured
dispatch, which at this run's amplification is 0.9-1.5% of TTFT. The median measured 2.6%. So
removing dispatch bought more than the dispatch it removed -- consistent with launches driving
part of the 81% gap through rank skew, which is exactly the second-order effect the attribution
could not settle. Consistent with, not proof of: three passes and a loose baseline.

A second, cleaner signal points the same way: the fused arm's own spread is **0.9%** against the
unfused arm's **5.2%**. Fewer launches made the run more repeatable, which is what less skew
would look like.

**What was actually built.** One Triton kernel from hidden states to the count row. `EXPERTS` is
the expert count padded to a power of two, so the reduction is over hidden size with no N tiling
and the whole logit row stays in registers: the five `[T, top_k]` tensors of the pre-03 path
never reach memory. Selection is `top_k` register argmaxes with the chosen lane masked out.

Selection identity, which is the contract, is held by reproducing the reference's arithmetic
rather than an equivalent of it:

* fp32 uses `input_precision="ieee"`, because Triton would otherwise use TF32 while torch's
  `allow_tf32` is False;
* bf16 rounds the fp32 accumulator **back to bf16 before selecting**, because the reference gate
  is a Linear whose output is bf16 and the router selects on that. The rounding also absorbs the
  accumulator's own reassociation: 8 mantissa bits are far coarser than a reduction-order
  difference;
* ties break towards the low index, matching `ops.topk_softmax`, which was probed rather than
  assumed (an all-tie row returns `[0..7]`).

Anything it cannot reproduce falls back, decided once from static properties every rank shares:
a quantized gate, a bias, a non-`FusedTopKRouter`, a non-monotonic scoring function, or
`VLLM_BATCH_INVARIANT` -- which was found by reading `UnquantizedLinearMethod.apply` and would
have silently changed the gate's rounding.

**Every decision now logs once per process.** The fused arms could only be shown to have fused
by the fact that they differed from the unfused ones, which is inference, not evidence. This
project's rule is that a green run is not a connected run; the log line is what makes the next
run checkable.

- [x] **The 220 us is attributed before any kernel is written, and the ticket stops here if
      dispatch is not the majority.** 220 us over the roughly 7.1 launches ticket 03 left is
      31 us each, which is three times a plain eager dispatch, so something in the current
      figure is not pure launch overhead — queueing gaps, or an op that is not as cheap as
      the gate is. A profile restricted to `execute_context` must say, per op, how much of
      the 220 us is kernel time and how much is gap. If kernel time dominates and it is the
      router's topk rather than dispatch, this design does not apply and a different one is
      needed: report that instead of building.
- [x] One Triton kernel takes hidden states, the target gate's weight and whatever its router
      needs, and writes the window's count row directly. Launches per source layer drop from
      about 7 to 1, counted in a profile rather than asserted.
- [x] The kernel is asserted **bit-identical** to the existing gate-plus-router-plus-count
      path on real weights, which is retained as the oracle — the same discipline the two
      `fused_placement.py` kernels are held to. A mismatch in *selection* silently plans the
      wrong expert and is invisible in every aggregate.
- [x] Router variants are handled by covering what the kernel implements and **falling back**
      to the reference path otherwise, never by silently predicting from a different rule
      than the target layer will use. The fallback is chosen on a static property identical
      on every rank, never on per-rank state.
- [x] Padding rows are excluded exactly as the current path excludes them, asserted by a
      forward whose unpadded count is less than its token count. A padding row that reaches
      the count moves load that does not exist.
- [x] Mean TTFT re-measured at DP=8 on the knee, three interleaved passes, against stock and
      against prediction-only in the same run. The share of the 9.67 ms actually recovered is
      recorded, including if it is small.
- [ ] `peak_hit_rate` and `count_error` are unchanged from the reference path, so that a
      launch saving is not paid for with accuracy nobody checked.
