# 08: Three-arm TTFT verdict and the stop gate

**What to build:** The decision, with its evidence, and an explicit answer to the stop gate the
spec commits to. Not more implementation.

Every TTFT figure in this project before 2026-08-29 compared placement against prediction,
because both arms enabled prediction and only the transfer budget differed. The disabled arm,
measured once it existed, put prediction and its infrastructure at **+7.6%** and placement at
**+31.5%** mean TTFT against a stock server, while the ceiling is **5.05% of a prefill step**.
This ticket is where the rebuilt path is held to the same standard.

**Blocked by:** 03 — Predicted counts from one Triton kernel (done); 07 — The overlap window
becomes the target layer's Attention (done); **13 — Four targets per collective**, and **09 —
Can a CUDA graph capture the prediction path**. The last two are new blockers added 2026-08-30:
prediction is now the entire overhead, 13 removes about half of it and 09 prices the other half,
so a verdict written before them measures a cost that is being removed. **17 — The spec says what
was measured** should also land first, since this ticket quotes the spec's ceiling.

**Status:** blocked. Do not write the verdict yet.

## What changed under this ticket, 2026-08-30

Tickets 06 and 11 measured the feature at DP=8 with the device path and the fused kernels, and
the single number this ticket was written to produce turns out to hide the finding:

    stock                169.68 ms      —
    prediction only      192.89 ms   +13.68%     2.6x the whole ceiling
    placing              183.24 ms    +9.38%     6.95 ms FASTER than prediction only

**The two halves have opposite signs.** Placement's own contribution is *negative cost* — the
arms differ in nothing but the transfer budget — and it captures 70-80% of the 5.26%-of-a-window
ceiling. Prediction pays for the window placement needs, and spends 2.6x what balance can ever
return. A stop gate applied to the sum would stop a mechanism that works because of the one that
funds it, so the criteria below are amended to require the split.

- [ ] Three arms — feature disabled, prediction only, placing — on `ko` and on `zh` as an
      independent second point. `ko` carries 89.6% of recoverable excess and `zh` 77.8%, so
      agreement between them is what separates a policy result from a dataset artifact.
- [ ] Mean and p99 TTFT for each arm, reported **against the disabled arm**. A figure quoted
      against the prediction-only arm answers a different question and must say so.
- [ ] **Placement's contribution and prediction's reported separately, with their signs.** The
      placing-versus-prediction-only difference is placement's own effect, and it has measured
      *negative* cost; the prediction-versus-stock difference is what the mechanism charges for
      the window. One combined percentage cannot express that and must not be the headline.
- [ ] **The operating point stated as closed-loop, and a fixed-rate arm added or its absence
      named.** Every TTFT figure on this branch fixes concurrency, not request rate
      (`request_rate: "inf"`), so `req/s` is `num_prompts / duration` and is the reciprocal of
      latency rather than independent evidence. The spec asks for lower TTFT "at equal request
      load", which a fixed-concurrency run does not literally provide: a slower arm receives less
      offered load. Either measure at a fixed rate all arms can sustain, or state plainly that
      the claim is about latency at fixed concurrency.
- [ ] Results in the required form: the share of measured recoverable excess recovered, not an
      absolute percentage, because an absolute figure cannot separate a weak policy from an
      operating point with little to recover.
- [ ] The expert-GEMM share of a prefill step re-measured, and the ceiling recomputed from it.
      Report the band, since the share is a mean across ranks spanning 8.74% to 15.56% while
      the step time is set by the peak rank.
- [ ] **The stop gate, evaluated and stated — on the halves, not the sum.** The spec's wording
      is "prediction and placement together cost more than half the measured ceiling", and taken
      literally that already fires while placement is *returning* 70-80% of the ceiling. Evaluate
      it as: does the expert-GEMM share clear 8% of a prefill window (measured 11.16%, so yes),
      and does **prediction alone** cost more than the ceiling after 13 and 09 have reported. A
      negative answer ends the project rather than becoming further tuning; this project has
      twice reached a conclusion only after building, and both times the arithmetic had been
      available earlier. If the gate as written and the gate as amended disagree, say so and let
      the operator choose — do not silently pick the one that permits continuing.
- [ ] The interconnect, the model shape and the operating point recorded with every number, so
      a conclusion about one node or one expert geometry is not mistaken for one about the
      policy.
- [ ] Where the residual cost goes, attributed rather than inferred, with any remaining gap
      named as unexplained rather than absorbed.
