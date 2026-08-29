# 08: Three-arm TTFT verdict and the stop gate

**What to build:** The decision, with its evidence, and an explicit answer to the stop gate the
spec commits to. Not more implementation.

Every TTFT figure in this project before 2026-08-29 compared placement against prediction,
because both arms enabled prediction and only the transfer budget differed. The disabled arm,
measured once it existed, put prediction and its infrastructure at **+7.6%** and placement at
**+31.5%** mean TTFT against a stock server, while the ceiling is **5.05% of a prefill step**.
This ticket is where the rebuilt path is held to the same standard.

**Blocked by:** 03 — Predicted counts from one Triton kernel; 07 — The overlap window becomes
the target layer's Attention.

**Status:** ready-for-agent

- [ ] Three arms — feature disabled, prediction only, placing — on `ko` and on `zh` as an
      independent second point. `ko` carries 89.6% of recoverable excess and `zh` 77.8%, so
      agreement between them is what separates a policy result from a dataset artifact.
- [ ] Mean and p99 TTFT for each arm, reported **against the disabled arm**. A figure quoted
      against the prediction-only arm answers a different question and must say so.
- [ ] Results in the required form: the share of measured recoverable excess recovered, not an
      absolute percentage, because an absolute figure cannot separate a weak policy from an
      operating point with little to recover.
- [ ] The expert-GEMM share of a prefill step re-measured, and the ceiling recomputed from it.
      Report the band, since the share is a mean across ranks spanning 8.74% to 15.56% while
      the step time is set by the peak rank.
- [ ] **The stop gate, evaluated and stated.** Stop and report a negative result if the expert
      GEMM share is below 8% of a prefill step, or if prediction and placement together cost
      more than half the measured ceiling. A negative answer here ends the project rather than
      becoming further tuning; this project has twice reached a conclusion only after building,
      and both times the arithmetic had been available earlier.
- [ ] The interconnect, the model shape and the operating point recorded with every number, so
      a conclusion about one node or one expert geometry is not mistaken for one about the
      policy.
- [ ] Where the residual cost goes, attributed rather than inferred, with any remaining gap
      named as unexplained rather than absorbed.
