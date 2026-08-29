# 10 — Prediction accuracy in the prefill regime

**What to build:** Measure how well the cross-layer gate predicts a target layer's
load **during prefill**, at the shapes and domains the TTFT work targets. Ticket 06
answered this question for decode: its runs used a 128-token decode, so almost every
scored forward carried tens of tokens, not the ~16k a prefill step carries. Prefill
routing is a different statistical regime — 1024 tokens per expert against 4 — and
the accuracy there is unmeasured.

This gates the prefill build decision rather than following it. Ticket 00's prefill
answer is an **oracle** bound: 50.4% of the per-layer excess removed with one
placement per layer, worth 5.4% of a prefill step here and 12.5% on an NVLink
machine. The realistic figure is that bound times the accuracy this ticket measures.
At recall@2 of 0.8 the prefill payoff is worth building; at 0.4 it is not.

**Blocked by:** 02 — vLLM cross-layer prediction and global predicted-load snapshot.

**Status:** answered 2026-08-25 — accuracy is sufficient; the blocker moved elsewhere

- [ ] Accuracy is measured on forwards whose post-allgather token count puts them in
      the prefill regime, and those forwards are identified by a recorded token count
      rather than by a magnitude threshold. The dump records assignments per layer for
      exactly this reason.
- [ ] Reported as hot-expert set overlap at the hot-set sizes a placement actually
      uses, and as per-expert count error, per lookahead distance, per domain. Reuse
      `bench/prediction_accuracy.py`; do not write a second scorer.
- [ ] Accuracy is reported for the **peak rank's** experts specifically, not pooled
      over all experts. The policy only ever considers candidates on the peak rank, so
      accuracy elsewhere does not bear on what it can do.
- [ ] **The number that decides the build**: the oracle placement benefit recomputed
      using *predicted* load to choose placements and *actual* load to score them, in
      the same forward. `bench/imbalance.py` already separates planning from
      evaluation, which is the seam this needs.
- [ ] Enough prefill forwards to support it. A run of 120 requests at concurrency 64
      with 2048-token prompts yields only about 6 full prefill steps, because one step
      admits eight of them; the 2026-08-25 survey hit exactly this and its prefill
      figures rest on 18 forwards across three domains. Size the request count from the
      number of prefill forwards wanted, not from the number of requests.
- [ ] A recommendation: whether the prefill payoff net of prediction error justifies
      building 07, 04 and 08 for it, and on which hardware.

---

## Answer (2026-08-25)

**Prefill prediction is good enough to place on.** Measured offline from the ticket 06
accuracy dumps, restricted to forwards whose recorded assignment count puts them in the
prefill regime (>= 4096 assignments; the largest carry 131072, or 1024 tokens per
expert). Code domain, 1024-token prompts, EP=8, 4 forwards per lookahead:

| lookahead | recall@2 | peak-rank recall@2 | peak-rank hit | excess removed, predicted | oracle |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8551 | 0.9091 | 0.892 | **34.1%** | 36.1% |
| 2 | 0.7645 | 0.9012 | 0.878 | **33.2%** | 36.8% |
| 3 | 0.7024 | 0.8958 | 0.827 | **32.4%** | 37.6% |

Prediction error costs about 3.6 points against the oracle at the configured lookahead
of 2, and no forward was made worse. Accuracy on the peak rank's own experts — the only
ones a placement can choose from — is **higher** than the pooled figure, 0.90 against
0.76, because the pooled figure is dominated by cold experts the policy never considers.

**This ticket's framing was wrong on one point.** It said the realistic figure is the
oracle bound "times the accuracy this ticket measures". A product like that cannot go
below zero and this quantity can: a misled placement adds load to the rank the layer is
actually waiting on, so it overshoots the baseline rather than merely failing to help.
It has to be measured by replaying predicted-chosen placements against actual load,
which is the `imbalance.plan_moves` / `imbalance.apply_moves` seam added for it. A test
pins the harmful case.

**The blocker is now the live path, not prediction.** The 2026-08-25 end-to-end run
placed 131 replicas per forward and removed **0.6%** of prefill excess, where the oracle
on that same dump removes **35.1%** at 43 placements and 59.9% at 131. The replicas are
activated — 344 log lines, maps rewritten — but the physical per-rank load equals
canonical ownership to **0.00%**, so no tokens reach any replica slot. `expert_load_pass`
is per physical expert and the dumped `rank_load` reshapes it rank-major including each
rank's replica slot, so that probe is not vacuous at either level.

Two candidate causes remain, neither yet confirmed:

  * `logical_replica_count` does not actually reach 2 for the activated expert, so
    `build_source_local_physical_map` has no second copy to split source ranks across.
  * The layer's routing reads its map earlier in the forward than the activation point
    at the top of the MoE forward, so it uses the pre-activation copy.

The exact-load path did achieve 23.9% through the same routing mechanism, so the
mechanism works and the defect is in the incremental in-forward publish.

**Recommendation:** the prefill payoff survives prediction error, so 07/04/08 are not
blocked by accuracy. They are blocked by this defect and by the SXM-vs-PCIe hardware
question. Fix the publish first: it is worth about 35% of prefill excess on this node,
and it is currently delivering 0.6% while costing 2.2x TTFT.
