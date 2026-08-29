# 11 — Evaluate quota-driven routing against source-rank routing

**What to build:** Nothing yet. Measure whether letting a replica absorb an
arbitrary token quota is worth overturning source-rank routing, and report a
recommendation. This is an evaluation ticket because what it would replace is a
correctness contract that is already implemented and verified 7 of 7 (ticket 03),
not an unbuilt design.

**Where this came from:** UltraEP (arXiv:2606.04101, `Dots-Infra/UltraEP`) plans
replication on exact post-gating load and gives each replica a quota
`q = min(need, target slack, expert's available load)`. Source-rank routing cannot
do that: it splits the source ranks across the copies, so one replica moves exactly
half of an expert. The measured gap is what this ticket exists to price.

**Blocked by:** 10 — Prediction accuracy in the prefill regime. Quota routing is
*more* sensitive to a wrong load estimate than a half-split is: a quota sized from
a bad prediction moves the wrong amount to the wrong rank, where a half-split at
least moves a bounded amount. Pricing it against a prediction whose accuracy is
unknown would answer the wrong question.

**Status:** ready-for-agent

## What the gap actually is

Measured on the 2026-08-25 survey (1 to 3 replicas per layer, one
replica slot per rank, `min_tokens_per_replica` = `BLOCK_SIZE_M`, baseline
1.4559x):

| replicas / layer | quota routing | source-rank half-split | quota ahead by |
| --- | --- | --- | --- |
| 1 | 1.2745x (39.8%) | 1.2947x (35.4%) | 4.4 pt |
| 2 | 1.1756x (61.5%) | 1.2261x (50.4%) | 11.1 pt |
| 3 | 1.1118x (75.5%) | 1.1921x (57.9%) | 17.6 pt |

At matched transfer budget that is roughly **12% to 23% more of the excess
removed**, growing with the replica count. It is *not* the order-of-magnitude gain
an earlier reading suggested: that came from UltraEP's `excess <= slack` feasibility
test, which is a relaxation — it ignores whether the replicated experts have that
much load to give. Under the real constraints quota routing reaches
1.2745x at one per layer, not 1.0.

## What it would cost

- **Ticket 03's contract is overturned.** "No source rank's chunk is ever split" is
  what makes the physical map a per-source-rank lookup with no per-token work. Quota
  routing needs a per-token assignment that hits a target count.
- **vLLM's existing per-token replica choice does not suffice.** It is a Knuth hash,
  so it is uniform-random across copies, not quota-driven. A new reroute path is
  needed; UltraEP writes its own CUDA kernel for this (`csrc/kernels/reroute.cu`).
- **Dispatch becomes irregular.** UltraEP names this directly and it is why their
  kernels are custom.

## Acceptance criteria

- [ ] The gap is re-measured on prefill forwards from mixed-domain traffic with a
      sample large enough to separate the two, using **predicted** load to size
      quotas and **actual** load to score them — the same in-sample/out-of-sample
      split ticket 10 establishes. An oracle comparison is not the answer here.
- [ ] The comparison holds transfer count equal, not replica count: a quota replica
      and a half-split replica each cost one transfer, so budget is the fair axis.
- [ ] The cost side is costed, not asserted: what a quota-driven reroute would add
      to the dispatch path, and whether it can stay inside the window measured for
      the weight transfer.
- [ ] A recommendation that states plainly whether the gain justifies replacing a
      verified contract, and if not, says so and closes this ticket.
- [ ] Whatever the answer, `min_tokens_per_replica` is adopted independently — it is
      cheap, it applies to both routing schemes, and for us its value is not a
      tunable: below one `BLOCK_SIZE_M` a replica saves no block, so it saves no
      time. UltraEP's 1024 is a guess where we know the quantum.
