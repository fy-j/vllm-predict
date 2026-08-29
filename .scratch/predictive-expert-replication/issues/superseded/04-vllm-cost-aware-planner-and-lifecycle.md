# 04 — vLLM cost-aware planner and replica lifecycle

**What to build:** Connect real Global predicted-load snapshots to deterministic placement of up to `max_replicas_per_layer` distinct experts, transfer, routing, replacement, and reclamation, using a validated offline cost profile seeded by ticket 00's measurements.

**Blocked by:** 00 — Feasibility checkpoint; 02 — vLLM cross-layer prediction and global predicted-load snapshot; 06 — Prediction accuracy versus lookahead distance; 08 — Replica activation; 09 — vLLM serving baseline report, for the under-load interconnect bandwidth the cost profile requires and which no other ticket owns; 10 — Prediction accuracy in the prefill regime, which gates the prefill build decision the way 00 gates the decode one.

**Status:** ready-for-agent, but **revised 2026-08-25 by measurement** — see the
revision note below before implementing. Three criteria changed shape and one
changed meaning; the lifecycle section is the part most affected.

**Partly implemented already.** A planner and a reduced lifecycle run on this branch:
`plan_replicas` chooses placements, `reconcile` keeps and reverts, and every forward
re-plans. What is genuinely missing is the residency and hotness state machine, any
consumer of the cost profile's four cost values, and the byte bound. Read
`CURRENT-STATUS.md`'s 2026-08-29 code audit before starting — finding 2 shows the
online planner is per layer and that this, not accuracy, is the 15%-versus-33% gap, so
this ticket's cost model has less to fix than it looks.

## Revision note (2026-08-25)

Measured on 1705 forwards across three domains (`bench/RESULTS.md`, 2026-08-25):

1. **The seam returns one globally ranked list, not a per-layer answer.** The
   transfer budget is global, and spending it uniformly per layer is 13% to
   5% behind ranking across layers at the same budget (critical path
   1.245 against 1.2258 at two per reachable layer). `max_replicas_per_layer` is a
   per-layer safety cap, not the allocation target; how many replicas a layer gets
   is a result of the prediction.
2. **`max_transfers_per_forward` must default to about 48 for prefill, not 4.** At 4
   the ceiling is 1.0% of a prefill step; at 48 it is 5.4% here and 12.5% on an
   NVLink machine, and 48 costs 432 MiB per forward against 6 GiB of activation
   traffic already in flight. The 4 was right for decode and wrong here.
3. **Cross-forward residency is harmful under mixed traffic and must not be relied
   on.** A placement planned on one forward and applied to another removes 24.2% of
   the excess within a domain and **-20.0%** across domains, worsening as the budget
   grows. Per-layer peak-rank stability is 0.33 within text and 0.39 pooled. So the
   policy re-plans every forward, and `min_residency_steps` / `hot_stable_steps` do
   not carry their decode meaning: they may only keep a placement that the *current*
   forward's prediction would choose again, never one inherited on faith.

- [ ] **A placement is reverted when the policy would no longer choose it.** The
      2026-08-25 bring-up path has no reversion at all: an empty plan leaves the layout
      untouched, so a replica placed once persists indefinitely, and a later placement
      into an occupied slot overwrites the earlier one silently because the
      one-slot-per-rank check only covers placements within a single plan. Measured,
      cross-forward persistence is **harmful** under mixed traffic (-20.0% across
      domains), so this is a correctness gap and not a tuning one: the code as it stands
      would be expected to lose to the baseline on mixed traffic.
- [ ] Every rank deterministically produces the same per-layer placements, or none, from the same snapshot, using `GreedyPeakReductionPolicy` and stable tie-breaking.
- [ ] A layer may place up to `max_replicas_per_layer` **different** logical experts, each on a different target rank chosen from the lightest ranks — a **cap**, not a target; the count comes from the global ranking. Replicating one expert onto several ranks is out of scope: it sheds at most that expert's own share times `K/(K+1)`, below the measured need at every fan-out.
- [ ] Candidates are drawn only from the layer's current peak rank, and are ranked by predicted reduction in that layer's peak rank time. Replicating an expert owned by any other rank cannot shorten the rank the layer waits on.
- [ ] The imbalance the policy reads is the per-layer critical-path kind. Summing layers before comparing ranks lets their peaks cancel and understates the need by about 3x on measured data.
- [ ] Across layers within one forward, `max_transfers_per_forward` bounds **transfers**, not layers: a layer placing K replicas spends K of the budget. The budget is spent on the highest-ranked candidates **across all layers**, never divided among them, and its prefill default is about 48 rather than 4. Approvals go by descending predicted benefit; remaining layers leave placement unchanged.
- [ ] The planner requires a cost profile matching model, dtype, EP size, logical expert count, and device; missing, invalid, or fingerprint-mismatched profiles fail startup. Transfer cost is computed from usable bandwidth measured under load, not idle bandwidth.
- [ ] `hot_load_ratio` acts only as a candidate pre-filter; the policy's positive-benefit test is the authoritative gate, and a lifecycle hot observation means the policy would select the same `(logical expert, target rank)` placement again.
- [ ] Residency never carries a placement the current forward's prediction would not choose again — measured, a placement inherited across forwards is worth +24.2% within one domain and **-20.0%** across domains. Subject to that, enforced minimum residency is the greater of configured `min_residency_steps` and the steps needed for predicted per-step gain to cover the exposed cost of **all** the layer's transfers, so a placement that cannot pay for itself is never created.
- [ ] Positive predicted benefit, transfer amortization, hot stability, replacement, and reclamation follow the approved lifecycle contract.
- [ ] Forward/version ownership prevents stale plan commits under a lookahead greater than one: a target accepts only the plan produced by the layer `prediction_lookahead_layers` before it in the same forward, each layer holds at most one inbound pending plan, and forward completion asserts none remain.
- [ ] Dummy forwards and prediction errors do not trigger online fallback.
- [ ] CPU tests cover planner outcomes and lifecycle transitions; distributed tests prove plans drive the activation path from ticket 08 correctly.
