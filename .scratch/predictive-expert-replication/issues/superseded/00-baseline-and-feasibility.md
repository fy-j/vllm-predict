# 00 — Feasibility checkpoint

**What to build:** No replication code. Answer one question with evidence: **does the measured EP rank imbalance convert into time on this node?** Everything downstream of a transfer is worth building only if it does. Reporting that does not bear on that question lives in ticket 09, so implementation waits on this decision alone.

**Blocked by:** None — run this before implementing any transfer machinery.

**Blocks:** 07, 04, 05.

**Status:** ANSWERED, and the answer differs by regime. **Decode: negative** (0.08% to 0.15% of a step; block quantization eats the imbalance). **Prefill: conditionally positive** (3.3% of prefill here, 7.8% on an NVLink machine) but only with `max_transfers_per_forward` raised from 4 to about 48 and with no reliance on cross-forward residency. Read both sections before acting. See `bench/RESULTS.md`, 2026-08-24 and 2026-08-25. In **decode** the limit is MoE kernel block granularity — not the policy, the interconnect, or prediction accuracy. In **prefill** quantization is already open, so the limit is the transfer budget and the routing granularity instead.

- [x] Interconnect and transfer cost. One BF16 expert is 9.00 MiB and moves in 176 to 177 us at about 53.5 GB/s, uniform across rank pairs. No NVLink on this node: every transfer traverses PCIe Gen5 x16.
- [x] Overlap window versus that cost. One layer's Attention is 23 to 31 us, so a lookahead of one hides 15% of a transfer while a lookahead of two hides 92%. This is why the lookahead is configurable rather than fixed at one.
- [x] Expert-weight read floor. A rank holds 144 MiB of local experts per layer, and a bmm proxy read it in 102 us. **That is not what the served kernel costs**: it loads only experts that have a block, so it costs 36 us/layer at one token per rank and reaches 94 us at 32. Superseded by the measured curve in the item below; the 102 us figure describes the proxy, not the floor.
- [x] Critical-path imbalance, measured per layer. Headroom is 41.1% median, consistent with the balancedness the server logs.
- [x] Concentration of that imbalance. Within a layer the peak rank must shed 39.6% (median) to equalize, its hottest expert carries 33.6% against 6.2% for an even split, and its top three carry 68.5%. This is what settled the placement policy on several distinct experts rather than one expert spread wider.
- [x] **MoE's share of TPOT, attributed rather than inferred.** Measured on pure-decode profile windows at concurrency 8, 64, 192 and 384: the expert GEMM is 1.44% to 3.56% of a decode step, or 2.94% to 5.30% of the attributed GPU time — a third to a half of the step is unattributed, so the second figure is an upper bound on the first, rising only slightly with batch. Collectives are 20x it, 1114 to 1520 us per layer, and are `allgather_reducescatter` token dispatch and combine rather than eager launch overhead — the GPU is busy, not idle. Their byte volume is two orders of magnitude too small to explain their duration, so they are arrival skew and latency; per-rank NCCL time does not track per-rank MoE time, so placement cannot move it.
- [x] MoE time over the expert-weight read floor at the reachable operating points. The 102 us floor came from a bmm proxy and does not describe the real kernel, which costs 36 us/layer at one token per rank because it loads only the experts that have blocks. Measured with the real kernel, MoE rises 36 to 94 us/layer from 1 to 32 tokens per rank, and the decode batch is read from the runner's own step annotation rather than derived from throughput. The binding constraint turned out to be different: `moe_align_block_size` rounds each expert up to `BLOCK_SIZE_M`, and at every reachable point tokens per expert (0.5 to 16) sits below it (16 to 64), so every touched expert costs exactly one block and imbalance is free. Crossing that needs M > 2048 concurrent decode tokens against about 464 this node can serve.
- [x] **A recommendation with its evidence**: the achievable gain net of transfer cost, and whether that justifies building the transfer machinery. The decision is the operator's, not automatic; no numeric threshold is fixed in advance because the threshold should follow the measured headroom rather than precede it. Where the evidence shows the limit is the interconnect or per-step overhead rather than the policy, say so, since those are conclusions about this node and do not transfer to a machine with NVLink or to the Ascend port.

**Tooling** already exists under `bench/`: a hardware probe, a serving harness with per-shape and per-domain pairing, a per-layer per-expert load dump behind an environment variable, a results table that derives the real decode batch, and a guard that rejects a benchmark run which measured nothing.


## Recommendation for **decode** (2026-08-24): stop; do not build 07, 04, or 05 for it

Perfect expert balance has a ceiling of **0.08% to 0.15% of a decode step** (0.13% to 0.24% against attributed GPU time alone, which is what this section first quoted; a third to a half of the step is unattributed),
against a transfer costing 175 us per expert. Measured at four operating points,
consistent across all of them.

The limit is **MoE kernel block granularity**, and naming it correctly matters
because it is not any of the things the design worried about:

- Not the placement policy. The policy could be optimal and recover the same
  0.15%, because a rank whose expert holds 3 tokens already costs what a rank
  holding 12 costs.
- Not the interconnect. PCIe hurts the collectives, which are 20x the expert
  GEMM, but placement cannot move collectives at all.
- Not prediction accuracy, which is why ticket 06 does not gate this decision.

What would have to change, in order of leverage:

1. **Tokens per expert above `BLOCK_SIZE_M`.** Needs more than 2048 concurrent
   decode tokens. A model with fewer, larger experts reaches it far sooner;
   Qwen3-30B-A3B's 128 experts at top-8 is close to the worst case.
2. **A smaller `BLOCK_SIZE_M` at large M.** No tuned `E=128,N=768` config exists
   for H100, and the H200 one uses 128 at M >= 1024, so tuning is not an obvious
   route.
3. **NVLink** raises MoE's *share* by shrinking the collectives, but leaves the
   quantization argument untouched. 8x H100 raises reachable M about 4x, to
   roughly 1800 — still under the threshold at this context.

This is a conclusion about this model shape and operating range. It transfers to
the Ascend port only insofar as that port shares them; a different expert count,
top-k, or block size changes the arithmetic and the question should be re-asked
there rather than inherited.


## Recommendation for **prefill / TTFT** (2026-08-25): conditionally proceed

The operator's goal moved to TTFT, which is a different regime and got its own
measurement (1705 forwards, three domains, two chunk budgets — `bench/RESULTS.md`,
2026-08-25). Prefill does **not** share decode's death: at 1024 tokens per expert
the kernel spends 8 blocks per expert, so quantization is open and the imbalance
converts into time.

- [x] **Prefill imbalance is real**: per-layer critical path 1.2801x to 1.5821x
  across text, code and math, against a 1.0112x multinomial floor. Combined baseline 1.4559x.
- [x] **The ceiling, with perfect prediction and one placement per layer**: 35.4%
  of the excess, which is 11.1% of MoE time, which is **3.3% of a prefill step
  here and 7.8% on an NVLink machine**. Two limits are already in that figure:
  only layers 5 to 47 can receive a placement, because a target is `lookahead`
  layers after a source and sources start after `prediction_skip_first_layers`, so
  the leading five layers are unreachable and carry 9.2% of the excess. The second
  figure is arithmetic from MoE's share of a prefill step, not a measurement.
- [x] **The binding constraint is `max_transfers_per_forward`, and its default of
  4 was sized for decode.** At 4 the ceiling is 0.5% and not worth building. At 48
  the transfers cost 432 MiB per forward against 6 GiB of activation traffic
  already in flight (+7%) and 8.5 ms serialized against a ~200 ms forward (4%).
- [x] **Cross-forward residency is harmful under mixed traffic**: a placement
  planned on one forward and applied to another removes 24.2% of the excess within
  one domain and **-20.0% across domains**, worsening as the budget grows. The
  policy must re-plan every forward.
- [x] **Raising `max_num_batched_tokens` is ruled out on both counts**: it does
  nothing for the feature (quantization was already open) and it makes mean TTFT
  35% to 46% worse, because a bigger chunk turns first tokens from a pipeline into
  a batch.

**What still gates a build decision:** prefill prediction accuracy is unmeasured —
ticket 06's runs were overwhelmingly decode forwards — and the payoff is the
oracle bound times that accuracy. That is ticket 10. And on this node the 5.4%
signal sits under a 119 ms interconnect term whose cross-rank spread is already
2.8x, so mechanism can be validated here but payoff cannot.
