# 09 — vLLM serving baseline report

**What to build:** The reference numbers ticket 05's three-way comparison is measured against, for the two agreed request shapes on the target node. Reporting only: no replication code, no decision attached. Split out of ticket 00 so that implementation waits on the feasibility decision alone and not on reporting completeness.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent, **revised 2026-08-25**: TTFT is now the primary metric
rather than one of several, since the feature's target moved to prefill. One of its
items is already answered — see the note below.

- [ ] Both fixed request shapes over **all three domains** — conversational text, code, and mathematical reasoning — not only the domain whose natural prompt length fits each shape. Ticket 05's three-way comparison requires all three for both shapes, and a run pairing one domain per shape (as `bench/RESULTS.md` 2026-08-24 did, leaving maths absent) cannot supply it. Pair each shape with the domain that fits it *first*, then cover the rest by trimming: 2k prompt / 1k decode and 1k prompt / 2k decode. Decode length forced; prompt length filtered from real samples, never padded with synthetic tokens.
- [ ] Concurrency levels that KV capacity can actually serve. At a 3072-token context, 181,040 KV tokens per rank allows about 58 sequences per rank, so 464 total is the hard ceiling and roughly 80% of that leaves room for prefill bursts. Report the **real decode batch** derived from throughput times mean TPOT alongside each nominal concurrency: they diverge once chunked prefill competes with decode for the per-step token budget.
- [ ] Fixed-RPS TPOT and maximum sustainable RPS under a p99 SLO. These need an **open-loop** run with a request rate set; a closed-loop concurrency sweep makes RPS an output rather than a control and cannot answer either.
- [ ] The SLO covers TTFT as well as TPOT. Measured p99 TPOT moves 5 to 12% from concurrency 64 to 384 while p99 TTFT moves 4 to 5.8x, so a TPOT-only SLO passes an operating point whose first-token latency has already collapsed.
- [ ] EP rank imbalance per domain and per request shape rather than pooled, reported as the per-layer critical-path kind. Summing layers before comparing ranks lets their peaks cancel: the measured shedding need is 39.6% aggregated correctly against 9.2% pooled, a factor of 4.3, and the hottest expert's share is 33.6% against 11.8%, a factor of 2.9. Quote the figure that matches the quantity.
- [ ] A uniform-routing control run establishes the imbalance floor, reported separately. Synthetically generated prompts sweep the vocabulary and route nearly uniformly, so they must never supply the headroom number.
- [ ] Peak memory and KV block count from a run with **no** EPLB, since enabling it for load recording allocates an extra transfer buffer and those two numbers cannot come from the same run.
- [ ] Per-layer interconnect utilization already consumed by token dispatch and combine, plus point-to-point bandwidth measured while those collectives run. The cost profile's usable-bandwidth field must come from this, not from an idle measurement.
- [ ] **A validated cost profile written from these measurements**, carrying the full runtime fingerprint, with `usable_transfer_bandwidth_bytes_per_us` taken from the under-load measurement rather than the idle figure. This was an item of ticket 00 before the split and was left in neither ticket; ticket 04 still requires it (`04`: "seeded by ticket 00's measurements", and its under-load bandwidth item), so **09 gates 04 for this one item** even though the dependency graph says otherwise. The profile currently on disk still carries the idle figure.
- [ ] End-to-end latency reported alongside TTFT and TPOT. Ticket 05 compares three configurations on it, so the baseline must exist; it was an item of ticket 00 before the split and was dropped.
- [ ] Interconnect facts recorded with every result: NVLink presence, PCIe generation and width, topology matrix.


## Already measured (2026-08-25)

**Raising `max_num_batched_tokens` from 2048 to 8192 makes mean TTFT 35% to 46%
worse**, p99 worse, TPOT 3 to 6% better, throughput flat, across text, code and
math at concurrency 64 with 2048-token prompts. The mechanism: at 2048 one prefill
per step pipelines first tokens out at steps 1, 2, 3, 4 — mean 2.5 step-times; at
8192 four prefills share one step four times as long, so all four wait 4
step-times. A bigger chunk turns first tokens from a pipeline into a batch.

This closes the hypothesis that the small chunk budget was itself the cause of the
queueing. It is a TTFT-against-TPOT trade, not a free gain, and it does nothing for
the feature: quantization is already open at the 2048 budget. Figures in
`bench/RESULTS.md`, 2026-08-25.
