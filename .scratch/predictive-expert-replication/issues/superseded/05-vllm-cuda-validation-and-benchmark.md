# 05 — vLLM CUDA validation and fair benchmark

**What to build:** Validate the completed CUDA PoC on the target eight-GPU RTX 5090 node and produce comparable serving and memory-capacity results for no replica, Native EPLB, and Predictive expert replication, judged against the headroom that ticket 00 measured.

**Blocked by:** 04 — vLLM cost-aware planner and replica lifecycle; 09 — vLLM serving baseline report (the three-way comparison needs one baseline measured the same way).

**Status:** ready-for-agent, **revised 2026-08-25**: the headline metric is **TTFT
under a fixed request rate**, not TPOT. Decode was measured to have no reachable
gain on this node, so a TPOT comparison would be measuring nothing. Request shapes
must be prefill-weighted, and the run must report enough prefill forwards to be
read against ticket 00's prefill section.

- [ ] Record GPU topology, NVLink presence, PCIe generation and width, P2P bandwidth and latency, and NCCL test results before interpreting serving performance.
- [ ] Run distributed correctness, repeated activation/replacement/reclamation, and slow-transfer tests with BF16 tolerance.
- [ ] Benchmark both fixed request shapes (2k prompt / 1k decode and 1k prompt / 2k decode) over conversational text, code, and mathematical reasoning, using the same shapes and domains ticket 09 establishes.
- [ ] Report the ratio of per-layer MoE time to the local expert-weight read floor for every configuration, and run at the highest ratio KV capacity allows rather than assuming a comfortably compute-bound regime is reachable.
- [ ] Report imbalance and recovered gain per domain and per request shape rather than pooled, and state whether the policy's value is domain-dependent.
- [ ] Compare the three baselines with identical model, request mix, and replica-slot budget; report fixed-RPS TPOT/TTFT, end-to-end latency, EP load distribution, and maximum sustainable RPS under a p99 SLO covering **both TPOT and TTFT** (ticket 09 measured p99 TTFT moving 4 to 5.8x across the concurrency range against 5 to 12% for TPOT, so a TPOT-only SLO admits an operating point whose first-token latency has already collapsed).
- [ ] Report the p99 TPOT delta at equal RPS and the sustainable-RPS delta at fixed p99 TPOT SLO, each expressed as the share of the imbalance headroom ticket 00 measured, so a small absolute gain is distinguishable from a small available gain.
- [ ] Report memory-capacity results separately, including KV blocks, peak memory, maximum concurrency, and OOM boundary.
- [ ] Report prediction quality per lookahead distance, predicted/actual peak load, transfer overlap, exposed wait, per-layer interconnect utilization, and startup normalization separately from request latency.
- [ ] Attribute any negative result to the interconnect where the evidence supports it, recording it as a conclusion about a PCIe-only node rather than about the policy, so the Ascend port is not prejudged.
