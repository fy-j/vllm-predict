# 05 — vLLM CUDA validation and fair benchmark

**What to build:** Validate the completed CUDA PoC on the target eight-GPU RTX 5090 node and produce comparable serving and memory-capacity results for no replica, Native EPLB, and Predictive expert replication.

**Blocked by:** 04 — vLLM cost-aware planner and replica lifecycle.

**Status:** ready-for-agent

- [ ] Record GPU topology, CUDA P2P bandwidth/latency, and NCCL test results before interpreting serving performance.
- [ ] Run distributed correctness, repeated activation/replacement/reclamation, and slow-transfer tests with BF16 tolerance.
- [ ] Compare the three baselines with identical model, request mix, and replica-slot budget; report fixed-RPS TPOT/TTFT and maximum sustainable RPS under the p99 TPOT SLO.
- [ ] Report memory-capacity results separately, including KV blocks, peak memory, maximum concurrency, and OOM boundary.
- [ ] Report prediction quality, predicted/actual peak load, transfer overlap, exposed wait, and startup normalization separately from request latency.
