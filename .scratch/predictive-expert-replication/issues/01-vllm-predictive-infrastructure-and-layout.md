# 01 — vLLM predictive infrastructure and fixed layout

**What to build:** Enable the vLLM CUDA PoC to opt into Predictive expert replication without enabling the Native EPLB controller. A supported Qwen3-30B-A3B BF16 worker starts with one replica slot per EP rank, fixed canonical ownership, normalized `16 canonical + 1 inactive slot` local layout, and canonical-only serving behavior before any replica plan is applied.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] The feature is disabled by default; predictive mode provisions Expert replication infrastructure while Native EPLB and Predictive expert replication together are rejected.
- [ ] Startup normalization finishes before readiness, leaves one inactive slot per rank, and records its startup duration separately from serving latency.
- [ ] The worker preserves reference output and does not run a Native EPLB placement controller in predictive mode.
- [ ] CPU or existing EPLB-state tests cover fixed layout, inactive rows, configuration validation, and disabled-mode compatibility.
