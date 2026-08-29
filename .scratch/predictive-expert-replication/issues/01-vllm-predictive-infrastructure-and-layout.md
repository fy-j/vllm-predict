# 01 — vLLM predictive infrastructure and fixed layout

**What to build:** Enable the vLLM CUDA PoC to opt into Predictive expert replication without enabling the Native EPLB controller. A supported Qwen3-30B-A3B BF16 worker starts with one replica slot per EP rank, fixed canonical ownership, normalized `16 canonical + 1 inactive slot` local layout, and canonical-only serving behavior before any replica plan is applied. An unsupported configuration or an unusable cost profile fails startup rather than serving with the feature silently inert.

**Blocked by:** None — can start immediately.

**Status:** done — implemented, unit-tested, and validated on the 8-GPU node

- [x] The feature is disabled by default; predictive mode provisions Expert replication infrastructure while Native EPLB and Predictive expert replication together are rejected.
- [x] The full configuration surface is validated, and unsupported values fail startup rather than implying functionality that does not exist: replica slots per rank, prediction lookahead, skipped leading layers, concurrent transfer bytes, transfers per forward, replicas per layer, hot-load ratio, hot stability, and minimum residency.
- [x] The approved runtime and model scope is enforced, including the EPLB preconditions that are otherwise skipped because replication settings are derived after parallel configuration has already been validated.
- [x] A cost profile is required and must carry the full runtime fingerprint. Model shape, dtype, EP size, and logical expert count are checked where the configuration is assembled; the device is checked in the worker. Both happen before readiness.
- [x] Predictive behavior lives in the replication state object that both model runners share, because the target model runs on the V1 model runner and duplicating the logic per runner would let the two diverge.
- [x] Startup normalization finishes before readiness, leaves one inactive slot per rank, clears those rows, and records its startup duration separately from serving latency.
- [x] Native rearrangement is refused on the request path in predictive mode while the profile-run reservation still runs, since predictive transfers need the same buffers reserved.
- [x] The worker preserves reference output and does not run a Native EPLB placement controller in predictive mode.
- [x] CPU or existing EPLB-state tests cover fixed layout, inactive rows, configuration validation, cost-profile rejection including fingerprint mismatch, the request-path rearrangement refusal, and disabled-mode compatibility.
