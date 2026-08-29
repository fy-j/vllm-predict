# 10: DeepSeek-V4-Flash integration

**What to build:** The same feature on DeepSeek-V4-Flash. A second integration, not a
configuration change.

It brings **no additional headroom**, and that is measured rather than assumed: expert GEMM is
**10.58%** of a prefill step against Qwen3-30B-A3B's 10.76%, giving ceilings of 5.19% and 5.05%.
The FLOP argument suggested double — 302 MFLOP of expert work per token against 75.5, halved
again by FP8 — but DSV4's attention is sparse MLA with hash compression at 9-13% of a step
against Qwen's 1%, so numerator and denominator moved together. It is sequenced last for
integration cost alone.

**Blocked by:** 08 — Three-arm TTFT verdict and the stop gate. A second integration is not paid
for before the first one pays.

**Status:** ready-for-agent

- [ ] The hooks reach DSV4's MoE. It runs its own mega-MoE path with its own EPLB state and
      expert-weight accessors, which bypasses the runner every current hook lives in, so this
      is the bulk of the work.
- [ ] The read-only prediction path carries what DSV4's router needs. Its router is the
      bias-corrected variant currently on the rejection list because routing depends on token
      identity the path does not forward. **The hash component may be an advantage rather than a
      cost**: it is a deterministic function of token ids, which are known at prediction time,
      so that part of the routing is exactly knowable for free. Establish whether that holds
      before treating it as a limitation.
- [ ] Quantised expert transfer, which moves in scope here: FP8 weights with block scales, and
      the scales travel with the weights or the replica is wrong.
- [ ] Environment facts that cost runs, encoded rather than remembered: the official
      `deepseek-ai` checkpoint loads where the `sgl-project` FP8 conversion does not (its
      gate/up fusion layout differs); `--kv-cache-dtype fp8` is required by the `fp8_ds_mla`
      layout; FlashInfer's JIT needs an `nvrtc.h` on the CUDA include path and an unversioned
      `libnvrtc.so`; and a base checkpoint has no chat template, so the completions endpoint is
      the one to use.
- [ ] The kernel classifier recognises DeepGEMM. A **grouped** scheduler is the tell for the
      expert GEMM; matching on the substring "gemm" is wrong because the attention projections
      are dense GEMMs under a similar name. Without this the expert-GEMM share reads near zero,
      which is a wrong number rather than an error.
- [ ] Three arms on DSV4, held to ticket 08's reporting standard.
