# 16: The block bar knows the dtype, and a failed rank does not hang the group

**What to build:** The two code-review findings that are reachable on a production path, fixed
together because each is small and neither is worth a context window alone. The other four
findings from the same review are debug-only and stay recorded in `CURRENT-STATUS.md`.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

## The block bar

`resolve_moe_block_size_m` takes a `dtype` and forwards it to `try_get_optimal_moe_config`, but
its only caller omits it, and there is no `block_shape` parameter at all. `get_default_config`
branches on exactly those two: for a block-quantised FP8 MoE it returns `BLOCK_SIZE_M=64`, and
with `dtype=None` it falls into the generic branch and returns **128**. The tuned-file lookup is
keyed on dtype as well, so even the tuned path resolves the wrong file.

That constant is **both** gates — `min_tokens_per_expert`, which suppresses a forward, and the
planner's `min_tokens`, which floors a placement's minimum move. Resolved 2x too high, every
prefill forward between 64 and 128 tokens per expert is suppressed and every placement shedding
64 to 127 tokens is refused. That is the exact failure the function's own docstring says it
exists to prevent: "a wrong value either reopens a regime that is settled negative or rejects
placements that would have paid." Harmless on BF16 Qwen, which is the approved scope; a trap for
ticket 10's FP8 DSV4.

- [ ] The resolved block size comes from the same `(dtype, block_shape)` the served MoE actually
      uses, and a test pins it against `try_get_optimal_moe_config` for a block-quantised
      configuration where the answer is 64 rather than 128.
- [ ] The log line distinguishes "resolved from the kernel configuration" from "fell back",
      because it currently claims the former either way, and removing that guess is the reason
      the function exists.

## The transport teardown

`OneSidedExpertTransfer.close()` calls the collective `nvshmem.finalize()`, and
`self._one_sided` is assigned only once `__init__` returns. A rank whose `nvshmem.init`
succeeded but whose symmetric staging allocation then failed logs the exception and skips
`close()`, while every other rank reaches `if self._one_sided is not None: self._one_sided.close()`
and blocks in `finalize()` forever. `agree_across_ranks` was added to turn exactly this class of
asymmetry into a clean fallback, and this reintroduces it on the cleanup side.

- [ ] A rank that fails after `nvshmem.init` still reaches the same teardown decision as its
      peers: either every rank finalizes or none does, agreed the way arming is agreed.
- [ ] A unit test with an injected all-reduce drives the asymmetric case — one rank failing
      after init — and asserts the group does not split. That is the shape ticket 06's
      `agree_across_ranks` tests already use.
- [ ] `close()` stays idempotent and still never raises: shutdown paths get called twice, a
      double free there is a segfault, and an exception on the way out loses results already
      produced.
