# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One kernel from a source layer's hidden states to its target's predicted counts.

Ticket 18. The unfused path issues the target gate's GEMM, the router's top-k and the
count as separate operators; this issues one. It extends ticket 03's counting kernel
upstream rather than adding a second one beside it.

**What this is worth, measured rather than projected.** An 8-rank attribution of a
DP=8 profile prices prediction's added window at 8.4% host dispatch, 1.8% device compute
and 81% gap. So fusing bounds out at the dispatch share -- about 0.9% of mean TTFT --
and not at the 52% the ticket set's "launches and compute" label suggested. That label
came from eliminating the collective and calling the residual launches; the residual is
mostly still waiting. Do not re-derive a larger number from the ticket text.

**Selection identity is the contract, not closeness.** A count that is a token off is
harmless. A *selection* that differs plans a replica for an expert the target layer will
not route to, and nothing this project measures would show it: the row still looks
plausible, the placement still activates, the excess it removes silently drops. So the
kernel reproduces the reference's arithmetic rather than an equivalent of it --
accumulate in fp32, round to the gate's own output dtype, then select -- and anything it
cannot reproduce falls back instead of approximating.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# Logged once per process, not per layer: 44 identical lines would be noise, and the
# decision is the same for every source in a model.
_REPORTED: set[str] = set()

# Measured, not chosen: 128 experts fuse at 30.8 us against the unfused 39.0 us, 256 at
# 87.5 against 58.8, and 384 (padding to 512) at 234.2 against 61.4. The crossover is
# between 128 and 256, so 256 is the last width that is allowed to try.
_MAX_FUSED_EXPERTS = 256


def _report_once(message: str) -> None:
    if message not in _REPORTED:
        _REPORTED.add(message)
        logger.info("Predictive expert replication: %s", message)


@triton.jit
def _fused_predict_kernel(
    hidden_ptr,
    gate_ptr,
    unpadded_ptr,
    out_ptr,
    num_tokens,
    hidden_size,
    stride_hm,
    stride_hk,
    stride_ge,
    stride_gk,
    num_logical_experts,
    TOP_K: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IEEE: tl.constexpr,
    ROUND_TO_BF16: tl.constexpr,
):
    """A token tile's logits, selection and counts, without leaving registers.

    `EXPERTS` is the expert count padded to a power of two, which is what makes the
    whole logit row hold: the reduction is over hidden size and the output is 128 wide,
    so there is no N tiling and no intermediate ever reaches memory. Five tensors of
    `[num_tokens, top_k]` in the pre-03 path become none.
    """
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    unpadded = tl.load(unpadded_ptr)
    # Padding rows are excluded here rather than after selection, so they cost no
    # arithmetic and, more to the point, cannot reach the atomics by a later edit.
    row_valid = rows < tl.minimum(num_tokens, unpadded)

    experts = tl.arange(0, EXPERTS)
    real_expert = experts < num_logical_experts

    acc = tl.zeros((BLOCK_M, EXPERTS), dtype=tl.float32)
    for start in range(0, hidden_size, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        in_hidden = cols < hidden_size
        left = tl.load(
            hidden_ptr + rows[:, None] * stride_hm + cols[None, :] * stride_hk,
            mask=row_valid[:, None] & in_hidden[None, :],
            other=0.0,
        )
        right = tl.load(
            gate_ptr + experts[None, :] * stride_ge + cols[:, None] * stride_gk,
            mask=in_hidden[:, None] & real_expert[None, :],
            other=0.0,
        )
        acc = tl.dot(left, right, acc, input_precision="ieee" if IEEE else "tf32")

    if ROUND_TO_BF16:
        # The reference gate is a Linear whose output is the input dtype, and the router
        # selects on *that*, so a kernel that selected on the fp32 accumulator would
        # order near-ties differently. Rounding here is what makes the two agree, and it
        # also absorbs the accumulator's own reassociation: bf16 keeps 8 mantissa bits,
        # far coarser than the difference a different reduction order introduces.
        acc = acc.to(tl.bfloat16).to(tl.float32)

    # Padded lanes must never win a slot, and an unselected expert must never be picked
    # twice, so both are excluded by the same sentinel.
    acc = tl.where(real_expert[None, :], acc, float("-inf"))

    for _ in tl.static_range(TOP_K):
        best = tl.argmax(acc, axis=1)
        tl.atomic_add(out_ptr + best, 1, mask=row_valid)
        acc = tl.where(experts[None, :] == best[:, None], float("-inf"), acc)


@dataclass(frozen=True)
class FusedPredictionPlan:
    """What the kernel needs that does not change between forwards.

    Held as a value rather than read per call so the decision to fuse is made once, from
    static properties every rank shares. A per-rank decision here would have one rank
    fusing and another not, and their predicted counts would then differ for a reason no
    log records -- the class of divergence that has deadlocked this branch before.
    """

    num_logical_experts: int
    top_k: int
    dtype: torch.dtype

    @property
    def experts_padded(self) -> int:
        return max(16, triton.next_power_of_2(self.num_logical_experts))


def predict_counts_fused(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    num_unpadded: torch.Tensor,
    plan: FusedPredictionPlan,
    out: torch.Tensor,
) -> torch.Tensor:
    """Count the target layer's predicted tokens per logical expert, in one launch.

    Args:
        hidden_states: Source-local hidden states before dispatch, `[num_tokens,
            hidden_size]`.
        gate_weight: The **target** layer's gate weight, `[num_logical_experts,
            hidden_size]`, unquantized and without bias.
        num_unpadded: Device scalar holding this ubatch's real token count.
        plan: The static properties the kernel specialises on.
        out: `[num_logical_experts]` int32 destination, zeroed here.

    Returns:
        `out`, for symmetry with the unfused counting path.
    """
    out.zero_()
    num_tokens = hidden_states.shape[0]
    if num_tokens == 0:
        return out

    hidden_size = hidden_states.shape[1]
    BLOCK_M = 64
    BLOCK_K = 64
    _fused_predict_kernel[(triton.cdiv(num_tokens, BLOCK_M),)](
        hidden_states,
        gate_weight,
        num_unpadded,
        out,
        num_tokens,
        hidden_size,
        hidden_states.stride(0),
        hidden_states.stride(1),
        gate_weight.stride(0),
        gate_weight.stride(1),
        plan.num_logical_experts,
        TOP_K=plan.top_k,
        EXPERTS=plan.experts_padded,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        IEEE=plan.dtype == torch.float32,
        ROUND_TO_BF16=plan.dtype == torch.bfloat16,
    )
    return out


def plan_fused_prediction(
    target_gate: object,
    target_router: object,
    num_logical_experts: int,
) -> FusedPredictionPlan | None:
    """The plan for this source/target pair, or None to keep the reference path.

    Every condition here is a **static** property of the model: a module type, a dtype,
    a routing rule. So every rank reaches the same answer. Nothing per-rank and nothing
    per-forward may enter this decision: one rank fusing while another does not would
    give the two different predicted counts, and the snapshot would carry the difference
    into a plan with no record of why.

    Returns None rather than raising: an unsupported gate or router is a normal
    configuration, and the reference path predicts it correctly at the cost this ticket
    is trying to remove.
    """
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if not envs.VLLM_PREDICTIVE_FUSED_PREDICT:
        _report_once("prediction is unfused, by VLLM_PREDICTIVE_FUSED_PREDICT=0")
        return None
    if envs.VLLM_BATCH_INVARIANT:
        # Batch-invariant mode swaps the gate's GEMM for one that rounds differently,
        # and selection is decided by that rounding. It is an env var, identical on
        # every rank, which is the only reason it may be read here.
        return None

    weight = getattr(target_gate, "weight", None)
    if weight is None or weight.dim() != 2:
        _report_once("falling back to the unfused path: the gate exposes no 2-D weight")
        return None
    if not isinstance(
        getattr(target_gate, "quant_method", None), UnquantizedLinearMethod
    ):
        _report_once("falling back to the unfused path: the gate is quantized")
        # A quantized gate applies scales the kernel does not implement. Predicting from
        # a different rule than the target layer will use is worse than not fusing.
        return None
    if getattr(target_gate, "bias", None) is not None:
        _report_once("falling back to the unfused path: the gate has a bias")
        return None
    if weight.shape[0] != num_logical_experts:
        _report_once(
            "falling back to the unfused path: the gate's output width is not "
            "the logical expert count"
        )
        return None
    if weight.dtype not in (torch.bfloat16, torch.float32):
        _report_once(
            "falling back to the unfused path: the gate's dtype is unsupported"
        )
        return None

    if type(target_router).__name__ != "FusedTopKRouter":
        _report_once(
            "falling back to the unfused path: the router is not FusedTopKRouter"
        )
        # Grouped top-k, correction bias and hash routing each change *selection*, which
        # is the one thing this kernel may not approximate.
        return None
    if getattr(target_router, "scoring_func", None) not in ("softmax", "sigmoid"):
        _report_once(
            "falling back to the unfused path: the router's scoring function is "
            "not monotonic in the logit"
        )
        return None
    # Above this the accumulator stops being the point. The kernel holds a whole
    # `BLOCK_M x next_pow2(experts)` fp32 row per program, which is what lets the logits
    # stay in registers; at 384 experts that pads to 512 and spills, and the fused path
    # measured 234 us against the unfused 61 us on this H100 -- bit-identical and four
    # times slower, with nothing in a TTFT number to say which ran. A DeepSeek-class
    # model is ticket 10's own target, so this width will be reached.
    if triton.next_power_of_2(num_logical_experts) > _MAX_FUSED_EXPERTS:
        _report_once(
            f"falling back to the unfused path: {num_logical_experts} experts pad past "
            f"{_MAX_FUSED_EXPERTS}, where the fused accumulator spills and loses"
        )
        return None

    top_k = getattr(target_router, "top_k", None)
    if not isinstance(top_k, int) or top_k < 1 or top_k > num_logical_experts:
        return None
    # Both scoring functions are monotonic in the logit, so a top-k on logits selects
    # what a top-k on scores would, and `topk_softmax` breaks an exact tie towards the
    # low index exactly as `tl.argmax` does. Selection needs neither the scores nor
    # their normalisation, both of which the reference computes and prediction discards.
    _report_once(
        f"prediction is fused into one kernel (top_k={top_k}, "
        f"experts={num_logical_experts}, dtype={weight.dtype})"
    )
    return FusedPredictionPlan(
        num_logical_experts=num_logical_experts, top_k=top_k, dtype=weight.dtype
    )
