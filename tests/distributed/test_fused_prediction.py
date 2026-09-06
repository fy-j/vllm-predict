# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ticket 18: one kernel takes hidden states to the predicted count row.

The contract these tests hold the kernel to is **selection identity**, not closeness.
A predicted count that is one token off changes nothing; a *selection* that differs
plans a replica for an expert the target layer will not route to, and that is invisible
in every aggregate this project measures — the count row still looks plausible, the
placement still activates, and the excess it removes silently drops. So the reference
path is retained as the oracle and equality is asserted against it, which is the same
discipline the two `fused_placement.py` kernels are held to.
"""

from __future__ import annotations

import pytest
import torch

from vllm.distributed.eplb.fused_prediction import (
    FusedPredictionPlan,
    predict_counts_fused,
)

NUM_LOGICAL = 128
TOP_K = 8
HIDDEN = 512

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused kernel is Triton on CUDA"
)


def _reference_counts(hidden, weight, num_unpadded, top_k, num_logical):
    """Gate, select and count exactly as the unfused path does."""
    logits = torch.nn.functional.linear(hidden, weight)
    ids = logits.topk(top_k, dim=1).indices[:num_unpadded]
    counts = torch.zeros(num_logical, dtype=torch.int32, device=hidden.device)
    if ids.numel():
        counts.scatter_add_(
            0,
            ids.reshape(-1).to(torch.int64),
            torch.ones_like(ids.reshape(-1), dtype=torch.int32),
        )
    return counts


def _plan(dtype):
    return FusedPredictionPlan(
        num_logical_experts=NUM_LOGICAL, top_k=TOP_K, dtype=dtype
    )


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_tokens", [1, 17, 2048])
def test_counts_equal_the_unfused_path(dtype, num_tokens):
    """The whole point: same hidden states in, same count row out."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    hidden = torch.randn(num_tokens, HIDDEN, device=device, dtype=dtype)
    weight = torch.randn(NUM_LOGICAL, HIDDEN, device=device, dtype=dtype) / HIDDEN**0.5
    unpadded = torch.tensor(num_tokens, device=device, dtype=torch.int32)
    out = torch.empty(NUM_LOGICAL, dtype=torch.int32, device=device)

    got = predict_counts_fused(hidden, weight, unpadded, _plan(dtype), out)

    expected = _reference_counts(hidden, weight, num_tokens, TOP_K, NUM_LOGICAL)
    assert torch.equal(got, expected), (got - expected).abs().max()


@requires_gpu
def test_padding_rows_never_reach_the_count():
    """A padding row moves load that does not exist, and no aggregate would show it."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, real = 512, 300
    hidden = torch.randn(num_tokens, HIDDEN, device=device, dtype=torch.bfloat16)
    weight = (
        torch.randn(NUM_LOGICAL, HIDDEN, device=device, dtype=torch.bfloat16)
        / HIDDEN**0.5
    )
    unpadded = torch.tensor(real, device=device, dtype=torch.int32)
    out = torch.empty(NUM_LOGICAL, dtype=torch.int32, device=device)

    got = predict_counts_fused(hidden, weight, unpadded, _plan(torch.bfloat16), out)

    assert int(got.sum()) == real * TOP_K
    assert torch.equal(got, _reference_counts(hidden, weight, real, TOP_K, NUM_LOGICAL))


@requires_gpu
def test_an_empty_forward_counts_nothing():
    device = torch.device("cuda")
    hidden = torch.zeros(0, HIDDEN, device=device, dtype=torch.bfloat16)
    weight = torch.zeros(NUM_LOGICAL, HIDDEN, device=device, dtype=torch.bfloat16)
    unpadded = torch.tensor(0, device=device, dtype=torch.int32)
    out = torch.empty(NUM_LOGICAL, dtype=torch.int32, device=device)

    got = predict_counts_fused(hidden, weight, unpadded, _plan(torch.bfloat16), out)

    assert int(got.sum()) == 0


@requires_gpu
def test_ties_break_towards_the_low_index_as_the_router_does():
    """`topk_softmax` breaks an exact tie by index, and a kernel that did not would
    disagree with the target layer on the one input where disagreement is certain."""
    device = torch.device("cuda")
    hidden = torch.zeros(4, HIDDEN, device=device, dtype=torch.float32)
    weight = torch.zeros(NUM_LOGICAL, HIDDEN, device=device, dtype=torch.float32)
    unpadded = torch.tensor(4, device=device, dtype=torch.int32)
    out = torch.empty(NUM_LOGICAL, dtype=torch.int32, device=device)

    got = predict_counts_fused(hidden, weight, unpadded, _plan(torch.float32), out)

    expected = torch.zeros(NUM_LOGICAL, dtype=torch.int32, device=device)
    expected[:TOP_K] = 4
    assert torch.equal(got, expected), got.nonzero().flatten().tolist()
