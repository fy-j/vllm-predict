# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused predicted-count kernel, tested against the path it replaces.

Ticket 03. The kernel collapses about twelve elementwise operations into one, and its
failure mode is a **wrong count** rather than a crash: nothing downstream would raise,
the
planner would simply choose a different expert and the placement would silently be worth
less. Three of this project's five historical defects were of that shape, so the
acceptance
criterion is equality against the retained reference rather than agreement with a
hand-written expectation.

Every test here therefore compares `count_logical_experts_triton` with
`count_logical_experts_reference` on the same inputs. The reference is the code that
shipped
and was measured, so it is the oracle; a test that encoded my own idea of the right
answer
would be weaker, because both implementations could then be wrong together in the way I
happened to imagine.
"""

import pytest
import torch

from vllm.distributed.eplb.predictive import (
    count_logical_experts_reference,
    count_logical_experts_triton,
)

NUM_LOGICAL = 128


def _run_both(logical_ids: torch.Tensor, unpadded: int, num_logical: int = NUM_LOGICAL):
    """Both implementations on identical inputs, returning their outputs."""
    device = logical_ids.device
    num_unpadded = torch.tensor([unpadded], dtype=torch.int32, device=device)
    ref = torch.zeros(num_logical, dtype=torch.int32, device=device)
    got = torch.zeros(num_logical, dtype=torch.int32, device=device)
    count_logical_experts_reference(logical_ids, num_unpadded, num_logical, ref)
    count_logical_experts_triton(logical_ids, num_unpadded, num_logical, got)
    return ref, got


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the kernel is CUDA-only"
)


@requires_gpu
@pytest.mark.parametrize(
    "num_tokens,topk", [(1, 1), (7, 8), (64, 8), (1024, 8), (4096, 6)]
)
def test_matches_the_reference_on_random_routing(num_tokens, topk):
    """The ordinary case, across the shapes prefill and decode actually produce."""
    torch.manual_seed(num_tokens * 31 + topk)
    ids = torch.randint(
        0, NUM_LOGICAL, (num_tokens, topk), dtype=torch.int64, device="cuda"
    )
    ref, got = _run_both(ids, unpadded=num_tokens)

    assert torch.equal(ref, got)
    assert int(got.sum()) == num_tokens * topk, (
        "every valid token-expert pair must be counted exactly once"
    )


@requires_gpu
@pytest.mark.parametrize("unpadded", [0, 1, 13, 63])
def test_padding_rows_are_excluded(unpadded):
    """Padding must contribute nothing.

    A dummy or padding-only forward has to predict zero load while still joining every
    collective, so this is the boundary the whole path depends on.
    """
    torch.manual_seed(unpadded + 7)
    ids = torch.randint(0, NUM_LOGICAL, (64, 8), dtype=torch.int64, device="cuda")
    ref, got = _run_both(ids, unpadded=unpadded)

    assert torch.equal(ref, got)
    assert int(got.sum()) == unpadded * 8


@requires_gpu
def test_an_all_padding_forward_counts_nothing():
    ids = torch.randint(0, NUM_LOGICAL, (32, 8), dtype=torch.int64, device="cuda")
    ref, got = _run_both(ids, unpadded=0)

    assert torch.equal(ref, got)
    assert int(got.sum()) == 0, "a dummy forward must predict zero load"


@requires_gpu
def test_an_empty_batch_counts_nothing():
    ids = torch.zeros((0, 8), dtype=torch.int64, device="cuda")
    ref, got = _run_both(ids, unpadded=0)

    assert torch.equal(ref, got)
    assert int(got.sum()) == 0


@requires_gpu
def test_ids_outside_the_logical_range_contribute_nothing():
    """An out-of-range id must not be folded into a neighbouring expert's count.

    Attributing load to the wrong expert is worse than dropping it: the planner would
    replicate an expert that is not hot. The reference clamps only to keep the scatter
    in
    bounds and masks the contribution separately, and the kernel has to do the same.
    """
    ids = torch.tensor(
        [[0, NUM_LOGICAL, -1, 5], [NUM_LOGICAL + 99, 3, -7, 0]],
        dtype=torch.int64,
        device="cuda",
    )
    ref, got = _run_both(ids, unpadded=2)

    assert torch.equal(ref, got)
    assert int(got.sum()) == 4, "only the four in-range ids may count"
    assert int(got[0]) == 2 and int(got[5]) == 1 and int(got[3]) == 1


@requires_gpu
def test_a_single_hot_expert_is_counted_in_full():
    """The skew the feature exists for: one expert taking every token."""
    ids = torch.full((512, 8), 42, dtype=torch.int64, device="cuda")
    ref, got = _run_both(ids, unpadded=512)

    assert torch.equal(ref, got)
    assert int(got[42]) == 512 * 8
    assert int(got.sum()) == 512 * 8


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_router_id_dtype_does_not_change_the_count(dtype):
    """Routers in the tree return either width, and neither may be mutated in place.

    The reference copies before masking precisely so it cannot write through to the
    tensor
    the router just handed back; the kernel must not write to its input either.
    """
    torch.manual_seed(5)
    ids = torch.randint(0, NUM_LOGICAL, (128, 8), dtype=dtype, device="cuda")
    before = ids.clone()
    ref, got = _run_both(ids, unpadded=100)

    assert torch.equal(ref, got)
    assert torch.equal(ids, before), "the router's tensor must not be modified"


@requires_gpu
def test_matches_the_reference_over_many_random_cases():
    """Randomised sweep, because the failure mode is a wrong number in a corner.

    Shapes, padding boundaries and expert counts all vary together; a bug that only
    shows
    up when the flattened length is not a multiple of the kernel's block size is exactly
    what a fixed-shape test would miss.
    """
    torch.manual_seed(1234)
    for _ in range(200):
        num_tokens = int(torch.randint(0, 300, (1,)).item())
        topk = int(torch.randint(1, 9, (1,)).item())
        num_logical = int(torch.randint(8, 257, (1,)).item())
        unpadded = int(torch.randint(0, num_tokens + 1, (1,)).item())
        ids = torch.randint(
            -4, num_logical + 4, (num_tokens, topk), dtype=torch.int64, device="cuda"
        )
        ref, got = _run_both(ids, unpadded=unpadded, num_logical=num_logical)
        assert torch.equal(ref, got), (
            f"mismatch at num_tokens={num_tokens} topk={topk} "
            f"num_logical={num_logical} unpadded={unpadded}"
        )


@requires_gpu
@pytest.mark.parametrize("shape", [(), (1,)])
def test_accepts_the_unpadded_scalar_in_either_shape(shape):
    """The runtime passes an element of a *list* of per-ubatch scalars.

    That element is 0-dim, while a hand-written test reaches for `[n]`. Both must work:
    the mismatch surfaces as Triton refusing to specialize an argument rather than as a
    wrong count, and only under the real caller.
    """
    ids = torch.randint(0, NUM_LOGICAL, (32, 8), dtype=torch.int64, device="cuda")
    unpadded = torch.zeros(shape, dtype=torch.int32, device="cuda")
    unpadded.fill_(20)
    out = torch.zeros(NUM_LOGICAL, dtype=torch.int32, device="cuda")

    count_logical_experts_triton(ids, unpadded, NUM_LOGICAL, out)

    assert int(out.sum()) == 20 * 8


@requires_gpu
def test_the_output_is_int32_and_zeroed_by_the_callee():
    """The snapshot is int32, and a stale buffer would silently inflate counts."""
    ids = torch.randint(0, NUM_LOGICAL, (16, 8), dtype=torch.int64, device="cuda")
    unpadded = torch.tensor([16], dtype=torch.int32, device="cuda")
    out = torch.full((NUM_LOGICAL,), 999, dtype=torch.int32, device="cuda")

    count_logical_experts_triton(ids, unpadded, NUM_LOGICAL, out)

    assert out.dtype == torch.int32
    assert int(out.sum()) == 16 * 8, "a pre-filled buffer must not leak into the count"
