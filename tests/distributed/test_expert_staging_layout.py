# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How one expert's weight tensors pack into the shared staging workspace.

Ticket 05. A one-sided put moves bytes into a peer's staging buffer, and the peer then
copies them into its replica row. The buffer is flat, but an expert is several tensors —
`w13` and `w2` for this model — so both sides have to agree on where each tensor's bytes
sit.

This is pinned by tests because the failure is silent. Disagree on the order and the
replica row receives `w2`'s bytes where `w13` belongs: nothing raises, no shape check
fires, the MoE kernel reads plausible-looking floats, and the output is quietly wrong
for the source ranks routed to that copy. Five defects in this project were of that
shape, and the cheapest guard is arithmetic that cannot be got wrong in two places at
once — one function, used by the send and the receive.
"""

import pytest
import torch

from vllm.distributed.eplb.expert_staging import (
    staging_layout,
    staging_workspace_bytes,
)


def _weights(rows=17, hidden=2048, inter=768, dtype=torch.bfloat16):
    """One layer's expert weights, shaped as this model's are."""
    return [
        torch.zeros(rows, 2 * inter, hidden, dtype=dtype),
        torch.zeros(rows, hidden, inter, dtype=dtype),
    ]


def test_the_layout_covers_every_tensor_exactly_once():
    """Offsets tile the buffer with no gap and no overlap.

    A gap would leave stale bytes inside the replica row; an overlap would have one
    tensor write over another. Neither raises, so both are asserted here.
    """
    tensors = _weights()
    layout = staging_layout(tensors)

    assert len(layout) == len(tensors)
    expected_offset = 0
    for (offset, nbytes), tensor in zip(layout, tensors):
        assert offset == expected_offset, "tensors must be laid out back to back"
        assert nbytes == tensor[0].numel() * tensor.element_size()
        expected_offset += nbytes


def test_the_workspace_is_sized_for_one_expert_not_one_layer():
    """The workspace holds a single expert, because one transfer is in flight at a time.

    Sizing it per layer would multiply the memory by the row count for no reason; sizing
    it per model would multiply it again by the layer count.
    """
    tensors = _weights(rows=17)
    one_expert = sum(t[0].numel() * t.element_size() for t in tensors)

    assert staging_workspace_bytes([tensors]) == one_expert
    assert staging_workspace_bytes([tensors]) * 17 == sum(
        t.numel() * t.element_size() for t in tensors
    )


def test_the_workspace_fits_the_largest_layer():
    """Layers are uniform in this model, but the workspace must not assume it.

    A model whose layers differ would otherwise get a buffer sized from whichever layer
    happened to be first, and the larger layers would write past its end.
    """
    small = _weights(inter=768)
    large = _weights(inter=1024)

    assert staging_workspace_bytes([small, large]) == staging_workspace_bytes([large])
    assert staging_workspace_bytes([small, large]) > staging_workspace_bytes([small])


def test_a_round_trip_through_a_flat_buffer_preserves_the_expert():
    """Pack an expert out, unpack it back, and require byte equality.

    This is the property the transfer needs and the one a wrong offset breaks. Running
    it through a real buffer rather than checking arithmetic catches an off-by-one that
    a comparison of offsets would agree with.
    """
    torch.manual_seed(0)
    source = [torch.randn(4, 6, 8, dtype=torch.float32), torch.randn(4, 8, 3)]
    destination = [torch.zeros_like(t) for t in source]
    buffer = torch.zeros(staging_workspace_bytes([source]), dtype=torch.uint8)
    row, target_row = 2, 3

    for (offset, nbytes), tensor in zip(staging_layout(source), source):
        buffer[offset : offset + nbytes] = tensor[row].reshape(-1).view(torch.uint8)
    for (offset, nbytes), tensor in zip(staging_layout(destination), destination):
        flat = tensor[target_row].reshape(-1).view(torch.uint8)
        flat.copy_(buffer[offset : offset + nbytes])

    for src, dst in zip(source, destination):
        assert torch.equal(src[row], dst[target_row])


def test_tensors_of_different_dtypes_are_measured_in_bytes_not_elements():
    """An expert may mix widths — FP8 weights carry BF16 or FP32 block scales.

    Counting elements rather than bytes would size the buffer wrongly the moment DSV4's
    quantised experts arrive, which is ticket 10.
    """
    tensors = [
        torch.zeros(4, 16, dtype=torch.float8_e4m3fn),
        torch.zeros(4, 2, dtype=torch.float32),
    ]

    layout = staging_layout(tensors)

    assert layout[0][1] == 16 * 1
    assert layout[1][1] == 2 * 4
    assert staging_workspace_bytes([tensors]) == 16 + 8


def test_an_empty_expert_list_is_rejected():
    """A layer with no weight tensors is a wiring error, not a zero-byte transfer."""
    with pytest.raises(ValueError, match="no weight tensors"):
        staging_layout([])


def test_a_non_contiguous_row_is_rejected_rather_than_silently_reordered():
    """`view(torch.uint8)` on a non-contiguous row would raise deep inside the transfer.

    Rejecting it here says which tensor is at fault, instead of failing at the copy with
    a message about strides.
    """
    tensor = torch.zeros(4, 6, 8).transpose(1, 2)

    with pytest.raises(ValueError, match="contiguous"):
        staging_layout([tensor])
