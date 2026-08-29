# SPDX-License-Identifier: Apache-2.0 SPDX-FileCopyrightText: Copyright contributors to
# the vLLM project
"""Where one expert's weight tensors sit inside the shared staging workspace.

Ticket 05. A one-sided put writes bytes into a peer's staging buffer and the peer copies
them into its replica row. The buffer is flat while an expert is several tensors, so
both sides must agree byte for byte on the layout.

Kept as one function used by both directions, because the failure is silent: disagree on
the order and the replica row takes `w2`'s bytes where `w13` belongs. Nothing raises, no
shape check fires, the MoE kernel reads plausible floats, and the output is quietly
wrong for whichever source ranks route to that copy.

**The workspace holds one expert, not one layer, and is shared by every layer**, because
at most one transfer is in flight. Sizing it per layer would multiply it by the row
count for nothing; the ordering constraint that makes the sharing safe — a transfer for
a later layer must not overwrite a replica a nearer layer still needs — belongs to the
caller and is stated in spec section 11.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def staging_layout(tensors: Sequence[torch.Tensor]) -> list[tuple[int, int]]:
    """Byte offset and length for each of one expert's weight tensors.

    Tensors are laid out back to back in the order given, so the caller must present
    them in the same order on both sides of a transfer. Sizes come from `element_size`,
    not element counts, because an expert may mix widths: FP8 weights carry their block
    scales in a wider dtype.

    Args:
        tensors: One layer's expert weight tensors, each `[rows, ...]`. Only row shape
            and dtype are read, never the contents.

    Returns: `(offset, nbytes)` per tensor, in the given order.

    Raises:
        ValueError: If there are no tensors, or if a tensor's row is not contiguous. The
            latter is caught here so the message names the tensor, rather than surfacing
            from a `view` deep inside the transfer as a complaint about strides.
    """
    if not tensors:
        raise ValueError(
            "an expert has no weight tensors to stage, which is a wiring error rather "
            "than a zero-byte transfer"
        )
    layout: list[tuple[int, int]] = []
    offset = 0
    for index, tensor in enumerate(tensors):
        row = tensor[0]
        if not row.is_contiguous():
            raise ValueError(
                f"expert weight tensor {index} has a non-contiguous row "
                f"{tuple(row.shape)} with strides {row.stride()}; the staging copy "
                f"reinterprets a row as bytes and cannot do that for a strided view"
            )
        nbytes = row.numel() * tensor.element_size()
        layout.append((offset, nbytes))
        offset += nbytes
    return layout


def staging_workspace_bytes(per_layer: Sequence[Sequence[torch.Tensor]]) -> int:
    """Bytes needed to stage one expert of the largest layer.

    Takes the maximum rather than the first layer's size: this model's layers are
    uniform, but a model whose layers differ would otherwise get a buffer sized from
    whichever layer came first, and the larger layers would write past its end.
    """
    return max(
        sum(nbytes for _, nbytes in staging_layout(tensors)) for tensors in per_layer
    )
