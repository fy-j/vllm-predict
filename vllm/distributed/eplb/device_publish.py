# SPDX-License-Identifier: Apache-2.0 SPDX-FileCopyrightText: Copyright contributors to
# the vLLM project
"""Publish a placement into the routing maps without reading the plan on the host.

Ticket 06. The plan is a device tensor and the transfer is issued from a kernel, so
publishing is the last place the host would have to learn what was placed. Routing
already consumes its maps as device tensors, so a scatter suffices — but every index in
that scatter comes from a plan value, and the obvious spelling is one `int()` per field.
That is one synchronisation per predicted layer, which is 5.28 ms of collective waiting
and 70% of what prediction costs: the entire quantity this path exists to remove.

Two techniques do all the work here.

**Write unconditionally at a clamped index, and choose the value with `where`.** A
guarded write needs a branch on device data, and there is no spare row to aim a no-op
at. Writing back what is already there is idempotent, so `index_put_` at `max(index, 0)`
with the old value under the mask has the same effect as not writing at all.

**Revert before placing.** A slot handed from one expert to another appears in both
sets, and reverting afterwards would clear the row just claimed — leaving the map
pointing at a canonical copy while the weights sat in a slot nothing routes to. The host
path orders it the same way for the same reason.

This handles **one replica per layer**, which is `max_replicas_per_layer = 1`, the
measured default: at cap 1 a budget covers all 43 reachable layers and removes 48.0% of
critical-path excess against a global-ranking oracle's 48.3%, where a cap of 2 covers 22
layers and removes 23.5%. A higher cap needs a different residency shape, so it is
rejected rather than silently mishandled.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerResidency:
    """Which expert occupies which rank's replica slot, for one layer, on the device.

    The host cannot hold this once the plan stops reaching it, and it is not only an
    accounting convenience: "already resident" is what makes a transfer free, which is
    what lets coverage ratchet up across forwards to every reachable layer.
    Reconstructing it by reading the plan would restore the synchronisation.

    Attributes:
        state: `[2]` int64, `(logical expert, target rank)`, both -1 when the layer
            holds no replica.
    """

    state: torch.Tensor

    @staticmethod
    def empty(device: torch.device) -> LayerResidency:
        """A layer holding no replica, as startup normalization leaves it."""
        return LayerResidency(torch.full((2,), -1, dtype=torch.int64, device=device))

    @property
    def expert(self) -> torch.Tensor:
        return self.state[0]

    @property
    def target(self) -> torch.Tensor:
        return self.state[1]


def publish_plan_on_device(
    plan: torch.Tensor,
    residency: LayerResidency,
    logical_to_physical: torch.Tensor,
    logical_replica_count: torch.Tensor,
    source_local: torch.Tensor,
    layout: torch.Tensor,
    per_rank_experts: int,
    replica_slots_per_rank: int,
    source_rank: int,
    slot: int = 0,
) -> torch.Tensor:
    """Make one layer's active replica exactly what `plan` asks for.

    `plan` is the layer's complete desired state, not an addition to it: `found == 0`
    means "this layer should hold no replica", and publishing that is what reverts
    whatever the last forward left. Skipping the call for a layer that planned nothing —
    as an earlier runner did — leaves reversion running only on layers that happened to
    receive a new placement, and active replicas then accumulate to 53-66 per forward
    against a budget of 43.

    Args:
        plan: `[4]` int64 `(found, logical_expert, target_rank, moved_x2)` from
            `plan_one_layer_on_device`, on the same device as the maps.
        residency: This layer's slot occupancy, read and updated in place.
        logical_to_physical: `[num_logical, width]` for this layer, mutated.
        logical_replica_count: `[num_logical]` for this layer, mutated.
        source_local: `[num_logical, 1]` map routing actually reads, mutated. Writing
            the global pair instead transfers the replica, describes it correctly, and
            publishes it where nothing reads: a measured run then activated 131 replicas
            per forward and removed 0.6% of prefill excess against an oracle's 35.1%.
        layout: `[ep_size, per_rank_experts + replica_slots_per_rank]`
            physical-to-logical rows for this layer, mutated. This is what
            `active_replicas` reads, so a row left stale here disagrees with routing
            about what is live.
        per_rank_experts: Canonical experts per rank.
        replica_slots_per_rank: Replica rows per rank.
        source_rank: The EP rank whose source-local map this is.
        slot: Which replica row of the target rank to use.

    Returns: A `[]` int64 device tensor, 1 when this call placed an expert that was not
    already resident and therefore needs a weight transfer, 0 otherwise. The caller
    charges its budget with this rather than with the plan, so a replica already in
    place costs nothing.

    Raises:
        ValueError: If the map is too narrow to hold a second copy, which would
            otherwise write out of bounds or silently drop the replica.
    """
    if logical_to_physical.shape[-1] < 2:
        raise ValueError(
            f"logical_to_physical is {logical_to_physical.shape[-1]} wide, so a second "
            f"copy cannot be recorded and the replica would never be routed to. "
            f"Widen it to at least 2."
        )
    stride = per_rank_experts + replica_slots_per_rank
    replica_column = per_rank_experts + slot
    # Column views, so every index that varies is a row and no device-side constant has
    # to be created. `torch.tensor(x, device="cuda")` is a host-to-device copy and
    # counts as a synchronising operation too, so building the column index on the
    # device would put back a smaller version of what this module removes.
    physical_first = logical_to_physical[:, 0]
    physical_second = logical_to_physical[:, 1]
    local_first = source_local[:, 0]
    slot_column = layout[:, replica_column]

    found = plan[0] > 0
    new_expert = plan[1]
    new_target = plan[2]
    old_expert = residency.expert
    old_target = residency.target

    resident = old_expert >= 0
    keep = found & resident & (old_expert == new_expert) & (old_target == new_target)
    revert = resident & ~keep
    place = found & ~keep

    # Reverted first. A slot handed from one expert to another appears in both sets, and
    # reverting afterwards would clear the row just claimed — leaving the map pointing
    # at a canonical copy while the weights sat in a slot nothing routes to.
    old_index = old_expert.clamp_min(0)
    old_canonical = _canonical_row(old_index, per_rank_experts, stride)
    minus_one = -torch.ones_like(old_canonical)
    _put(physical_first, old_index, revert, old_canonical)
    _put(physical_second, old_index, revert, minus_one)
    _put(logical_replica_count, old_index, revert, torch.ones_like(old_canonical))
    _put(local_first, old_index, revert, old_canonical)
    _put(slot_column, old_target.clamp_min(0), revert, minus_one)

    # Then placed. Copies are ordered by ascending physical row, because that is the
    # order `compute_logical_maps` discovers them in as it walks the slots — and a
    # replica can land *below* its canonical row, so which index holds which copy is not
    # fixed. Assuming `[canonical, replica]` puts the wrong row in the map for exactly
    # those cases, and the planner produces them routinely.
    new_index = new_expert.clamp_min(0)
    new_canonical = _canonical_row(new_index, per_rank_experts, stride)
    replica_row = new_target.clamp_min(0) * stride + replica_column
    lower = torch.minimum(new_canonical, replica_row)
    upper = torch.maximum(new_canonical, replica_row)
    _put(physical_first, new_index, place, lower)
    _put(physical_second, new_index, place, upper)
    _put(logical_replica_count, new_index, place, torch.full_like(new_canonical, 2))
    # `build_source_local_physical_map` picks copy `source_rank % count`, so half the
    # ranks keep the canonical row and half take the replica. Publishing the copy this
    # rank does not route to is the defect that activated 131 replicas and sent them no
    # tokens.
    _put(local_first, new_index, place, lower if source_rank % 2 == 0 else upper)
    _put(slot_column, new_target.clamp_min(0), place, new_expert)

    residency.state.copy_(
        torch.where(
            found,
            torch.stack([new_expert, new_target]),
            torch.full_like(residency.state, -1),
        )
    )
    return place.to(torch.int64)


def _put(
    column: torch.Tensor,
    row: torch.Tensor,
    mask: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Write `value` at `row` of `column`, but only where `mask` holds.

    A guarded write needs a branch on device data and there is no spare row to aim a
    no-op at, so the write always happens and the value is chosen instead: under a false
    mask it is whatever was already there, which is idempotent.

    `column.index_copy_` and not `column[row] = value`: the latter reads as the same
    thing and goes through a path that synchronises, which is the one cost this module
    exists to avoid.
    """
    index = row.view(1)
    current = column.index_select(0, index).squeeze(0)
    column.index_copy_(0, index, torch.where(mask, value, current).view(1))


def _canonical_row(
    logical_expert: torch.Tensor, per_rank_experts: int, stride: int
) -> torch.Tensor:
    """Device counterpart of `canonical_row_of`, kept as a tensor throughout."""
    owner = logical_expert // per_rank_experts
    return owner * stride + logical_expert % per_rank_experts
