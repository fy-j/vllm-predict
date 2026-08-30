# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan, charge and publish a placement in two kernels instead of 132.

Ticket 07's profile found where the feature's remaining cost is, and it is not the
transfer — `put_expert` runs at a p50 of 1.2 us. Per placed layer the device path
launched **46 kernels to plan, 19 to charge the budget and 67 to publish**, all of them
tiny elementwise and reduce ops over 128 integers, and in an eager engine each one is
also a host-side dispatch. Measured against the prediction-only arm, placement added
**53 kernel launches and 0.95 ms of host time per layer** — about 42 ms per forward —
and the GPU sat idle for 78% of a prefill window because the host could not feed it.

This is ticket 03's defect in a new place and takes the same fix. Moving the plan onto
the device removed 5.28 ms per layer of host *synchronisation* and put 0.95 ms per layer
of host *dispatch* back; that was the right trade where the synchronisation dominated
and a bad one otherwise. Fusing costs nothing in either direction.

Two kernels, not one, because the halves run at different points in the forward: the
plan is made at the predicting layer's MoE tail and the publish at the target layer's
MoE head, one Attention block later, which is the window the transfer hides in.

**Everything here is exact integer arithmetic in doubled units.** The retained
implementations use float64 because one replica sheds exactly *half* an expert. Doubling
the load makes that an integer, so every comparison means what it does in Python and no
rank can round differently from another — which matters more than speed, since each rank
derives the plan locally and a plan that differs by rank pairs a send with no receive.

The tensor versions in `predictive_planner` and `device_publish` stay as the oracles
these are tested against, exactly as the host planner is the oracle for those.
"""

from __future__ import annotations

import math

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _plan_and_charge_kernel(
    load_ptr,
    residency_ptr,
    budget_ptr,
    spent_ptr,
    placed_total_ptr,
    transfer_ptr,
    publish_ptr,
    ep_size,
    per_rank,
    min_tokens_x2,
    EP: tl.constexpr,
    PR: tl.constexpr,
):
    """One layer's placement decision and its budget charge, in one program.

    Every reduction is over at most `ep_size * per_rank` integers, so a single program
    does all of it and no cross-block agreement is needed — which also removes the last
    place where two ranks could reduce in a different order and disagree.
    """
    ranks = tl.arange(0, EP)
    offs = tl.arange(0, PR)
    rank_ok = ranks < ep_size
    off_ok = offs < per_rank

    # Everything runs in doubled units, so a shed half-expert is an integer. Note what
    # doubling does to the shed amount: `moved` is half an expert, so `2 * moved` is the
    # expert's own count — the *undoubled* load. Using the doubled load here sheds twice
    # what a replica can, which overshoots the target and refused 40 of 200 placements
    # the tensor planner accepts. Padded lanes contribute zero.
    index = ranks[:, None] * per_rank + offs[None, :]
    valid = rank_ok[:, None] & off_ok[None, :]
    load1 = tl.load(load_ptr + index, mask=valid, other=0).to(tl.int64)
    load2 = load1 * 2

    rank_load2 = tl.sum(load2, axis=1)
    peak_load2 = tl.max(tl.where(rank_ok, rank_load2, -1))
    # The *first* maximum, which is what `max(range(n), key=...)` returns on the host,
    # so ties go to the lower rank id there and here alike.
    peak = tl.min(tl.where(rank_ok & (rank_load2 == peak_load2), ranks, EP))
    moved2 = tl.sum(tl.where(ranks[:, None] == peak, load1, 0), axis=0)

    # Every (expert offset, target rank) pair against every rank's resulting load. The
    # peak sheds `moved2` and the target gains it; nothing else moves.
    trial = (
        rank_load2[None, None, :]
        - moved2[:, None, None] * (ranks == peak)[None, None, :].to(tl.int64)
        + moved2[:, None, None]
        * (ranks[None, :, None] == ranks[None, None, :]).to(tl.int64)
    )
    # -1 rather than a large negative: every real trial load is non-negative, because a
    # rank cannot shed more than it holds.
    gain2 = peak_load2 - tl.max(tl.where(rank_ok[None, None, :], trial, -1), axis=2)

    admissible = (
        (moved2[:, None] >= min_tokens_x2)
        & (moved2[:, None] > 0)
        & (ranks[None, :] != peak)
        & (gain2 > 0)
        & off_ok[:, None]
        & rank_ok[None, :]
    )
    total2 = tl.sum(tl.where(rank_ok, rank_load2, 0))
    found = tl.where((tl.max(admissible.to(tl.int64)) > 0) & (total2 > 0), 1, 0)

    # Strongest gain, then the lowest expert id, then the lowest target id. The
    # flattened order is expert-major, so the lowest flat index at the best gain is the
    # planner's tie-break, and taking it from an equality mask keeps the choice
    # independent of how the reduction was scheduled.
    lowest = -9223372036854775807
    masked = tl.where(admissible, gain2, lowest)
    best = tl.max(masked)
    flat = offs[:, None] * ep_size + ranks[None, :]
    winner = tl.min(tl.where(masked == best, flat, EP * PR))
    offset = winner // ep_size
    target = winner % ep_size

    # Zeroed rather than left as garbage when nothing was found: the winner index is 0
    # over an all-rejected mask, so these would otherwise name a placement that was
    # refused.
    expert = tl.where(found == 1, peak * per_rank + offset, 0)
    target = tl.where(found == 1, target, 0)
    moved_x2 = tl.where(found == 1, tl.sum(tl.where(offs == offset, moved2, 0)), 0)

    # Charged for what moves, not for what is planned: a replica already resident needs
    # no transfer, so in a steady state the honest charge is zero and the budget bounds
    # churn rather than coverage.
    res_expert = tl.load(residency_ptr + 0)
    res_target = tl.load(residency_ptr + 1)
    resident = tl.where(res_expert >= 0, 1, 0)
    same = tl.where((res_expert == expert) & (res_target == target), 1, 0)
    keep = found * resident * same
    needs = found * (1 - keep)
    spent = tl.load(spent_ptr)
    affordable = needs * tl.where(spent < tl.load(budget_ptr), 1, 0)
    tl.store(spent_ptr, spent + affordable)
    tl.store(placed_total_ptr, tl.load(placed_total_ptr) + affordable)

    # The transfer moves only what was charged; the publish describes the layer's whole
    # desired state, so a resident replica is still published and an empty plan reverts.
    tl.store(transfer_ptr + 0, affordable)
    tl.store(transfer_ptr + 1, expert)
    tl.store(transfer_ptr + 2, target)
    tl.store(transfer_ptr + 3, moved_x2)
    tl.store(publish_ptr + 0, tl.maximum(keep, affordable))
    tl.store(publish_ptr + 1, expert)
    tl.store(publish_ptr + 2, target)
    tl.store(publish_ptr + 3, moved_x2)


@triton.jit
def _publish_kernel(
    plan_ptr,
    residency_ptr,
    l2p_ptr,
    count_ptr,
    local_ptr,
    layout_ptr,
    l2p_stride,
    local_stride,
    layout_stride,
    per_rank,
    stride,
    replica_column,
    take_upper,
):
    """Make one layer's routing maps say exactly what `plan` asks for.

    Masked stores replace the retained version's clamp-and-`where` dance: it wrote
    unconditionally at a clamped index with the old value under a false mask, because
    `index_copy_` has no mask. A Triton store does, so a guarded write is guarded.

    **Revert before placing.** A slot handed from one expert to another appears in both
    sets, and reverting afterwards would clear the row just claimed — leaving the map
    pointing at a canonical copy while the weights sat in a slot nothing routes to.
    Stores in one program run in program order, which is what holds that ordering here.
    """
    found = tl.where(tl.load(plan_ptr + 0) > 0, 1, 0)
    new_expert = tl.load(plan_ptr + 1)
    new_target = tl.load(plan_ptr + 2)
    old_expert = tl.load(residency_ptr + 0)
    old_target = tl.load(residency_ptr + 1)

    resident = tl.where(old_expert >= 0, 1, 0)
    same = tl.where((old_expert == new_expert) & (old_target == new_target), 1, 0)
    keep = found * resident * same
    revert = (resident * (1 - keep)) == 1
    place = (found * (1 - keep)) == 1

    old_index = tl.maximum(old_expert, 0)
    old_canonical = (old_index // per_rank) * stride + old_index % per_rank
    tl.store(l2p_ptr + old_index * l2p_stride + 0, old_canonical, mask=revert)
    tl.store(l2p_ptr + old_index * l2p_stride + 1, -1, mask=revert)
    tl.store(count_ptr + old_index, 1, mask=revert)
    tl.store(local_ptr + old_index * local_stride, old_canonical, mask=revert)
    tl.store(
        layout_ptr + tl.maximum(old_target, 0) * layout_stride + replica_column,
        -1,
        mask=revert,
    )

    # Copies are ordered by ascending physical row, because that is the order
    # `compute_logical_maps` discovers them in as it walks the slots — and a replica can
    # land *below* its canonical row, which the planner produces routinely. Assuming
    # `[canonical, replica]` puts the wrong row in the map for exactly those cases.
    new_index = tl.maximum(new_expert, 0)
    new_canonical = (new_index // per_rank) * stride + new_index % per_rank
    replica_row = tl.maximum(new_target, 0) * stride + replica_column
    lower = tl.minimum(new_canonical, replica_row)
    upper = tl.maximum(new_canonical, replica_row)
    tl.store(l2p_ptr + new_index * l2p_stride + 0, lower, mask=place)
    tl.store(l2p_ptr + new_index * l2p_stride + 1, upper, mask=place)
    tl.store(count_ptr + new_index, 2, mask=place)
    # `build_source_local_physical_map` picks copy `source_rank % count`, so half the
    # ranks keep the canonical row and half take the replica. Publishing the copy this
    # rank does not route to is the defect that activated 131 replicas and sent them no
    # tokens.
    tl.store(
        local_ptr + new_index * local_stride,
        tl.where(take_upper == 1, upper, lower),
        mask=place,
    )
    tl.store(
        layout_ptr + tl.maximum(new_target, 0) * layout_stride + replica_column,
        new_expert,
        mask=place,
    )

    tl.store(residency_ptr + 0, tl.where(found == 1, new_expert, -1))
    tl.store(residency_ptr + 1, tl.where(found == 1, new_target, -1))


def plan_and_charge_fused(
    predicted: torch.Tensor,
    residency: torch.Tensor,
    budget: torch.Tensor,
    spent: torch.Tensor,
    placed_total: torch.Tensor,
    transfer_plan: torch.Tensor,
    publish_plan: torch.Tensor,
    ep_size: int,
    min_tokens: float,
) -> None:
    """Plan one layer, charge the budget, and write both plan rows. One launch.

    Replaces `plan_one_layer_on_device` plus the residency and budget arithmetic around
    it — 65 launches measured, against this one.

    Args:
        predicted: `[num_logical]` predicted load, integer-valued and already reduced
            across the EP group, so every rank sees identical values.
        residency: `[2]` int64 `(expert, target)` this layer holds, `-1` for none. Read,
            not written; the publish updates it.
        budget: `[]` int64 transfers this forward may still pay for.
        spent: `[]` int64 charged so far this forward, updated in place.
        placed_total: `[]` int64 lifetime counter, updated in place. It is the only
            evidence a replica was placed at all once the host stops learning the plan.
        transfer_plan: `[4]` int64 written with `(charged, expert, target, moved_x2)`.
        publish_plan: `[4]` int64 written with `(desired, expert, target, moved_x2)`,
            where `desired` covers a replica already resident, which needs no
            transfer.
        ep_size: EP group size. Expert `e` belongs to rank `e // (num_logical/ep_size)`.
        min_tokens: Refuse a placement moving less than this. Below one `BLOCK_SIZE_M` a
            replica saves no block, so it saves no time.

    Raises:
        ValueError: If the expert count does not divide the EP size, which would make
            the rank of an expert ambiguous.
    """
    num_logical = predicted.numel()
    if num_logical % ep_size != 0:
        raise ValueError(
            f"{num_logical} logical experts do not divide across {ep_size} EP ranks."
        )
    per_rank = num_logical // ep_size
    _plan_and_charge_kernel[(1,)](
        predicted,
        residency,
        budget,
        spent,
        placed_total,
        transfer_plan,
        publish_plan,
        ep_size,
        per_rank,
        # `moved >= min_tokens` for integer `moved2 = 2m` is `moved2 >= ceil(2m)`:
        # integer comparisons throughout, and the same placements pass.
        math.ceil(2 * min_tokens),
        EP=triton.next_power_of_2(ep_size),
        PR=triton.next_power_of_2(per_rank),
    )


def publish_plan_fused(
    plan: torch.Tensor,
    residency: torch.Tensor,
    logical_to_physical: torch.Tensor,
    logical_replica_count: torch.Tensor,
    source_local: torch.Tensor,
    layout: torch.Tensor,
    per_rank_experts: int,
    replica_slots_per_rank: int,
    source_rank: int,
    slot: int = 0,
) -> None:
    """Publish one layer's complete desired state. One launch.

    Replaces `publish_plan_on_device`, which measured 67 launches for the same writes.
    Arguments and semantics are identical to it, which the tests assert by equality
    against it over a randomised sequence of plans.

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
    _publish_kernel[(1,)](
        plan,
        residency,
        logical_to_physical,
        logical_replica_count,
        source_local,
        layout,
        logical_to_physical.stride(0),
        source_local.stride(0),
        layout.stride(0),
        per_rank_experts,
        per_rank_experts + replica_slots_per_rank,
        per_rank_experts + slot,
        source_rank % 2,
    )
