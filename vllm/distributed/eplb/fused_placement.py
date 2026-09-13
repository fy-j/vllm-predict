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

    # Ticket 15. A layer that wanted a different replica and could not afford it keeps
    # the one it has: its weights are still in the row, keeping them costs nothing, and
    # the alternative is worse than doing nothing. Before this, `publish` took the
    # revert branch and cleared the layer, so at the default budget of 4 over 44
    # reachable layers a traffic shift re-placed 4 and **reverted 40**, then needed ten
    # forwards to ratchet coverage back -- the budget bounding coverage, which is
    # exactly what the charge-for-what-moves rule above exists to prevent.
    #
    # Narrow on purpose: `found == 0` still reverts. That is the planner saying no
    # placement helps this layer's load, which is a decision and not a budget accident;
    # the snapshot counts logical experts, so it is unaffected by what is resident and
    # cannot be a self-fulfilling one.
    hold = found * (1 - keep) * (1 - affordable) * resident
    pub_found = tl.maximum(tl.maximum(keep, affordable), hold)
    pub_expert = tl.where(hold == 1, res_expert, expert)
    pub_target = tl.where(hold == 1, res_target, target)
    pub_moved = tl.where(hold == 1, 0, moved_x2)

    # The transfer moves only what was charged; the publish describes the layer's whole
    # desired state, so a resident replica is still published and an empty plan reverts.
    tl.store(transfer_ptr + 0, affordable)
    tl.store(transfer_ptr + 1, expert)
    tl.store(transfer_ptr + 2, target)
    tl.store(transfer_ptr + 3, moved_x2)
    tl.store(publish_ptr + 0, pub_found)
    tl.store(publish_ptr + 1, pub_expert)
    tl.store(publish_ptr + 2, pub_target)
    tl.store(publish_ptr + 3, pub_moved)


@triton.jit
def _plan_replicas_kernel(
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
    CAP: tl.constexpr,
    SLOTS: tl.constexpr,
):
    """One layer's `CAP` placements, their budget charge and both plan tables.

    `_plan_and_charge_kernel` generalised. The loop unrolls, and each step's result is
    folded into a `SLOTS`-wide register vector rather than a Python list, which the
    Triton frontend turns into a tuple.

    Two orderings matter and they are deliberately different. **The budget is charged in
    gain order**, so when a layer cannot afford every replica it keeps the ones worth
    most. **The rows are stored in expert order**, so a slot keeps its physical row
    across forwards; a set that reappeared in a different order would otherwise move
    between rows and re-transfer, and four times the transfer rate at unchanged traffic
    measured +27% mean TTFT.

    Residency is matched **as a set**, not slot by slot: whether a replica is already
    held is a property of the layer, not of the position it happens to occupy, and a
    positional test turns any reordering into a silent wholesale re-transfer.

    Asserted bit-identical to `plan_layer_replicas_on_device`, which is retained as the
    oracle. A plan that differs by rank pairs a put with a peer expecting nothing.
    """
    ranks = tl.arange(0, EP)
    offs = tl.arange(0, PR)
    slots = tl.arange(0, SLOTS)
    rank_ok = ranks < ep_size
    off_ok = offs < per_rank
    valid = rank_ok[:, None] & off_ok[None, :]
    slot_ok = slots < CAP

    index = ranks[:, None] * per_rank + offs[None, :]
    owned2 = tl.load(load_ptr + index, mask=valid, other=0).to(tl.int64) * 2
    # A replica's half lands on a rank that does not own that logical expert, so it is
    # carried apart: adding it to a column of `owned2` would corrupt whatever expert the
    # target holds there, invisibly, because the rank totals would still look right.
    extra2 = tl.zeros((EP,), dtype=tl.int64)
    spent_mask = tl.zeros((EP, PR), dtype=tl.int32)
    total2 = tl.sum(tl.where(valid, owned2, 0))

    res_expert = tl.load(residency_ptr + slots * 2 + 0, mask=slot_ok, other=-1)
    res_target = tl.load(residency_ptr + slots * 2 + 1, mask=slot_ok, other=-1)

    out_found = tl.zeros((SLOTS,), dtype=tl.int64)
    out_expert = tl.zeros((SLOTS,), dtype=tl.int64)
    out_target = tl.zeros((SLOTS,), dtype=tl.int64)
    out_moved = tl.zeros((SLOTS,), dtype=tl.int64)
    out_charged = tl.zeros((SLOTS,), dtype=tl.int64)
    out_desired = tl.zeros((SLOTS,), dtype=tl.int64)

    budget = tl.load(budget_ptr)
    spent_now = tl.load(spent_ptr)
    placed = tl.load(placed_total_ptr)

    for i in range(CAP):
        rank_load2 = tl.sum(owned2, axis=1) + extra2
        peak_load2 = tl.max(tl.where(rank_ok, rank_load2, -1))
        peak = tl.min(tl.where(rank_ok & (rank_load2 == peak_load2), ranks, EP))
        on_peak = ranks[:, None] == peak
        # One replica per logical expert: a spent expert offers nothing to move and so
        # fails `moved2 > 0` without a rule of its own. Re-picking one would need a
        # third physical copy of it; refusing costs 2.82% of the benefit, measured.
        usable = (spent_mask == 0) & valid
        # In doubled units a shed half is the expert's own raw count, which is
        # `owned2 // 2` and exact because `owned2` is twice an integer. Summing the
        # doubled value sheds twice what a replica can, the overshoot that refused 40 of
        # 200 placements when this file was first written.
        moved2 = tl.sum(tl.where(on_peak & usable, owned2 // 2, 0), axis=0)

        trial = (
            rank_load2[None, None, :]
            - moved2[:, None, None] * (ranks == peak)[None, None, :].to(tl.int64)
            + moved2[:, None, None]
            * (ranks[None, :, None] == ranks[None, None, :]).to(tl.int64)
        )
        gain2 = peak_load2 - tl.max(tl.where(rank_ok[None, None, :], trial, -1), axis=2)

        admissible = (
            (moved2[:, None] >= min_tokens_x2)
            & (moved2[:, None] > 0)
            & (ranks[None, :] != peak)
            & (gain2 > 0)
            & off_ok[:, None]
            & rank_ok[None, :]
        )
        found = tl.where((tl.max(admissible.to(tl.int64)) > 0) & (total2 > 0), 1, 0)

        lowest = -9223372036854775807
        masked = tl.where(admissible, gain2, lowest)
        best = tl.max(masked)
        flat = offs[:, None] * ep_size + ranks[None, :]
        winner = tl.min(tl.where(masked == best, flat, EP * PR))
        offset = winner // ep_size
        target = winner % ep_size

        expert = tl.where(found == 1, peak * per_rank + offset, 0)
        target = tl.where(found == 1, target, 0)
        moved_x2 = tl.where(found == 1, tl.sum(tl.where(offs == offset, moved2, 0)), 0)

        here = slots == i
        out_found = tl.where(here, found, out_found)
        out_expert = tl.where(here, expert, out_expert)
        out_target = tl.where(here, target, out_target)
        out_moved = tl.where(here, moved_x2, out_moved)

        chosen = on_peak & (offs[None, :] == offset) & (found == 1)
        owned2 = tl.where(chosen, owned2 - moved_x2, owned2)
        extra2 = tl.where(ranks == target, extra2 + moved_x2, extra2)
        spent_mask = spent_mask + tl.where(chosen, 1, 0)

    # A refused row sorts last: its expert is 0 and would otherwise take slot 0 and
    # shift every real row the moment one placement is refused.
    huge = 1 << 40
    key = tl.where(out_found == 1, out_expert, huge)
    earlier = (key[None, :] < key[:, None]) | (
        (key[None, :] == key[:, None]) & (slots[None, :] < slots[:, None])
    )
    slot_of = tl.sum(tl.where(earlier & slot_ok[None, :], 1, 0).to(tl.int64), axis=1)

    # Residency is matched **by slot**, after the sort, not as a set. Set membership
    # looks kinder -- a replica held anywhere counts as held -- but it is unsound with
    # positional slots: an expert resident in slot 0 and planned into slot 2 would be
    # charged nothing while the publish points its map at slot 2's physical row, where
    # its weights are not. Positional matching can re-transfer a replica whose position
    # shifted, which is bounded waste; the other is silent corruption.
    #
    # The charge still runs in **gain order**, which is why the sort comes first: the
    # slot a row lands in has to be known before its residency can be read, and a layer
    # that cannot afford every replica must keep the ones worth most.
    # The publish table can name a *different* replica from the transfer table -- the
    # resident one, when the budget cannot afford the planned one. Seeded with the
    # planned values so a slot the loop leaves untouched still describes its own plan.
    out_pub_expert = out_expert
    out_pub_target = out_target
    out_pub_moved = out_moved
    for i in range(CAP):
        # Padding lanes carry a slot index too, so they are excluded here rather than
        # only at the stores: one of them landing on `i` would double the gather.
        at = (slot_of == i) & slot_ok
        found = tl.sum(tl.where(at, out_found, 0))
        expert = tl.sum(tl.where(at, out_expert, 0))
        target = tl.sum(tl.where(at, out_target, 0))
        slot = tl.sum(tl.where(at, slot_of, 0))
        moved = tl.sum(tl.where(at, out_moved, 0))

        res_e = tl.sum(tl.where(slots == slot, res_expert, 0))
        res_t = tl.sum(tl.where(slots == slot, res_target, 0))
        resident = tl.where(res_e >= 0, 1, 0)
        same = tl.where((res_e == expert) & (res_t == target), 1, 0)
        keep = found * resident * same
        needs = found * (1 - keep)
        affordable = needs * tl.where(spent_now < budget, 1, 0)
        spent_now += affordable
        placed += affordable
        # Ticket 15, per slot: a slot whose new replica the budget cannot afford keeps
        # the one it holds rather than being cleared. Because the charge runs in gain
        # order, the slots that lose out are the ones worth least, and each of them
        # falls back to something already paid for instead of to nothing.
        hold = found * (1 - keep) * (1 - affordable) * resident
        out_charged = tl.where(at, affordable, out_charged)
        out_desired = tl.where(
            at, tl.maximum(tl.maximum(keep, affordable), hold), out_desired
        )
        out_pub_expert = tl.where(
            at, tl.where(hold == 1, res_e, expert), out_pub_expert
        )
        out_pub_target = tl.where(
            at, tl.where(hold == 1, res_t, target), out_pub_target
        )
        out_pub_moved = tl.where(at, tl.where(hold == 1, 0, moved), out_pub_moved)

    tl.store(spent_ptr, spent_now)
    tl.store(placed_total_ptr, placed)

    tl.store(transfer_ptr + slot_of * 4 + 0, out_charged, mask=slot_ok)
    tl.store(transfer_ptr + slot_of * 4 + 1, out_expert, mask=slot_ok)
    tl.store(transfer_ptr + slot_of * 4 + 2, out_target, mask=slot_ok)
    tl.store(transfer_ptr + slot_of * 4 + 3, out_moved, mask=slot_ok)
    tl.store(publish_ptr + slot_of * 4 + 0, out_desired, mask=slot_ok)
    tl.store(publish_ptr + slot_of * 4 + 1, out_pub_expert, mask=slot_ok)
    tl.store(publish_ptr + slot_of * 4 + 2, out_pub_target, mask=slot_ok)
    tl.store(publish_ptr + slot_of * 4 + 3, out_pub_moved, mask=slot_ok)


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
    if num_logical == 0:
        # `plan_replicas` returns no placement for this and names the all-zero dummy
        # forward as the reason, so it must not become a crash: `per_rank` would be 0
        # and the kernel would fail to compile on `tl.arange(0, 0)`.
        transfer_plan.zero_()
        publish_plan.zero_()
        return
    # Unit innermost stride, because the kernel indexes `load_ptr + rank * per_rank +
    # off`. The tensor versions this replaces handle arbitrary strides, and every
    # equality test builds contiguous tensors, so a strided view would read the wrong
    # elements in silence.
    if predicted.stride(-1) != 1:
        raise ValueError(
            "the predicted load must have unit innermost stride; the fused planner "
            "indexes it directly. Pass a contiguous tensor."
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


def plan_replicas_fused(
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
    """Plan a layer's whole replica set, charge the budget, write both plan tables.

    `plan_and_charge_fused` with a slot dimension. The cap is taken from `residency`'s
    leading dimension rather than passed, so the three tables cannot disagree about how
    many slots a layer has -- a disagreement that would write one table's row into
    another's slot and be invisible in every aggregate.

    Args:
        predicted: `[num_logical]` reduced predicted load, unit innermost stride.
        residency: `[cap, 2]` int64 `(expert, target)` per slot, `-1` for empty.
        budget: `[]` int64 transfers still affordable this forward.
        spent: `[]` int64 charged so far, updated in place.
        placed_total: `[]` int64 lifetime counter, updated in place.
        transfer_plan: `[cap, 4]` int64, written.
        publish_plan: `[cap, 4]` int64, written.
        ep_size: EP group size.
        min_tokens: Refuse a placement moving less than this.
    """
    cap = residency.shape[0]
    for name, tensor, want in (
        ("residency", residency, (cap, 2)),
        ("transfer_plan", transfer_plan, (cap, 4)),
        ("publish_plan", publish_plan, (cap, 4)),
    ):
        if tuple(tensor.shape) != want or tensor.stride(-1) != 1:
            raise ValueError(
                f"{name} must be a contiguous {want} tensor for a cap of {cap}, "
                f"got shape {tuple(tensor.shape)} stride {tensor.stride()}."
            )
    num_logical = predicted.numel()
    if num_logical % ep_size != 0:
        raise ValueError(
            f"{num_logical} logical experts do not divide across {ep_size} EP ranks."
        )
    if predicted.stride(-1) != 1:
        raise ValueError(
            "the predicted load must have unit innermost stride; the fused planner "
            "indexes it directly. Pass a contiguous tensor."
        )
    _plan_replicas_kernel[(1,)](
        predicted,
        residency,
        budget,
        spent,
        placed_total,
        transfer_plan,
        publish_plan,
        ep_size,
        num_logical // ep_size,
        math.ceil(2 * min_tokens),
        EP=triton.next_power_of_2(ep_size),
        PR=triton.next_power_of_2(num_logical // ep_size),
        CAP=cap,
        SLOTS=triton.next_power_of_2(cap),
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
    for name, tensor in (
        ("logical_to_physical", logical_to_physical),
        ("logical_replica_count", logical_replica_count),
        ("source_local", source_local),
        ("layout", layout),
    ):
        # Same reason as the planner's: the kernel adds a column offset to a row
        # pointer, so only `stride(0)` is passed and the innermost stride must be 1.
        if tensor.stride(-1) != 1:
            raise ValueError(
                f"{name} must have unit innermost stride; the fused publish writes "
                f"columns by offset. Pass a contiguous tensor."
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


@triton.jit
def _publish_replicas_kernel(
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
    first_replica_column,
    take_upper,
    CAP: tl.constexpr,
):
    """Make one layer's routing maps say exactly what a `CAP`-row plan asks for.

    `_publish_kernel` with a slot dimension, and the ordering rule is stronger here.
    The single-slot version reverts before it places because a slot handed from one
    expert to another appears in both sets. With several slots the same hazard runs
    *between* them: slot 0's outgoing expert can be slot 2's incoming one, so every
    revert must precede every place, not merely its own slot's place. Two unrolled
    passes give that, since stores within one program run in program order.
    """
    for i in range(CAP):
        old_expert = tl.load(residency_ptr + i * 2 + 0)
        old_target = tl.load(residency_ptr + i * 2 + 1)
        found = tl.where(tl.load(plan_ptr + i * 4 + 0) > 0, 1, 0)
        new_expert = tl.load(plan_ptr + i * 4 + 1)
        new_target = tl.load(plan_ptr + i * 4 + 2)

        resident = tl.where(old_expert >= 0, 1, 0)
        same = tl.where((old_expert == new_expert) & (old_target == new_target), 1, 0)
        keep = found * resident * same
        revert = (resident * (1 - keep)) == 1

        old_index = tl.maximum(old_expert, 0)
        old_canonical = (old_index // per_rank) * stride + old_index % per_rank
        tl.store(l2p_ptr + old_index * l2p_stride + 0, old_canonical, mask=revert)
        tl.store(l2p_ptr + old_index * l2p_stride + 1, -1, mask=revert)
        tl.store(count_ptr + old_index, 1, mask=revert)
        tl.store(local_ptr + old_index * local_stride, old_canonical, mask=revert)
        tl.store(
            layout_ptr
            + tl.maximum(old_target, 0) * layout_stride
            + first_replica_column
            + i,
            -1,
            mask=revert,
        )

    for i in range(CAP):
        old_expert = tl.load(residency_ptr + i * 2 + 0)
        old_target = tl.load(residency_ptr + i * 2 + 1)
        found = tl.where(tl.load(plan_ptr + i * 4 + 0) > 0, 1, 0)
        new_expert = tl.load(plan_ptr + i * 4 + 1)
        new_target = tl.load(plan_ptr + i * 4 + 2)

        resident = tl.where(old_expert >= 0, 1, 0)
        same = tl.where((old_expert == new_expert) & (old_target == new_target), 1, 0)
        keep = found * resident * same
        place = (found * (1 - keep)) == 1

        # Copies are ordered by ascending physical row, because that is the order
        # `compute_logical_maps` discovers them in, and a replica can land *below* its
        # canonical row, which the planner produces routinely.
        new_index = tl.maximum(new_expert, 0)
        new_canonical = (new_index // per_rank) * stride + new_index % per_rank
        replica_row = tl.maximum(new_target, 0) * stride + first_replica_column + i
        lower = tl.minimum(new_canonical, replica_row)
        upper = tl.maximum(new_canonical, replica_row)
        tl.store(l2p_ptr + new_index * l2p_stride + 0, lower, mask=place)
        tl.store(l2p_ptr + new_index * l2p_stride + 1, upper, mask=place)
        tl.store(count_ptr + new_index, 2, mask=place)
        # `build_source_local_physical_map` picks copy `source_rank % count`, so half
        # the ranks keep the canonical row and half take the replica. Publishing the
        # copy this rank does not route to is the defect that activated 131 replicas
        # and sent them no tokens.
        tl.store(
            local_ptr + new_index * local_stride,
            tl.where(take_upper == 1, upper, lower),
            mask=place,
        )
        tl.store(
            layout_ptr
            + tl.maximum(new_target, 0) * layout_stride
            + first_replica_column
            + i,
            new_expert,
            mask=place,
        )

        tl.store(residency_ptr + i * 2 + 0, tl.where(found == 1, new_expert, -1))
        tl.store(residency_ptr + i * 2 + 1, tl.where(found == 1, new_target, -1))


def publish_replicas_fused(
    plan: torch.Tensor,
    residency: torch.Tensor,
    logical_to_physical: torch.Tensor,
    logical_replica_count: torch.Tensor,
    source_local: torch.Tensor,
    layout: torch.Tensor,
    per_rank_experts: int,
    replica_slots_per_rank: int,
    source_rank: int,
) -> None:
    """Publish a layer's whole desired replica set. One launch.

    `publish_plan_fused` with a slot dimension. Slot `i` owns replica column `i` of
    every rank's layout, so two replicas planned onto the same target rank never
    contend for a row.

    Raises:
        ValueError: If the map is too narrow to hold a second copy, or if there are
            fewer replica columns than plan rows -- either would write out of bounds or
            drop a replica in silence.
    """
    cap = plan.shape[0]
    if logical_to_physical.shape[-1] < 2:
        raise ValueError(
            f"logical_to_physical is {logical_to_physical.shape[-1]} wide, so a second "
            f"copy cannot be recorded and the replica would never be routed to. "
            f"Widen it to at least 2."
        )
    if replica_slots_per_rank < cap:
        raise ValueError(
            f"a plan of {cap} rows needs at least {cap} replica columns per rank, but "
            f"there are {replica_slots_per_rank}. Two slots would share a physical row "
            f"and one replica's weights would be read as the other's."
        )
    if tuple(residency.shape) != (cap, 2) or tuple(plan.shape) != (cap, 4):
        raise ValueError(
            f"plan {tuple(plan.shape)} and residency {tuple(residency.shape)} must "
            f"agree on the slot count."
        )
    for name, tensor in (
        ("plan", plan),
        ("residency", residency),
        ("logical_to_physical", logical_to_physical),
        ("logical_replica_count", logical_replica_count),
        ("source_local", source_local),
        ("layout", layout),
    ):
        if tensor.stride(-1) != 1:
            raise ValueError(
                f"{name} must have unit innermost stride; the fused publish writes "
                f"columns by offset. Pass a contiguous tensor."
            )
    _publish_replicas_kernel[(1,)](
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
        per_rank_experts,
        source_rank % 2,
        CAP=cap,
    )
