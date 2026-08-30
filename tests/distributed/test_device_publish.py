# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publishing a placement without the host learning what it is.

Ticket 06. The plan is a device tensor from ticket 04, and the transfer is issued from a
kernel, so publishing is the last place the host would have to read it. Routing already
consumes its maps as device tensors, so a scatter suffices — but every index in that
scatter depends on plan values, which means the writes have to be tensor-indexed rather
than indexed by Python ints.

`apply_replica_maps` plus the layout edit is the oracle: it is the code that was
measured working end to end, and the acceptance criterion is that the device path
produces the same maps. Not "similar" — routing reads these, so a disagreement sends
tokens to a row holding another expert's weights, and nothing raises.

The randomised sequence is the load-bearing test. What breaks an implementation like
this is the transitions, not the states: a slot handed from one expert to another, a
placement that repeats, a forward that places nothing, and a replica whose row sits
*below* its own canonical row.
"""

import pytest
import torch

from vllm.distributed.eplb.device_publish import LayerResidency, publish_plan_on_device
from vllm.distributed.eplb.predictive_planner import (
    Placement,
    apply_replica_maps,
    canonical_row_of,
)

EP_SIZE = 8
PER_RANK = 4
NUM_LOGICAL = EP_SIZE * PER_RANK
SLOTS = 1


def fresh_maps():
    """The routing maps as startup normalization leaves them: one copy each."""
    canonical = torch.tensor(
        [canonical_row_of(e, PER_RANK, SLOTS) for e in range(NUM_LOGICAL)],
        dtype=torch.int64,
    )
    logical_to_physical = torch.full((NUM_LOGICAL, 2), -1, dtype=torch.int64)
    logical_to_physical[:, 0] = canonical
    replica_count = torch.ones(NUM_LOGICAL, dtype=torch.int64)
    source_local = canonical.view(NUM_LOGICAL, 1).clone()
    layout = torch.full((EP_SIZE, PER_RANK + SLOTS), -1, dtype=torch.int64)
    for expert in range(NUM_LOGICAL):
        layout[expert // PER_RANK, expert % PER_RANK] = expert
    return logical_to_physical, replica_count, source_local, layout


def host_publish(state, plan, source_rank):
    """The oracle: the host path, driven by the same plan.

    `state` is `(logical_to_physical, replica_count, source_local, layout, occupant)`.
    """
    logical_to_physical, replica_count, source_local, layout, occupant = state
    found, expert, target, _moved = (int(v) for v in plan.tolist())
    placements = [Placement(0, expert, 0, target, 1.0)] if found else []
    reverted = apply_replica_maps(
        logical_to_physical=logical_to_physical,
        logical_replica_count=replica_count,
        source_local=source_local,
        placements=placements,
        per_rank_experts=PER_RANK,
        replica_slots_per_rank=SLOTS,
        source_rank=source_rank,
        slot_occupant=occupant,
    )
    for target_rank, _reverted_expert in reverted:
        layout[target_rank, PER_RANK] = -1
    for placement in placements:
        layout[placement.target_rank, PER_RANK] = placement.logical_expert


def device_publish(state, residency, plan, source_rank):
    """The path under test, in the same shape so the two can be compared."""
    logical_to_physical, replica_count, source_local, layout = state
    publish_plan_on_device(
        plan=plan,
        residency=residency,
        logical_to_physical=logical_to_physical,
        logical_replica_count=replica_count,
        source_local=source_local,
        layout=layout,
        per_rank_experts=PER_RANK,
        replica_slots_per_rank=SLOTS,
        source_rank=source_rank,
    )


def both(plans, source_rank=0):
    """Drive the same plan sequence through both paths and return the two states."""
    host = (*fresh_maps(), {})
    device = fresh_maps()
    residency = LayerResidency.empty(device=torch.device("cpu"))
    for plan in plans:
        host_publish(host, plan, source_rank)
        device_publish(device, residency, plan, source_rank)
    return host[:4], device


def plan(found, expert=0, target=0, moved=2):
    return torch.tensor([found, expert, target, moved], dtype=torch.int64)


def assert_same(host, device):
    names = ("logical_to_physical", "replica_count", "source_local", "layout")
    for name, want, got in zip(names, host, device):
        assert torch.equal(want, got), f"{name} disagrees:\nhost {want}\ndevice {got}"


def test_a_single_placement_publishes_the_same_maps_as_the_host_path():
    """The base case, and the one that decides whether any token reaches the replica."""
    host, device = both([plan(1, expert=9, target=5)])

    assert_same(host, device)


def test_a_replica_below_its_own_canonical_row_is_ordered_the_same_way():
    """Copies are ordered by ascending physical row, and a replica can be the lower one.

    Expert 9 is owned by rank 2 with canonical row 11, and a replica on rank 0 sits at
    row 4. Assuming `[canonical, replica]` puts the wrong row in the map for exactly
    this case, and it is a case the planner produces routinely.
    """
    host, device = both([plan(1, expert=9, target=0)])

    assert_same(host, device)
    assert int(device[0][9, 0]) < int(device[0][9, 1])


def test_placing_nothing_reverts_what_the_last_forward_left():
    """An empty plan is the reversion path, not a no-op.

    Reversion is free — a map edit with no transfer — and keeping an unwanted replica is
    not neutral: it goes on shedding half of an expert that may no longer be hot onto a
    rank that may now be the peak.
    """
    host, device = both([plan(1, expert=9, target=5), plan(0)])

    assert_same(host, device)
    assert int(device[1][9]) == 1, "the reverted expert is back to one copy"
    assert int(device[3][5, PER_RANK]) == -1, "the slot reads inactive again"


def test_a_slot_handed_from_one_expert_to_another_releases_before_it_is_reclaimed():
    """The transition that a naive implementation gets wrong in the writes' order.

    Reverting after placing would clear the row that was just claimed, leaving the map
    pointing at a canonical copy while the weights sat in a slot nothing routes to.
    """
    host, device = both([plan(1, expert=9, target=5), plan(1, expert=2, target=5)])

    assert_same(host, device)
    assert int(device[3][5, PER_RANK]) == 2
    assert int(device[1][9]) == 1
    assert int(device[1][2]) == 2


def test_repeating_a_placement_changes_nothing():
    """The steady state: consecutive prefill forwards want the same replica.

    This is what makes coverage ratchet up rather than churn, so it has to be idempotent
    in the maps as well as free in transfers.
    """
    once, _ = both([plan(1, expert=9, target=5)])
    host, device = both([plan(1, expert=9, target=5)] * 3)

    assert_same(host, device)
    assert_same(once, device)


def test_moving_a_replica_to_a_different_rank_frees_the_old_slot():
    """Same expert, new target. Both rows have to be corrected, in the right order."""
    host, device = both([plan(1, expert=9, target=5), plan(1, expert=9, target=3)])

    assert_same(host, device)
    assert int(device[3][5, PER_RANK]) == -1
    assert int(device[3][3, PER_RANK]) == 9


@pytest.mark.parametrize("source_rank", range(EP_SIZE))
def test_the_source_local_map_agrees_for_every_source_rank(source_rank):
    """Each rank publishes its own map, and the copy it picks depends on its own id.

    `build_source_local_physical_map` picks copy `source_rank % count`, so half the
    ranks keep the canonical row and half take the replica. Getting this wrong is the
    defect that activated 131 replicas and routed zero tokens to them.
    """
    host, device = both([plan(1, expert=9, target=5)], source_rank=source_rank)

    assert_same(host, device)


def test_it_agrees_over_a_long_random_sequence_of_plans():
    """The transitions are what break this, and a sequence is how they get exercised."""
    generator = torch.Generator().manual_seed(7)
    plans = []
    for _ in range(200):
        found = int(torch.randint(0, 4, (1,), generator=generator)) > 0
        expert = int(torch.randint(0, NUM_LOGICAL, (1,), generator=generator))
        target = int(torch.randint(0, EP_SIZE, (1,), generator=generator))
        plans.append(plan(int(found), expert=expert, target=target))

    host, device = both(plans)

    assert_same(host, device)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a host read is only observable on CUDA"
)
def test_publishing_performs_no_host_read():
    """The ticket's headline criterion, at the seam where publishing decides it.

    Every index here comes from the plan, so the temptation is one `int()` per field —
    which is one synchronisation per predicted layer, the whole cost being removed.
    """
    device = tuple(t.cuda() for t in fresh_maps())
    residency = LayerResidency.empty(device=torch.device("cuda"))
    # Both plans moved to the device *before* the guard: a host-to-device copy counts as
    # a synchronising operation too, and one in the test would fail it for the wrong
    # reason.
    place = plan(1, expert=9, target=5).cuda()
    nothing = plan(0).cuda()

    torch.cuda.set_sync_debug_mode("error")
    try:
        device_publish(device, residency, place, source_rank=0)
        device_publish(device, residency, nothing, source_rank=0)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert int(device[3][5, PER_RANK]) == -1
