# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ticket B: a layer may fill more than one replica slot.

The measurements this rests on, from offline replay against real DP=8 dumps: one
replica per layer removes 32.1% of critical-path excess, two 49.5%, four 68.1%. The
replica count is a placement-side parameter and placement is *negative cost*, so this
buys benefit without moving prediction's bill -- which is the only lever left with the
right order of magnitude.
"""

from __future__ import annotations

import pytest
import torch

from vllm.distributed.eplb.predictive_planner import (
    plan_layer_replicas_on_device,
    plan_one_layer_on_device,
)

EP = 4
PER_RANK = 4
MIN_TOKENS = 1.0


def _load(values):
    return torch.tensor(values, dtype=torch.int64)


def test_cap_one_is_the_single_replica_planner_exactly():
    """The retained planner is the oracle, so cap 1 must not merely agree in spirit."""
    torch.manual_seed(0)
    for _ in range(50):
        load = torch.randint(0, 500, (EP * PER_RANK,), dtype=torch.int64)

        one = plan_one_layer_on_device(load, EP, MIN_TOKENS)
        many = plan_layer_replicas_on_device(load, EP, MIN_TOKENS, cap=1)

        assert torch.equal(many[0], one), (load.tolist(), many[0], one)


def test_a_logical_expert_is_never_replicated_twice_in_a_layer():
    """A second copy of the same expert would need a third physical row for it, so a
    wider `logical_to_physical_map` and a `logical_replica_count` above two. The
    unconstrained greedy would ask for that in 61% of layers; it costs 2.82% of the
    benefit to refuse, measured on the same dumps."""
    load = _load([900, 800, 700, 600] + [10] * 12)

    plan = plan_layer_replicas_on_device(load, EP, MIN_TOKENS, cap=4)

    chosen = [int(row[1]) for row in plan if int(row[0])]
    assert len(chosen) == len(set(chosen)), chosen


def test_rows_are_sorted_by_expert_so_an_unchanged_set_keeps_its_slots():
    """Slots are matched against residency position by position. A stable set arriving
    in a different order would read as a wholesale change and re-transfer every replica,
    and four times the transfer rate at unchanged traffic measured +27% mean TTFT."""
    load = _load([900, 800, 700, 600] + [10] * 12)

    plan = plan_layer_replicas_on_device(load, EP, MIN_TOKENS, cap=4)

    found = [int(row[1]) for row in plan if int(row[0])]
    assert found == sorted(found), found


def test_refused_rows_sort_last_and_are_zero():
    """A not-found row's expert is 0, which would sort it first and shift every real
    row's slot the moment one placement is refused."""
    load = _load([100] * 16)

    plan = plan_layer_replicas_on_device(load, EP, MIN_TOKENS * 10_000, cap=3)

    assert int(plan[:, 0].sum()) == 0
    assert torch.equal(plan, torch.zeros_like(plan))


def test_each_replica_is_planned_against_what_the_previous_one_left():
    """Planning every replica against the original load would pick the same shed twice
    over, and the second would remove nothing that the first had not."""
    load = _load([1000, 900, 20, 20] + [30] * 12)

    plan = plan_layer_replicas_on_device(load, EP, MIN_TOKENS, cap=2)

    assert int(plan[0, 0]) and int(plan[1, 0]), plan
    assert int(plan[0, 1]) != int(plan[1, 1])


def test_more_replicas_never_remove_less_excess():
    """The property the whole ticket rests on. A greedy that credited the receiving rank
    to the wrong column would still look monotone in aggregate, so this is checked on
    the planner's own arithmetic rather than on a summary."""
    torch.manual_seed(1)

    def peak_after(load, cap):
        owned = load.view(EP, PER_RANK).to(torch.float64).clone()
        extra = torch.zeros(EP, dtype=torch.float64)
        for row in plan_layer_replicas_on_device(load, EP, MIN_TOKENS, cap):
            if not int(row[0]):
                continue
            expert, target = int(row[1]), int(row[2])
            half = row[3].to(torch.float64) / 2.0
            owned[expert // PER_RANK, expert % PER_RANK] -= half
            extra[target] += half
        return float((owned.sum(dim=1) + extra).max())

    for _ in range(20):
        load = torch.randint(0, 400, (EP * PER_RANK,), dtype=torch.int64)
        peaks = [peak_after(load, cap) for cap in (1, 2, 3, 4)]
        assert peaks == sorted(peaks, reverse=True), peaks


@pytest.mark.parametrize("cap", [0, -1])
def test_a_layer_needs_at_least_one_slot(cap):
    with pytest.raises(ValueError, match="at least one replica slot"):
        plan_layer_replicas_on_device(_load([1] * 16), EP, MIN_TOKENS, cap)


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused planner is Triton on CUDA"
)


def _fused(load, ep_size, min_tokens, cap):
    from vllm.distributed.eplb.fused_placement import plan_replicas_fused

    device = load.device
    residency = torch.full((cap, 2), -1, dtype=torch.int64, device=device)
    budget = torch.tensor(1 << 30, dtype=torch.int64, device=device)
    spent = torch.zeros((), dtype=torch.int64, device=device)
    placed = torch.zeros((), dtype=torch.int64, device=device)
    transfer = torch.zeros(cap, 4, dtype=torch.int64, device=device)
    publish = torch.zeros(cap, 4, dtype=torch.int64, device=device)
    plan_replicas_fused(
        load, residency, budget, spent, placed, transfer, publish, ep_size, min_tokens
    )
    return publish


@requires_gpu
@pytest.mark.parametrize("cap", [1, 2, 4])
def test_the_kernel_is_bit_identical_to_the_oracle(cap):
    """The oracle is retained precisely so this can be asserted rather than trusted.

    A plan that differs by rank pairs a put with a peer expecting nothing, and every
    rank derives its own from the same snapshot -- so agreement has to be exact, not
    close, and it has to survive the tie-breaks.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    for trial in range(200):
        load = torch.randint(0, 400, (EP * PER_RANK,), dtype=torch.int64, device=device)
        if trial % 4 == 0:
            # Ties are where an order-dependent reduction would diverge between ranks.
            load = (load // 50) * 50

        want = plan_layer_replicas_on_device(load, EP, MIN_TOKENS, cap)
        got = _fused(load, EP, MIN_TOKENS, cap)

        assert torch.equal(got[:, 1:], want[:, 1:]), (
            f"trial {trial} cap {cap}\nload {load.tolist()}\ngot\n{got}\nwant\n{want}"
        )
        assert torch.equal(got[:, 0] > 0, want[:, 0] > 0), (trial, got, want)


@requires_gpu
def test_an_unchanged_set_is_kept_and_costs_nothing():
    """The steady state has to be free, or the budget buys churn instead of coverage.

    Residency is matched by slot, and the publish writes it back in the planner's own
    expert order, so a set that has not changed comes back in the same order and every
    slot reads as held. Matching as a *set* instead would look kinder here and is
    unsound: an expert resident in slot 0 and planned into slot 2 would be charged
    nothing while its map pointed at slot 2's physical row, where its weights are not.
    """
    from vllm.distributed.eplb.fused_placement import plan_replicas_fused

    device = torch.device("cuda")
    load = torch.tensor(
        [900, 800, 700, 600] + [10] * 12, dtype=torch.int64, device=device
    )
    cap = 3
    first = _fused(load, EP, MIN_TOKENS, cap)
    residency = first[:, 1:3].clone().contiguous()

    budget = torch.tensor(1 << 30, dtype=torch.int64, device=device)
    spent = torch.zeros((), dtype=torch.int64, device=device)
    placed = torch.zeros((), dtype=torch.int64, device=device)
    transfer = torch.zeros(cap, 4, dtype=torch.int64, device=device)
    publish = torch.zeros(cap, 4, dtype=torch.int64, device=device)

    plan_replicas_fused(
        load, residency, budget, spent, placed, transfer, publish, EP, MIN_TOKENS
    )

    assert int(spent) == 0, f"re-transferred what it already held: {transfer}"
    assert int(publish[:, 0].sum()) == cap, publish
    assert torch.equal(publish[:, 1:3], residency), (publish, residency)


@requires_gpu
def test_the_budget_is_spent_on_the_strongest_gains_first():
    """A layer that cannot afford every replica must keep the ones worth most, so the
    charge runs in gain order even though the rows are stored in expert order."""
    from vllm.distributed.eplb.fused_placement import plan_replicas_fused

    device = torch.device("cuda")
    load = torch.tensor(
        [900, 800, 700, 600] + [10] * 12, dtype=torch.int64, device=device
    )
    cap = 3
    full = _fused(load, EP, MIN_TOKENS, cap)

    residency = torch.full((cap, 2), -1, dtype=torch.int64, device=device)
    budget = torch.tensor(1, dtype=torch.int64, device=device)
    spent = torch.zeros((), dtype=torch.int64, device=device)
    placed = torch.zeros((), dtype=torch.int64, device=device)
    transfer = torch.zeros(cap, 4, dtype=torch.int64, device=device)
    publish = torch.zeros(cap, 4, dtype=torch.int64, device=device)

    plan_replicas_fused(
        load, residency, budget, spent, placed, transfer, publish, EP, MIN_TOKENS
    )

    assert int(spent) == 1 and int(transfer[:, 0].sum()) == 1
    charged = transfer[transfer[:, 0] > 0][0]
    # The first replica the greedy found is the strongest, and it is the one in `full`
    # whose moved load is largest.
    strongest = full[full[:, 3].argmax()]
    assert int(charged[1]) == int(strongest[1]), (charged, full)


@requires_gpu
def test_the_tables_must_agree_about_how_many_slots_a_layer_has():
    """A cap mismatch would write one table's row into another's slot, and nothing
    downstream would show it."""
    from vllm.distributed.eplb.fused_placement import plan_replicas_fused

    device = torch.device("cuda")
    load = torch.ones(EP * PER_RANK, dtype=torch.int64, device=device)
    zero = torch.zeros((), dtype=torch.int64, device=device)
    with pytest.raises(ValueError, match="must be a contiguous"):
        plan_replicas_fused(
            load,
            torch.full((3, 2), -1, dtype=torch.int64, device=device),
            torch.tensor(1, dtype=torch.int64, device=device),
            zero.clone(),
            zero.clone(),
            torch.zeros(2, 4, dtype=torch.int64, device=device),
            torch.zeros(3, 4, dtype=torch.int64, device=device),
            EP,
            MIN_TOKENS,
        )


NUM_LOGICAL = EP * PER_RANK


def _fresh_maps(device, slots):
    canonical = torch.empty(NUM_LOGICAL, dtype=torch.int64, device=device)
    for expert in range(NUM_LOGICAL):
        canonical[expert] = (expert // PER_RANK) * (
            PER_RANK + slots
        ) + expert % PER_RANK
    l2p = torch.full((NUM_LOGICAL, 2), -1, dtype=torch.int64, device=device)
    l2p[:, 0] = canonical
    layout = torch.full((EP, PER_RANK + slots), -1, dtype=torch.int64, device=device)
    for expert in range(NUM_LOGICAL):
        layout[expert // PER_RANK, expert % PER_RANK] = expert
    return [
        l2p,
        torch.ones(NUM_LOGICAL, dtype=torch.int64, device=device),
        canonical.view(NUM_LOGICAL, 1).clone(),
        layout,
    ]


@requires_gpu
@pytest.mark.parametrize("source_rank", [0, 1])
def test_a_one_slot_publish_matches_the_single_slot_kernel(source_rank):
    """The retained single-slot publish is the oracle at cap 1, over transitions rather
    than states: what breaks a publish is a slot changing hands."""
    from vllm.distributed.eplb.fused_placement import (
        publish_plan_fused,
        publish_replicas_fused,
    )

    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(11)
    want, got = _fresh_maps(device, 1), _fresh_maps(device, 1)
    want_res = torch.full((2,), -1, dtype=torch.int64, device=device)
    got_res = torch.full((1, 2), -1, dtype=torch.int64, device=device)

    for step in range(200):
        found = int(torch.randint(0, 4, (1,), generator=generator).item()) > 0
        expert = int(torch.randint(0, NUM_LOGICAL, (1,), generator=generator).item())
        target = int(torch.randint(0, EP, (1,), generator=generator).item())
        row = [1, expert, target, 8] if found else [0, 0, 0, 0]

        publish_plan_fused(
            torch.tensor(row, dtype=torch.int64, device=device),
            want_res,
            want[0],
            want[1],
            want[2],
            want[3],
            PER_RANK,
            1,
            source_rank,
        )
        publish_replicas_fused(
            torch.tensor([row], dtype=torch.int64, device=device),
            got_res,
            got[0],
            got[1],
            got[2],
            got[3],
            PER_RANK,
            1,
            source_rank,
        )

        for name, a, b in zip(("l2p", "count", "local", "layout"), want, got):
            assert torch.equal(a, b), f"step {step} {name}\n{a}\n{b}"
        assert torch.equal(want_res, got_res.view(-1))


@requires_gpu
def test_an_expert_moving_between_slots_keeps_its_map():
    """The hazard several slots introduce: slot 0's outgoing expert can be slot 2's
    incoming one. Reverting and placing slot by slot would clear the row just claimed,
    leaving the map pointing at a canonical copy while the weights sat in a slot nothing
    routes to -- and every aggregate would still look right."""
    from vllm.distributed.eplb.fused_placement import publish_replicas_fused

    device = torch.device("cuda")
    slots = 3
    maps = _fresh_maps(device, slots)
    residency = torch.full((slots, 2), -1, dtype=torch.int64, device=device)

    def publish(rows):
        publish_replicas_fused(
            torch.tensor(rows, dtype=torch.int64, device=device),
            residency,
            maps[0],
            maps[1],
            maps[2],
            maps[3],
            PER_RANK,
            slots,
            0,
        )

    publish([[1, 5, 1, 8], [1, 9, 2, 8], [0, 0, 0, 0]])
    assert int(maps[1][5]) == 2 and int(maps[1][9]) == 2

    # Expert 5 moves from slot 0 to slot 1; expert 9 leaves entirely.
    publish([[1, 3, 1, 8], [1, 5, 2, 8], [0, 0, 0, 0]])

    assert int(maps[1][9]) == 1, "the departed replica was not reverted"
    assert int(maps[1][5]) == 2, "the moved replica lost its second copy"
    assert int(maps[1][3]) == 2
    replica_row = 2 * (PER_RANK + slots) + PER_RANK + 1
    assert replica_row in maps[0][5].tolist(), (maps[0][5], replica_row)
    assert int(maps[3][2, PER_RANK + 1]) == 5, maps[3]


@requires_gpu
def test_more_plan_rows_than_replica_columns_is_refused():
    """Two slots sharing a physical row read one replica's weights as the other's."""
    from vllm.distributed.eplb.fused_placement import publish_replicas_fused

    device = torch.device("cuda")
    maps = _fresh_maps(device, 1)
    with pytest.raises(ValueError, match="replica columns per rank"):
        publish_replicas_fused(
            torch.zeros(3, 4, dtype=torch.int64, device=device),
            torch.full((3, 2), -1, dtype=torch.int64, device=device),
            maps[0],
            maps[1],
            maps[2],
            maps[3],
            PER_RANK,
            1,
            0,
        )


def test_the_transfer_fakes_accept_what_the_coordinator_actually_passes():
    """The stand-in must not drift from the thing it stands in for.

    This branch has paid for that drift twice: `bind_prediction_target` gained arguments
    in the fakes and not in `MoERunner`, so 563 unit tests passed and the first 8-GPU
    server died on a `TypeError`; and the coordinator gained `num_slots` here while four
    fakes still had the old signature. Binding the coordinator's real call against the
    real `transfer` catches both directions without a GPU.
    """
    import inspect

    from vllm.distributed.eplb.device_coordinator import DevicePlacementCoordinator
    from vllm.distributed.eplb.device_transfer import DeviceExpertTransfer

    source = inspect.getsource(DevicePlacementCoordinator.plan_and_launch)
    call = source[source.index("self.transfer.transfer(") :]
    passed = {
        line.split("=")[0].strip()
        for line in call.splitlines()
        if "=" in line and line.strip().endswith(",") and "==" not in line
    }
    accepted = set(inspect.signature(DeviceExpertTransfer.transfer).parameters)

    unknown = {name for name in passed if name.isidentifier()} - accepted
    assert not unknown, f"the coordinator passes {unknown}, which transfer() refuses"


# --------------------------------------------------------------------------
# Ticket 15: a spent budget must not revert a resident replica
# --------------------------------------------------------------------------


def _charge(load, residency, budget, spent, cap=1, min_tokens=MIN_TOKENS):
    """Run one layer through the fused planner and hand back both plan tables."""
    from vllm.distributed.eplb.fused_placement import plan_replicas_fused

    device = load.device
    transfer = torch.zeros((cap, 4), dtype=torch.int64, device=device)
    publish = torch.zeros((cap, 4), dtype=torch.int64, device=device)
    placed = torch.zeros((), dtype=torch.int64, device=device)
    plan_replicas_fused(
        load,
        residency,
        torch.tensor(budget, dtype=torch.int64, device=device),
        spent,
        placed,
        transfer,
        publish,
        EP,
        min_tokens,
    )
    return transfer, publish


@requires_gpu
def test_a_spent_budget_keeps_the_replica_a_layer_already_holds():
    """The defect: an unaffordable *different* replica used to clear a good one.

    Reverting never touches weights, so the resident replica's row still holds its
    expert and keeping it is free. Clearing it costs that layer's whole benefit until
    the budget ratchets back around, which at the default budget of 4 over 44 reachable
    layers takes about ten forwards.
    """
    device = torch.device("cuda")
    load = torch.full((EP * PER_RANK,), 100, dtype=torch.int64, device=device)
    load[2] = 9000
    resident = torch.tensor([[2, 5]], dtype=torch.int64, device=device)
    # Traffic shifts: a different expert is now hottest, and the budget is gone.
    load[2] = 100
    load[9] = 9000
    spent = torch.tensor(4, dtype=torch.int64, device=device)

    transfer, publish = _charge(load, resident, budget=4, spent=spent)

    assert transfer[0, 0].item() == 0, "an unaffordable placement must not be charged"
    assert publish[0, 0].item() == 1, "the layer must still describe a replica"
    assert publish[0, 1].item() == 2, "and it must be the resident one, not the refused"
    assert publish[0, 2].item() == 5
    assert spent.item() == 4, "keeping what is already there charges nothing"


@requires_gpu
def test_a_spent_budget_with_nothing_resident_still_places_nothing():
    """The fix must not invent a replica for a layer that never had one."""
    device = torch.device("cuda")
    load = torch.full((EP * PER_RANK,), 100, dtype=torch.int64, device=device)
    load[9] = 9000
    empty = torch.full((1, 2), -1, dtype=torch.int64, device=device)
    spent = torch.tensor(4, dtype=torch.int64, device=device)

    transfer, publish = _charge(load, empty, budget=4, spent=spent)

    assert transfer[0, 0].item() == 0
    assert publish[0, 0].item() == 0


@requires_gpu
def test_a_layer_the_planner_refuses_still_reverts():
    """Narrow by design: `found == 0` is a judgement about load, not a budget accident.

    The snapshot counts logical experts, so it is unaffected by what is resident and a
    balanced layer cannot keep itself balanced by holding a replica. Reverting there is
    the documented behaviour and this fix must not quietly widen into it.
    """
    device = torch.device("cuda")
    flat = torch.full((EP * PER_RANK,), 100, dtype=torch.int64, device=device)
    resident = torch.tensor([[2, 5]], dtype=torch.int64, device=device)
    spent = torch.zeros((), dtype=torch.int64, device=device)

    _, publish = _charge(flat, resident, budget=43, spent=spent, min_tokens=1e9)

    assert publish[0, 0].item() == 0, "no admissible placement still means no replica"


@requires_gpu
def test_a_traffic_shift_does_not_collapse_coverage_to_the_budget():
    """The behaviour the ticket is really about, and it needs traffic that *moves*.

    A first version of this test held each layer's hot expert fixed and passed against
    the unfixed kernel: with the plan equal to residency every layer takes the `keep`
    branch and nothing is ever charged, so no budget pressure arises and there is no
    defect to see. The sawtooth needs the plan to *differ* from what is resident while
    the budget is gone -- which is what a domain switch does to every layer at once.

    Unfixed, the shift forward places `budget` layers and reverts the rest, so coverage
    drops to the budget. Fixed, the layers that cannot afford their new replica keep
    their old one: coverage holds while the new set is paid for a few layers per
    forward.
    """
    device = torch.device("cuda")
    LAYERS, BUDGET = 8, 2

    def loads_for(offset):
        out = []
        for layer in range(LAYERS):
            load = torch.full((EP * PER_RANK,), 100, dtype=torch.int64, device=device)
            load[(layer * 2 + offset) % (EP * PER_RANK)] = 9000
            out.append(load)
        return out

    residency = [
        torch.full((1, 2), -1, dtype=torch.int64, device=device) for _ in range(LAYERS)
    ]

    def run_forward(loads):
        spent = torch.zeros((), dtype=torch.int64, device=device)
        for layer in range(LAYERS):
            _, publish = _charge(loads[layer], residency[layer], BUDGET, spent)
            # Stands in for the publish kernel's residency update.
            if publish[0, 0].item():
                residency[layer] = publish[None, 0, 1:3].clone()
            else:
                residency[layer] = torch.full(
                    (1, 2), -1, dtype=torch.int64, device=device
                )
        return sum(int(r[0, 0].item() >= 0) for r in residency)

    before = loads_for(1)
    warmup = [run_forward(before) for _ in range(6)]
    assert warmup[-1] == LAYERS, f"coverage never ratcheted up: {warmup}"

    # Every layer's best expert changes at once, which is what a domain switch does.
    after = loads_for(0)
    during = [run_forward(after) for _ in range(4)]

    assert min(during) == LAYERS, (
        f"coverage collapsed on the traffic shift: {warmup} then {during}. A layer "
        f"that cannot afford its new replica must keep the one it has."
    )


@requires_gpu
def test_the_earliest_layer_wins_when_the_budget_runs_out_mid_forward():
    """Ticket 15 asks for this to be stated, not merely to happen.

    Across layers the budget is first-come-first-served by layer index, because layers
    are visited in increasing order and each plans from its own row -- no layer can see
    a later one's prediction to rank against. Within a layer it is gain order, so a
    layer that cannot afford every slot keeps the replicas worth most. The two orders
    are different on purpose and only the second is a choice.
    """
    device = torch.device("cuda")
    spent = torch.zeros((), dtype=torch.int64, device=device)
    empty = torch.full((1, 2), -1, dtype=torch.int64, device=device)
    charged = []
    for layer in range(4):
        load = torch.full((EP * PER_RANK,), 100, dtype=torch.int64, device=device)
        # Later layers are hotter, so a gain-ranked scheme would prefer them.
        load[layer * 2 + 1] = 1000 * (layer + 1)
        transfer, _ = _charge(load, empty, budget=2, spent=spent)
        charged.append(int(transfer[0, 0].item()))

    assert charged == [1, 1, 0, 0], (
        f"the budget went to {charged}; it must go to the earliest layers, not the "
        f"hottest, because no layer can see a later one's prediction"
    )
