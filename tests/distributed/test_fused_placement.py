# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused kernels decide and publish exactly what the tensor versions do.

Ticket 07's profile: per placed layer the device path launched 46 kernels to plan, 19
to charge the budget and 67 to publish, and in an eager engine each is also a host
dispatch — 0.95 ms per layer, about 42 ms per forward, with the GPU idle for 78% of a
prefill window. The fusion is worth having only if it is *the same decision*, so both
halves are tested by equality against the implementations they replace, over randomised
inputs rather than hand-picked ones.

That shape is deliberate and this project's history is the argument for it: the retained
host planner exists to be the oracle for the tensor planner, and the tensor planner is
now the oracle for this. A test written against what I expected the answer to be would
have agreed with my own misreading.

The arithmetic is exact rather than approximately equal, and it has to be: every rank
derives the plan locally from the same snapshot, so a plan that differs by one rank
pairs a sender with no receiver and hangs the engine.
"""

import pytest
import torch

from vllm.distributed.eplb.device_publish import LayerResidency, publish_plan_on_device
from vllm.distributed.eplb.fused_placement import (
    plan_and_charge_fused,
    publish_plan_fused,
)
from vllm.distributed.eplb.predictive_planner import (
    canonical_row_of,
    plan_one_layer_on_device,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused kernels are CUDA-only"
)

EP_SIZE = 8
PER_RANK = 4
NUM_LOGICAL = EP_SIZE * PER_RANK
SLOTS = 1
MIN_TOKENS = 4.0


def reference_plan_and_charge(
    predicted: torch.Tensor,
    residency: torch.Tensor,
    budget: torch.Tensor,
    spent: torch.Tensor,
    placed_total: torch.Tensor,
    ep_size: int,
    min_tokens: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The arithmetic `plan_and_launch` used to inline, kept here as the oracle.

    Copied rather than imported because the coordinator now calls the fused kernel, and
    a test that imports the code under test as its own reference proves nothing.
    """
    raw = plan_one_layer_on_device(predicted, ep_size, min_tokens)
    found = raw[0] > 0
    keep = (
        found
        & (residency[0] >= 0)
        & (residency[0] == raw[1])
        & (residency[1] == raw[2])
    )
    needs = found & ~keep
    affordable = needs & (spent < budget)
    charged = affordable.to(torch.int64)
    spent += charged
    placed_total += charged
    transfer = torch.stack([charged, raw[1], raw[2], raw[3]])
    publish = torch.stack([(keep | affordable).to(torch.int64), raw[1], raw[2], raw[3]])
    return transfer, publish


def random_load(generator: torch.Generator, device: torch.device) -> torch.Tensor:
    """A snapshot with the shapes that break planners: zeros, ties, one hot expert."""
    kind = int(torch.randint(0, 4, (1,), generator=generator).item())
    if kind == 0:
        load = torch.zeros(NUM_LOGICAL, dtype=torch.int32)
    elif kind == 1:
        load = torch.full((NUM_LOGICAL,), 7, dtype=torch.int32)
    elif kind == 2:
        load = torch.randint(0, 3, (NUM_LOGICAL,), generator=generator).to(torch.int32)
    else:
        load = torch.randint(0, 40, (NUM_LOGICAL,), generator=generator).to(torch.int32)
        load[int(torch.randint(0, NUM_LOGICAL, (1,), generator=generator))] = 4096
    return load.to(device)


@pytest.mark.parametrize("min_tokens", [0.5, 4.0, 128.0])
def test_the_fused_plan_agrees_with_the_tensor_planner(min_tokens):
    """Bit-identical plans over 200 randomised snapshots, including the awkward ones.

    All-zero load and an exactly balanced layer must both place nothing; a single hot
    expert must place; ties must break the same way, which is what the equality-mask
    argmax is for.
    """
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(20260830)
    mismatches = []
    for case in range(200):
        predicted = random_load(generator, device)
        resident = int(torch.randint(-1, NUM_LOGICAL, (1,), generator=generator).item())
        target = int(torch.randint(0, EP_SIZE, (1,), generator=generator).item())
        budget_value = int(torch.randint(0, 3, (1,), generator=generator).item())

        def state(resident=resident, target=target, budget_value=budget_value):
            return (
                torch.tensor([resident, target], dtype=torch.int64, device=device),
                torch.tensor(budget_value, dtype=torch.int64, device=device),
                torch.zeros((), dtype=torch.int64, device=device),
                torch.zeros((), dtype=torch.int64, device=device),
            )

        residency, budget, spent, placed = state()
        want_transfer, want_publish = reference_plan_and_charge(
            predicted, residency, budget, spent, placed, EP_SIZE, min_tokens
        )
        want = (want_transfer.tolist(), want_publish.tolist(), int(spent), int(placed))

        residency, budget, spent, placed = state()
        got_transfer = torch.zeros(4, dtype=torch.int64, device=device)
        got_publish = torch.zeros(4, dtype=torch.int64, device=device)
        plan_and_charge_fused(
            predicted,
            residency,
            budget,
            spent,
            placed,
            got_transfer,
            got_publish,
            EP_SIZE,
            min_tokens,
        )
        got = (got_transfer.tolist(), got_publish.tolist(), int(spent), int(placed))
        if got != want:
            mismatches.append((case, predicted.tolist(), want, got))

    assert not mismatches, (
        f"{len(mismatches)} of 200 plans differ from the tensor planner; first: "
        f"{mismatches[0]}"
    )


def fresh_maps(device: torch.device):
    """The routing maps as startup normalization leaves them: one copy each."""
    canonical = torch.tensor(
        [canonical_row_of(e, PER_RANK, SLOTS) for e in range(NUM_LOGICAL)],
        dtype=torch.int64,
        device=device,
    )
    logical_to_physical = torch.full(
        (NUM_LOGICAL, 2), -1, dtype=torch.int64, device=device
    )
    logical_to_physical[:, 0] = canonical
    layout = torch.full(
        (EP_SIZE, PER_RANK + SLOTS), -1, dtype=torch.int64, device=device
    )
    for expert in range(NUM_LOGICAL):
        layout[expert // PER_RANK, expert % PER_RANK] = expert
    return (
        logical_to_physical,
        torch.ones(NUM_LOGICAL, dtype=torch.int64, device=device),
        canonical.view(NUM_LOGICAL, 1).clone(),
        layout,
    )


@pytest.mark.parametrize("source_rank", [0, 1])
def test_the_fused_publish_agrees_over_a_sequence_of_plans(source_rank):
    """What breaks a publish is the transitions, so this drives a sequence, not states.

    A slot changing hands, a placement repeating, a forward placing nothing, and a
    replica whose row sits below its own canonical row are the four that matter, and a
    random walk of 200 plans hits all of them.
    """
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(31415)
    want_state = fresh_maps(device)
    got_state = fresh_maps(device)
    want_residency = LayerResidency.empty(device)
    got_residency = torch.full((2,), -1, dtype=torch.int64, device=device)

    for step in range(200):
        found = int(torch.randint(0, 4, (1,), generator=generator).item()) > 0
        expert = int(torch.randint(0, NUM_LOGICAL, (1,), generator=generator).item())
        target = int(torch.randint(0, EP_SIZE, (1,), generator=generator).item())
        plan_values = [1, expert, target, 8] if found else [0, 0, 0, 0]
        plan = torch.tensor(plan_values, dtype=torch.int64, device=device)

        publish_plan_on_device(
            plan=plan,
            residency=want_residency,
            logical_to_physical=want_state[0],
            logical_replica_count=want_state[1],
            source_local=want_state[2],
            layout=want_state[3],
            per_rank_experts=PER_RANK,
            replica_slots_per_rank=SLOTS,
            source_rank=source_rank,
        )
        publish_plan_fused(
            plan=plan,
            residency=got_residency,
            logical_to_physical=got_state[0],
            logical_replica_count=got_state[1],
            source_local=got_state[2],
            layout=got_state[3],
            per_rank_experts=PER_RANK,
            replica_slots_per_rank=SLOTS,
            source_rank=source_rank,
        )
        for name, want, got in zip(
            ("logical_to_physical", "replica_count", "source_local", "layout"),
            want_state,
            got_state,
        ):
            assert torch.equal(want, got), (
                f"step {step} with plan {plan_values}: {name} differs\n"
                f"want {want.tolist()}\ngot  {got.tolist()}"
            )
        assert torch.equal(want_residency.state, got_residency), (
            f"step {step}: residency differs, want "
            f"{want_residency.state.tolist()} got {got_residency.tolist()}"
        )


def test_the_fused_path_launches_two_kernels_where_the_tensor_path_launched_132():
    """The reason the fusion exists, asserted rather than assumed.

    Counted through the profiler, because the launch count is the cost: 132 tiny kernels
    per placed layer were 0.95 ms of host time per layer and about 42 ms per forward.
    """
    from torch.profiler import ProfilerActivity, profile

    device = torch.device("cuda")
    predicted = random_load(torch.Generator().manual_seed(7), device)
    predicted[3] = 4096
    residency = torch.full((2,), -1, dtype=torch.int64, device=device)
    budget = torch.tensor(43, dtype=torch.int64, device=device)
    spent = torch.zeros((), dtype=torch.int64, device=device)
    placed = torch.zeros((), dtype=torch.int64, device=device)
    transfer = torch.zeros(4, dtype=torch.int64, device=device)
    publish = torch.zeros(4, dtype=torch.int64, device=device)
    maps = fresh_maps(device)

    def one_layer():
        plan_and_charge_fused(
            predicted,
            residency,
            budget,
            spent,
            placed,
            transfer,
            publish,
            EP_SIZE,
            MIN_TOKENS,
        )
        publish_plan_fused(
            plan=publish,
            residency=residency,
            logical_to_physical=maps[0],
            logical_replica_count=maps[1],
            source_local=maps[2],
            layout=maps[3],
            per_rank_experts=PER_RANK,
            replica_slots_per_rank=SLOTS,
            source_rank=0,
        )

    one_layer()  # compile both kernels first; the first launch of each is a load
    torch.accelerator.synchronize()

    repeats = 8
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(repeats):
            one_layer()
        torch.accelerator.synchronize()
    launches = sum(
        event.count
        for event in prof.key_averages()
        if event.self_device_time_total > 0 or event.device_time_total > 0
    )

    assert launches / repeats <= 4, (
        f"{launches / repeats:.1f} kernels per placed layer, where the point of this "
        f"module is two"
    )


@pytest.mark.parametrize(
    "ep_size,per_rank",
    [
        (8, 16),  # Qwen3-30B-A3B at the approved EP size: 128 logical experts
        (2, 64),  # what this node can actually run since it lost six GPUs
        (4, 3),  # per-rank count that is not a power of two, so the mask matters
    ],
)
def test_the_fused_plan_agrees_at_the_real_expert_geometries(ep_size, per_rank):
    """The kernel's block sizes are `constexpr`, so each geometry compiles its own.

    A fusion verified only at the shape the other tests use would be unverified at the
    shape that serves requests.
    """
    device = torch.device("cuda")
    num_logical = ep_size * per_rank
    generator = torch.Generator().manual_seed(99)
    for case in range(40):
        load = torch.randint(0, 50, (num_logical,), generator=generator).to(torch.int32)
        if case % 3 == 0:
            load[int(torch.randint(0, num_logical, (1,), generator=generator))] = 9000
        predicted = load.to(device)
        residency = torch.tensor([-1, -1], dtype=torch.int64, device=device)
        budget = torch.tensor(43, dtype=torch.int64, device=device)

        want_spent = torch.zeros((), dtype=torch.int64, device=device)
        want_placed = torch.zeros((), dtype=torch.int64, device=device)
        want_transfer, want_publish = reference_plan_and_charge(
            predicted, residency, budget, want_spent, want_placed, ep_size, MIN_TOKENS
        )

        spent = torch.zeros((), dtype=torch.int64, device=device)
        placed = torch.zeros((), dtype=torch.int64, device=device)
        transfer = torch.zeros(4, dtype=torch.int64, device=device)
        publish = torch.zeros(4, dtype=torch.int64, device=device)
        plan_and_charge_fused(
            predicted,
            residency,
            budget,
            spent,
            placed,
            transfer,
            publish,
            ep_size,
            MIN_TOKENS,
        )

        assert transfer.tolist() == want_transfer.tolist(), (
            f"EP={ep_size} per_rank={per_rank} case {case}: transfer plan differs"
        )
        assert publish.tolist() == want_publish.tolist()
        assert int(spent) == int(want_spent) and int(placed) == int(want_placed)


def test_an_empty_snapshot_places_nothing_rather_than_failing_to_compile():
    """`plan_replicas` returns no placement for this, so the fused path must not crash.

    A dummy or padding-only forward is the case, and it is the one the retained planner
    names in its own docstring. With no experts `per_rank` is 0 and the kernel would
    fail to compile on `tl.arange(0, 0)` — a crash where the tensor version places
    nothing.
    """
    device = torch.device("cuda")
    transfer = torch.full((4,), 7, dtype=torch.int64, device=device)
    publish = torch.full((4,), 7, dtype=torch.int64, device=device)
    plan_and_charge_fused(
        torch.zeros(0, dtype=torch.int32, device=device),
        torch.tensor([-1, -1], dtype=torch.int64, device=device),
        torch.tensor(43, dtype=torch.int64, device=device),
        torch.zeros((), dtype=torch.int64, device=device),
        torch.zeros((), dtype=torch.int64, device=device),
        transfer,
        publish,
        EP_SIZE,
        MIN_TOKENS,
    )
    assert transfer.tolist() == [0, 0, 0, 0]
    assert publish.tolist() == [0, 0, 0, 0]


def test_a_strided_snapshot_is_refused_rather_than_read_wrong():
    """The kernel indexes the load directly, so a strided view would read garbage.

    The tensor planner it replaces handles any stride, and every equality test above
    builds contiguous tensors — exactly the difference a test would not catch.
    """
    device = torch.device("cuda")
    strided = torch.zeros((NUM_LOGICAL, 2), dtype=torch.int32, device=device)[:, 0]
    with pytest.raises(ValueError, match="unit innermost stride"):
        plan_and_charge_fused(
            strided,
            torch.tensor([-1, -1], dtype=torch.int64, device=device),
            torch.tensor(43, dtype=torch.int64, device=device),
            torch.zeros((), dtype=torch.int64, device=device),
            torch.zeros((), dtype=torch.int64, device=device),
            torch.zeros(4, dtype=torch.int64, device=device),
            torch.zeros(4, dtype=torch.int64, device=device),
            EP_SIZE,
            MIN_TOKENS,
        )
