# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The device planner, tested against the host planner it must agree with exactly.

Ticket 04. Moving the placement decision to the device is what lets the plan stop
reaching the host, and the host synchronisation it removes is 70% of what prediction
costs. But every rank derives the plan locally from the same snapshot and no plan is
broadcast, so **a plan that differs by rank pairs a sender with no receiver and hangs
the engine**. Agreement is therefore a correctness property and the acceptance criterion
is equality, not similarity.

`plan_replicas` is the oracle. It is the code that was measured, and keeping it in the
tree for this comparison is the reason it was not deleted when the orchestration was
replaced.

The randomised sweep is the load-bearing test. The cases that break an implementation
like this are ties, all-zero load, and loads where relocating the peak does not lower it
— none of which a hand-picked example reliably contains, and all of which a few hundred
random draws do.
"""

import pytest
import torch

from vllm.distributed.eplb.predictive_planner import (
    plan_one_layer_on_device,
    plan_replicas,
)


def _host(load: torch.Tensor, ep_size: int, min_tokens: float):
    """The host planner's answer, as `(expert, target, moved_x2)` or None."""
    chosen = plan_replicas(
        load.view(1, -1).to(torch.float64), ep_size, budget=1, min_tokens=min_tokens
    )
    if not chosen:
        return None
    p = chosen[0]
    return (p.logical_expert, p.target_rank, round(p.moved_load * 2))


def _device(load: torch.Tensor, ep_size: int, min_tokens: float):
    """The device planner's answer in the same shape, so the two can be compared."""
    out = plan_one_layer_on_device(load, ep_size, min_tokens)
    found, expert, target, moved_x2 = (int(v) for v in out.tolist())
    return (expert, target, moved_x2) if found else None


def _both(load: torch.Tensor, ep_size: int = 8, min_tokens: float = 0.0):
    return _host(load, ep_size, min_tokens), _device(load, ep_size, min_tokens)


def test_a_single_hot_expert_is_replicated_onto_the_lightest_rank():
    """The case the feature exists for, as a readable anchor before the sweep."""
    load = torch.ones(128, dtype=torch.int64)
    load[3] = 400  # rank 0 owns experts 0..15

    host, device = _both(load)

    assert host == device
    assert device is not None and device[0] == 3


def test_an_all_zero_layer_places_nothing():
    """A dummy or padding-only forward must produce no placement on either path."""
    host, device = _both(torch.zeros(128, dtype=torch.int64))

    assert host is None and device is None


def test_a_perfectly_balanced_layer_places_nothing():
    """With nothing to gain, the positive-benefit test must refuse on both paths.

    Moving half an expert off the peak of an already even layer makes the target the new
    peak, so the gain is not positive and relocating the peak is not lowering it.
    """
    host, device = _both(torch.full((128,), 10, dtype=torch.int64))

    assert host is None and device is None


def test_a_move_below_the_minimum_is_refused():
    """Below one block per expert a replica saves no block, so it saves no time."""
    load = torch.ones(128, dtype=torch.int64)
    load[3] = 400

    host, device = _both(load, min_tokens=1e9)

    assert host is None and device is None


def test_the_minimum_is_compared_against_the_moved_half_not_the_expert():
    """Source-rank routing sheds half an expert, so the floor applies to the half.

    Getting this wrong by a factor of two would admit placements that save no block, or
    refuse ones that do, and neither would fail loudly.
    """
    load = torch.ones(128, dtype=torch.int64)
    load[3] = 400  # moved is 200

    assert _both(load, min_tokens=200.0) != (None, None), "200 must be admissible"
    below, below_device = _both(load, min_tokens=201.0)
    assert below is None and below_device is None, "201 must not be"


def test_ties_between_equally_hot_experts_go_to_the_lowest_id():
    """The tie-break is the plan's determinism, so it is tested directly.

    Two experts on the peak rank with identical load give identical gain; the host
    planner orders by the `Placement` tuple, which resolves to the lowest expert id. A
    device argmax left to its own tie behaviour would be free to pick either, and two
    ranks picking differently is the deadlock this comparison guards.
    """
    load = torch.ones(128, dtype=torch.int64)
    load[5] = 300
    load[9] = 300  # same rank, same load

    host, device = _both(load)

    assert host == device
    assert device is not None and device[0] == 5


def test_ties_between_equally_light_targets_go_to_the_lowest_rank():
    load = torch.ones(128, dtype=torch.int64)
    load[3] = 400  # peak is rank 0; every other rank is equally light

    host, device = _both(load)

    assert host == device
    assert device is not None and device[1] == 1


def test_ties_for_the_peak_rank_go_to_the_lowest_rank():
    """Identical rank totals: the host takes the first, so must the device."""
    load = torch.ones(128, dtype=torch.int64)
    load[3] = 400
    load[19] = 400  # rank 0 and rank 1 now carry the same total

    host, device = _both(load)

    assert host == device


@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_it_agrees_at_every_ep_size(ep_size):
    torch.manual_seed(ep_size)
    load = torch.randint(0, 50, (128,), dtype=torch.int64)
    load[7] = 500

    host, device = _both(load, ep_size=ep_size)

    assert host == device


@pytest.mark.parametrize("min_tokens", [0.0, 1.0, 64.0, 128.0])
def test_it_agrees_over_many_random_layers(min_tokens):
    """The sweep. Skew, flatness and near-ties all appear here without being contrived.

    Loads are drawn so that a plain uniform draw and a heavily skewed draw both occur,
    since the first exercises the positive-benefit refusal and the second the ordinary
    placement.
    """
    torch.manual_seed(int(min_tokens) + 11)
    for trial in range(300):
        load = torch.randint(0, 40, (128,), dtype=torch.int64)
        if trial % 3 == 0:
            load[int(torch.randint(0, 128, (1,)))] = int(torch.randint(100, 900, (1,)))
        if trial % 7 == 0:
            load[:] = load[0]  # perfectly flat
        host, device = _both(load, min_tokens=min_tokens)
        assert host == device, (
            f"disagreement at trial {trial}, min_tokens={min_tokens}: "
            f"host={host} device={device}"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="agreement must hold on CUDA too"
)
@pytest.mark.parametrize("min_tokens", [0.0, 128.0])
def test_it_agrees_on_cuda(min_tokens):
    """The reduction runs on the device in production, so the sweep runs there too.

    A CPU-only equality would not catch an ordering-dependent reduction, which is the
    failure mode that makes two ranks disagree.
    """
    torch.manual_seed(7)
    for _ in range(100):
        load = torch.randint(0, 40, (128,), dtype=torch.int64)
        load[int(torch.randint(0, 128, (1,)))] = int(torch.randint(100, 900, (1,)))
        host = _host(load, 8, min_tokens)
        device = _device(load.cuda(), 8, min_tokens)
        assert host == device


def test_the_result_is_a_device_tensor_the_transfer_can_read_without_the_host():
    """The point of the ticket: consumable without a host round trip."""
    load = torch.ones(128, dtype=torch.int64)
    load[3] = 400

    out = plan_one_layer_on_device(load, 8, 0.0)

    assert out.dtype == torch.int64
    assert out.device == load.device
    assert tuple(out.shape) == (4,)
