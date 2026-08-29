# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the in-forward placement coordinator.

The coordinator is what makes the transfer hideable. Planning from *predicted* load at
layer `i` is not a compromise for accuracy's sake — it is the only thing that creates a
window: routing maps logical experts to physical rows before dispatch, so a placement
decided from layer `i`'s own measured load would have to transfer before that layer's
dispatch begins, leaving nothing to overlap. Predicting `i + lookahead` buys the
intervening layers' compute.

Three phases, one per layer boundary, matching spec section 9:

* `record_prediction` at layer `i`, after its snapshot: start an async copy of the
  snapshot to pinned host memory. No host sync, because this runs inside the forward.
* `plan_and_launch` at the next layer: the copy has landed, so the deterministic planner
  runs on the CPU snapshot and the P2P goes out on the predictive stream.
* `activate` at layer `i + lookahead`, before its routing: wait for the transfer and
  publish that layer's map row.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vllm.distributed.eplb.predictive_coordinator import PlacementCoordinator


class _FakeComm:
    def __init__(self):
        self.sends: list[tuple[int, int]] = []
        self.recvs: list[tuple[int, int]] = []
        self.executed = 0

    def add_send(self, tensors, dst_rank, expert_id):
        self.sends.append((dst_rank, expert_id))

    def add_recv(self, tensors, src_rank, expert_id):
        self.recvs.append((src_rank, expert_id))

    def execute(self):
        self.executed += 1

    def set_stream(self, stream):
        pass


class _FakeEvent:
    """Stands in for a CUDA event; records that it was waited on."""

    def __init__(self):
        self.synchronized = 0

    def record(self, stream=None):
        pass

    def synchronize(self):
        self.synchronized += 1


def _coordinator(
    ep_size=4,
    per_rank=4,
    slots=1,
    layers=6,
    budget=4,
    lookahead=2,
    max_per_layer=8,
    min_tokens_per_expert=0.0,
):
    weights = [[torch.zeros(per_rank + slots, 3)] for _ in range(layers)]
    return PlacementCoordinator(
        ep_size=ep_size,
        ep_rank=0,
        canonical_per_rank=per_rank,
        replica_slots_per_rank=slots,
        num_layers=layers,
        lookahead=lookahead,
        budget=budget,
        max_per_layer=max_per_layer,
        min_tokens_per_expert=min_tokens_per_expert,
        min_tokens=0.0,
        expert_weights=weights,
        expert_buffer=[torch.zeros(per_rank + slots, 3)],
        communicator=_FakeComm(),
        event_factory=_FakeEvent,
    )


def _skewed(num_logical=16):
    """A load vector whose first expert dominates, so a placement is worthwhile."""
    load = torch.ones(num_logical)
    load[0] = 40.0
    return load


class TestPhasesHappenAtTheRightLayers:
    def test_recording_does_not_plan_or_send(self):
        """Planning at the recording layer would need a host sync inside the forward."""
        coordinator = _coordinator()
        coordinator.record_prediction(source_layer=0, predicted=_skewed())
        assert coordinator.communicator.executed == 0
        assert coordinator.pending_activation(2) is None

    def test_launching_happens_after_recording_not_with_it(self):
        coordinator = _coordinator()
        coordinator.record_prediction(source_layer=0, predicted=_skewed())
        coordinator.plan_and_launch()
        assert coordinator.communicator.executed == 1
        assert coordinator.pending_activation(2) is not None

    def test_the_target_layer_is_the_source_plus_lookahead(self):
        coordinator = _coordinator(lookahead=2)
        coordinator.record_prediction(source_layer=3, predicted=_skewed())
        coordinator.plan_and_launch()
        assert coordinator.pending_activation(5) is not None
        assert coordinator.pending_activation(4) is None

    def test_launching_with_nothing_recorded_is_a_no_op(self):
        coordinator = _coordinator()
        coordinator.plan_and_launch()
        assert coordinator.communicator.executed == 0

    def test_activation_waits_for_the_transfer(self):
        coordinator = _coordinator()
        coordinator.record_prediction(source_layer=0, predicted=_skewed())
        coordinator.plan_and_launch()
        placements = coordinator.activate(2)
        assert placements
        assert coordinator.last_event.synchronized == 1

    def test_activating_a_layer_with_nothing_pending_returns_nothing(self):
        coordinator = _coordinator()
        assert coordinator.activate(2) == []

    def test_activation_clears_the_pending_entry(self):
        coordinator = _coordinator()
        coordinator.record_prediction(source_layer=0, predicted=_skewed())
        coordinator.plan_and_launch()
        coordinator.activate(2)
        assert coordinator.pending_activation(2) is None


class TestNoHostSyncOnTheRecordingPath:
    """Recording runs inside the forward and must not stall it."""

    def test_the_snapshot_copy_into_host_memory_is_non_blocking(self):
        """A blocking device-to-host copy here would stall the forward every layer."""
        coordinator = _coordinator()
        seen = {}
        real = torch.Tensor.copy_

        def spy(self, other, *args, **kwargs):
            if self.device.type == "cpu":
                seen["non_blocking"] = kwargs.get("non_blocking", False)
            return real(self, other, *args, **kwargs)

        torch.Tensor.copy_ = spy
        try:
            coordinator.record_prediction(source_layer=0, predicted=_skewed())
        finally:
            torch.Tensor.copy_ = real
        assert seen.get("non_blocking") is True

    def test_the_host_buffer_is_pinned_when_cuda_is_available(self):
        coordinator = _coordinator()
        buffer = coordinator.host_snapshot_buffer(16)
        assert buffer.device.type == "cpu"
        if torch.cuda.is_available():
            assert buffer.is_pinned()


class TestBudgetAndLegality:
    def test_the_budget_bounds_placements_per_target_layer(self):
        coordinator = _coordinator(budget=2)
        load = torch.ones(16)
        load[0] = 40.0
        load[1] = 30.0
        load[2] = 20.0
        coordinator.record_prediction(source_layer=0, predicted=load)
        coordinator.plan_and_launch()
        assert len(coordinator.pending_activation(2)) <= 2

    def test_a_balanced_prediction_launches_nothing(self):
        coordinator = _coordinator()
        coordinator.record_prediction(source_layer=0, predicted=torch.ones(16))
        coordinator.plan_and_launch()
        assert coordinator.communicator.executed == 0
        assert coordinator.pending_activation(2) is None

    def test_a_target_past_the_last_layer_is_refused(self):
        coordinator = _coordinator(layers=6, lookahead=2)
        with pytest.raises(ValueError, match="beyond the last layer"):
            coordinator.record_prediction(source_layer=5, predicted=_skewed())


class TestWaitingIsNeverPerRank:
    """Whether a copy or a transfer has landed is per-rank timing.

    Two ranks disagreeing would plan from different data, or route against different
    maps. Both waits are therefore unconditional. This lesson cost two silent
    deadlocks earlier in this branch, once in a diagnostic and once in a placement
    driver, so it is pinned rather than remembered.
    """

    def test_neither_phase_polls_an_event(self):
        import inspect

        from vllm.distributed.eplb import predictive_coordinator

        source = inspect.getsource(predictive_coordinator)
        assert ".query()" not in source, (
            "event.query() is per-rank timing; use synchronize() so every rank "
            "proceeds on the same layer"
        )

    def test_both_phases_wait(self):
        coordinator = _coordinator()
        coordinator.record_prediction(source_layer=0, predicted=_skewed())
        coordinator.plan_and_launch()  # waits for the snapshot copy
        coordinator.activate(2)  # waits for the transfer
        assert coordinator.last_event.synchronized == 1

    def test_the_coordinator_takes_no_per_rank_flag(self):
        """Nothing like `is_dummy` may reach a decision that gates a collective."""
        import inspect

        from vllm.distributed.eplb.predictive_coordinator import PlacementCoordinator

        for name in ("record_prediction", "plan_and_launch", "activate"):
            params = list(
                inspect.signature(getattr(PlacementCoordinator, name)).parameters
            )
            assert "is_dummy" not in params


class TestTheSnapshotEventBelongsToTheCopyStream:
    """The wait on the snapshot copy must not be vacuous.

    A first version recorded that event on the predictive stream while the copy was
    enqueued on the current one. The wait then guaranteed nothing, the planner read a
    partly-filled host buffer whose contents differ by rank, the ranks produced plans
    that did not pair, and the engine deadlocked with no error. A whole run died of it.
    """

    def test_the_copy_event_is_recorded_on_the_current_stream(self):
        recorded: list[object] = []

        class _Recorder:
            def record(self, stream=None):
                recorded.append(stream)

            def synchronize(self):
                pass

        coordinator = _coordinator()
        coordinator.stream = "a-different-stream"
        coordinator._event_factory = _Recorder
        coordinator.record_prediction(source_layer=0, predicted=_skewed())

        assert recorded == [None], (
            "the snapshot event must be recorded on the stream the copy was enqueued "
            "on, which is the current one, not the predictive stream"
        )


class TestTheTransferBudgetIsPerForwardNotPerLayer:
    """`max_transfers_per_forward` caps the forward, not each layer.

    Each layer plans from a one-row snapshot, so handing every layer the full budget
    spent it once per layer: a run measured 131 replicas against a budget of 43,
    roughly 23 ms of PCIe transfer per forward, charged to decode forwards too.
    """

    def test_the_budget_is_shared_across_the_layers_of_one_forward(self):
        coordinator = _coordinator(budget=3, layers=16)
        placed = 0
        for source in range(0, 14):
            coordinator.record_prediction(source_layer=source, predicted=_skewed())
            placed += len(coordinator.plan_and_launch())
        assert placed <= 3, (
            f"{placed} replicas placed across the forward against a budget of 3; the "
            "budget is being spent per layer"
        )

    def test_the_budget_refreshes_when_the_next_forward_begins(self):
        coordinator = _coordinator(budget=2, layers=16)
        first = 0
        for source in range(0, 14):
            coordinator.record_prediction(source_layer=source, predicted=_skewed())
            first += len(coordinator.plan_and_launch())
        # A target that does not advance marks a new forward.
        coordinator._pending.clear()
        second = 0
        for source in range(0, 14):
            coordinator.record_prediction(source_layer=source, predicted=_skewed())
            second += len(coordinator.plan_and_launch())
        assert first > 0 and second > 0, (
            "the second forward placed nothing, so the per-forward budget never reset"
        )


class TestEveryLayerIsPublishedEveryForward:
    """A layer that plans nothing must still be published, so it can be reverted.

    `placements` is a layer's complete desired set, so an empty one means "hold no
    replica here". Gating the publish on a non-empty activation — as the runner first
    did — left reversion running only on layers that happened to receive a new
    placement, and active replicas stayed at a measured 53-66 per forward against a
    budget of 43 even after reversion was written.
    """

    def test_a_layer_with_no_pending_plan_is_still_published(self):
        published: list[tuple[int, list]] = []
        coordinator = _coordinator()
        coordinator._publish = lambda layer, placements: published.append(
            (layer, placements)
        )

        result = coordinator.activate_and_publish(layer=3)

        assert result == []
        assert published == [(3, [])], (
            "a layer with nothing pending must be published with an empty set, which "
            "is what reverts what the previous forward left there"
        )

    def test_a_layer_with_a_pending_plan_publishes_it(self):
        published: list[tuple[int, list]] = []
        coordinator = _coordinator()
        coordinator._publish = lambda layer, placements: published.append(
            (layer, placements)
        )
        coordinator.record_prediction(source_layer=1, predicted=_skewed())
        plan = coordinator.plan_and_launch()
        assert plan, "the fixture must produce a plan for this test to mean anything"

        result = coordinator.activate_and_publish(layer=3)

        assert result == plan
        assert published == [(3, plan)]

    def test_the_runner_publishes_unconditionally(self):
        """Pins the call site, because the gate lived there rather than here."""
        source = Path(
            "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
        ).read_text()
        assert "activate_and_publish" in source, (
            "the runner must publish through activate_and_publish"
        )
        assert "if activated:" not in source, (
            "the runner must not gate the publish on a non-empty activation; that "
            "leaves stale replicas live on every layer that planned nothing"
        )


class TestEachLayerGetsOnlyItsShareOfTheBudget:
    """A layer must not spend the whole forward's transfer budget.

    Every layer plans on its own row, so handing it the full remaining budget let the
    earliest layers consume it: a measured run covered 6-7 layers at ~5 replicas each
    and removed 5.0% of prefill excess, where the same budget over ~31 layers removes
    35.0% offline. The critical path is the sum of all 48 layers' peaks, so no single
    layer is worth more than a fraction of it, and the second and later replicas on one
    layer chase progressively smaller experts.
    """

    def test_a_layer_cannot_take_more_than_the_cap(self):
        coordinator = _coordinator(budget=40, layers=16)
        coordinator.max_per_layer = 2
        coordinator.record_prediction(source_layer=0, predicted=_skewed())

        plan = coordinator.plan_and_launch()

        assert len(plan) <= 2, (
            f"one layer took {len(plan)} placements against a per-layer cap of 2, so "
            f"the early layers will exhaust the forward's budget"
        )

    def test_the_budget_still_reaches_the_later_layers(self):
        coordinator = _coordinator(budget=40, layers=16)
        coordinator.max_per_layer = 1
        covered = 0
        for source in range(14):
            coordinator.record_prediction(source_layer=source, predicted=_skewed())
            if coordinator.plan_and_launch():
                covered += 1

        assert covered >= 10, (
            f"only {covered} layers were covered; a cap of 1 with a budget of 40 must "
            f"reach every layer that has skew, which is what breadth buys"
        )

    def test_the_cap_never_exceeds_the_remaining_forward_budget(self):
        coordinator = _coordinator(budget=3, layers=16)
        coordinator.max_per_layer = 8
        placed = 0
        for source in range(14):
            coordinator.record_prediction(source_layer=source, predicted=_skewed())
            placed += len(coordinator.plan_and_launch())

        assert placed <= 3, (
            f"{placed} placements against a forward budget of 3; the per-layer cap "
            f"must not override the per-forward budget"
        )


class TestDecodeForwardsRunNoPlacement:
    """Below one block per expert, balancing cannot buy anything.

    `moe_align_block_size` pads every expert to a multiple of `BLOCK_SIZE_M`, so an
    imbalance smaller than one block costs nothing to begin with — ticket 00's
    inequality. Decode never clears that bar on this node, and measured -1.2%
    imbalance for a 10% TPOT cost. So a decode forward runs the machinery not at all:
    it must not plan, must not transfer, and must not publish, because publishing an
    empty desired set would revert what prefill placed and force it to be sent again.
    """

    def test_a_forward_below_one_block_per_expert_plans_nothing(self):
        coordinator = _coordinator(min_tokens_per_expert=128.0)
        # 4 experts, 100 assignments -> 25 per expert, well under a block.
        thin = torch.tensor([[70.0, 10.0, 10.0, 10.0]])
        coordinator.record_prediction(source_layer=0, predicted=thin)

        assert coordinator.plan_and_launch() == []
        assert coordinator.communicator.executed == 0, (
            "a gated forward must issue no transfer at all"
        )

    def test_a_forward_above_one_block_per_expert_plans_normally(self):
        coordinator = _coordinator(min_tokens_per_expert=128.0)
        fat = torch.tensor([[7000.0, 1000.0, 1000.0, 1000.0]])
        coordinator.record_prediction(source_layer=0, predicted=fat)

        assert coordinator.plan_and_launch(), (
            "a prefill-sized forward must still place; the gate is on tokens per "
            "expert, not on skew"
        )

    def test_a_gated_forward_leaves_what_prefill_placed_alone(self):
        published: list[tuple[int, list]] = []
        coordinator = _coordinator(min_tokens_per_expert=128.0)
        coordinator._publish = lambda layer, placements: published.append(
            (layer, placements)
        )
        # A prefill forward places something.
        coordinator.record_prediction(
            source_layer=0, predicted=torch.tensor([[7000.0, 1000.0, 1000.0, 1000.0]])
        )
        coordinator.plan_and_launch()
        coordinator.activate_and_publish(layer=2)
        assert published, "the prefill forward must have published"
        published.clear()

        # A decode forward follows.
        coordinator.record_prediction(
            source_layer=0, predicted=torch.tensor([[70.0, 10.0, 10.0, 10.0]])
        )
        coordinator.plan_and_launch()
        coordinator.activate_and_publish(layer=2)

        assert published == [], (
            "a gated forward must not publish; publishing an empty desired set would "
            "revert the replica prefill placed and force it to be transferred again"
        )


class TestOnlyTheDifferenceIsTransferred:
    """A replica already resident needs no transfer.

    Its row still holds that expert's weights: reverting never touches weights and
    nothing else writes a replica row. Repeating the whole set every forward was the
    fixed cost that left TTFT at 1122-1129 ms across four runs whose imbalance benefit
    varied more than threefold.
    """

    def _plan_once(self, coordinator, load):
        coordinator.record_prediction(source_layer=0, predicted=load)
        plan = coordinator.plan_and_launch()
        coordinator.activate_and_publish(layer=2)
        return plan

    def test_an_unchanged_plan_transfers_nothing_the_second_time(self):
        coordinator = _coordinator()
        load = _skewed()
        first = self._plan_once(coordinator, load)
        assert first, "the fixture must place something"
        after_first = coordinator.communicator.executed

        second = self._plan_once(coordinator, load)

        assert second == first, "the desired set is unchanged"
        assert coordinator.communicator.executed == after_first, (
            "a replica already resident must not be transferred again"
        )

    def test_the_full_desired_set_is_still_published(self):
        """Only the transfer shrinks. The published set stays complete, or reversion
        would drop replicas that are meant to stay."""
        published: list[tuple[int, list]] = []
        coordinator = _coordinator()
        coordinator._publish = lambda layer, placements: published.append(
            (layer, placements)
        )
        load = _skewed()
        first = self._plan_once(coordinator, load)
        self._plan_once(coordinator, load)

        assert published[-1][1] == first, (
            "the second forward must publish the whole desired set, not just what it "
            "transferred"
        )

    def test_a_changed_plan_transfers_the_new_replica(self):
        coordinator = _coordinator()
        self._plan_once(coordinator, _skewed())
        before = coordinator.communicator.executed

        # Put the skew on an expert owned by a different rank, so the desired set
        # genuinely changes rather than being reordered.
        moved = torch.ones(16)
        moved[8] = 40.0
        second = self._plan_once(coordinator, moved)
        assert second and second[0].logical_expert == 8, (
            f"the fixture must plan a different replica for this to test anything, "
            f"got {second}"
        )

        assert coordinator.communicator.executed > before, (
            "a replica that is not resident must still be transferred"
        )


class TestTheSnapshotStaysInteger:
    """The planner's input must not round, and must not depend on summation order.

    The counts are int32 and their sum is exact in int64. A float snapshot happens to
    be safe for the host planner, which promotes to float64, but a device-side argmax
    over floats is order-dependent: two ranks reducing the same values in a different
    block order can select different experts, and the plan must be identical on every
    rank or the replicas and the routing maps disagree. Keeping it integer makes that
    a property of the data rather than something a kernel has to be careful about.
    """

    def test_the_host_buffer_is_an_integer_dtype(self):
        coordinator = _coordinator()
        buffer = coordinator.host_snapshot_buffer(16)
        assert not buffer.dtype.is_floating_point, (
            f"the snapshot buffer is {buffer.dtype}; a float reduction of it is not "
            f"order-independent, which the rank-identical plan depends on"
        )

    def test_counts_survive_the_copy_exactly(self):
        coordinator = _coordinator()
        counts = torch.zeros(2, 16, dtype=torch.int32)
        counts[0, 3] = 16_777_217  # 2**24 + 1, the first integer float32 cannot hold
        counts[1, 3] = 0
        coordinator.record_prediction(source_layer=0, predicted=counts)

        assert coordinator._recorded is not None
        _target, host, _event = coordinator._recorded
        assert host[3].item() == 16_777_217, (
            f"got {host[3].item()}; a float32 snapshot rounds this to 16777216"
        )

    def test_the_summation_is_promoted_before_it_can_overflow(self):
        """int32 counts summed across ranks must not wrap."""
        coordinator = _coordinator()
        counts = torch.full((8, 16), 2**28, dtype=torch.int32)
        coordinator.record_prediction(source_layer=0, predicted=counts)

        assert coordinator._recorded is not None
        _target, host, _event = coordinator._recorded
        assert host[0].item() == 8 * 2**28, (
            f"got {host[0].item()}; the sum wrapped in int32 instead of promoting"
        )
