# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-forward placement coordination for Predictive expert replication.

Planning from *predicted* load rather than measured load is what makes the transfer
hideable, and that is the reason the design predicts at all. Routing maps logical
experts to physical rows **before** dispatch, so a placement decided from a layer's own
measured load would have to finish transferring before that same layer's dispatch begins
— leaving no window and exposing the whole transfer. Predicting `i + lookahead` buys the
intervening layers' compute to hide it in. Spec sections 9 and 10.

Three phases, one per layer boundary:

1. `record_prediction`, at the source layer once its snapshot is ready: start an
   **async** copy of the snapshot into pinned host memory. Nothing else, because this
   runs inside the forward and a device-to-host sync here would stall it.
2. `plan_and_launch`, at the next layer boundary: the copy has landed, so the
   deterministic planner runs on the host snapshot and the P2P goes out on the
   predictive stream.
3. `activate`, at the target layer before its routing: wait for the transfer, hand back
   the placements so the caller can publish that one layer's map.

An earlier version of this path planned once per forward from exact load in `step()`,
after the forward had finished. That has no window by construction — the information
does not exist until the work it would have hidden behind is done — and it measured 53%
worse TTFT for that reason.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from typing import Protocol

import torch

from vllm.distributed.eplb.predictive_planner import (
    Placement,
    plan_replicas,
    reconcile,
    transfer_replicas,
)


class _Communicator(Protocol):
    """What this path needs of the EPLB communicator.

    Narrower than `EplbCommunicator` on purpose: it states the four methods used
    here, so tests can substitute a recorder without a process group, and typing still
    catches a call the real communicator does not answer.
    """

    def set_stream(self, cuda_stream: torch.cuda.Stream | None) -> None: ...

    def add_send(
        self, tensors: list[torch.Tensor], dst_rank: int, expert_id: int
    ) -> None: ...

    def add_recv(
        self, tensors: list[torch.Tensor], src_rank: int, expert_id: int
    ) -> None: ...

    def execute(self) -> None: ...


class _TransferEvent(Protocol):
    """The slice of `torch.cuda.Event` this path uses.

    Declared so a test can substitute a recorder without a CUDA context, and so the
    three calls that matter are named. `synchronize` blocks the host and is used only
    where the host must read the data; `wait` orders a stream and is used everywhere
    else. Neither is ever conditional — see the collective-safety rule.
    """

    def record(self, stream: torch.cuda.Stream | None = ...) -> None: ...

    def synchronize(self) -> None: ...

    def wait(self, stream: torch.cuda.Stream | None = ...) -> None: ...


class PlacementCoordinator:
    """Drives predict, plan, transfer and activate across layer boundaries.

    Attributes:
        communicator: The EPLB communicator; exposed for tests to inspect.
        last_event: The most recently recorded transfer event, for tests.
    """

    # False, and it cannot be otherwise: `plan_and_launch` synchronises on the snapshot
    # copy, so calling it at the predicting layer's tail stalls the layer that just
    # issued that copy. Launching a layer later is what buys the copy a layer of compute
    # to land in, at the cost of an overlap window one whole MoE layer wider than the
    # design wants — and at `lookahead = 1` of no window at all, which configuration
    # validation rejects for this path. Ticket 07 moves the launch for the device path
    # only, where nothing waits on the host.
    launch_at_predicting_layer_tail = False

    def __init__(
        self,
        ep_size: int,
        ep_rank: int,
        canonical_per_rank: int,
        replica_slots_per_rank: int,
        num_layers: int,
        lookahead: int,
        budget: int,
        max_per_layer: int,
        min_tokens_per_expert: float,
        min_tokens: float,
        expert_weights: Sequence[Sequence[torch.Tensor]],
        expert_buffer: Sequence[torch.Tensor],
        communicator: _Communicator,
        stream: torch.cuda.Stream | None = None,
        publish: Callable[[int, list[Placement]], None] | None = None,
        event_factory: Callable[[], _TransferEvent] | None = None,
    ):
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.canonical_per_rank = canonical_per_rank
        self.replica_slots_per_rank = replica_slots_per_rank
        self.num_layers = num_layers
        self.lookahead = lookahead
        self.budget = budget
        self.min_tokens = min_tokens
        self.expert_weights = expert_weights
        self.expert_buffer = expert_buffer
        self.communicator = communicator
        self._spent = 0
        self._last_target = -1
        self.max_per_layer = max_per_layer
        self.min_tokens_per_expert = min_tokens_per_expert
        self._suppressed = False
        self._active: dict[int, set[tuple[int, int, int]]] = {}
        self.stream = stream
        self._publish = publish
        self._event_factory = event_factory or torch.cuda.Event
        self.last_event: _TransferEvent | None = None

        # One recording in flight: the snapshot buffer is reused, so a second recording
        # before `plan_and_launch` would overwrite the first.
        self._recorded: tuple[int, torch.Tensor, _TransferEvent] | None = None
        self._pending: dict[int, tuple[list[Placement], _TransferEvent]] = {}
        self._host_buffer: torch.Tensor | None = None

    def host_snapshot_buffer(self, size: int) -> torch.Tensor:
        """Pinned host buffer the snapshot is copied into, allocated once.

        Pinned so the copy can be async; a pageable destination would make
        `non_blocking` a no-op and stall the forward.
        """
        if self._host_buffer is None or self._host_buffer.numel() < size:
            self._host_buffer = torch.empty(
                size, dtype=torch.int64, pin_memory=torch.cuda.is_available()
            )
        return self._host_buffer[:size]

    def note_forward_token_load(self, tokens_per_expert: float | None) -> None:
        """Open a forward and decide whether it may change placement at all.

        Call once per forward, before its first `plan_and_launch`, with a token count
        **every rank agrees on** — `num_tokens_across_dp_cpu` is the result of the DP
        coordination all-reduce and is already on the host, so this costs nothing and
        cannot diverge by rank.

        This exists because suppression cannot be decided from the snapshot alone. On a
        decode forward the runner skips prediction on this same bar, so nothing is ever
        recorded, so `plan_and_launch` returns at its `_recorded is None` guard *before*
        reaching the snapshot check — and suppression stayed at whatever the last
        prefill forward left it. `activate_and_publish` then published an empty set on
        every layer, which is the revert path, on every decode and every
        `execute_dummy_batch`. That silently undid `reconcile`'s "transfer only the
        difference": with `_active` emptied between them, consecutive prefill forwards
        re-sent the whole set.

        Args:
            tokens_per_expert: This forward's tokens per logical expert, after the
                allgather. Below `min_tokens_per_expert` the MoE kernel pads every
                expert to the same block count, so no placement can save time.
        """
        self._spent = 0
        # `None` means the DP token count was unavailable, which happens on the
        # diagnostic paths and is exactly where `_prediction_is_worth_it` returns True.
        # So this must not suppress: the two gates have to agree, and disagreement is
        # what made every decode forward revert all 48 layers.
        self._suppressed = (
            tokens_per_expert is not None
            and tokens_per_expert <= self.min_tokens_per_expert
        )

    def record_prediction(self, source_layer: int, predicted: torch.Tensor) -> None:
        """Start copying a source layer's predicted load to the host. No planning.

        Args:
            source_layer: The layer that produced the prediction.
            predicted: Predicted per-logical-expert load for `source_layer +
                lookahead`. May be a `[ep_size, num_logical]` snapshot or already
                summed.

        Raises:
            ValueError: If the target layer would be past the last one, which means
                the caller bound a source it should not have.
        """
        target = source_layer + self.lookahead
        if target >= self.num_layers:
            raise ValueError(
                f"layer {source_layer} predicts layer {target}, which is beyond the "
                f"last layer ({self.num_layers - 1}); this source should not be bound."
            )
        summed = predicted if predicted.dim() == 1 else predicted.sum(dim=0)
        host = self.host_snapshot_buffer(summed.numel())
        # Integer end to end. The counts are int32 and their sum is exact in int64, so
        # nothing here rounds and nothing depends on summation order. A float snapshot
        # is safe on the host planner, which promotes to float64 anyway, but a
        # device-side argmax over floats is order-dependent: two ranks reducing the same
        # values in a different block order can pick different experts, and the plan has
        # to be identical on every rank. Keeping it integer makes that constructive
        # rather than something the kernel has to be careful about.
        host.copy_(summed.to(dtype=torch.int64), non_blocking=True)
        event = self._event_factory()
        # Recorded on the **current** stream, which is where the copy was enqueued.
        # Recording it on the predictive stream instead — as a first version did — makes
        # the later wait vacuous: it waits on a stream the copy never touched, the
        # planner then reads a partly-filled host buffer whose contents differ by rank,
        # the ranks produce plans that do not pair, and the engine deadlocks with no
        # error. That is what a whole run died of.
        event.record()
        self._recorded = (target, host, event)

    def plan_and_launch(self) -> list[Placement]:
        """Plan from the recorded snapshot and launch its transfers.

        Called one layer boundary after `record_prediction`, so the async copy has had a
        layer of compute to land in. The wait is unconditional rather than a poll: a
        per-rank decision about whether the copy arrived would let ranks plan from
        different data and produce plans that do not pair.
        """
        if self._recorded is None:
            return []
        target, host, copy_event = self._recorded
        self._recorded = None
        copy_event.synchronize()
        # `max_transfers_per_forward` caps transfers across the whole forward, not per
        # layer. Passing the full budget to each layer's one-row plan spent it 43 times
        # over: a run measured 131 replicas per forward against a budget of 43, roughly
        # 23 ms of PCIe transfer per forward on this node, charged to every decode
        # forward as well. Layers are visited in increasing order within a forward, so a
        # target that does not advance means a new forward has begun.
        if target <= self._last_target:
            self._spent = 0
        self._last_target = target
        # Below one block per expert the MoE kernel pads every expert to the same number
        # of blocks, so the imbalance costs nothing and balancing it saves nothing —
        # ticket 00's inequality. Decode never clears this bar on this node and measured
        # -1.2% imbalance for a 10% TPOT cost, so a decode forward runs the placement
        # machinery not at all: no plan, no transfer, and no publish, so what prefill
        # put there stays and does not have to be sent again.
        #
        # `host` is the allgathered snapshot and identical on every rank, so this
        # decision is too. Gating a transfer on per-rank state is the deadlock this
        # branch has already hit twice. Assigns rather than only setting True.
        # `note_forward_token_load` is the authority for a forward, and it decides from
        # the DP-agreed padded count; this decides from the unpadded snapshot, which is
        # the more accurate of the two. Having the snapshot only ever set True meant a
        # boundary reset had to clear it, and that reset silently overrode the runner —
        # harmless while both thresholds are the same value, and a revert of every layer
        # the moment they diverge.
        tokens_per_expert = float(host.sum()) / host.numel()
        self._suppressed = tokens_per_expert <= self.min_tokens_per_expert
        if self._suppressed:
            return []
        remaining = self.budget - self._spent
        if remaining <= 0:
            return []
        # One layer's row, because that is all that exists yet: when this layer plans
        # for `target`, no later layer's prediction has been computed. So the
        # cross-layer ranking `plan_replicas` implements never has a second candidate
        # here, and the forward's budget is spent **first-come-first-served by layer
        # index**. That makes `max_per_layer` the allocation target rather than a safety
        # cap, and it decides coverage: at cap `k` a budget of `b` reaches `b/k` layers,
        # always the lowest-indexed ones.
        #
        # Note what charging for transfers means, since the arithmetic above no longer
        # holds across forwards: a layer whose replica is already resident costs
        # nothing, so coverage ratchets up over successive forwards until every
        # reachable layer holds one. That is the intended trade — coverage is free once
        # resident and the budget's job is to bound *churn*, which it still does: a
        # domain switch invalidates every replica and the next forward may then move
        # only `budget` of them. The offline control
        # `imbalance.plan_moves_in_layer_order` charges every placement regardless of
        # residency, so it models the first forward under the old accounting, and the
        # `b/k` figures it produced describe that rather than the steady state.
        #
        # Coverage is what drives benefit, because a layer's second replica chases a
        # much smaller expert than its first. Handing each layer the whole remaining
        # budget covered 6-7 layers at ~5 replicas each and removed 5.0% of prefill
        # excess; a cap of 2 covers 22 and removes 16.9%. Offline at equal budget, a cap
        # of 1 covers all 43 reachable layers and removes 48.0% against a global-ranking
        # oracle's 48.3% — which is why the default is 1. See `bench/RESULTS.md`,
        # 2026-08-29, and `imbalance.plan_moves_in_layer_order`, which is this
        # allocation reproduced offline so the oracle has a fair control.
        plan = plan_replicas(
            host.unsqueeze(0),
            self.ep_size,
            min(remaining, self.max_per_layer),
            self.min_tokens,
        )
        if not plan:
            return []
        # `plan_replicas` numbers layers from its own input, which held one row.
        plan = [
            Placement(
                target,
                p.logical_expert,
                p.source_rank,
                p.target_rank,
                p.moved_load,
            )
            for p in plan
        ]
        # Only the difference is sent. A replica already resident needs no transfer: its
        # row still holds that expert's weights, because reverting never touches weights
        # and nothing else writes a replica row. With coverage steady at 22 layers,
        # consecutive prefill forwards want largely the same set, so this is most of the
        # 43 transfers that were being repeated every forward.
        #
        # `active` comes from this coordinator's own published history, which is driven
        # by plans identical on every rank, so the transfer set stays identical across
        # ranks — the property the sends and receives pair on.
        active = self._active.get(target, set())
        _keep, _revert, to_transfer = reconcile(active, plan)
        # The transfer belongs on the ordered predictive stream, so it overlaps the
        # layers between here and the target rather than serializing into the forward.
        if self.stream is not None:
            self.communicator.set_stream(self.stream)
        # And it has to be ordered *behind* what the compute stream has already
        # enqueued. `transfer_replicas` drains into a replica row, and the previous
        # forward's MoE kernel may still be reading that row: nothing in stream ordering
        # connects the two otherwise, and since `activate` became a stream wait rather
        # than a host block the CPU runs further ahead, widening the window rather than
        # closing it. Recorded on the compute stream explicitly and before entering the
        # other one, because `record()` takes the *current* stream — inside
        # `torch.cuda.stream(self.stream)` that is the predictive stream and the wait
        # would be on its own work. The device path had exactly that bug.
        compute = torch.cuda.current_stream() if self.stream is not None else None
        if compute is not None:
            barrier = self._event_factory()
            barrier.record(compute)
        with self._transfer_stream():
            if compute is not None and self.stream is not None:
                self.stream.wait_event(barrier)
            transfer_replicas(
                to_transfer,
                expert_weights=self.expert_weights,
                expert_buffer=self.expert_buffer,
                ep_rank=self.ep_rank,
                per_rank_experts=self.canonical_per_rank,
                communicator=self.communicator,
            )
            event = self._event_factory()
            event.record(self.stream)
        if self.stream is not None:
            self.communicator.set_stream(None)
        self.last_event = event
        self._pending[target] = (plan, event)
        # Charged for what actually moved, not for what was planned. Counting the whole
        # plan made a knob named for transfers bound *coverage* instead: a replica
        # already resident needs no transfer, because its row still holds that expert's
        # weights and nothing else writes a replica row, so in a steady state the honest
        # charge is zero. Bytes in flight are bounded separately, which is what
        # `max_concurrent_transfer_bytes` is for.
        self._spent += len(to_transfer)
        return plan

    def activate_and_publish(self, layer: int) -> list[Placement]:
        """Wait for `layer`'s transfer and publish its **complete** desired set.

        Publishing unconditionally is the point. `placements` is the layer's whole
        desired set, so an empty one means "this layer should hold no replica" and
        publishing it is what reverts whatever the last forward left there. Skipping the
        call when a layer planned nothing — as the runner first did — left reversion
        running only on layers that happened to receive a new placement, and active
        replicas stayed at a measured 53-66 per forward against a budget of 43 even
        after reversion was written.

        Returns:
            The placements activated, for the caller's own accounting.
        """
        if self._suppressed:
            # Leave this layer exactly as the last unsuppressed forward left it.
            return []
        placements = self.activate(layer)
        self.publish(layer, placements)
        self._active[layer] = {
            (p.layer, p.logical_expert, p.target_rank) for p in placements
        }
        return placements

    def publish(self, layer: int, placements: list[Placement]) -> None:
        """Hand an activation to whoever owns the routing maps.

        A seam so the runner never reaches into the EPLB state, and so the coordinator
        stays testable without one.
        """
        if self._publish is not None:
            self._publish(layer, placements)

    def _transfer_stream(self):
        """Context for the transfer: the predictive stream, or the current one."""
        if self.stream is None:
            return contextlib.nullcontext()
        return torch.cuda.stream(self.stream)

    def pending_activation(self, layer: int) -> list[Placement] | None:
        """Placements waiting to be activated at `layer`, if any."""
        entry = self._pending.get(layer)
        return None if entry is None else entry[0]

    def activate(self, layer: int) -> list[Placement]:
        """Order `layer`'s transfer ahead of its use and hand back what to activate.

        The wait is unconditional. Whether a transfer has landed is per-rank timing, and
        two ranks disagreeing would leave them routing against different maps.

        It is a *stream* wait, not a host one. Everything that consumes the weights —
        the map writes here, then this layer's MoE kernel — is enqueued on the current
        stream afterwards, so stream ordering is the whole requirement and blocking the
        host bought nothing.
        """
        entry = self._pending.pop(layer, None)
        if entry is None:
            return []
        placements, event = entry
        # `wait()` with no argument orders the *current* stream behind the event.
        # `synchronize()` would block the host instead, which this site never needed:
        # everything that consumes the weights — the map writes below, then this layer's
        # MoE kernel — is enqueued on the current stream afterwards, so stream ordering
        # is the whole requirement. Blocking the CPU only threw away its run-ahead, and
        # a placed run made 126 `cudaEventSynchronize` calls with GPU occupancy falling
        # 86.8% -> 52.9%. About half of those were here.
        #
        # The host copy in `plan_and_launch` is not this and must stay a real
        # `synchronize()`: the planner reads that buffer on the host. Only a device-side
        # plan removes that one, which is the rest of ticket 13.
        event.wait()
        return placements
