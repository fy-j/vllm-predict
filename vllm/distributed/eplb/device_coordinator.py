# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0 SPDX-FileCopyrightText: Copyright contributors to
# the vLLM project
"""In-forward placement with the plan never leaving the device.

Ticket 06. Same three phases and the same layer boundaries as `PlacementCoordinator`,
with every decision that depended on a plan value moved onto the device:

* the plan itself, from `plan_one_layer_on_device`;
* the transfer, from `DeviceExpertTransfer`'s two kernels;
* publishing, from `publish_plan_on_device`;
* residency and the transfer budget, which are what made the host need to know in the
  first place — "already resident" is what makes a transfer free, and free is what lets
  coverage ratchet up across forwards to every reachable layer.

What is left on the host is layer indices and configuration. Those are not data: the
layer being visited is known from the call, and no rank's copy of it can differ.

**Collectivity survives by construction, which the host path could not promise.** The
kernels and the barrier launch on every rank whatever the plan says, and a plan that
placed nothing makes them return immediately rather than making the host skip a
collective. The one remaining rank-agreed gate is suppression, and it is decided from
`num_tokens_across_dp_cpu` — the DP coordination all-reduce's result, already on the
host and identical everywhere.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Sequence

import torch

from vllm.distributed.eplb.device_publish import LayerResidency, publish_plan_on_device
from vllm.distributed.eplb.device_transfer import DeviceExpertTransfer, WeightPointers
from vllm.distributed.eplb.predictive_planner import (
    plan_one_layer_on_device,
    replica_row_of,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# Forwards between reports of the device-side placement counter. Large enough that the
# one synchronisation it costs is noise, small enough that a short benchmark still
# produces the line the runners look for. Lowerable, because it is the only evidence
# that a replica was placed at all and a functional run is a few forwards long — at 50
# a smoke test cannot tell a working path from an inert one.
_REPORT_EVERY = int(os.environ.get("VLLM_PREDICTIVE_PLACEMENT_REPORT_EVERY", "50"))

# Debug only: tells a crash inside the transfer apart from one in everything
# around it. Left in because that distinction took three server runs to make.
_SKIP_TRANSFER = bool(int(os.environ.get("VLLM_PREDICTIVE_SKIP_DEVICE_TRANSFER", "0")))


class LayerMaps:
    """One layer's routing tensors, as the device publish path needs them.

    Held per layer so publishing is an indexed write rather than a rebuild.
    `compute_logical_maps` inverts the whole physical layout with a Python loop and an
    `.item()`, measured at 4.707 ms per layer — 202 ms for a forward touching 43 of
    them, on the critical path.
    """

    def __init__(
        self,
        logical_to_physical: torch.Tensor,
        logical_replica_count: torch.Tensor,
        source_local: torch.Tensor,
        source_local_replica_count: torch.Tensor,
        layout: torch.Tensor,
    ):
        self.logical_to_physical = logical_to_physical
        self.logical_replica_count = logical_replica_count
        self.source_local = source_local
        self.source_local_replica_count = source_local_replica_count
        self.layout = layout


class DevicePlacementCoordinator:
    """Placement coordination that never reads a plan on the host.

    Attributes:
        last_event: The most recently recorded transfer event, for tests.
    """

    def __init__(
        self,
        ep_size: int,
        ep_rank: int,
        canonical_per_rank: int,
        replica_slots_per_rank: int,
        num_layers: int,
        lookahead: int,
        budget: int,
        min_tokens: float,
        min_tokens_per_expert: float,
        pointers: Sequence[WeightPointers],
        maps: Sequence[LayerMaps],
        transfer: DeviceExpertTransfer,
        device: torch.device,
        stream: torch.cuda.Stream | None = None,
        event_factory: Callable[[], torch.cuda.Event] | None = None,
        slot: int = 0,
    ):
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.canonical_per_rank = canonical_per_rank
        self.replica_slots_per_rank = replica_slots_per_rank
        self.num_layers = num_layers
        self.lookahead = lookahead
        self.min_tokens = min_tokens
        self.min_tokens_per_expert = min_tokens_per_expert
        self.pointers = pointers
        self.maps = maps
        self.transfer = transfer
        self.stream = stream
        self.slot = slot
        self.replica_row = replica_row_of(canonical_per_rank, slot)
        self._event_factory = event_factory or torch.cuda.Event
        self.last_event: torch.cuda.Event | None = None

        # Residency and the budget live on the device, because the host no longer learns
        # what was placed and reconstructing either by reading the plan would put the
        # synchronisation straight back.
        self._residency = [
            LayerResidency.empty(device) for _ in range(max(num_layers, 1))
        ]
        self._budget = torch.tensor(budget, dtype=torch.int64, device=device)
        self._spent = torch.zeros((), dtype=torch.int64, device=device)
        self._zero = torch.zeros((), dtype=torch.int64, device=device)

        # One row per layer, owned for this coordinator's lifetime, because the transfer
        # kernels read the plan on the **predictive** stream tens of microseconds after
        # the host enqueued them. A per-layer temporary would be freed the moment
        # `plan_and_launch` returns, and PyTorch's allocator hands a freed block to the
        # next allocation on the stream that allocated it without waiting for another
        # stream to be done with it — that is what `record_stream` exists to declare,
        # and this path never called it. Measured with the production classes
        # (`bench/probe_plan_lifetime.py`): the freed block came back on the 5th
        # allocation and the put moved the expert that had overwritten the plan instead
        # of the one the plan named. In a server the overwrite is not another plan but
        # whatever the forward allocated there, so `pe` is not a rank, and the put lands
        # NVSHMEM's proxy thread in a segfault — which is exactly the crash that kept
        # `device_issued_transfer` switched off.
        self._transfer_plans = torch.zeros(
            (max(num_layers, 1), 4), dtype=torch.int64, device=device
        )

        # Evidence that the path did something, counted on the device and read rarely.
        # `analyse_e2e.py` and the bench runners treat an activation log line as the
        # only cheap signal that an arm was not silently inert, and five defects on this
        # branch were each individually enough to make it inert. The host cannot count
        # placements per forward any more, so it counts them on the device and reads the
        # total once every `_REPORT_EVERY` forwards — one synchronisation per
        # hundred-odd forwards, against one per predicted layer.
        self._placed_total = torch.zeros((), dtype=torch.int64, device=device)
        self._forwards = 0

        self._recorded: tuple[int, torch.Tensor] | None = None
        self._pending: dict[int, tuple[torch.Tensor, torch.cuda.Event]] = {}
        self._last_target = -1
        self._suppressed = False
        self._logged_layers: set[int] = set()

    def note_forward_token_load(self, tokens_per_expert: float) -> None:
        """Decide once per forward whether placement runs at all.

        Below one `BLOCK_SIZE_M` per expert the MoE kernel pads every expert to the same
        number of blocks, so the imbalance costs nothing and balancing it saves nothing.
        Decode never clears that bar on this hardware and measured -1.2% imbalance for a
        10% TPOT cost, so a decode forward runs none of this — and what prefill put
        there stays, rather than having to be sent again.

        Taken **before** anything is recorded, and from a value every rank agrees on.
        Deriving it from a snapshot instead is what previously made every decode and
        every dummy forward publish an empty set on all 48 layers, reverting everything.
        """
        self._suppressed = tokens_per_expert <= self.min_tokens_per_expert
        self._forwards += 1
        if self._forwards % _REPORT_EVERY == 0:
            logger.info(
                "Predictive expert replication: activated %d device-issued replica(s) "
                "over %d forwards.",
                int(self._placed_total),
                self._forwards,
            )

    def record_prediction(self, source_layer: int, predicted: torch.Tensor) -> None:
        """Keep a source layer's predicted load. No copy, no event, no planning.

        The host path copied this to pinned memory and synchronised on it a layer later.
        Nothing here leaves the device, so the snapshot is simply held.

        Raises:
            ValueError: If the target layer would be past the last one, which means the
                caller bound a source it should not have.
        """
        target = source_layer + self.lookahead
        if target >= self.num_layers:
            raise ValueError(
                f"layer {source_layer} predicts layer {target}, beyond the last layer "
                f"({self.num_layers - 1}); this source should not be bound."
            )
        summed = predicted if predicted.dim() == 1 else predicted.sum(dim=0)
        self._recorded = (target, summed)

    def plan_and_launch(self) -> list:
        """Plan the recorded layer and launch its transfer, all on the device.

        Returns an empty list: the host does not learn what was planned, which is the
        point. Callers use it for nothing.
        """
        if self._suppressed or self._recorded is None:
            return []
        target, predicted = self._recorded
        self._recorded = None

        # Layers are visited in increasing order within a forward, so a target that does
        # not advance means a new forward has begun. A host-side layer index, not plan
        # data.
        if target <= self._last_target:
            self._spent.zero_()
        self._last_target = target

        raw = plan_one_layer_on_device(predicted, self.ep_size, self.min_tokens)
        residency = self._residency[target]
        found = raw[0] > 0
        keep = (
            found
            & (residency.expert >= 0)
            & (residency.expert == raw[1])
            & (residency.target == raw[2])
        )
        # Charged for what moves, not for what is planned. A replica already resident
        # needs no transfer, so in a steady state the honest charge is zero and the
        # budget bounds churn rather than coverage.
        needs = found & ~keep
        affordable = needs & (self._spent < self._budget)
        charged = affordable.to(torch.int64)
        self._spent += charged
        self._placed_total += charged

        publish_plan = torch.stack(
            [(keep | affordable).to(torch.int64), raw[1], raw[2], raw[3]]
        )
        # Publishing runs on the compute stream and the plan stays referenced in
        # `_pending` until it does, so a temporary is safe there. The transfer's is not:
        # it goes into this layer's own row, which nothing frees. The row is rewritten
        # only by a later forward, and `activate_and_publish` has made the compute
        # stream wait on the transfer event by then, so a rewrite cannot overtake a
        # kernel still reading it.
        transfer_plan = self._transfer_plans[target]
        transfer_plan.copy_(
            torch.stack([affordable.to(torch.int64), raw[1], raw[2], raw[3]])
        )

        with self._transfer_stream():
            # The predictive stream is ordered behind everything the compute stream has
            # enqueued, which is how the drain avoids overwriting a replica row the
            # previous forward's MoE is still reading. Nothing in stream ordering
            # connects the two otherwise, and a per-layer read event would be the same
            # wait taken later.
            barrier = self._event_factory()
            barrier.record()
            if self.stream is not None:
                self.stream.wait_event(barrier)
            if not _SKIP_TRANSFER:
                self.transfer.transfer(
                    transfer_plan,
                    self.pointers[target],
                    self.replica_row,
                    self.stream or torch.cuda.current_stream(),
                )
            event = self._event_factory()
            event.record(self.stream)
        self.last_event = event
        self._pending[target] = (publish_plan, event)
        if target not in self._logged_layers:
            self._logged_layers.add(target)
            logger.info(
                "Predictive expert replication: device-issued transfer launched for "
                "layer %d",
                target,
            )
        return []

    def activate_and_publish(self, layer: int) -> list:
        """Wait for `layer`'s transfer and publish its complete desired state.

        Publishing unconditionally is the point: the plan is the layer's whole desired
        state, so `found == 0` means "this layer should hold no replica" and publishing
        that is what reverts whatever the last forward left. Skipping it for a layer
        that planned nothing left reversion running only where a new placement happened
        to land, and active replicas stayed at 53-66 per forward against a budget of 43.
        """
        if self._suppressed:
            # Leave this layer exactly as the last unsuppressed forward left it.
            return []
        pending = self._pending.pop(layer, None)
        if pending is None:
            # No prediction was ever recorded for this layer, so it has never held a
            # replica and there is nothing to revert.
            return []
        plan, event = pending
        # A stream wait, not a host one. Everything that consumes the weights is
        # enqueued on the current stream afterwards, so stream ordering is the whole
        # requirement.
        event.wait()
        maps = self.maps[layer]
        publish_plan_on_device(
            plan=plan,
            residency=self._residency[layer],
            logical_to_physical=maps.logical_to_physical,
            logical_replica_count=maps.logical_replica_count,
            source_local=maps.source_local,
            layout=maps.layout,
            per_rank_experts=self.canonical_per_rank,
            replica_slots_per_rank=self.replica_slots_per_rank,
            source_rank=self.ep_rank,
        )
        # All-ones, so the shared routing path's per-token replica choice stays a
        # lookup:
        # with one copy on offer a rank's chunk cannot be split.
        maps.source_local_replica_count.fill_(1)
        return []

    def _transfer_stream(self):
        """Context for the transfer: the predictive stream, or the current one."""
        if self.stream is None:
            return contextlib.nullcontext()
        return torch.cuda.stream(self.stream)
