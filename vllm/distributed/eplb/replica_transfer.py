# SPDX-License-Identifier: Apache-2.0 SPDX-FileCopyrightText: Copyright contributors to
# the vLLM project
"""Land a predicted expert in its target's replica row, without the host reading it.

Ticket 05. The pynccl path this replaces is not slow because of the wire: transfers
measure 0.32% of GPU time. It is slow because `ncclSend`'s peer argument is a host
`c_int` consumed when the host enqueues, so the host must know which expert goes where,
so it must read a device tensor — 5.28 ms of collective waiting per predicted layer, 70%
of everything prediction adds. A one-sided put has no peer to name at enqueue time and
no matching receive to pair with, which is what lets the plan stay on the device.

Three orderings make that safe, and every one of them is silent when it is wrong. They
are kept here, in one place, rather than spread between the coordinator and the
communicator:

1. **Arrival.** A put returns to its issuer long before the bytes are visible to the
   peer, and the peer has no local event that observes them. The probe that concluded a
   plain stream event was enough had a `dist.barrier()` inside the checked region doing
   the work. So arrival is established by NVSHMEM's **stream-ordered barrier**: no host
   involvement, and safe against the divergence that has deadlocked this branch twice,
   because it is collective and the plan is identical on every rank by construction.
   Every rank calls it the same number of times whatever its own role, which is why the
   chunking below counts the whole plan's bytes and not this rank's.
2. **The workspace.** One staging buffer, shared by every layer, so a later layer's put
   must not overwrite bytes an earlier layer has not drained into its row yet.
3. **The row.** A replica row is live memory: the previous layer's dispatch may still be
   reading it while this transfer's copy is enqueued on the predictive stream, and
   nothing in stream ordering connects the two streams. The consumer hands over the
   event it recorded after that read, and the copy waits on it.

Bytes in flight are bounded by chunking a layer's placements, not by refusing any: at
the default of one expert a layer's placements move one after another, so the cost to
hide is the serialized span rather than a single transfer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch

from vllm.distributed.eplb.predictive_planner import (
    Placement,
    local_row_of,
    replica_row_of,
)


class _Put(Protocol):
    """What this engine needs of a one-sided transport.

    Narrower than `OneSidedExpertTransfer` so a test can drive the orderings through a
    fake fabric on the CPU, which is where all three of them are actually decided.
    """

    expert_bytes: int

    def put_expert(
        self,
        tensors: Sequence[torch.Tensor],
        row: int,
        dst_rank: int,
        stream: torch.cuda.Stream | None = ...,
    ) -> None: ...

    def copy_into_row(
        self,
        tensors: Sequence[torch.Tensor],
        row: int,
        stream: torch.cuda.Stream | None = ...,
    ) -> None: ...

    def barrier(self, stream: torch.cuda.Stream | None = ...) -> None: ...


class _Event(Protocol):
    """The slice of `torch.cuda.Event` used here.

    `wait` orders a stream and costs the host nothing; `synchronize` blocks it and
    appears only in the opt-in exposed-time accounting, never on the per-forward path.
    """

    def record(self, stream: torch.cuda.Stream | None = ...) -> None: ...

    def wait(self, stream: torch.cuda.Stream | None = ...) -> None: ...

    def synchronize(self) -> None: ...

    def elapsed_time(self, other) -> float: ...


@dataclass
class TransferReport:
    """What one layer's transfer did, for the cost model to be checked against.

    Attributes:
        chunks: Barrier rounds the layer took. Above 1 the byte cap serialized it, and
            the span to hide is that many transfers rather than one.
        sent: Experts this rank put to a peer.
        received: Experts this rank copied into a replica row.
    """

    chunks: int = 0
    sent: int = 0
    received: int = 0


@dataclass
class _Exposed:
    """A wait's two markers, read later so the wait itself stays off the host."""

    before: _Event
    after: _Event


@dataclass
class ReplicaTransferEngine:
    """Per-rank transfer of predicted experts into replica rows.

    Attributes:
        put: The one-sided transport.
        ep_rank: This rank's id in the EP group.
        per_rank_experts: Canonical rows per rank, which fixes where the replica row is.
        max_concurrent_bytes: Cap on bytes in flight. Defaults to one expert.
        slot: Which inactive row to land in.
    """

    put: _Put
    ep_rank: int
    per_rank_experts: int
    max_concurrent_bytes: int | None = None
    slot: int = 0
    event_factory: Callable[..., _Event] = torch.cuda.Event

    _pending: dict[int, _Event] = field(default_factory=dict, init=False)
    _row_read: dict[int, _Event] = field(default_factory=dict, init=False)
    _last_drain: _Event | None = field(default=None, init=False)
    _exposed: list[_Exposed] = field(default_factory=list, init=False)

    def note_replica_row_read(self, layer: int, event: _Event) -> None:
        """Register the event recorded after a MoE kernel read this layer's replica row.

        Held per layer rather than globally: a layer's row is only overwritten by that
        layer's own transfer, and a single latest-read event would make every layer wait
        for whichever layer ran last.
        """
        self._row_read[layer] = event

    def transfer_layer(
        self,
        layer: int,
        placements: Sequence[Placement],
        tensors: Sequence[torch.Tensor],
        stream: torch.cuda.Stream | None = None,
    ) -> TransferReport:
        """Move one layer's planned experts into their targets' replica rows.

        Args:
            layer: The layer being placed, used to match the consumer's read event.
            placements: That layer's plan, identical on every rank.
            tensors: This rank's weight tensors for `layer`, each `[rows, ...]`.
            stream: The predictive stream, or None to use the current one.

        Returns: What the layer's transfer did.

        Raises:
            ValueError: If two placements target the same rank, which one slot cannot
                hold and which would land one expert's bytes on top of the other's.
        """
        report = TransferReport()
        if not placements:
            return report

        targets = [p.target_rank for p in placements]
        if len(set(targets)) != len(targets):
            raise ValueError(
                f"layer {layer} plans two replicas on one rank ({sorted(targets)}); a "
                f"rank already holding a replica for this layer cannot hold a second."
            )

        replica_row = replica_row_of(self.per_rank_experts, self.slot)
        for chunk in self._chunks(placements):
            for placement in chunk:
                if self.ep_rank == placement.source_rank:
                    source_row = local_row_of(
                        placement.logical_expert, self.per_rank_experts
                    )
                    self.put.put_expert(
                        tensors, source_row, placement.target_rank, stream
                    )
                    report.sent += 1
            # Collective, and called by every rank in the group whatever its role above.
            self.put.barrier(stream)
            for placement in chunk:
                if self.ep_rank == placement.target_rank:
                    read = self._row_read.pop(layer, None)
                    if read is not None:
                        read.wait(stream)
                    self.put.copy_into_row(tensors, replica_row, stream)
                    report.received += 1
            drain = self.event_factory()
            drain.record(stream)
            self._last_drain = drain
            report.chunks += 1

        assert self._last_drain is not None
        self._pending[layer] = self._last_drain
        return report

    def wait(
        self,
        layer: int,
        stream: torch.cuda.Stream | None = None,
        measure_exposed: bool = False,
    ) -> None:
        """Order the current stream behind `layer`'s transfer, then forget it.

        A layer with nothing pending is the ordinary case rather than a fault: once a
        replica is resident it is not transferred again, so in a steady state most
        layers activate without having moved anything.

        Args:
            layer: The layer about to route to its replica row.
            stream: Stream to order, or None for the current one.
            measure_exposed: Record markers around the wait so `drain_exposed_ms` can
                report how long the consumer stood still. Off by default because reading
                the markers needs a host synchronisation.
        """
        event = self._pending.pop(layer, None)
        if event is None:
            return
        if not measure_exposed:
            event.wait(stream)
            return
        # `enable_timing`, or `elapsed_time` refuses the pair. The drain event above
        # needs no timing and does not ask for it, because a timed event costs a
        # device-side write on every record and that one is on the per-forward path.
        before = self.event_factory(enable_timing=True)
        after = self.event_factory(enable_timing=True)
        before.record(stream)
        event.wait(stream)
        after.record(stream)
        self._exposed.append(_Exposed(before, after))

    def drain_exposed_ms(self) -> list[float]:
        """Milliseconds each measured wait spent exposed, clearing the record.

        Synchronizes, so this belongs at a step boundary and never inside a forward. An
        exposed wait is the feature paying rather than earning, which is why it is
        reported instead of being smoothed into the step time.
        """
        out = []
        for pair in self._exposed:
            pair.after.synchronize()
            out.append(pair.before.elapsed_time(pair.after))
        self._exposed.clear()
        return out

    def _chunks(self, placements: Sequence[Placement]) -> list[list[Placement]]:
        """Split a layer's placements into groups that fit the byte cap.

        Counts every placement's bytes, not only the ones this rank sends, so all ranks
        produce the same number of groups and therefore the same number of barriers. A
        cap smaller than one expert would otherwise yield an empty group forever, so a
        single placement always forms a group whatever the cap says.
        """
        cap = self.max_concurrent_bytes or self.put.expert_bytes
        chunks: list[list[Placement]] = []
        used = 0
        for placement in placements:
            if not chunks or used + self.put.expert_bytes > cap:
                chunks.append([placement])
                used = self.put.expert_bytes
            else:
                chunks[-1].append(placement)
                used += self.put.expert_bytes
        return chunks
