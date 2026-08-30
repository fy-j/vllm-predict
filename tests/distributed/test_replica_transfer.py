# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one-sided transfer lands the expert, in the right order, within its byte budget.

Ticket 05. The engine under test moves one expert into a peer's replica row without the
host learning the plan, so what has to be pinned is the *ordering*, and every one of
these orderings is invisible when it is wrong: the weights are plausible floats either
way and only the model's output degrades.

Driven through a fake fabric rather than 8 GPUs, because every property here except the
interconnect itself is arithmetic and event bookkeeping. The fabric holds one staging
buffer per rank, so a put really does write the peer's memory and byte equality is a
real assertion. The 8-rank counterpart is `bench/probe_replica_transfer.py`, which
covers what a fake cannot: that NVSHMEM's put and barrier mean on the wire what they
mean here.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from vllm.distributed.eplb.predictive_planner import Placement
from vllm.distributed.eplb.replica_transfer import ReplicaTransferEngine

PER_RANK = 4
EP_SIZE = 8


class FakeEvent:
    """A CUDA event's record/wait pair, logged instead of enqueued."""

    def __init__(
        self,
        fabric: "FakeFabric",
        owner: int,
        tag: str = "drain",
        enable_timing: bool = False,
    ):
        self._log = fabric
        self._owner = owner
        self.tag = tag
        self.enable_timing = enable_timing
        self.recorded = False

    def record(self, stream=None) -> None:
        self.recorded = True
        self._log.note(("record", self._owner, self.tag))

    def wait(self, stream=None) -> None:
        self._log.note(("wait", self._owner, self.tag))

    def synchronize(self) -> None:
        self._log.note(("synchronize", self._owner, self.tag))

    def elapsed_time(self, other) -> float:
        if not (self.enable_timing and other.enable_timing):
            raise ValueError("Both events must be created with enable_timing=True")
        return 1.5


class FakeFabric:
    """One staging buffer per rank, a real barrier, and a log of what happened.

    The barrier is a genuine `threading.Barrier` because the ordering it provides is the
    property under test: a receiver that copies before the sender has put reads zeros,
    and a fake that let it do so would pass tests the wire would fail. Ranks therefore
    run concurrently, one thread each, exactly as they do in a real EP group.

    The timeout matters. A rank that skips the barrier — the deadlock this design is
    most exposed to — would otherwise hang the suite instead of failing it.
    """

    def __init__(self, expert_bytes: int, ep_size: int = EP_SIZE):
        self.expert_bytes = expert_bytes
        self.staging = {
            rank: torch.zeros(expert_bytes, dtype=torch.uint8)
            for rank in range(ep_size)
        }
        self.log: list = []
        self.barrier = threading.Barrier(ep_size, timeout=10)
        self._lock = threading.Lock()

    def note(self, entry: tuple) -> None:
        with self._lock:
            self.log.append(entry)


class FakePut:
    """One rank's view of the fabric, shaped like `OneSidedExpertTransfer`."""

    def __init__(self, fabric: FakeFabric, rank: int):
        self._fabric = fabric
        self.rank = rank
        self.expert_bytes = fabric.expert_bytes

    def put_expert(self, tensors, row, dst_rank, stream=None) -> None:
        from vllm.distributed.eplb.expert_staging import staging_layout

        self._fabric.note(("put", self.rank, dst_rank))
        buffer = self._fabric.staging[dst_rank]
        for (offset, nbytes), tensor in zip(staging_layout(tensors), tensors):
            buffer[offset : offset + nbytes] = tensor[row].reshape(-1).view(torch.uint8)

    def copy_into_row(self, tensors, row, stream=None) -> None:
        from vllm.distributed.eplb.expert_staging import staging_layout

        self._fabric.note(("copy", self.rank, row))
        buffer = self._fabric.staging[self.rank]
        for (offset, nbytes), tensor in zip(staging_layout(tensors), tensors):
            flat = tensor[row].reshape(-1).view(torch.uint8)
            flat.copy_(buffer[offset : offset + nbytes])

    def barrier(self, stream=None) -> None:
        self._fabric.note(("barrier", self.rank, None))
        self._fabric.barrier.wait()


def make_weights(seed: int, layers: int = 2, rows: int = PER_RANK + 1):
    """One rank's expert weights per layer, distinct per rank so a wrong source shows.

    Shaped as the model's are: a list per layer, and within a layer one tensor per
    weight, each indexed by physical row.
    """
    generator = torch.Generator().manual_seed(seed)
    return [
        [
            torch.randn(rows, 6, 8, generator=generator),
            torch.randn(rows, 8, 3, generator=generator),
        ]
        for _ in range(layers)
    ]


def _expert_bytes() -> int:
    """One expert's bytes, as the engine sizes its workspace."""
    return sum(w[0].numel() * w.element_size() for w in make_weights(0)[0])


def build(ep_size: int = EP_SIZE, max_concurrent_bytes: int | None = None):
    """A fabric, per-rank weights, and one engine per rank sharing that fabric."""
    expert_bytes = _expert_bytes()
    fabric = FakeFabric(expert_bytes, ep_size)
    weights = {rank: make_weights(rank + 1) for rank in range(ep_size)}
    engines = {
        rank: ReplicaTransferEngine(
            put=FakePut(fabric, rank),
            ep_rank=rank,
            per_rank_experts=PER_RANK,
            max_concurrent_bytes=max_concurrent_bytes,
            event_factory=lambda r=rank, **kwargs: FakeEvent(fabric, r, **kwargs),
        )
        for rank in range(ep_size)
    }
    return fabric, weights, engines


def run_layer(engines, weights, layer, placements):
    """Every rank runs the same layer with the same plan, concurrently.

    Concurrently because the barrier is real: run serially, a receiver scheduled before
    its sender would deadlock at a barrier no one else has reached yet.
    """
    with ThreadPoolExecutor(max_workers=len(engines)) as pool:
        futures = {
            rank: pool.submit(
                engine.transfer_layer, layer, placements, weights[rank][layer]
            )
            for rank, engine in engines.items()
        }
    return {rank: future.result() for rank, future in futures.items()}


def test_the_replica_row_holds_the_source_ranks_bytes():
    """The transferred row is byte-identical to the canonical row it came from.

    The whole feature rests on this: source-rank routing sends a share of an expert's
    tokens to this row, and a row holding the wrong bytes returns wrong logits for those
    tokens with nothing raising.
    """
    fabric, weights, engines = build()
    # Logical expert 9 is canonically rank 2's third row; rank 5 receives a copy.
    placement = Placement(0, 9, 2, 5, 100.0)

    run_layer(engines, weights, 0, [placement])

    source_row, replica_row = 9 - 2 * PER_RANK, PER_RANK
    for source, target in zip(weights[2][0], weights[5][0]):
        assert torch.equal(source[source_row], target[replica_row])


def test_repeated_transfers_over_every_rank_pair_stay_byte_identical():
    """Reusing one workspace must not leave a residue that survives into the next pair.

    A stale tail would show up here and nowhere else: a second transfer of a smaller
    payload into the same buffer would otherwise inherit the first one's bytes.
    """
    fabric, weights, engines = build()

    for source_rank in range(EP_SIZE):
        for target_rank in range(EP_SIZE):
            if source_rank == target_rank:
                continue
            expert = source_rank * PER_RANK + 1
            plan = [Placement(0, expert, source_rank, target_rank, 1.0)]
            run_layer(engines, weights, 0, plan)
            for source, target in zip(weights[source_rank][0], weights[target_rank][0]):
                assert torch.equal(source[1], target[PER_RANK])


def test_the_workspace_is_drained_before_the_next_layer_overwrites_it():
    """Layer `i`'s copy out of the workspace precedes layer `i + 1`'s put into it.

    One workspace shared by every layer is the memory decision; this ordering is what
    makes it safe. Violated, the replica row for layer `i` silently receives layer `i +
    1`'s expert.
    """
    fabric, weights, engines = build()

    run_layer(engines, weights, 0, [Placement(0, 9, 2, 5, 1.0)])
    run_layer(engines, weights, 1, [Placement(1, 1, 0, 5, 1.0)])

    kinds = [(kind, rank) for kind, rank, _ in fabric.log if kind in ("put", "copy")]
    assert kinds.index(("copy", 5)) < kinds.index(("put", 0)), (
        "layer 1's put overwrote the workspace before layer 0's copy drained it"
    )


def test_the_copy_waits_for_the_kernel_that_last_read_the_row():
    """Overwriting a replica row must follow the MoE kernel that was reading it.

    The row is live: the previous layer's dispatch may still be reading it while this
    transfer's copy is enqueued on another stream, and nothing in stream ordering
    connects the two.
    """
    fabric, weights, engines = build()
    read = FakeEvent(fabric, 5, tag="row-read")
    engines[5].note_replica_row_read(0, read)

    run_layer(engines, weights, 0, [Placement(0, 9, 2, 5, 1.0)])

    order = [(kind, tag) for kind, _rank, tag in fabric.log if kind in ("wait", "copy")]
    assert ("wait", "row-read") in order
    copy_at = next(i for i, (kind, _) in enumerate(order) if kind == "copy")
    assert order.index(("wait", "row-read")) < copy_at


def test_a_rank_with_no_role_still_takes_part_in_the_barrier():
    """Arrival is collective, so the barrier count must not depend on role.

    A rank that skipped the barrier because it neither sends nor receives would leave
    the two that do waiting for it forever. This is the deadlock shape that has already
    cost this branch two debugging sessions.
    """
    fabric, weights, engines = build()

    run_layer(engines, weights, 0, [Placement(0, 9, 2, 5, 1.0)])

    barriers = [rank for kind, rank, _ in fabric.log if kind == "barrier"]
    assert sorted(barriers) == list(range(EP_SIZE))


def test_the_byte_cap_serialises_a_layers_transfers_rather_than_dropping_one():
    """Two placements over the cap go one after another, and both still land.

    Refusing the second would silently halve a layer's placement while the planner
    believed it had both, so the cap has to serialise instead.
    """
    fabric, weights, engines = build(max_concurrent_bytes=None)
    plan = [Placement(0, 9, 2, 5, 1.0), Placement(0, 1, 0, 6, 1.0)]

    reports = run_layer(engines, weights, 0, plan)

    assert reports[2].chunks == 2, "one expert in flight means one transfer at a time"
    for source_rank, target_rank, row in ((2, 5, 1), (0, 6, 1)):
        for source, target in zip(weights[source_rank][0], weights[target_rank][0]):
            assert torch.equal(source[row], target[PER_RANK])
    # The second put must follow the first chunk's barrier, or the two share a window
    # wider than the cap allows.
    kinds = [kind for kind, _rank, _ in fabric.log]
    puts = [i for i, kind in enumerate(kinds) if kind == "put"]
    assert kinds.index("barrier") < puts[1]


def test_a_cap_wide_enough_lets_a_layers_transfers_share_one_window():
    """Raising the cap is what buys concurrency, so it must actually change behaviour.

    Guards against a cap that is read and then ignored — the defect ticket 02 found in
    its predecessor, where the knob existed and nothing consulted it.
    """
    fabric, weights, engines = build(max_concurrent_bytes=2 * _expert_bytes())
    plan = [Placement(0, 9, 2, 5, 1.0), Placement(0, 1, 0, 6, 1.0)]

    reports = run_layer(engines, weights, 0, plan)

    assert reports[2].chunks == 1


def test_the_chunking_is_identical_on_every_rank():
    """Every rank must make the same number of chunks, whatever its own role.

    Chunking from this rank's own bytes rather than the whole plan's would give a sender
    and a bystander different barrier counts, which hangs the group.
    """
    fabric, weights, engines = build()
    plan = [Placement(0, 9, 2, 5, 1.0), Placement(0, 1, 0, 6, 1.0)]

    reports = run_layer(engines, weights, 0, plan)

    assert len({report.chunks for report in reports.values()}) == 1


def test_activation_waits_for_the_transfer_and_reports_the_exposed_time():
    """A transfer that has not landed makes the consumer wait, and says how long.

    The alternative designs are both wrong: cancelling loses a placement the routing map
    already advertises, and falling back to canonical routing contradicts the map. So it
    waits — and the wait is the number ticket 08 needs, because an exposed transfer is
    the feature paying instead of earning.
    """
    fabric, weights, engines = build()
    run_layer(engines, weights, 0, [Placement(0, 9, 2, 5, 1.0)])

    engines[5].wait(0, measure_exposed=True)

    assert [kind for kind, _r, tag in fabric.log if tag == "drain" and kind == "wait"]
    # `elapsed_time` refuses a pair of events not created for timing, which is a defect
    # that appears only on a device — the fake refuses it too for that reason.
    assert engines[5].drain_exposed_ms() == pytest.approx([1.5])
    assert engines[5].drain_exposed_ms() == []


def test_waiting_for_a_layer_that_transferred_nothing_is_not_an_error():
    """A layer whose replica was already resident transfers nothing and still activates.

    Most forwards are in that steady state, so treating a missing event as a fault would
    fail almost every forward.
    """
    _fabric, _weights, engines = build()

    engines[3].wait(7)


def test_a_plan_placing_two_replicas_on_one_rank_is_refused():
    """One slot per layer cannot hold two experts, and the copy would not say so.

    The layout builder rejects this already; refusing it here too means the weights are
    never moved for a plan that cannot be expressed.
    """
    _fabric, weights, engines = build()
    plan = [Placement(0, 9, 2, 5, 1.0), Placement(0, 1, 0, 5, 1.0)]

    with pytest.raises(ValueError, match="already"):
        run_layer(engines, weights, 0, plan)
