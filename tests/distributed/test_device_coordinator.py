# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The plan the transfer kernels read is the plan the planner wrote.

Ticket 06. Everything else about the device path was verified against the host path it
replaces — the plan bit-identically, the publish against `apply_replica_maps`, the
transfer at 112/112 bytes over every rank pair — and it still crashed a real server,
because none of those checks let a plan tensor age. They all synchronised immediately
after launching, holding the plan in a local variable while they did.

What a server does instead: `plan_and_launch` builds the plan on the compute stream,
launches two kernels that read it on the **predictive** stream, and returns. The
temporary died there, PyTorch's allocator handed its block to the next allocation on the
compute stream, and the kernels — still queued behind a 40 us transfer — read whatever
the forward had put there. `pe` is then not a rank, and NVSHMEM's proxy thread
segfaults.

This test ages the plan the way a forward does: it delays the consumer on another
stream, allocates on the compute stream in between, and asks what the consumer saw.
It needs no NVSHMEM: the defect is in the ownership of the plan, not the transport.
"""

import time

import pytest
import torch

from vllm.distributed.eplb.device_coordinator import (
    DevicePlacementCoordinator,
    LayerMaps,
)
from vllm.distributed.eplb.predictive_planner import (
    canonical_row_of,
    plan_one_layer_on_device,
)

EP_SIZE = 4
PER_RANK = 4
NUM_LOGICAL = EP_SIZE * PER_RANK
NUM_LAYERS = 6
LOOKAHEAD = 2
MIN_TOKENS = 1.0

# Long enough that the host finishes its allocations while the consumer is still queued,
# which the test then asserts rather than assumes.
_DELAY_CYCLES = 400_000_000


class DelayedReader:
    """A transfer that reads the plan late, as a queued kernel does.

    The real `put_expert` reads the plan from device memory when it runs, not when it is
    launched. A fake that reads it eagerly would pass whatever the ownership rules are,
    which is precisely how this defect survived three rounds of verification.
    """

    def __init__(self):
        self.address: int | None = None
        self.late_read: torch.Tensor | None = None
        self.done: torch.cuda.Event | None = None

    def transfer(self, plan, pointers, replica_row, stream) -> None:
        self.address = plan.data_ptr()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(_DELAY_CYCLES)
            self.late_read = plan.clone()
            self.done = torch.cuda.Event()
            self.done.record(stream)


def hot_load(device: torch.device) -> torch.Tensor:
    """A snapshot with one clearly hottest expert on one clearly busiest rank."""
    load = torch.full((NUM_LOGICAL,), 4, dtype=torch.int32, device=device)
    load[0] = 400
    return load


def churn(device: torch.device, attempts: int = 256) -> list[torch.Tensor]:
    """Allocate `[4]` int64 blocks on the compute stream, as a forward does anyway.

    Returned so the caller keeps them alive; a freed block would not stay overwritten.
    """
    return [
        torch.full((4,), -12345, dtype=torch.int64, device=device)
        for _ in range(attempts)
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_transfer_reads_the_planned_expert_and_not_what_replaced_it():
    """A plan that ages across a stream boundary still names the planned expert.

    Fails before the fix with the plan's block handed to another allocation and the
    consumer reading `-12345`, which is a `pe` of -12345 in the real kernel.
    """
    device = torch.device("cuda")
    # The first `torch.stack` in a process blocks the host for 50-100 ms while its
    # kernel loads. Left unwarmed it drains the delay this test depends on, the consumer
    # reads the plan before the allocations land, and the test passes vacuously.
    torch.stack([torch.zeros((), dtype=torch.int64, device=device)] * 4)
    torch.zeros(4, dtype=torch.int64, device=device).clone()
    torch.accelerator.synchronize()

    reader = DelayedReader()
    coordinator = DevicePlacementCoordinator(
        ep_size=EP_SIZE,
        ep_rank=0,
        canonical_per_rank=PER_RANK,
        replica_slots_per_rank=1,
        num_layers=NUM_LAYERS,
        lookahead=LOOKAHEAD,
        budget=NUM_LAYERS,
        min_tokens=MIN_TOKENS,
        min_tokens_per_expert=1.0,
        pointers=[None] * NUM_LAYERS,
        maps=[None] * NUM_LAYERS,
        transfer=reader,
        device=device,
        stream=torch.cuda.Stream(),
    )

    predicted = hot_load(device)
    expected = plan_one_layer_on_device(predicted, EP_SIZE, MIN_TOKENS)
    coordinator.note_forward_token_load(1024.0)
    coordinator.record_prediction(0, predicted)
    coordinator.plan_and_launch()

    blocks = churn(device)
    assert reader.done is not None
    still_queued = not reader.done.query()
    recycled = any(block.data_ptr() == reader.address for block in blocks)
    torch.accelerator.synchronize()

    assert still_queued, (
        "the consumer had already read the plan before the allocations landed, so this "
        "test proves nothing; lengthen the delay or check what drained it"
    )
    assert not recycled, (
        "the plan's memory was handed to another allocation while the transfer was "
        "still queued to read it"
    )
    assert reader.late_read is not None
    assert reader.late_read.tolist() == expected.tolist(), (
        f"the transfer read {reader.late_read.tolist()} where the planner wrote "
        f"{expected.tolist()}"
    )


def layer_maps(device: torch.device) -> LayerMaps:
    """The routing maps as startup normalization leaves them: one copy each."""
    canonical = torch.tensor(
        [canonical_row_of(e, PER_RANK, 1) for e in range(NUM_LOGICAL)],
        dtype=torch.int64,
        device=device,
    )
    logical_to_physical = torch.full(
        (NUM_LOGICAL, 2), -1, dtype=torch.int64, device=device
    )
    logical_to_physical[:, 0] = canonical
    layout = torch.full((EP_SIZE, PER_RANK + 1), -1, dtype=torch.int64, device=device)
    for expert in range(NUM_LOGICAL):
        layout[expert // PER_RANK, expert % PER_RANK] = expert
    return LayerMaps(
        logical_to_physical=logical_to_physical,
        logical_replica_count=torch.ones(NUM_LOGICAL, dtype=torch.int64, device=device),
        source_local=canonical.view(NUM_LOGICAL, 1).clone(),
        source_local_replica_count=torch.ones(
            NUM_LOGICAL, dtype=torch.int64, device=device
        ),
        layout=layout,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_per_forward_path_never_waits_for_the_device():
    """The ticket's headline criterion, measured rather than grepped.

    A host synchronisation has a dozen spellings — `.item()`, `int()`, an indexing
    expression, a `cudaMemcpy` inside a helper — and `set_sync_debug_mode("error")`
    misses some: it says nothing about the 50-100 ms the first `torch.stack` in a
    process spends loading its kernel. What every spelling has in common is
    observable, so that is what this asserts: with 100 ms queued on the compute
    stream, a path that does not wait for the device returns in microseconds, and one
    that waits cannot.
    """
    device = torch.device("cuda")
    coordinator = DevicePlacementCoordinator(
        ep_size=EP_SIZE,
        ep_rank=0,
        canonical_per_rank=PER_RANK,
        replica_slots_per_rank=1,
        num_layers=NUM_LAYERS,
        lookahead=LOOKAHEAD,
        budget=NUM_LAYERS,
        min_tokens=MIN_TOKENS,
        min_tokens_per_expert=1.0,
        pointers=[None] * NUM_LAYERS,
        maps=[layer_maps(device) for _ in range(NUM_LAYERS)],
        transfer=type("Noop", (), {"transfer": lambda *_: None})(),
        device=device,
        stream=torch.cuda.Stream(),
    )
    predicted = hot_load(device)

    def one_forward() -> float:
        """Time the host through the three phases of one predicted layer."""
        start = time.perf_counter()
        coordinator.note_forward_token_load(1024.0)
        coordinator.record_prediction(0, predicted)
        coordinator.plan_and_launch()
        coordinator.activate_and_publish(LOOKAHEAD)
        return (time.perf_counter() - start) * 1e3

    # Warm every kernel the path uses, for the same reason as above: the first use of
    # each blocks the host while it loads, which is not a synchronisation on the path.
    one_forward()
    torch.accelerator.synchronize()

    torch.cuda._sleep(200_000_000)  # ~100 ms on the compute stream
    elapsed = one_forward()
    queued_work_remains = not torch.cuda.current_stream().query()
    torch.accelerator.synchronize()

    assert queued_work_remains, (
        "the queued work had already drained, so this measurement proves nothing"
    )
    assert elapsed < 20.0, (
        f"the per-forward path took {elapsed:.1f} ms of host time with ~100 ms queued "
        f"on the device, so it waited for the device somewhere"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_each_layer_keeps_its_own_plan_while_several_are_in_flight():
    """Layers overlap by design, so one layer's plan must not be another's buffer.

    At `prediction_lookahead_layers = 2` several plans are legitimately in flight across
    the model at once. A single shared plan buffer would pass the test above and still
    hand layer `L+1`'s expert to layer `L`'s transfer.
    """
    device = torch.device("cuda")
    torch.stack([torch.zeros((), dtype=torch.int64, device=device)] * 4)
    torch.accelerator.synchronize()

    seen: list[tuple[int, list[int]]] = []

    class Recorder:
        def transfer(self, plan, pointers, replica_row, stream) -> None:
            seen.append((plan.data_ptr(), plan.tolist()))

    coordinator = DevicePlacementCoordinator(
        ep_size=EP_SIZE,
        ep_rank=0,
        canonical_per_rank=PER_RANK,
        replica_slots_per_rank=1,
        num_layers=NUM_LAYERS,
        lookahead=LOOKAHEAD,
        budget=NUM_LAYERS,
        min_tokens=MIN_TOKENS,
        min_tokens_per_expert=1.0,
        pointers=[None] * NUM_LAYERS,
        maps=[None] * NUM_LAYERS,
        transfer=Recorder(),
        device=device,
        stream=torch.cuda.Stream(),
    )
    coordinator.note_forward_token_load(1024.0)
    for source_layer in (0, 1):
        load = torch.full((NUM_LOGICAL,), 4, dtype=torch.int32, device=device)
        # A different hottest expert per layer, so the two plans must differ.
        load[source_layer] = 400
        coordinator.record_prediction(source_layer, load)
        coordinator.plan_and_launch()
    torch.accelerator.synchronize()

    assert len(seen) == 2
    (first_address, first_plan), (second_address, second_plan) = seen
    assert first_address != second_address, (
        "both layers' transfers were aimed at the same plan buffer, so the second "
        "overwrote a plan the first may not have read yet"
    )
    assert first_plan != second_plan


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_plan_left_over_from_a_previous_forward_is_an_invariant_violation():
    """Every plan is produced and consumed inside one forward, so none may survive it.

    Ticket 07 shortens the distance between the launch and the wait to a single
    Attention block, which makes a plan that outlives its forward far more likely to be
    read by the wrong layer than a fallback anyone would want. It is therefore a
    violation and not a recovery: a stale plan names an expert chosen from another
    forward's load, so activating it moves load onto a rank that may now be the peak.
    """
    device = torch.device("cuda")
    coordinator = DevicePlacementCoordinator(
        ep_size=EP_SIZE,
        ep_rank=0,
        canonical_per_rank=PER_RANK,
        replica_slots_per_rank=1,
        num_layers=NUM_LAYERS,
        lookahead=1,
        budget=NUM_LAYERS,
        min_tokens=MIN_TOKENS,
        min_tokens_per_expert=1.0,
        pointers=[None] * NUM_LAYERS,
        maps=[layer_maps(device) for _ in range(NUM_LAYERS)],
        transfer=type("Noop", (), {"transfer": lambda *_: None})(),
        device=device,
        stream=torch.cuda.Stream(),
    )
    predicted = hot_load(device)

    coordinator.note_forward_token_load(1024.0)
    coordinator.record_prediction(0, predicted)
    coordinator.plan_and_launch()
    # The forward ends without layer 1 ever being visited, which is the sequence a
    # binding error or an early exit produces.
    with pytest.raises(RuntimeError, match="pending plan"):
        coordinator.note_forward_token_load(1024.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_target_refuses_a_plan_produced_in_another_forward():
    """A plan carries the forward it was produced in, and a mismatch raises.

    The plans are keyed by target layer, so a layer can only ever be handed a plan aimed
    at it — but not necessarily one from this forward. Forward identity is what closes
    that, and it is checked on the host from a counter, so it costs no device read.
    """
    device = torch.device("cuda")
    coordinator = DevicePlacementCoordinator(
        ep_size=EP_SIZE,
        ep_rank=0,
        canonical_per_rank=PER_RANK,
        replica_slots_per_rank=1,
        num_layers=NUM_LAYERS,
        lookahead=1,
        budget=NUM_LAYERS,
        min_tokens=MIN_TOKENS,
        min_tokens_per_expert=1.0,
        pointers=[None] * NUM_LAYERS,
        maps=[layer_maps(device) for _ in range(NUM_LAYERS)],
        transfer=type("Noop", (), {"transfer": lambda *_: None})(),
        device=device,
        stream=torch.cuda.Stream(),
    )
    predicted = hot_load(device)
    coordinator.note_forward_token_load(1024.0)
    coordinator.record_prediction(0, predicted)
    coordinator.plan_and_launch()
    # Forge the sequence the emptiness check would otherwise catch first: the plan stays
    # pending while a new forward begins.
    coordinator._forward_id += 1

    with pytest.raises(RuntimeError, match="another forward"):
        coordinator.activate_and_publish(1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_plan_produced_and_consumed_in_one_forward_is_accepted():
    """The control for the two above: the real sequence must not raise."""
    device = torch.device("cuda")
    coordinator = DevicePlacementCoordinator(
        ep_size=EP_SIZE,
        ep_rank=0,
        canonical_per_rank=PER_RANK,
        replica_slots_per_rank=1,
        num_layers=NUM_LAYERS,
        lookahead=1,
        budget=NUM_LAYERS,
        min_tokens=MIN_TOKENS,
        min_tokens_per_expert=1.0,
        pointers=[None] * NUM_LAYERS,
        maps=[layer_maps(device) for _ in range(NUM_LAYERS)],
        transfer=type("Noop", (), {"transfer": lambda *_: None})(),
        device=device,
        stream=torch.cuda.Stream(),
    )
    predicted = hot_load(device)

    for _ in range(3):
        coordinator.note_forward_token_load(1024.0)
        coordinator.record_prediction(0, predicted)
        coordinator.plan_and_launch()
        coordinator.activate_and_publish(1)
    torch.accelerator.synchronize()


def test_the_device_coordinator_asks_to_be_launched_at_the_predicting_layers_tail():
    """The capability the runner dispatches on, stated by the class that has it.

    Cheap to assert and worth pinning: if this flips, the window silently grows by a
    whole MoE layer and every measurement still looks reasonable.
    """
    assert DevicePlacementCoordinator.launch_at_predicting_layer_tail is True

    from vllm.distributed.eplb.predictive_coordinator import PlacementCoordinator

    assert PlacementCoordinator.launch_at_predicting_layer_tail is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_transfer_waits_for_the_compute_stream_before_reading_anything():
    """The transfer must not overtake the work the compute stream has already enqueued.

    Two things depend on it. The plan buffer is written on the compute stream and read
    by the kernels on the predictive one, so without ordering the put can read a row
    before the write lands — the same "`pe` is not a rank" that segfaulted NVSHMEM's
    proxy thread, arriving through visibility instead of through lifetime. And
    `drain_expert` writes a replica row the previous forward's MoE may still be reading.

    The bug this pins was a vacuous wait: `barrier.record()` inside
    `with torch.cuda.stream(self.stream)` records on the **predictive** stream, since
    that is the current one there, so `self.stream.wait_event(barrier)` waited on its
    own event. This project has shipped that mistake once before, on the snapshot copy.

    Tested by visibility rather than by reading the source: a value written on the
    compute stream behind a long delay must be the value the transfer sees.
    """
    device = torch.device("cuda")
    stream = torch.cuda.Stream()
    marker = torch.zeros((), dtype=torch.int64, device=device)
    seen: list[torch.Tensor] = []

    class ReadsTheMarker:
        def transfer(self, plan, pointers, replica_row, transfer_stream) -> None:
            with torch.cuda.stream(transfer_stream):
                seen.append(marker.clone())

    coordinator = DevicePlacementCoordinator(
        ep_size=EP_SIZE,
        ep_rank=0,
        canonical_per_rank=PER_RANK,
        replica_slots_per_rank=1,
        num_layers=NUM_LAYERS,
        lookahead=1,
        budget=NUM_LAYERS,
        min_tokens=MIN_TOKENS,
        min_tokens_per_expert=1.0,
        pointers=[None] * NUM_LAYERS,
        maps=[layer_maps(device) for _ in range(NUM_LAYERS)],
        transfer=ReadsTheMarker(),
        device=device,
        stream=stream,
    )
    predicted = hot_load(device)
    # Warm every kernel this path uses, or the first use of each blocks the host long
    # enough to drain the delay below and the test passes without ordering anything.
    coordinator.note_forward_token_load(1024.0)
    coordinator.record_prediction(0, predicted)
    coordinator.plan_and_launch()
    coordinator.activate_and_publish(1)
    torch.accelerator.synchronize()
    seen.clear()

    torch.cuda._sleep(200_000_000)  # ~100 ms of compute-stream work in front
    marker.fill_(7)
    coordinator.note_forward_token_load(1024.0)
    coordinator.record_prediction(0, predicted)
    coordinator.plan_and_launch()
    coordinator.activate_and_publish(1)
    torch.accelerator.synchronize()

    assert seen and int(seen[0]) == 7, (
        f"the transfer read {int(seen[0]) if seen else 'nothing'} where the compute "
        f"stream had written 7, so it ran ahead of work already enqueued there"
    )
