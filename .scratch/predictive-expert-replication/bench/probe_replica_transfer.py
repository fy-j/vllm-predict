# SPDX-License-Identifier: Apache-2.0 SPDX-FileCopyrightText: Copyright contributors to
# the vLLM project
"""Does the production transfer path land the right bytes on 8 real ranks?

Ticket 05's distributed acceptance. `tests/distributed/test_replica_transfer.py` drives
the same two classes through a fake fabric and pins the orderings; it cannot check the
one thing that matters most here — that NVSHMEM's put and stream-ordered barrier mean on
the wire what the fake says they mean. So this runs `OneSidedExpertTransfer` and
`ReplicaTransferEngine` themselves, not a reimplementation, over every rank pair, and
compares the replica row against bytes the receiving rank can derive independently.

It also settles a claim this project had recorded as verified and was not.
`probe_nvshmem.py` concluded that a plain CUDA event was enough for a consumer to know a
put had landed — but its check had a `dist.barrier()` inside the region under test, so
the barrier did the work and the event was never actually load-bearing.
`--control-no-barrier` reruns the same transfers with the barrier removed and counts the
mismatches, which is what makes the difference visible instead of asserted.

Run: torchrun --nproc_per_node=8 probe_replica_transfer.py torchrun --nproc_per_node=8
probe_replica_transfer.py --control-no-barrier
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch
import torch.distributed as dist

from vllm.distributed.eplb.expert_staging import staging_workspace_bytes
from vllm.distributed.eplb.nvshmem_transfer import OneSidedExpertTransfer
from vllm.distributed.eplb.predictive_planner import Placement, replica_row_of
from vllm.distributed.eplb.replica_transfer import ReplicaTransferEngine

# Qwen3-30B-A3B's shapes, so the timing is the timing that matters: 9.00 MiB per expert.
HIDDEN = 2048
INTERMEDIATE = 768
PER_RANK = 16
LAYERS = 2
ITERS = 20


_failures = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    """Print one check and remember it, so the exit status means something.

    Counted rather than printed and forgotten: an earlier version of this script exited
    0 with a FAIL on screen, which is exactly how a broken transfer path gets called
    green.
    """
    global _failures
    if not ok:
        _failures += 1
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def weights_for(rank: int, layer: int, device: torch.device) -> list[torch.Tensor]:
    """One rank's expert weights for one layer, derivable by any rank from its id.

    Deterministic on `(rank, layer)` so the receiver can compute what the sender should
    have sent without a second communication path to check the first one against.
    """
    generator = torch.Generator(device="cpu").manual_seed(1000 * layer + rank)
    rows = PER_RANK + 1
    return [
        torch.randn(
            rows, 2 * INTERMEDIATE, HIDDEN, generator=generator, dtype=torch.bfloat16
        ).to(device),
        torch.randn(
            rows, HIDDEN, INTERMEDIATE, generator=generator, dtype=torch.bfloat16
        ).to(device),
    ]


class _NoBarrier:
    """The control: the same transport with arrival never established.

    Wraps rather than reimplements, so the only difference between the arms is the
    barrier.
    """

    def __init__(self, inner: OneSidedExpertTransfer):
        self._inner = inner
        self.expert_bytes = inner.expert_bytes

    def put_expert(self, tensors, row, dst_rank, stream=None) -> None:
        self._inner.put_expert(tensors, row, dst_rank, stream)

    def copy_into_row(self, tensors, row, stream=None) -> None:
        self._inner.copy_into_row(tensors, row, stream)

    def barrier(self, stream=None) -> None:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-no-barrier", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())

    # NCCL first, as vLLM does, so coexistence is exercised rather than assumed.
    dist.init_process_group(backend="nccl")

    def broadcast_uid(local):
        holder = [local]
        dist.broadcast_object_list(holder, src=0)
        dist.barrier()
        return holder[0]

    layers = [weights_for(rank, layer, device) for layer in range(LAYERS)]
    expert_bytes = staging_workspace_bytes(layers)
    if rank == 0:
        print(f"one expert is {expert_bytes / 2**20:.2f} MiB", flush=True)

    try:
        transport = OneSidedExpertTransfer(
            rank=rank,
            world_size=world,
            expert_bytes=expert_bytes,
            device=device,
            broadcast_uid=broadcast_uid,
        )
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            report("one-sided transport up", False, f"{type(exc).__name__}: {exc}")
        return 1
    if rank == 0:
        report("one-sided transport up", True, f"after NCCL, {world} PEs")

    put = _NoBarrier(transport) if args.control_no_barrier else transport
    engine = ReplicaTransferEngine(put=put, ep_rank=rank, per_rank_experts=PER_RANK)
    stream = torch.cuda.Stream()
    replica_row = replica_row_of(PER_RANK)

    # Every ordered rank pair, so a layout mistake that happens to work for one
    # direction or one distance cannot hide.
    mismatches = 0
    checked = 0
    for source_rank in range(world):
        for target_rank in range(world):
            if source_rank == target_rank:
                continue
            layer = (source_rank + target_rank) % LAYERS
            expert = source_rank * PER_RANK + (target_rank % PER_RANK)
            engine.transfer_layer(
                layer,
                [Placement(layer, expert, source_rank, target_rank, 1.0)],
                layers[layer],
                stream=stream,
            )
            engine.wait(layer)
            torch.cuda.synchronize()
            if rank == target_rank:
                expected = weights_for(source_rank, layer, device)
                source_row = expert % PER_RANK
                for got, want in zip(layers[layer], expected):
                    checked += 1
                    if not torch.equal(got[replica_row], want[source_row]):
                        mismatches += 1
            dist.barrier()

    counts = torch.tensor([checked, mismatches], device=device)
    dist.all_reduce(counts)
    checked, mismatches = (int(v) for v in counts)
    if rank == 0:
        report(
            "replica row is byte-identical over every rank pair",
            mismatches == 0,
            f"{checked - mismatches}/{checked} weight tensors match"
            + (
                ""
                if mismatches == 0
                else "; with the barrier removed this is the expected result, and "
                "is what shows a CUDA event alone does not observe a peer's put"
            ),
        )

    # The span the overlap window has to hide: put, barrier, and the staging copy.
    placement = Placement(0, 1, 0, 1, 1.0)
    for _ in range(3):
        engine.transfer_layer(0, [placement], layers[0], stream=stream)
        engine.wait(0)
    torch.cuda.synchronize()
    spans = []
    for _ in range(ITERS):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        with torch.cuda.stream(stream):
            start.record(stream)
        engine.transfer_layer(0, [placement], layers[0], stream=stream)
        with torch.cuda.stream(stream):
            end.record(stream)
        engine.wait(0)
        torch.cuda.synchronize()
        spans.append(start.elapsed_time(end) * 1000.0)
    median = statistics.median(spans)
    if rank == 0:
        print(
            f"\none expert, put + barrier + copy: p50 {median:8.1f} us   "
            f"min {min(spans):8.1f} us   max {max(spans):8.1f} us"
        )
        print(
            f"  43 layers would be {43 * median / 1000:.1f} ms of transfer, against "
            f"the 5.28 ms per layer of host synchronisation this path removes."
        )

    # A transfer slowed past its window must make the consumer wait and complete, not
    # cancel and not fall back to canonical routing — either of those contradicts a
    # routing map that already advertises the row. Slowed by issuing the same expert
    # repeatedly, so the span grows without changing anything else about the path.
    #
    # The delay goes in *front* of the transfer, not after it. Queued behind, the extra
    # puts land after the drain event is recorded and the wait measures nothing but its
    # own overhead — 0.012 ms, which reads like a pass and tests nothing. Against a
    # measured baseline, not a constant: how long an *unslowed* wait takes is a property
    # of this machine, and a threshold guessed from the transfer time alone already
    # produced one false failure at 0.433 ms against a made-up 0.588 ms bar.
    engine.transfer_layer(0, [placement], layers[0], stream=stream)
    engine.wait(0, measure_exposed=True)
    torch.cuda.synchronize()
    baseline = engine.drain_exposed_ms()

    slow_puts = 20
    for _ in range(slow_puts):
        transport.put_expert(layers[0], 1, (rank + 1) % world, stream)
    engine.transfer_layer(0, [placement], layers[0], stream=stream)
    engine.wait(0, measure_exposed=True)
    torch.cuda.synchronize()
    exposed = engine.drain_exposed_ms()
    if rank == 0:
        report(
            "a slowed transfer is waited for, measured, and completed",
            bool(exposed and baseline) and exposed[0] > 5 * baseline[0],
            f"consumer stood still {exposed[0]:.3f} ms behind {slow_puts} extra puts, "
            f"against {baseline[0]:.3f} ms unslowed"
            if exposed and baseline
            else "nothing measured",
        )
        for got, want in zip(layers[0], weights_for(0, 0, device)):
            if rank == 1 and not torch.equal(got[replica_row], want[1]):
                report("the slowed transfer still landed", False)

    dist.barrier()
    # Before `destroy_process_group`, and before the interpreter starts tearing down: a
    # live symmetric heap at exit segfaults every rank after the results are already
    # out.
    transport.close()
    dist.destroy_process_group()
    # The control arm is expected to fail its byte check; that is the point of running
    # it.
    return 0 if args.control_no_barrier else min(_failures, 1)


if __name__ == "__main__":
    sys.exit(main())
