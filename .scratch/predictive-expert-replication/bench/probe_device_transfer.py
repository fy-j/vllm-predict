# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Does the kernel-issued transfer land the right bytes, from a plan never read on host?

Ticket 06's transfer half, verified the way ticket 05's was: the production classes over
every ordered rank pair, with the receiving rank deriving what it should have got
independently. What a fake cannot check is exactly what matters here — that the kernel
reads the plan correctly, that the barrier makes the bytes visible, and that the byte
arithmetic over several weight tensors of different shapes agrees between the two
kernels.

The plan tensors are built on the device before the timed region, because a
host-to-device copy is a synchronising operation too and one inside the region would
measure the wrong thing.

Run: torchrun --nproc_per_node=8 probe_device_transfer.py
"""

from __future__ import annotations

import os
import statistics
import sys

import torch
import torch.distributed as dist

from vllm.distributed.eplb.device_transfer import DeviceExpertTransfer, WeightPointers
from vllm.distributed.eplb.nvshmem_transfer import OneSidedExpertTransfer
from vllm.distributed.eplb.predictive_planner import replica_row_of

HIDDEN = 2048
INTERMEDIATE = 768
PER_RANK = 16
LAYERS = 2
ITERS = 20

_failures = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    """Print one check and remember it, so the exit status means something."""
    global _failures
    if not ok:
        _failures += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def weights_for(rank: int, layer: int, device: torch.device) -> list[torch.Tensor]:
    """One rank's expert weights for one layer, derivable by any rank from its id.

    Deterministic on `(rank, layer)` so the receiver can compute what the sender should
    have sent, without a second communication path to check the first one against.
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


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")

    def broadcast_uid(local):
        holder = [local]
        dist.broadcast_object_list(holder, src=0)
        dist.barrier()
        return holder[0]

    layers = [weights_for(rank, layer, device) for layer in range(LAYERS)]
    pointers = [WeightPointers.build(tensors) for tensors in layers]
    expert_bytes = max(p.total_bytes for p in pointers)
    if rank == 0:
        print(f"one expert is {expert_bytes / 2**20:.2f} MiB", flush=True)

    transport = OneSidedExpertTransfer(
        rank=rank,
        world_size=world,
        expert_bytes=expert_bytes,
        device=device,
        broadcast_uid=broadcast_uid,
    )
    try:
        transfer = DeviceExpertTransfer(
            staging=transport._staging, ep_rank=rank, per_rank_experts=PER_RANK
        )
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            report("kernels build and register", False, f"{type(exc).__name__}: {exc}")
        transport.close()
        return 1
    if rank == 0:
        report("kernels build and register", True, "after nvshmem.init")

    stream = torch.cuda.Stream()
    replica_row = replica_row_of(PER_RANK)

    # Every ordered rank pair, so a byte-arithmetic mistake that happens to work for one
    # direction or one expert offset cannot hide.
    mismatches = 0
    checked = 0
    for source_rank in range(world):
        for target_rank in range(world):
            if source_rank == target_rank:
                continue
            layer = (source_rank + target_rank) % LAYERS
            expert = source_rank * PER_RANK + (target_rank % PER_RANK)
            plan = torch.tensor(
                [1, expert, target_rank, 2], dtype=torch.int64, device=device
            )
            transfer.transfer(plan, pointers[layer], replica_row, stream)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            if rank == target_rank:
                expected = weights_for(source_rank, layer, device)
                for got, want in zip(layers[layer], expected):
                    checked += 1
                    if not torch.equal(got[replica_row], want[expert % PER_RANK]):
                        mismatches += 1
            dist.barrier()

    counts = torch.tensor([checked, mismatches], dtype=torch.int64, device=device)
    dist.all_reduce(counts)
    checked, mismatches = (int(v) for v in counts)
    if rank == 0:
        report(
            "replica row is byte-identical over every rank pair",
            mismatches == 0,
            f"{checked - mismatches}/{checked} weight tensors match",
        )

    # A plan that placed nothing must move nothing, decided on the device.
    before = layers[0][0][replica_row].clone()
    empty = torch.zeros(4, dtype=torch.int64, device=device)
    transfer.transfer(empty, pointers[0], replica_row, stream)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    untouched = torch.tensor(
        [int(torch.equal(before, layers[0][0][replica_row]))],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(untouched)
    if rank == 0:
        report(
            "an empty plan leaves the replica row alone",
            int(untouched) == world,
            f"{int(untouched)}/{world} ranks untouched",
        )

    plan = torch.tensor(
        [1, (rank + 1) % world * PER_RANK, (rank + 1) % world, 2],
        dtype=torch.int64,
        device=device,
    )
    for _ in range(3):
        transfer.transfer(plan, pointers[0], replica_row, stream)
    torch.cuda.synchronize()
    spans = []
    for _ in range(ITERS):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        with torch.cuda.stream(stream):
            start.record(stream)
        transfer.transfer(plan, pointers[0], replica_row, stream)
        with torch.cuda.stream(stream):
            end.record(stream)
        torch.cuda.synchronize()
        spans.append(start.elapsed_time(end) * 1000.0)
    if rank == 0:
        median = statistics.median(spans)
        print(
            f"\nput + barrier + drain, one expert: p50 {median:8.1f} us   "
            f"min {min(spans):8.1f} us   max {max(spans):8.1f} us"
        )
        print(
            "  Ticket 05's host-issued route measured 53.7 us for the same payload, "
            "and paid a host synchronisation of 5.28 ms per layer for the privilege."
        )

    dist.barrier()
    transfer.close()
    transport.close()
    dist.destroy_process_group()
    return min(_failures, 1)


if __name__ == "__main__":
    sys.exit(main())
