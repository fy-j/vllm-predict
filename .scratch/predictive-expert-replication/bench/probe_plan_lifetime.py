# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Does the plan tensor still hold the plan when the kernel that reads it runs?

Ticket 06's open crash: wired into a real server, `put_expert` reached all 48 layers and
then segfaulted inside NVSHMEM's proxy thread, while `probe_device_transfer.py` passed
112/112 byte checks and a standalone pipeline of 48 transfers per round survived every
time. The difference is not the transport. It is **who still owns the plan**.

`DevicePlacementCoordinator.plan_and_launch` built the `[4]` plan on the compute stream,
launched two kernels that read it on the **predictive** stream, and returned — dropping
the last reference. PyTorch's allocator returns a freed block to the pool of the
stream it was allocated on and hands it to the next allocation there without waiting
for another stream to finish with it; that is what `record_stream` exists to declare,
and nothing in `vllm/distributed/eplb/` called it. So the compute stream overwrote the
plan while the kernels were still queued behind a 40 us transfer, and the put read what
forward had allocated: a `pe` that is not a rank, an expert id that is not an expert.

Both existing probes miss it for the same reason — they hold `plan` in a local variable
and synchronise immediately after launching. A server hits it because a real forward
allocates continuously on the compute stream while the predictive stream lags.

Three arms:

    control          plan held alive                     -> the planned expert arrives
    production       the coordinator plans and launches,  -> the planned expert arrives
                     then the compute stream allocates       and its buffer is untouched
    dropped plan     a plan nothing owns (opt-in)         -> reproduces the defect

Run: torchrun --nproc_per_node=2 probe_plan_lifetime.py
     PROBE_DROPPED_PLAN=1 torchrun --nproc_per_node=2 probe_plan_lifetime.py
     PROBE_INVALID_PEER=1 torchrun --nproc_per_node=2 probe_plan_lifetime.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

from vllm.distributed.eplb.device_coordinator import DevicePlacementCoordinator
from vllm.distributed.eplb.device_transfer import DeviceExpertTransfer, WeightPointers
from vllm.distributed.eplb.nvshmem_transfer import OneSidedExpertTransfer
from vllm.distributed.eplb.predictive_planner import (
    plan_one_layer_on_device,
    replica_row_of,
)

HIDDEN = 2048
INTERMEDIATE = 768
PER_RANK = 16
NUM_LAYERS = 4
LOOKAHEAD = 2
MIN_TOKENS = 1.0

# Long enough that the host finishes its allocations while the transfer is still queued.
# Asserted, not assumed: an arm that measured a completed transfer proves nothing.
_SLEEP_CYCLES = 400_000_000

_failures = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    """Print one check and remember it, so the exit status means something."""
    global _failures
    if not ok:
        _failures += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def weights_for(rank: int, device: torch.device) -> list[torch.Tensor]:
    """One rank's expert weights, derivable by any rank from its id."""
    generator = torch.Generator(device="cpu").manual_seed(7000 + rank)
    rows = PER_RANK + 1
    return [
        torch.randn(
            rows, 2 * INTERMEDIATE, HIDDEN, generator=generator, dtype=torch.bfloat16
        ).to(device),
        torch.randn(
            rows, HIDDEN, INTERMEDIATE, generator=generator, dtype=torch.bfloat16
        ).to(device),
    ]


def hot_load(hottest: int, device: torch.device, num_logical: int) -> torch.Tensor:
    """A snapshot every rank agrees on, with one expert clearly the hottest.

    The planner is deterministic on this, so a receiver can derive what should arrive
    without being told — the same property the feature relies on to avoid a broadcast.
    """
    load = torch.full((num_logical,), 8, dtype=torch.int32, device=device)
    load[hottest] = 4096
    return load


def overwrite_freed_block(address: int, poison: list[int], device) -> tuple[bool, list]:
    """Allocate `[4]` int64 blocks on the compute stream, as a forward does anyway.

    Returns whether one of them was handed the plan's block, and the blocks themselves
    so the caller keeps them alive — a freed block would not stay overwritten.
    """
    keep = []
    hit = False
    for _ in range(256):
        block = torch.tensor(poison, dtype=torch.int64, device=device)
        keep.append(block)
        hit = hit or block.data_ptr() == address
    return hit, keep


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        print("needs at least 2 ranks", file=sys.stderr)
        return 2
    torch.accelerator.set_device_index(rank % torch.accelerator.device_count())
    device = torch.device("cuda", rank % torch.accelerator.device_count())
    dist.init_process_group(backend="nccl")

    def broadcast_uid(local):
        holder = [local]
        dist.broadcast_object_list(holder, src=0)
        dist.barrier()
        return holder[0]

    weights = weights_for(rank, device)
    pointers = WeightPointers.build(weights)
    transport = OneSidedExpertTransfer(
        rank=rank,
        world_size=world,
        expert_bytes=pointers.total_bytes,
        device=device,
        broadcast_uid=broadcast_uid,
    )
    transfer = DeviceExpertTransfer(
        staging=transport._staging, ep_rank=rank, per_rank_experts=PER_RANK
    )
    stream = torch.cuda.Stream()
    replica_row = replica_row_of(PER_RANK)
    num_logical = PER_RANK * world

    # Rank 0 owns the hot expert, so rank 0 sends and someone else receives. Recomputed
    # from the sender's id on the receiver, not communicated: the check must not travel
    # the path it is checking.
    sender = 0
    sender_weights = weights_for(sender, device) if rank != sender else weights

    # The first `torch.stack` in a process blocks the host for 50-100 ms while its
    # kernel loads — measured, call 0 at 50.62 ms draining a side stream, calls 1-3 at
    # 0.01-0.04 ms not. It is not a per-call synchronisation, but left unwarmed here it
    # drains the delay every arm below depends on and every arm passes vacuously.
    torch.stack([torch.zeros((), dtype=torch.int64, device=device)] * 4)
    torch.accelerator.synchronize()

    def replica_holds(expert: int) -> bool:
        """Whether this rank's replica row holds `expert`'s weights."""
        return bool(
            torch.equal(weights[0][replica_row], sender_weights[0][expert % PER_RANK])
            and torch.equal(
                weights[1][replica_row], sender_weights[1][expert % PER_RANK]
            )
        )

    def check(name: str, expert: int, target: int, must_hold: bool) -> None:
        holds = int(replica_holds(expert)) if rank == target else 0
        agreed = torch.tensor([holds], dtype=torch.int64, device=device)
        dist.all_reduce(agreed)
        # int64 rather than bool: `all_reduce` with SUM over a bool tensor saturates
        # back to bool, which once made eight agreeing ranks report as one.
        got = int(agreed) == 1
        if rank == 0:
            report(
                name,
                got == must_hold,
                f"the replica row holds expert {expert}"
                if got
                else "it does not hold it",
            )

    def clear_replica(target: int) -> None:
        if rank == target:
            weights[0][replica_row].zero_()
            weights[1][replica_row].zero_()
        torch.accelerator.synchronize()
        dist.barrier()

    # Control: the plan stays referenced, which is what both existing probes do, and it
    # is why they pass while a server crashes.
    expert = 3
    target = 1
    clear_replica(target)
    held = torch.tensor([1, expert, target, 2], dtype=torch.int64, device=device)
    transfer.transfer(held, pointers, replica_row, stream)
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()
    dist.barrier()
    check("plan held alive: the planned expert arrives", expert, target, True)

    # Production: the coordinator owns the plan, and the compute stream allocates while
    # the transfer is queued — which is what a forward does and what no probe did.
    predicted = hot_load(expert, device, num_logical)
    planned = plan_one_layer_on_device(predicted, world, MIN_TOKENS)
    planned_expert, planned_target = int(planned[1]), int(planned[2])
    coordinator = DevicePlacementCoordinator(
        ep_size=world,
        ep_rank=rank,
        canonical_per_rank=PER_RANK,
        replica_slots_per_rank=1,
        num_layers=NUM_LAYERS,
        lookahead=LOOKAHEAD,
        budget=NUM_LAYERS,
        min_tokens=MIN_TOKENS,
        min_tokens_per_expert=1.0,
        pointers=[pointers] * NUM_LAYERS,
        maps=[None] * NUM_LAYERS,
        transfer=transfer,
        device=device,
        stream=stream,
    )
    # One full pass first, for the same reason `torch.stack` is warmed above: the
    # planner is dozens of operations and the first use of each loads its kernel, which
    # the host long enough to drain the queued delay the measurement depends on.
    coordinator.note_forward_token_load(1024.0)
    coordinator.record_prediction(0, predicted)
    coordinator.plan_and_launch()
    torch.accelerator.synchronize()
    dist.barrier()

    clear_replica(planned_target)
    if rank == 0:
        print(
            f"[plan] the planner chose expert {planned_expert} -> rank "
            f"{planned_target}, and no rank was told",
            flush=True,
        )
    with torch.cuda.stream(stream):
        torch.cuda._sleep(_SLEEP_CYCLES)
    coordinator.note_forward_token_load(1024.0)
    coordinator.record_prediction(0, predicted)
    coordinator.plan_and_launch()
    launched = coordinator.last_event
    assert launched is not None
    plan_address = coordinator._transfer_plans[LOOKAHEAD].data_ptr()
    recycled, blocks = overwrite_freed_block(
        plan_address, [1, (expert + 1) % PER_RANK, planned_target, 2], device
    )
    still_queued = not launched.query()
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()
    dist.barrier()
    if rank == 0:
        report(
            "the transfer was still queued while the compute stream allocated",
            still_queued,
            "otherwise this arm proves nothing",
        )
        report(
            "the plan's buffer was never handed to another allocation",
            not recycled,
            f"{len(blocks)} allocations of its size and shape",
        )
    check(
        "the coordinator's transfer moved the expert it planned",
        planned_expert,
        planned_target,
        True,
    )
    del blocks

    # The defect, on purpose: a plan nothing owns. Opt-in, because it is expected to
    # move the wrong expert and there is no value in a permanently red check.
    if os.environ.get("PROBE_DROPPED_PLAN") == "1":
        clear_replica(target)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(_SLEEP_CYCLES)

        def launch_and_forget(good: int) -> int:
            plan = torch.stack(
                [
                    torch.tensor(v, dtype=torch.int64, device=device)
                    for v in (1, good, target, 2)
                ]
            )
            address = plan.data_ptr()
            transfer.transfer(plan, pointers, replica_row, stream)
            return address

        wrong = expert + 1
        address = launch_and_forget(expert)
        overwrite_freed_block(address, [1, wrong, target, 2], device)
        torch.cuda.current_stream().wait_stream(stream)
        torch.accelerator.synchronize()
        dist.barrier()
        check(
            "dropped plan: the transfer moved what overwrote it, not what was planned",
            wrong,
            target,
            True,
        )

    # And the link to the server's crash: a plan overwritten by something that is not a
    # plan names a peer that does not exist. Expected to abort, which is the point.
    if os.environ.get("PROBE_INVALID_PEER") == "1":
        if rank == 0:
            print(
                f"\n[arm] invalid peer — a put at PE 12345 of {world}. A segfault "
                f"in NVSHMEM's proxy thread here is the server's crash, reproduced.",
                flush=True,
            )
        dist.barrier()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(_SLEEP_CYCLES)
        plan = torch.stack(
            [
                torch.tensor(v, dtype=torch.int64, device=device)
                for v in (1, expert, target, 2)
            ]
        )
        address = plan.data_ptr()
        transfer.transfer(plan, pointers, replica_row, stream)
        del plan
        overwrite_freed_block(address, [1, expert, 12345, 2], device)
        torch.cuda.current_stream().wait_stream(stream)
        torch.accelerator.synchronize()
        dist.barrier()
        if rank == 0:
            report(
                "an invalid peer did not abort, so garbage alone is not the crash", True
            )

    dist.barrier()
    transfer.close()
    transport.close()
    dist.destroy_process_group()
    return min(_failures, 1)


if __name__ == "__main__":
    sys.exit(main())
