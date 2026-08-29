# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Can the replica land in the model's own weights through PyTorch symmetric memory?

`probe_nvshmem_register.py` ruled out NVSHMEM's `register_external_tensor`: registration
needs a dynamic VMM heap, the VMM heap needs a user handle PyTorch does not create, and the
two settings exclude each other. That left a symmetric-heap staging buffer plus a local
copy as the only NVSHMEM route.

But PyTorch has its own symmetric memory, and **this repository already uses it** for
custom all-reduce (`vllm/distributed/device_communicators/symm_mem.py`). It offers what the
NVSHMEM route could not:

  * `get_mem_pool(device)` returns a `torch.cuda.MemPool`, so a tensor can be *allocated*
    inside the symmetric pool with ordinary `torch.empty`. Weight loading and EPLB's
    `rearrange` keep working on an ordinary torch tensor — no NVSHMEM allocator underneath
    the model's weights.
  * `rendezvous(tensor, group)` then makes it addressable by every rank.
  * `get_buffer(peer, sizes, dtype)` hands back a **torch tensor view of the peer's
    memory**, so writing one expert is `peer_view[row].copy_(my_row)` — a slice, which
    NVSHMEM's tracking table refused.
  * `buffer_ptrs_dev` is a **device-resident array of peer base pointers**, which is what
    lets a Triton kernel compute the destination on device and keep the host out entirely.

What this probe has to establish, in order, because each would sink the design alone:

  1. A realistically shaped expert-weight tensor can be allocated in the symmetric pool.
  2. It rendezvous, and `is_symm_mem_tensor` agrees.
  3. It is still an ordinary tensor for local compute — the MoE kernel reads it every layer.
  4. A **row-granular** write into a peer's replica row lands, ordered by a CUDA event.
  5. It scales to all 48 layers, about 7.3 GiB per rank, not just one tensor.
  6. What a 6 MiB row costs, against NVSHMEM's 33 us put plus a 6.3 us local copy.

Shapes are Qwen3-30B-A3B's at EP=8: 17 rows per rank (16 canonical + 1 replica), w13
[17, 2*768, 2048] and w2 [17, 2048, 768] in bf16, 153 MiB per layer per rank.

Run:
    torchrun --nproc_per_node=8 probe_symm_mem.py
"""

from __future__ import annotations

import os
import statistics
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

HIDDEN = 2048
MOE_INTERMEDIATE = 768
ROWS = 17
NUM_LAYERS = 48
REPLICA_ROW = 16
ITERS = 30


def report(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")
    group = dist.group.WORLD
    group_name = group.group_name
    symm_mem.enable_symm_mem_for_group(group_name)

    w13_shape = (ROWS, 2 * MOE_INTERMEDIATE, HIDDEN)
    w2_shape = (ROWS, HIDDEN, MOE_INTERMEDIATE)
    row_bytes = 2 * MOE_INTERMEDIATE * HIDDEN * 2

    # (1) Allocate inside the symmetric pool with ordinary torch, which is the whole point:
    # the model's weight loader would do exactly this and keep plain tensor semantics.
    try:
        pool = symm_mem.get_mem_pool(device)
        with torch.cuda.use_mem_pool(pool):
            w13 = torch.zeros(w13_shape, dtype=torch.bfloat16, device=device)
            w2 = torch.zeros(w2_shape, dtype=torch.bfloat16, device=device)
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            report("allocate weights in the symmetric pool", False, f"{type(exc).__name__}: {exc}")
        return 1
    if rank == 0:
        report(
            "allocate weights in the symmetric pool",
            True,
            f"w13 {tuple(w13.shape)} + w2 {tuple(w2.shape)}, "
            f"{(w13.numel() + w2.numel()) * 2 / 2**20:.0f} MiB",
        )

    # (2) rendezvous, and confirm torch agrees this is symmetric memory.
    try:
        hdl13 = symm_mem.rendezvous(w13, group_name)
        symm_mem.rendezvous(w2, group_name)
        is_symm = torch._C._distributed_c10d._SymmetricMemory.is_symm_mem_tensor(w13)
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            report("rendezvous", False, f"{type(exc).__name__}: {exc}")
        return 1
    if rank == 0:
        report(
            "rendezvous",
            bool(is_symm),
            f"world {hdl13.world_size}, is_symm_mem_tensor {is_symm}",
        )

    # (3) Still an ordinary tensor for the kernel that reads it every layer.
    try:
        w13[0].fill_(1.0)
        probe = torch.randn(64, HIDDEN, dtype=torch.bfloat16, device=device)
        out = probe @ w13[0].t()
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(out).all())
        detail = f"matmul on a canonical row is finite, {tuple(out.shape)}"
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    if rank == 0:
        report("still usable for local compute", ok, detail)
    if not ok:
        return 1

    # (4) The real operation: write one expert row into a peer's replica row.
    peer = (rank + 1) % world
    w13[REPLICA_ROW].fill_(-1.0)
    w13[0].fill_(float(rank + 1))
    torch.cuda.synchronize()
    hdl13.barrier()

    stream = torch.cuda.Stream()
    try:
        peer_view = hdl13.get_buffer(peer, w13_shape, torch.bfloat16)
        with torch.cuda.stream(stream):
            peer_view[REPLICA_ROW].copy_(w13[0])
        done = torch.cuda.Event()
        with torch.cuda.stream(stream):
            done.record(stream)
        torch.cuda.current_stream().wait_event(done)
        torch.cuda.synchronize()
        hdl13.barrier()
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            report("row-granular write into a peer's replica row", False, f"{type(exc).__name__}: {exc}")
        return 1

    expected = float(((rank - 1) % world) + 1)
    got = float(w13[REPLICA_ROW, 0, 0].item())
    if rank == 0:
        report(
            "row-granular write into a peer's replica row",
            got == expected,
            f"replica row holds {got}, expected {expected} from rank {(rank - 1) % world}",
        )

    # (5) All 48 layers, which is the real memory ask rather than one tensor.
    layers = [(w13, w2)]
    try:
        with torch.cuda.use_mem_pool(pool):
            for _ in range(NUM_LAYERS - 1):
                a = torch.zeros(w13_shape, dtype=torch.bfloat16, device=device)
                b = torch.zeros(w2_shape, dtype=torch.bfloat16, device=device)
                layers.append((a, b))
        for a, b in layers[1:]:
            symm_mem.rendezvous(a, group_name)
            symm_mem.rendezvous(b, group_name)
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            report(
                f"all {NUM_LAYERS} layers",
                False,
                f"failed at layer {len(layers)}: {type(exc).__name__}: {exc}",
            )
        return 1
    total = sum((a.numel() + b.numel()) * 2 for a, b in layers)
    if rank == 0:
        report(f"all {NUM_LAYERS} layers", True, f"{total / 2**30:.2f} GiB per rank")

    # (6) What a row costs.
    for _ in range(5):
        with torch.cuda.stream(stream):
            peer_view[REPLICA_ROW].copy_(w13[0])
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        with torch.cuda.stream(stream):
            start.record(stream)
            peer_view[REPLICA_ROW].copy_(w13[0])
            end.record(stream)
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    if rank == 0:
        med = statistics.median(times)
        print(
            f"\n{row_bytes / 2**20:.2f} MiB row -> peer replica row: p50 {med:.1f} us "
            f"({row_bytes / (med * 1e-6) / 1e9:.0f} GB/s), min {min(times):.1f}, max {max(times):.1f}\n"
            f"  NVSHMEM staged route: 33.0 us put + 6.3 us local copy = 39.3 us for 9 MiB.\n"
            f"  buffer_ptrs_dev present: {hdl13.buffer_ptrs_dev is not None} "
            f"(a device-side pointer table, so a Triton kernel needs no host)",
            flush=True,
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
