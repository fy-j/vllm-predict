# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Can NVSHMEM put directly into the model's own expert weights? Answer before building.

A one-sided put needs its *destination* to be addressable by the sender, which NVSHMEM
provides through the symmetric heap: allocations made in the same order land at the same
offset on every rank, so a sender computes the peer address arithmetically and the CPU
never has to know the plan. The model's expert weights are not symmetric-heap memory —
PyTorch's caching allocator placed them at unrelated addresses on each rank — so there
are two ways to land a replica:

  (b) put into a symmetric staging buffer, then copy locally into the replica row.
      One extra device copy: 6.3 us for Qwen's 9 MiB, 16.8 us for DSV4's 24 MiB at the
      measured 3.00 TB/s.
  (a) `register_external_tensor` the weight tensor itself and put straight into its
      replica row. No staging, no copy — but it puts NVSHMEM constraints on an
      allocation that weight loading and EPLB's `rearrange` also use.

The operator chose (a), which makes registration a load-bearing premise rather than an
option. This probe tests it at the real scale, because the failure modes would otherwise
surface inside weight loading or `rearrange`, far from this feature:

  1. Does registering a torch-allocated tensor work at all, after torch and NCCL are up?
  2. Does it hold for **all 48 layers**, about 7.3 GiB per rank? Registration may have a
     size or count limit, and 48 x 153 MiB is the real ask, not one tensor.
  3. Can a put target a **row slice** of a registered tensor, which is what writing one
     replica row means, rather than the whole buffer?
  4. Does the tensor still work normally for local compute afterwards? If registration
     pins or relocates it, the MoE kernel that reads it is what breaks.
  5. Is a plain CUDA event enough to order the consumer, as it was for the heap path?

Shapes are Qwen3-30B-A3B's, EP=8: 16 canonical + 1 replica row per rank, w13
[17, 2*768, 2048] and w2 [17, 2048, 768] in bf16, so 153 MiB per layer per rank.

**ANSWER (2026-08-29): option (a) is not available on this stack.** Registration and the
symmetric heap have mutually exclusive requirements, four checks deep:

  1. NVSHMEM's per-device memory resource must exist first. It is created lazily on the
     first symmetric-heap allocation (`nvshmem/core/memory.py:89`), so registering before
     any allocation fails with "device that is not initialized with NVSHMEM" — which reads
     like a device-binding bug and is not one. One `nvshmem.core.tensor((1,), uint8)` fixes it.
  2. The buffer size must be a multiple of the heap granularity, 512 MiB by default.
     A layer's w13 is 102 MiB and w2 51 MiB, so 48 layers would need 49 GiB per rank.
     `NVSHMEM_CUMEM_GRANULARITY=2097152` lowers it to 2 MiB, and one padding row (18 rows,
     not 17) makes both tensors align.
  3. The buffer must be CUDA VMM allocated. PyTorch's caching allocator uses `cudaMalloc`,
     so a plain tensor fails with "Please check if buffer is allocated using CUDA VMM API".
     `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` makes torch use VMM and clears this.
  4. **And then `cuMemMap` fails with `CUDA_ERROR_NOT_SUPPORTED`**: the handle type torch
     creates its VMM allocations with is not one NVSHMEM can map into its heap, and torch
     does not expose that as a knob.

`NVSHMEM_DISABLE_CUDA_VMM=1` makes step 4 disappear and registration *report* success for
all 48 layers, 7.59 GiB per rank — but the first real put then fails with "Buffer
registration requires dynamic VMM heap". So the registration was not symmetric. The two
settings are mutually exclusive: registration needs the VMM heap, and the VMM heap needs a
mappable user handle that torch does not provide.

**Consequence: land replicas through a symmetric-heap staging buffer (option b).** Its
cost is one local device copy, 6.3 us for Qwen's 9 MiB and 16.8 us for DSV4's 24 MiB at the
measured 3.00 TB/s, against a 67.87 ms prefill step — 0.01%. There is also an option (c):
allocate the expert weights *from* the symmetric heap with `nvshmem.core.tensor()` rather
than registering torch memory. That is mechanically proven by `probe_nvshmem.py`, but it
puts NVSHMEM's allocator underneath the model's own weights, which weight loading and
EPLB's `rearrange` both use — a deeper intrusion than registration would have been.

A row-granular put needs `nvshmem.bindings.putmem_on_stream(dst_ptr, src_ptr, bytes, pe,
stream)`. `nvshmem.core.put` resolves its arguments through nvshmem's tracking table, which
holds whole allocations, so passing a row slice raises "Tensor not tracked by nvshmem".

Run:
    NVSHMEM_CUMEM_GRANULARITY=2097152 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        torchrun --nproc_per_node=8 probe_nvshmem_register.py
"""

from __future__ import annotations

import os
import statistics
import sys

import torch
import torch.distributed as dist

HIDDEN = 2048
MOE_INTERMEDIATE = 768
# 18, not 17. `register_external_buffer` requires the buffer size to be a multiple of the
# heap granularity, and w2 at 17 rows is 51 MiB — not a multiple of even the smallest
# granularity CUDA VMM allows (2 MiB), since a w2 row is 3 MiB. One padding row makes both
# tensors align: w13 18x6 = 108 MiB, w2 18x3 = 54 MiB, both multiples of 2 MiB.
ROWS = 18  # 16 canonical + 1 replica slot + 1 alignment pad
NUM_LAYERS = 48
REPLICA_ROW = 16
ITERS = 30


def report(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")

    import nvshmem.core as nvshmem

    try:
        from cuda.core import Device
    except ImportError:
        from cuda.core.experimental import Device  # type: ignore

    # Bind NVSHMEM to the *same* device torch is using. `Device()` with no argument takes
    # cuda.core's current device, which does not follow `torch.cuda.set_device`, and then
    # registration fails with "device that is not initialized with NVSHMEM" — the tensors
    # live on one device and NVSHMEM was initialised on another.
    local = rank % torch.cuda.device_count()
    dev = Device(local)
    dev.set_current()

    uid = [nvshmem.get_unique_id(empty=rank != 0)]
    dist.broadcast_object_list(uid, src=0)
    dist.barrier()
    nvshmem.init(
        device=dev, uid=uid[0], rank=rank, nranks=world, initializer_method="uid"
    )
    if rank == 0:
        report("NVSHMEM up after torch and NCCL", True, f"{nvshmem.n_pes()} PEs")

    # `register_external_tensor` needs NVSHMEM's memory resource for this device to
    # already exist, and that is created lazily on the *first* symmetric-heap allocation
    # (`memory.py:89`). Registering before any allocation fails with "device that is not
    # initialized with NVSHMEM", which reads like a device-binding problem and is not one.
    # One byte is enough to force it.
    _mr_anchor = nvshmem.tensor((1,), dtype=torch.uint8)
    if rank == 0:
        report("symmetric-heap anchor allocated", True, "forces the per-device MR")

    # (1) and (2): register every layer, which is the real ask.
    w13_bytes = ROWS * 2 * MOE_INTERMEDIATE * HIDDEN * 2
    w2_bytes = ROWS * HIDDEN * MOE_INTERMEDIATE * 2
    per_layer = w13_bytes + w2_bytes
    weights, registered = [], []
    try:
        for _ in range(NUM_LAYERS):
            w13 = torch.zeros(
                ROWS, 2 * MOE_INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda"
            )
            w2 = torch.zeros(
                ROWS, HIDDEN, MOE_INTERMEDIATE, dtype=torch.bfloat16, device="cuda"
            )
            weights.append((w13, w2))
            # Collective: every PE must register in the same order.
            registered.append(
                (
                    nvshmem.register_external_tensor(w13),
                    nvshmem.register_external_tensor(w2),
                )
            )
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            report(
                f"register all {NUM_LAYERS} layers",
                False,
                f"failed after {len(registered)} layers "
                f"({len(registered) * per_layer / 2**30:.2f} GiB): "
                f"{type(exc).__name__}: {exc}",
            )
        return 1
    if rank == 0:
        report(
            f"register all {NUM_LAYERS} layers",
            True,
            f"{NUM_LAYERS * per_layer / 2**30:.2f} GiB per rank, "
            f"{per_layer / 2**20:.0f} MiB per layer",
        )

    # (4): the tensor must still be an ordinary tensor for the kernel that reads it.
    w13, w2 = weights[0]
    try:
        w13[0].fill_(1.0)
        probe = torch.randn(64, HIDDEN, dtype=torch.bfloat16, device="cuda")
        out = probe @ w13[0].t()
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(out).all())
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    else:
        detail = f"matmul on a registered row gives finite output, {tuple(out.shape)}"
    if rank == 0:
        report("registered tensor still usable for local compute", ok, detail)
    if not ok:
        return 1

    # (3): put into the *replica row* of the peer's registered tensor, not the whole thing.
    peer = (rank + 1) % world
    stream = torch.cuda.Stream()
    src_w13, _ = registered[0]
    payload = float(rank + 1)
    with torch.cuda.stream(stream):
        w13[REPLICA_ROW].fill_(-1.0)
    torch.cuda.synchronize()
    dist.barrier()

    # Each rank fills a *canonical* row with its own id and puts that row into the peer's
    # replica row: the shape of a real transfer, one expert out of a layer's block.
    with torch.cuda.stream(stream):
        w13[0].fill_(payload)
    torch.cuda.synchronize()
    dist.barrier()

    # A row-granular put cannot go through `nvshmem.core.put`: that resolves its
    # arguments through nvshmem's tracking table, which holds the whole registered
    # tensor and not a slice of it, so a slice raises "Tensor not tracked by nvshmem".
    # The bindings layer takes raw pointers, which is what writing one expert row into a
    # layer's block means and what a device-side implementation would emit anyway.
    import nvshmem.bindings as nvb

    row_bytes = 2 * MOE_INTERMEDIATE * HIDDEN * 2
    base = w13.data_ptr()
    dst_ptr = base + REPLICA_ROW * row_bytes
    src_ptr = base  # canonical row 0
    try:
        with torch.cuda.stream(stream):
            nvb.putmem_on_stream(dst_ptr, src_ptr, row_bytes, peer, stream.cuda_stream)
        done = torch.cuda.Event()
        with torch.cuda.stream(stream):
            done.record(stream)
        torch.cuda.current_stream().wait_event(done)
        torch.cuda.synchronize()
        dist.barrier()
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            report("row-granular put via bindings", False, f"{type(exc).__name__}: {exc}")
        return 1

    expected = float(((rank - 1) % world) + 1)
    got = float(w13[REPLICA_ROW, 0, 0].item())
    if rank == 0:
        report(
            "row-granular put into a registered weight tensor",
            got == expected,
            f"replica row holds {got}, expected {expected} "
            f"(from rank {(rank - 1) % world}), {row_bytes / 2**20:.0f} MiB",
        )

    for _ in range(5):
        with torch.cuda.stream(stream):
            nvb.putmem_on_stream(dst_ptr, src_ptr, row_bytes, peer, stream.cuda_stream)
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        with torch.cuda.stream(stream):
            start.record(stream)
            nvb.putmem_on_stream(dst_ptr, src_ptr, row_bytes, peer, stream.cuda_stream)
            end.record(stream)
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    if rank == 0:
        med = statistics.median(times)
        print(
            f"\n{row_bytes / 2**20:.2f} MiB direct put into a weight row: "
            f"p50 {med:8.1f} us  ({row_bytes / (med * 1e-6) / 1e9:6.1f} GB/s)\n"
            f"  a staged put measured 33.0 us plus a 6.3 us local copy, so option (a) "
            f"wins if this is at or under about 39 us.",
            flush=True,
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
