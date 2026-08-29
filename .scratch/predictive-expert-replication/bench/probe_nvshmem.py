# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Does device-initiated P2P work here? Answer before building anything on it.

The host synchronisation this feature stalls on is not an artifact of stream choice.
`ncclSend`'s peer is a `ctypes.c_int` consumed when the host enqueues, so the host
must know which expert goes to which rank, so it must read a device tensor, so it must
wait for the GPU. Measured: GPU occupancy 87% -> 53%, one `cudaEventSynchronize` of
11.02 ms.

One-sided put removes the host from that loop: the address is computed on the device
and there is no matching receive to pair with. That needs NVSHMEM, which needs NVLink,
which this 8x RTX 5090 node does not have. This script is what to run on the 8x H200
before integrating, and it answers the three things that would each invalidate the
design on their own:

  1. Is a stream-ordered put available? Only a stream-ordered API lets the target layer
     wait with `wait_event` — a device-side wait. A device-side-only put needs flags and
     polling, and polling is per-rank timing, which is how this branch deadlocked twice.
  2. Can the symmetric heap be allocated after torch and NCCL are already up? vLLM
     initialises those first and we do not get to go earlier.
  3. Can NVSHMEM and NCCL coexist in one process? vLLM's all2all backend keeps using
     NCCL for token dispatch either way.

It also measures the one number that decides how much of the rest is needed: the
latency of a 9.00 MiB put. At PCIe it is 289 us, and hiding it drove several rounds of
design. If NVLink makes it ~13 us, the planning delay can stay at one layer and the
lookahead does not have to rise at all, so the prediction accuracy cost is zero.

Run:
    torchrun --nproc_per_node=8 probe_nvshmem.py
"""

from __future__ import annotations

import inspect
import os
import statistics
import sys

import torch
import torch.distributed as dist

EXPERT_BYTES = 9 * 1024 * 1024  # one Qwen3-30B-A3B expert, as measured
ITERS = 50


def report(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # (3) NCCL first, exactly as vLLM does, so coexistence is actually exercised.
    dist.init_process_group(backend="nccl")
    probe = torch.ones(1024, device="cuda")
    dist.all_reduce(probe)
    if rank == 0:
        report("NCCL up before NVSHMEM", probe[0].item() == world)

    try:
        import nvshmem.core as nvshmem  # type: ignore
    except ImportError:
        try:
            import pynvshmem as nvshmem  # type: ignore
        except ImportError:
            if rank == 0:
                report(
                    "NVSHMEM importable", False,
                    "neither nvshmem.core nor pynvshmem; install NVSHMEM's Python "
                    "bindings with `uv pip install --extra-index-url "
                    "https://pypi.nvidia.com nvshmem4py-cu13`. They are not on public "
                    "PyPI, and `nvidia-nvshmem-cu13` alone ships only the C library. "
                    "Without them this design cannot be built at all and the fallback "
                    "is raising the planning delay, which costs prediction accuracy.",
                )
            return 1
    if rank == 0:
        report("NVSHMEM importable", True, nvshmem.__name__)

    # (2) Symmetric heap after torch and NCCL are already initialised.
    #
    # UID bootstrap, not MPI: vLLM is not launched under mpirun and `mpi4py` must be
    # built against the same MPI it runs with, which is a dependency this project will
    # not take on. `get_unique_id(empty=True)` is how a non-root rank makes a
    # correctly-typed receptacle for the broadcast — it is not an error path.
    try:
        # `cuda.core.Device` since cuda-python 13; it was `cuda.core.experimental`
        # before, and nvshmem4py's own docs still say the old path.
        try:
            from cuda.core import Device  # type: ignore
        except ImportError:
            from cuda.core.experimental import Device  # type: ignore

        uid = [
            nvshmem.get_unique_id(empty=rank != 0)  # type: ignore[attr-defined]
        ]
        dist.broadcast_object_list(uid, src=0)
        dist.barrier()
        nvshmem.init(  # type: ignore[attr-defined]
            device=Device(),
            uid=uid[0],
            rank=rank,
            nranks=world,
            initializer_method="uid",
        )
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            report("symmetric heap after NCCL", False, f"{type(exc).__name__}: {exc}")
        return 1
    if rank == 0:
        report("symmetric heap after NCCL", True, f"{nvshmem.n_pes()} PEs")

    # `tensor`, not `empty`: nvshmem4py allocates symmetric memory and hands back a
    # torch view in one call. `register_external_tensor` also exists, which matters
    # more than this probe does — see the note at the end of this file.
    try:
        send = nvshmem.tensor((EXPERT_BYTES,), dtype=torch.uint8)  # type: ignore
        recv = nvshmem.tensor((EXPERT_BYTES,), dtype=torch.uint8)  # type: ignore
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            report("9 MiB symmetric buffers", False, f"{type(exc).__name__}: {exc}")
        return 1
    send.fill_(rank + 1)
    recv.zero_()

    # (1) Stream-ordered put, so a plain CUDA event can order the consumer.
    peer = (rank + 1) % world
    stream = torch.cuda.Stream()
    # Stream-orderedness is not a separate entry point here: `put` itself takes the
    # stream and raises NotImplementedError when it is None. So the check is that the
    # parameter exists, not that a `*_on_stream` name does.
    put = getattr(nvshmem, "put", None)
    takes_stream = put is not None and "stream" in inspect.signature(put).parameters
    if not takes_stream:
        if rank == 0:
            report(
                "stream-ordered put", False,
                "no host-initiated put taking a stream. A device-side-only put needs "
                "a flag and polling, and polling is per-rank timing — the divergence "
                "class that deadlocked this branch twice. Reconsider before using it.",
            )
        return 1
    if rank == 0:
        report("stream-ordered put", True, "nvshmem.core.put(..., stream=)")

    def one_put() -> None:
        put(recv, send, peer, stream=stream)  # type: ignore[misc]

    # Correctness: the neighbour's payload must land, and a CUDA event must be enough
    # to know it did. If `wait_event` is not sufficient the data will be wrong here.
    one_put()
    done = torch.cuda.Event()
    with torch.cuda.stream(stream):
        done.record(stream)
    torch.cuda.current_stream().wait_event(done)
    torch.cuda.synchronize()
    dist.barrier()
    expected = ((rank - 1) % world) + 1
    got = int(recv[0].item())
    if rank == 0:
        report(
            "wait_event orders the put", got == expected,
            f"expected {expected}, got {got}"
            + ("" if got == expected else "; a stream event is not sufficient, so the "
               "consumer needs an NVSHMEM fence instead"),
        )

    # The number that decides how much hiding the transfer still needs.
    for _ in range(5):
        one_put()
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        with torch.cuda.stream(stream):
            start.record(stream)
            put(recv, send, peer, stream=stream)  # type: ignore[misc]
            end.record(stream)
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    if rank == 0:
        med = statistics.median(times)
        gbs = EXPERT_BYTES / (med * 1e-6) / 1e9
        print(f"\n9.00 MiB put: p50 {med:8.1f} us  ({gbs:6.1f} GB/s)")
        print(f"              min {min(times):8.1f} us   max {max(times):8.1f} us")
        print(f"  PCIe on the 5090 node measured 289 us. 43 per forward is "
              f"{43 * med / 1000:.1f} ms here against 12.4 ms there.")
        print("  Under ~50 us the planning delay can stay at one layer and the "
              "lookahead need not rise, so prediction accuracy costs nothing.")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
