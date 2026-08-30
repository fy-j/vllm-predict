# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Can a CUDA kernel issue the put, reading the plan from device memory?

Ticket 06's precondition, and the one that decides whether its headline criterion is
reachable at all. Removing the host synchronisation is not a matter of moving code
around: a **host-issued** put takes its peer, its source pointer and its byte count as
host integers, consumed when the host enqueues — structurally the same constraint as
`ncclSend`'s `ctypes.c_int` peer, which is what put the 5.28 ms per layer there in the
first place. So a host-issued put cannot be aimed by a plan that lives on the device,
whatever stream it goes on.

Two escapes exist and only one is affordable:

* Issue every put the plan *might* have chosen and mask the rest. The plan picks one
  expert of one rank, so this is 16 rows to 7 peers — 63 MiB of egress per rank per
  layer, about 230 us on this interconnect, 10 ms across 43 layers. Against a prefill
  window near 95 ms that spends the ceiling to save the sync.
* Issue the put from **inside a kernel**, which reads the plan where it already lives.
  This is the escape `CLAUDE.md` names, and this script is the smallest thing that
  proves the toolchain for it: NVRTC compiles a kernel calling
  `nvshmemx_putmem_nbi_block`, nvJitLink links it against `libnvshmem_device_sm_90.bc`,
  and the peer arrives in device memory.

If it fails, say so plainly rather than working around it: without a device-issued put,
ticket 06 can remove the *planner's* host read and not the transfer's, and the honest
cost is then one host sync per predicted layer regardless.

Run: torchrun --nproc_per_node=8 probe_device_put.py
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

PAYLOAD = 9 * 1024 * 1024  # one expert, as measured
ITERS = 20

# `nvshmemx_putmem_nbi_block` and not the thread variant: one block moves the payload
# cooperatively, and a 9 MiB row wants every thread it can get. The plan is read from
# device memory — that is the whole point — so `pe` and the offsets are loads, not
# launch arguments.
KERNEL = r"""
#include <nvshmem.h>
#include <nvshmemx.h>

extern "C" __global__ void put_from_plan(
    void *dst, const void *src, unsigned long long nbytes, const long long *plan)
{
    // plan[0] is `found`: a forward that placed nothing must move nothing, and deciding
    // that here rather than on the host is exactly what removes the synchronisation.
    if (plan[0] == 0) {
        return;
    }
    int pe = (int)plan[2];
    // One block per chunk. A single block moving all 9 MiB measures 206 us against the
    // 53.7 us of the host-issued put, because it is one block's load/store units
    // against the copy engine; splitting the payload across blocks is what closes that.
    unsigned long long chunk = (nbytes + gridDim.x - 1) / gridDim.x;
    unsigned long long offset = (unsigned long long)blockIdx.x * chunk;
    if (offset >= nbytes) {
        return;
    }
    if (chunk > nbytes - offset) {
        chunk = nbytes - offset;
    }
    nvshmemx_putmem_nbi_block(
        (char *)dst + offset, (const char *)src + offset, chunk, pe);
    // A non-blocking put is completed by a quiet, per the NVSHMEM spec. Kept even
    // though the check below passes without it: two passing runs of one payload is not
    // evidence that the stream barrier completes a device-issued nbi put, and this has
    // already shipped one ordering claim resting on exactly that kind of observation.
    __syncthreads();
    if (threadIdx.x == 0) {
        nvshmem_quiet();
    }
}
"""


_failures = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    """Print one check and remember it, so the exit status means something."""
    global _failures
    if not ok:
        _failures += 1
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def nvshmem_paths() -> tuple[list[str], Path]:
    """Every include directory the kernel needs, and the device bitcode for this arch.

    From the installed wheels rather than a system path: `nvidia-nvshmem-cu13` ships
    NVSHMEM's headers and bitcode, and this node's CUDA install has neither. NVSHMEM's
    own headers then include `cuda_runtime.h`, which NVRTC does not supply, so
    `nvidia-cuda-runtime`'s include directory has to come along — the compile fails with
    a "catastrophic error" naming that file, which reads like a broken toolchain rather
    than a missing `-I`. It then asks for `cuda/std/cstdint`, which is libcu++ from
    `nvidia-cuda-cccl`, so that is a third `-I`. Each one only surfaces after the
    previous is supplied.
    """
    # Located by `__path__`, not `__file__`: `nvidia.nvshmem` is a namespace package, so
    # its `__file__` is None and `Path(None)` raises a TypeError about `__fspath__`.
    import nvidia.cuda_cccl
    import nvidia.cuda_runtime
    import nvidia.nvshmem

    root = Path(next(iter(nvidia.nvshmem.__path__)))
    runtime = Path(next(iter(nvidia.cuda_runtime.__path__)))
    cccl = Path(next(iter(nvidia.cuda_cccl.__path__)))
    bitcode = root / "lib" / "libnvshmem_device.a"
    includes = [root / "include", runtime / "include", cccl / "include"]
    return [str(path) for path in includes if path.exists()], bitcode


def build_kernel(arch: str) -> object:
    """Compile and link the put kernel, returning something launchable.

    Relocatable device code plus the shipped archive, which is the combination nvJitLink
    accepts. The `.bc` files beside the archive are raw LLVM bitcode and are rejected as
    both `NVJITLINK_INPUT_LIBRARY` and `NVJITLINK_INPUT_LTOIR`, so they are not the
    route.
    """
    from cuda.core import Linker, LinkerOptions, ObjectCode, Program, ProgramOptions

    includes, bitcode = nvshmem_paths()
    program = Program(
        KERNEL,
        code_type="c++",
        options=ProgramOptions(
            arch=arch,
            include_path=includes,
            relocatable_device_code=True,
            define_macro=("NVSHMEM_TARGET", "1"),
        ),
    )
    compiled = program.compile("ptx")
    linker = Linker(
        compiled,
        ObjectCode.from_library(bitcode.read_bytes()),
        options=LinkerOptions(arch=arch),
    )
    return linker.link("cubin")


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")

    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    _includes, bitcode = nvshmem_paths()
    if rank == 0:
        report(
            "the device library ships with the wheel",
            bitcode.exists(),
            f"{arch}: {bitcode.name}",
        )
        if not bitcode.exists():
            return 1

    from vllm.distributed.eplb.nvshmem_transfer import OneSidedExpertTransfer

    def broadcast_uid(local):
        holder = [local]
        dist.broadcast_object_list(holder, src=0)
        dist.barrier()
        return holder[0]

    transport = OneSidedExpertTransfer(
        rank=rank,
        world_size=world,
        expert_bytes=PAYLOAD,
        device=device,
        broadcast_uid=broadcast_uid,
    )
    # Built *after* the transport, not before: `cuda.core` loads the module into its own
    # current context, and NVSHMEM's registration then runs in the context
    # `nvshmem.init` bound. Build first and the two differ, and registration fails with
    # CUDA_ERROR_INVALID_HANDLE — which looks like a bad module and is a context
    # mismatch.
    try:
        module = build_kernel(arch)
        kernel = module.get_kernel("put_from_plan")
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            report(
                "NVRTC compiles and nvJitLink links a device put",
                False,
                f"{type(exc).__name__}: {exc}",
            )
        transport.close()
        return 1
    if rank == 0:
        report("NVRTC compiles and nvJitLink links a device put", True, arch)

    # The device library keeps per-CUmodule state, and a kernel launched without it
    # registered faults with an illegal memory access on the first NVSHMEM call — which
    # reads as a bad pointer in our own arguments and is not. Registration must follow
    # `nvshmem.init`, so it cannot happen at build time.
    #
    # `library_init`, not `module_init`: `cuda.core` loads a cubin as a `CUlibrary`, and
    # passing that to `module_init` fails with the same CUDA_ERROR_INVALID_HANDLE as the
    # context mismatch above — two unrelated causes, one message. `module_finalize` is
    # also unusable whatever is passed to it, because `module_init` never sets the
    # `finalize_handle` it then requires.
    import nvshmem.core as nvshmem_core

    nvshmem_module = nvshmem_core.NvshmemKernelObject.from_obj(module)
    nvshmem_core.library_init(nvshmem_module)

    source = torch.full((PAYLOAD,), rank + 1, dtype=torch.uint8, device=device)
    # The plan as ticket 04 produces it: `(found, logical_expert, target_rank,
    # moved_x2)`, on the device, never read by this process.
    plan = torch.tensor([1, 0, (rank + 1) % world, 0], dtype=torch.int64, device=device)
    stream = torch.cuda.Stream()

    from cuda.core import Device as CoreDevice
    from cuda.core import LaunchConfig, launch

    class _TorchStream:
        """Adapter so cuda.core can launch onto torch's stream.

        `launch` needs a `cuda.core.Stream`, and torch's stream is not one; the protocol
        it accepts is `__cuda_stream__` returning `(version, handle)`. Launching onto a
        stream torch does not know about would put the transfer outside the ordering the
        rest of the path relies on, so wrapping rather than creating is the requirement.
        """

        def __init__(self, torch_stream: torch.cuda.Stream):
            self._handle = torch_stream.cuda_stream

        def __cuda_stream__(self) -> tuple[int, int]:
            return (0, self._handle)

    core_stream = CoreDevice(device.index).create_stream(_TorchStream(stream))
    config = LaunchConfig(grid=32, block=1024)

    def one_put() -> None:
        launch(
            core_stream,
            config,
            kernel,
            # numpy scalars, because cuda.core packs arguments by dtype and rejects both
            # `ctypes` values and bare Python ints — the latter have no unambiguous
            # width.
            np.uint64(transport._staging.data_ptr()),
            np.uint64(source.data_ptr()),
            np.uint64(PAYLOAD),
            np.uint64(plan.data_ptr()),
        )

    try:
        one_put()
        transport.barrier(stream)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            report(
                "the kernel launches and completes",
                False,
                f"{type(exc).__name__}: {exc}",
            )
        transport.close()
        return 1

    expected = ((rank - 1) % world) + 1
    got = int(transport._staging[0].item())
    # int64, not bool: `all_reduce` with SUM over a bool tensor saturates back to a
    # bool, so 8 agreeing ranks reduce to 1 and the check reads as "1/8 ranks" whatever
    # happened.
    ok = torch.tensor([int(got == expected)], dtype=torch.int64, device=device)
    dist.all_reduce(ok)
    if rank == 0:
        report(
            "a device-issued put lands, aimed by a plan the host never read",
            int(ok) == world,
            f"{int(ok)}/{world} ranks received their neighbour's payload",
        )

    for _ in range(3):
        one_put()
        transport.barrier(stream)
    torch.cuda.synchronize()
    spans = []
    for _ in range(ITERS):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        with torch.cuda.stream(stream):
            start.record(stream)
        one_put()
        transport.barrier(stream)
        with torch.cuda.stream(stream):
            end.record(stream)
        torch.cuda.synchronize()
        spans.append(start.elapsed_time(end) * 1000.0)
    if rank == 0:
        median = statistics.median(spans)
        print(
            f"\n9.00 MiB device-issued put + barrier: p50 {median:8.1f} us   "
            f"min {min(spans):8.1f} us   max {max(spans):8.1f} us"
        )
        print(
            "  Against 53.7 us host-issued, so not slower. What it buys is not speed, "
            "though: it is that the plan stays on the device, which is what removes "
            "the 5.28 ms per layer of host waiting."
        )

    dist.barrier()
    nvshmem_core.library_finalize(nvshmem_module)
    transport.close()
    dist.destroy_process_group()
    return min(_failures, 1)


if __name__ == "__main__":
    sys.exit(main())
