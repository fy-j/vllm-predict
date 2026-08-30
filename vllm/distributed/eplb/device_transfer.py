# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Move an expert's weights from a kernel, so the plan never reaches the host.

Ticket 06. Ticket 05's transport works and is fast — 53.7 us for a 9.00 MiB expert — but
it cannot remove the cost this ticket is about, because a **host-issued** put takes its
peer, its source pointer and its byte count as host integers consumed when the host
enqueues. That is the same constraint `ncclSend` has, and it is where the 5.28 ms per
predicted layer comes from: the host must know which expert goes where, so it must read
a device tensor, so it must wait. Measured, that is 70% of everything prediction adds,
against a perfect-balance ceiling of about 5%.

So the put is issued from inside a kernel, which reads the plan where it already lives.
Measured at **47.6 us against 53.7 us host-issued**, so this is not a trade: it is the
same time with the host removed. The alternative of issuing every put the plan might
have chosen and masking the rest was ruled out by arithmetic — 16 rows to 7 peers is
about 230 us per layer, 10 ms across 43 layers.

Two kernels and one barrier, because arrival cannot be established without one:

1. `put_expert` runs on every rank, and the rank that owns the chosen expert puts its
   row into the target's staging buffer. Whether this rank is that owner is decided from
   the plan, on the device.
2. `nvshmemx_barrier_all_on_stream`, host-issued and needing no plan value, which is
   what makes the bytes visible. **A CUDA event is not sufficient and never was** — the
   probe that concluded otherwise had a `dist.barrier()` inside the region it checked,
   and removing the barrier leaves 51 of 112 weight tensors wrong.
3. `drain_expert` runs on every rank, and the target copies its staging buffer into its
   replica row.

Weight addresses come from a pointer table built once at startup. That is host work, but
it happens before any forward, and nothing in it depends on a plan.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# One kernel per direction, in one compilation unit so the launch bounds and the byte
# arithmetic cannot drift apart between them. `nvshmemx_putmem_nbi_block` moves the
# payload with a whole block; a single block for all 9 MiB measured 206 us against 47.6
# us for 32, because one block is its own load/store units rather than the copy engine.
_SOURCE = r"""
#include <nvshmem.h>
#include <nvshmemx.h>

// (found, logical_expert, target_rank, moved_x2), exactly as plan_one_layer_on_device
// returns it. Read on the device and nowhere else.
#define PLAN_FOUND 0
#define PLAN_EXPERT 1
#define PLAN_TARGET 2

extern "C" __global__ void put_expert(
    const long long *plan,
    const long long *weight_ptrs,
    const long long *row_bytes,
    const long long *row_strides,
    int num_tensors,
    char *staging,
    int my_pe,
    int per_rank_experts)
{
    if (plan[PLAN_FOUND] == 0) {
        return;
    }
    long long expert = plan[PLAN_EXPERT];
    if ((int)(expert / per_rank_experts) != my_pe) {
        return;
    }
    int pe = (int)plan[PLAN_TARGET];
    long long row = expert % per_rank_experts;

    // Blocks are handed out across (tensor, chunk) so an expert's several tensors move
    // in parallel rather than one after another.
    int chunks = gridDim.x / num_tensors;
    if (chunks < 1) {
        chunks = 1;
    }
    int tensor = blockIdx.x / chunks;
    int chunk = blockIdx.x % chunks;
    if (tensor >= num_tensors) {
        return;
    }

    long long offset = 0;
    for (int t = 0; t < tensor; ++t) {
        offset += row_bytes[t];
    }
    long long total = row_bytes[tensor];
    long long span = (total + chunks - 1) / chunks;
    long long start = (long long)chunk * span;
    if (start >= total) {
        return;
    }
    if (span > total - start) {
        span = total - start;
    }

    const char *src =
        (const char *)weight_ptrs[tensor] + row * row_strides[tensor] + start;
    char *dst = staging + offset + start;
    nvshmemx_putmem_nbi_block(dst, src, (size_t)span, pe);
    // A non-blocking put is completed by a quiet, per the NVSHMEM spec. The barrier
    // that follows appears to complete it as well, but this project has already shipped
    // one ordering claim resting on that kind of observation and it cost a session.
    __syncthreads();
    if (threadIdx.x == 0) {
        nvshmem_quiet();
    }
}

extern "C" __global__ void drain_expert(
    const long long *plan,
    const long long *weight_ptrs,
    const long long *row_bytes,
    const long long *row_strides,
    int num_tensors,
    const char *staging,
    int my_pe,
    int replica_row)
{
    if (plan[PLAN_FOUND] == 0 || (int)plan[PLAN_TARGET] != my_pe) {
        return;
    }
    int chunks = gridDim.x / num_tensors;
    if (chunks < 1) {
        chunks = 1;
    }
    int tensor = blockIdx.x / chunks;
    int chunk = blockIdx.x % chunks;
    if (tensor >= num_tensors) {
        return;
    }

    long long offset = 0;
    for (int t = 0; t < tensor; ++t) {
        offset += row_bytes[t];
    }
    long long total = row_bytes[tensor];
    long long span = (total + chunks - 1) / chunks;
    long long start = (long long)chunk * span;
    if (start >= total) {
        return;
    }
    if (span > total - start) {
        span = total - start;
    }

    char *dst = (char *)weight_ptrs[tensor]
        + (long long)replica_row * row_strides[tensor] + start;
    const char *src = staging + offset + start;

    // 16 bytes per thread where both sides allow it. Expert weights are row-major and
    // 256-byte aligned in practice, but a tail is handled rather than assumed: a
    // quantised layer carries scale tensors whose rows are much smaller.
    long long index = (long long)threadIdx.x;
    long long step = (long long)blockDim.x;
    bool aligned = (((uintptr_t)dst | (uintptr_t)src) & 15) == 0;
    long long vectors = aligned ? span / 16 : 0;
    for (long long v = index; v < vectors; v += step) {
        ((uint4 *)dst)[v] = ((const uint4 *)src)[v];
    }
    for (long long b = vectors * 16 + index; b < span; b += step) {
        dst[b] = src[b];
    }
}
"""

_GRID_CHUNKS_PER_TENSOR = 16
_BLOCK = 512

# Debug only, for bisecting a crash between the kernels and the barrier. `full` is
# the real path; `no-barrier` produces wrong bytes on purpose and must not serve.
_STAGE = os.environ.get("VLLM_PREDICTIVE_DEVICE_TRANSFER_STAGE", "full")


def _nvshmem_build_paths() -> tuple[list[str], Path]:
    """Include directories and the device archive, all from installed wheels.

    Not from a system CUDA install, which on this node has neither. Each of these only
    surfaces once the previous one is supplied, and each failure names something other
    than the missing include: NVSHMEM's headers want `cuda_runtime.h` (a "catastrophic
    error" that reads as a broken toolchain) and then `cuda/std/cstdint` from libcu++.
    """
    import nvidia.cuda_cccl
    import nvidia.cuda_runtime
    import nvidia.nvshmem

    # `__path__`, not `__file__`: these are namespace packages, so `__file__` is None
    # and `Path(None)` raises a TypeError about `__fspath__`.
    root = Path(next(iter(nvidia.nvshmem.__path__)))
    includes = [
        root / "include",
        Path(next(iter(nvidia.cuda_runtime.__path__))) / "include",
        Path(next(iter(nvidia.cuda_cccl.__path__))) / "include",
    ]
    return [
        str(p) for p in includes if p.exists()
    ], root / "lib" / "libnvshmem_device.a"


@dataclass
class WeightPointers:
    """Where one layer's expert weight rows live, as device tensors.

    Built once at startup: the addresses are fixed for the model's lifetime, so nothing
    here is per-forward work, and nothing in it depends on a plan.

    Attributes:
        base: `[num_tensors]` int64 base address of each weight tensor.
        row_bytes: `[num_tensors]` bytes in one expert's row of each tensor.
        row_strides: `[num_tensors]` bytes between consecutive rows. Equal to
            `row_bytes` for a contiguous tensor, and kept separately rather than assumed
            so a padded or otherwise strided layout is moved correctly instead of
            silently sheared.
        total_bytes: One expert's bytes across all tensors, which is the staging size.
    """

    base: torch.Tensor
    row_bytes: torch.Tensor
    row_strides: torch.Tensor
    total_bytes: int

    @staticmethod
    def build(tensors: Sequence[torch.Tensor]) -> WeightPointers:
        """Read the addresses of one layer's weight tensors.

        Raises:
            ValueError: If a tensor has no rows to index or a non-contiguous row, either
                of which would make the byte arithmetic here describe something other
                than what the kernel copies.
        """
        if not tensors:
            raise ValueError(
                "a layer has no weight tensors to transfer, which is a wiring error "
                "rather than a zero-byte transfer"
            )
        base, row_bytes, row_strides = [], [], []
        for index, tensor in enumerate(tensors):
            if tensor.dim() < 2 or tensor.shape[0] < 1:
                raise ValueError(
                    f"expert weight tensor {index} has shape {tuple(tensor.shape)}, "
                    f"which has no expert rows to move"
                )
            row = tensor[0]
            if not row.is_contiguous():
                raise ValueError(
                    f"expert weight tensor {index} has a non-contiguous row "
                    f"{tuple(row.shape)} with strides {row.stride()}; the transfer "
                    f"moves a row as bytes and cannot do that for a strided view"
                )
            base.append(tensor.data_ptr())
            row_bytes.append(row.numel() * tensor.element_size())
            row_strides.append(tensor.stride(0) * tensor.element_size())
        device = tensors[0].device
        as_tensor = lambda values: torch.tensor(  # noqa: E731
            values, dtype=torch.int64, device=device
        )
        return WeightPointers(
            base=as_tensor(base),
            row_bytes=as_tensor(row_bytes),
            row_strides=as_tensor(row_strides),
            total_bytes=sum(row_bytes),
        )


class DeviceExpertTransfer:
    """The two kernels and the barrier, launched from a plan that stays on the device.

    One instance per worker. Building the kernels needs NVSHMEM already initialised,
    because registration binds to the context `nvshmem.init` bound — build first and the
    module lands in `cuda.core`'s own context, which fails with
    `CUDA_ERROR_INVALID_HANDLE` and looks like a bad module rather than a context
    mismatch.
    """

    def __init__(self, staging: torch.Tensor, ep_rank: int, per_rank_experts: int):
        """Compile, link and register the kernels.

        Args:
            staging: The symmetric staging buffer, one expert wide, from
                `OneSidedExpertTransfer`.
            ep_rank: This rank's PE id, which must be its EP rank — the plan names
                target ranks in EP terms and the put addresses PEs.
            per_rank_experts: Canonical experts per rank, which is how the kernel
                derives the source rank from the chosen expert without being told it.
        """
        from cuda.core import (
            Device,
            LaunchConfig,
            Linker,
            LinkerOptions,
            ObjectCode,
            Program,
            ProgramOptions,
        )

        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
        includes, archive = _nvshmem_build_paths()
        if not archive.exists():
            raise RuntimeError(
                f"NVSHMEM's device library is missing at {archive}, so the put cannot "
                f"be issued from a kernel. Install `nvshmem4py-cu13` from "
                f"https://pypi.nvidia.com."
            )
        program = Program(
            _SOURCE,
            code_type="c++",
            options=ProgramOptions(
                arch=arch,
                include_path=includes,
                relocatable_device_code=True,
            ),
        )
        # Relocatable PTX against the archive. The `.bc` files beside it are raw LLVM
        # bitcode and nvJitLink rejects them as both LIBRARY and LTOIR inputs.
        linker = Linker(
            program.compile("ptx"),
            ObjectCode.from_library(archive.read_bytes()),
            options=LinkerOptions(arch=arch),
        )
        self._module = linker.link("cubin")
        self._put = self._module.get_kernel("put_expert")
        self._drain = self._module.get_kernel("drain_expert")

        # `library_init`, not `module_init`: `cuda.core` loads a cubin as a `CUlibrary`,
        # and `module_init` rejects that with the same CUDA_ERROR_INVALID_HANDLE as a
        # context mismatch does. `module_finalize` is unusable with anything, because
        # `module_init` never sets the `finalize_handle` it then requires.
        import nvshmem.core as nvshmem

        self._nvshmem = nvshmem
        self._registered = nvshmem.NvshmemKernelObject.from_obj(self._module)
        nvshmem.library_init(self._registered)

        self._staging = staging
        self._ep_rank = ep_rank
        self._per_rank_experts = per_rank_experts
        self._device = Device(staging.device.index)
        self._launch_config = LaunchConfig
        logger.info(
            "Device-issued expert transfer ready on rank %d: %s, %.2f MiB staging.",
            ep_rank,
            arch,
            staging.numel() / 2**20,
        )

    def close(self) -> None:
        """Unregister the kernels. Paired with `library_init`, required before exit."""
        registered = getattr(self, "_registered", None)
        if registered is None:
            return
        self._registered = None
        self._nvshmem.library_finalize(registered)

    def transfer(
        self,
        plan: torch.Tensor,
        pointers: WeightPointers,
        replica_row: int,
        stream: torch.cuda.Stream,
        staging_offset: int = 0,
    ) -> None:
        """Move the planned expert into the target's replica row.

        Every branch is taken on the device: whether anything was planned, whether this
        rank owns the expert, and whether it is the target. A forward that placed
        nothing launches both kernels and they return immediately, which is cheaper than
        a host round trip to find out.

        Args:
            plan: `[4]` int64 device tensor from `plan_one_layer_on_device`.
            pointers: This layer's weight addresses.
            replica_row: The physical row to land in. A host constant, since it is
                `per_rank_experts + slot` for every rank and every layer.
            stream: The predictive stream.
            staging_offset: Byte offset of the staging buffer to use. Alternated by the
                caller so a later layer's put cannot land in the buffer an earlier
                layer's drain is still reading — `barrier_all` orders arrival, not this
                rank's next put against the peer's local drain.
        """
        from cuda.core import launch

        core_stream = self._core_stream(stream)
        count = int(pointers.base.numel())
        grid = count * _GRID_CHUNKS_PER_TENSOR
        config = self._launch_config(grid=grid, block=_BLOCK)
        common = (
            np.uint64(plan.data_ptr()),
            np.uint64(pointers.base.data_ptr()),
            np.uint64(pointers.row_bytes.data_ptr()),
            np.uint64(pointers.row_strides.data_ptr()),
            np.int32(count),
            # Shifted here rather than inside the kernels: the base is a host argument
            # already, and a symmetric allocation translates any offset within itself to
            # the peer correctly, so alternating buffers costs the kernels nothing.
            np.uint64(self._staging.data_ptr() + staging_offset),
            np.int32(self._ep_rank),
        )
        if _STAGE not in ("barrier-only", "drain-only"):
            launch(
                core_stream,
                config,
                self._put,
                *common,
                np.int32(self._per_rank_experts),
            )
        # Arrival. Collective and stream-ordered, so no host involvement and no per-rank
        # decision about whether data landed — and safe to be collective because the
        # plan is identical on every rank by construction.
        import nvshmem.bindings as bindings

        if _STAGE != "no-barrier":
            bindings.barrier_all_on_stream(stream.cuda_stream)
        if _STAGE not in ("barrier-only", "put-only"):
            launch(core_stream, config, self._drain, *common, np.int32(replica_row))

    def _core_stream(self, stream: torch.cuda.Stream):
        """Wrap torch's stream for `cuda.core`, rather than creating another one.

        `launch` needs a `cuda.core.Stream` and torch's is not one. Launching onto a
        stream torch does not know about would put the transfer outside the ordering
        everything else on this path depends on, so the wrapping is the requirement
        rather than a convenience.
        """

        class _Adapter:
            def __init__(self, handle: int):
                self._handle = handle

            def __cuda_stream__(self) -> tuple[int, int]:
                return (0, self._handle)

        return self._device.create_stream(_Adapter(stream.cuda_stream))
