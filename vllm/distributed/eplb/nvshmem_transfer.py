# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Move one expert's weights with a one-sided put, so the host never learns the plan.

Ticket 05. The host currently stalls once per predicted layer, and it is structural
rather than a stream-placement mistake: `ncclSend`'s peer is a `ctypes.c_int` consumed
when the host enqueues, so the host must know which expert goes to which rank, so it
must read a device tensor. Measured, that synchronisation is 5.28 ms of collective
waiting per source layer — 70% of everything prediction adds — and no amount of stream
juggling removes it.

A one-sided put removes the host from the loop: the destination address is computed from
the symmetric heap's offset arithmetic, and there is no matching receive to pair with.

**The route is a symmetric staging buffer plus a local copy**, and it is chosen from
measurement rather than preference: a 9.00 MiB put takes 33.0 us at 285 GB/s and the
local copy 6.3 us at the measured 3.00 TB/s, so 39.3 us together against a prefill
window near 95 ms. Two alternatives were probed and must not be retried blindly:

* Registering the model's own weight tensors so a put could target them directly is
  **not available on this stack**. Registration needs a dynamic VMM heap; the VMM heap
  needs a user allocation handle PyTorch does not create; the two settings exclude each
  other. Four checks deep, recorded in `bench/probe_nvshmem_register.py`.
* PyTorch symmetric memory **does** work and is the better shape if the staging copy
  ever matters, but it needs a memory-pool context around expert-weight creation inside
  the MoE and quantisation paths, where this route touches the weight allocation not at
  all.

Two API facts that cost time and are easy to hit again. `register_external_tensor` and
the symmetric allocator both need NVSHMEM's per-device memory resource to exist, and
that is created lazily on the **first** heap allocation — registering or slicing before
any allocation fails with "device that is not initialized with NVSHMEM", which reads
like a device-binding bug and is not one. And a **row-granular** put must go through
`nvshmem.bindings.putmem_on_stream`: `nvshmem.core.put` resolves its arguments through
nvshmem's tracking table, which holds whole allocations, so passing a slice raises
"Tensor not tracked by nvshmem".
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm.distributed.eplb.expert_staging import staging_layout
from vllm.logger import init_logger

logger = init_logger(__name__)


def nvshmem_unavailable_reason() -> str | None:
    """Why a one-sided put cannot be used here, or None if it can.

    Checked as a capability rather than assumed, and reported as a reason rather than a
    bool so a run that falls back says which of several unrelated things was missing.
    All of them have been hit on this project's hardware: no NVLink on the first node,
    no Python bindings on the second even with the C library present, and a CUDA include
    path without `nvrtc.h`.
    """
    if not torch.cuda.is_available():
        return "no CUDA device"
    try:
        import nvshmem.core  # noqa: F401
    except ImportError:
        return (
            "nvshmem4py is not installed. It is not on public PyPI: "
            "`uv pip install --extra-index-url https://pypi.nvidia.com "
            "nvshmem4py-cu13`. "
            "`nvidia-nvshmem-cu13` alone ships only the C library"
        )
    try:
        from cuda.core import Device  # noqa: F401
    except ImportError:
        try:
            from cuda.core.experimental import Device  # noqa: F401
        except ImportError:
            return "cuda.core is not importable, so NVSHMEM cannot be bound to a device"
    return None


class OneSidedExpertTransfer:
    """One symmetric staging buffer, shared by every layer, written by a one-sided put.

    Lifetime is the worker's: NVSHMEM is initialised once, after torch and NCCL are
    already up, which was verified to work rather than assumed — vLLM initialises those
    first and this code does not get to go earlier.

    Attributes:
        expert_bytes: Size of the staging buffer, one expert of the largest layer.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        expert_bytes: int,
        device: torch.device,
        broadcast_uid,
        buffers: int = 1,
    ):
        """Initialise NVSHMEM and allocate the staging buffer.

        Args:
            rank: This rank's id within the group the put addresses.
            world_size: That group's size.
            expert_bytes: One expert's staging bytes, from `staging_workspace_bytes`.
                Kept as one expert whatever `buffers` says, since it is also the
                default cap on bytes in flight.
            device: The device torch is using. **Must** be the device NVSHMEM binds to:
                `cuda.core.Device()` with no argument takes cuda.core's current device,
                which does not follow `torch.cuda.set_device`, and the mismatch surfaces
                much later as a registration or slicing failure.
            broadcast_uid: Callable taking the rank-0 unique id and returning the same
                object on every rank. Injected rather than importing a process group
                here, so the bootstrap can be exercised without one.
            buffers: How many expert-sized staging buffers to allocate back to back. The
                device path asks for two and alternates by layer parity, because
                `barrier_all` orders arrival and not one rank's next put against the
                peer's drain out of the same workspace.

        Raises:
            RuntimeError: If NVSHMEM is unavailable, with the reason.
        """
        reason = nvshmem_unavailable_reason()
        if reason is not None:
            raise RuntimeError(f"one-sided expert transfer is unavailable: {reason}")

        import nvshmem.core as nvshmem

        try:
            from cuda.core import Device
        except ImportError:
            from cuda.core.experimental import Device  # type: ignore

        # UID bootstrap, not MPI: vLLM is not launched under `mpirun`, and `mpi4py` has
        # to be built against the same MPI distribution it runs with.
        cuda_device = Device(device.index if device.index is not None else 0)
        cuda_device.set_current()
        uid = broadcast_uid(nvshmem.get_unique_id(empty=rank != 0))
        nvshmem.init(
            device=cuda_device,
            uid=uid,
            rank=rank,
            nranks=world_size,
            initializer_method="uid",
        )

        # The first heap allocation is what creates NVSHMEM's per-device memory
        # resource, and everything else needs it to exist. Allocating the staging buffer
        # is that allocation, so nothing here has to force it separately.
        self.expert_bytes = expert_bytes
        self._staging = nvshmem.tensor((expert_bytes * buffers,), dtype=torch.uint8)
        self._nvshmem = nvshmem
        self._rank = rank
        logger.info(
            "One-sided expert transfer ready: %d PEs, %d x %.2f MiB staging buffer.",
            nvshmem.n_pes(),
            buffers,
            expert_bytes / 2**20,
        )

    def close(self) -> None:
        """Free the staging buffer and shut NVSHMEM down.

        Required, not tidiness: leaving the symmetric heap alive into interpreter exit
        segfaults every rank *after* the work has finished and every result has been
        printed, which looks like a transfer bug and is not one.

        The buffer goes back through `free_tensor`, not merely out of scope. NVSHMEM
        keeps its own reference count beside Python's, so dropping the last Python
        reference leaves the allocation tracked; finalize then reports every buffer as
        leaked and segfaults on the way out. Dropping the reference is what looks
        obviously sufficient and is not.
        """
        staging = getattr(self, "_staging", None)
        if staging is None:
            return
        self._staging = None
        self._nvshmem.free_tensor(staging)
        self._nvshmem.finalize()

    def put_expert(
        self,
        tensors: Sequence[torch.Tensor],
        row: int,
        dst_rank: int,
        stream: torch.cuda.Stream,
    ) -> None:
        """Write one expert's weights into `dst_rank`'s staging buffer.

        Stream-ordered on the issuer, which is **not** the same as visible to the peer:
        the put returns to this rank's stream well before the bytes land, and the peer
        holds no event that observes them. `barrier` is what establishes arrival. The
        earlier probe that concluded a plain CUDA event was sufficient had a
        `dist.barrier()` inside the region it was checking, so it measured the barrier
        rather than the event.

        Args:
            tensors: The layer's expert weight tensors.
            row: The physical row holding the expert to send.
            dst_rank: The receiving rank.
            stream: The predictive stream to enqueue on.
        """
        import nvshmem.bindings as bindings

        base = self._staging.data_ptr()
        for (offset, nbytes), tensor in zip(staging_layout(tensors), tensors):
            source = tensor[row]
            # Pointer form, because the destination is a slice of the peer's buffer and
            # `nvshmem.core.put` only resolves whole tracked allocations.
            bindings.putmem_on_stream(
                base + offset,
                source.data_ptr(),
                nbytes,
                dst_rank,
                stream.cuda_stream,
            )

    def barrier(self, stream: torch.cuda.Stream | None = None) -> None:
        """Make every put issued before this point visible to its target rank.

        Collective and stream-ordered: no host synchronisation, and no per-rank decision
        about whether data arrived. A point-to-point signal would be cheaper, but the
        plan is identical on every rank by construction, so the collective costs nothing
        in generality and stays clear of the divergence class that deadlocked this
        branch twice. It also replaces a collective that was already there — the pynccl
        `execute()` this path removes was called on every rank per layer for the same
        reason.
        """
        import nvshmem.bindings as bindings

        target = stream if stream is not None else torch.cuda.current_stream()
        bindings.barrier_all_on_stream(target.cuda_stream)

    def copy_into_row(
        self,
        tensors: Sequence[torch.Tensor],
        row: int,
        stream: torch.cuda.Stream,
    ) -> None:
        """Copy this rank's staging buffer into `row` of its own weight tensors.

        The caller is responsible for the two orderings that make the shared buffer
        safe:
        this must follow the `barrier` that made the put visible, and it must follow the
        event recorded when a MoE kernel last read the row being overwritten.
        `ReplicaTransferEngine` owns both.
        """
        with torch.cuda.stream(stream):
            for (offset, nbytes), tensor in zip(staging_layout(tensors), tensors):
                flat = tensor[row].reshape(-1).view(torch.uint8)
                flat.copy_(self._staging[offset : offset + nbytes])
