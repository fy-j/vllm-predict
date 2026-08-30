# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-layer predicted-load estimation for Predictive expert replication.

The current sparse MoE evaluates the *next* sparse MoE's gate and router on its own
source-local hidden states, producing that layer's predicted logical-expert load before
token dispatch. An AllGather over the EPLB group then gives every EP
rank the same `[source rank, logical expert]` Global predicted-load snapshot, so
a deterministic planner can run locally on every rank without a plan broadcast.

This path is read-only with respect to placement: it never applies the EPLB
logical-to-physical mapping and never records actual expert load.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from vllm.distributed.parallel_state import get_eplb_group
from vllm.triton_utils import tl, triton
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if TYPE_CHECKING:
    from vllm.distributed.eplb.eplb_state import EplbLayerState
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
        FusedMoERouter,
    )
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


def build_predictive_physical_map(
    num_layers: int,
    num_logical_experts: int,
    ep_size: int,
    replica_slots_per_rank: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Build the fixed predictive physical-to-logical map.

    Rank `r` owns logical experts `[r * canonical, (r + 1) * canonical)` in its
    leading physical rows, where `canonical = num_logical_experts // ep_size`.
    Its trailing `replica_slots_per_rank` rows are inactive, marked `-1`, which both
    keeps canonical ownership fixed and gives every rank room to receive one predictive
    replica.

    Args:
        num_layers: Number of sparse MoE layers.
        num_logical_experts: Logical experts in the model.
        ep_size: EP group size.
        replica_slots_per_rank: Inactive replica rows reserved per rank.

    Returns:
        A `[num_layers, ep_size * (canonical + replica_slots_per_rank)]` map.

    Raises:
        ValueError: If the logical experts do not divide evenly across ranks.
    """
    if num_logical_experts % ep_size != 0:
        raise ValueError(
            f"Predictive expert replication needs {num_logical_experts} logical "
            f"experts to divide EP size {ep_size}."
        )
    canonical_per_rank = num_logical_experts // ep_size
    local_rows = canonical_per_rank + replica_slots_per_rank
    layout = torch.full(
        (num_layers, ep_size * local_rows), -1, dtype=dtype, device=device
    )
    canonical = torch.arange(num_logical_experts, dtype=dtype, device=device).view(
        ep_size, canonical_per_rank
    )
    layout.view(num_layers, ep_size, local_rows)[:, :, :canonical_per_rank] = canonical
    return layout


def build_source_local_physical_map(
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    source_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the one physical copy this source rank uses for each logical expert.

    This is the Source-local physical map. It is returned in the shape the shared
    routing path already accepts, `[num_logical_experts, 1]` alongside an all-ones
    replica count, so that path's per-token replica choice degenerates to a plain
    lookup. Source-rank routing is then true *by construction*: with one copy on offer
    there is nothing left for a per-token decision to split, so a rank's chunk cannot be
    divided across copies. Nothing in the shared routing kernel changes, which also
    leaves the other model families that call it untouched.

    Ranks are spread across the available copies by `source_rank % replicas`, which is
    deterministic and identical on every rank, so each rank derives its own row without
    a broadcast. A plan may later override the choice per expert; this is the
    placement-free default.

    Args:
        logical_to_physical_map: `[num_logical_experts, max_replicas]` for one
            layer, holding each logical expert's physical rows and `-1` padding.
        logical_replica_count: `[num_logical_experts]` copies per logical expert.
        source_rank: The EP rank whose map is being built.

    Returns:
        The `[num_logical_experts, 1]` physical row to route to, and the all-ones
        replica count that pins it.
    """
    counts = logical_replica_count.clamp(min=1).to(torch.int64)
    replica_index = torch.remainder(source_rank, counts).unsqueeze(1)
    chosen = logical_to_physical_map.gather(1, replica_index)
    return chosen, torch.ones_like(logical_replica_count)


def count_logical_experts_reference(
    logical_ids: torch.Tensor,
    num_unpadded: torch.Tensor,
    num_logical_experts: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Count predicted tokens per logical expert, the elementwise way.

    This is the implementation that shipped and was measured, kept as the **oracle** the
    fused kernel is tested against rather than as a fallback. It is not on the serving
    path.

    An id outside the logical range contributes nothing rather than being folded into a
    neighbouring expert's count, which would attribute load to a real expert and make
    the planner replicate one that is not hot. The weight excludes it and the clamp only
    keeps `scatter_add_` in bounds. Deliberately not validated on the host, since that
    would synchronise every layer of every forward.

    Args:
        logical_ids: `[num_tokens, topk]` selected logical experts, either int width.
        num_unpadded: Device scalar holding this rank's valid token count, either 0-dim
            or single-element. Rows at or past it are padding and must not be counted.
        num_logical_experts: Logical expert count, which bounds the output.
        out: `[num_logical_experts]` int32 destination, zeroed here.

    Returns:
        `out`, for convenience.
    """
    out.zero_()
    num_tokens = logical_ids.shape[0]
    if num_tokens == 0:
        return out

    # `num_unpadded` is used unindexed: the runtime passes a 0-dim scalar out of a list
    # of per-ubatch counts, and indexing that raises. Both a 0-dim and a one-element
    # tensor broadcast correctly against `arange`, and the kernel reads element 0 of
    # either.
    is_valid = torch.arange(num_tokens, device=logical_ids.device) < num_unpadded
    flat_ids = logical_ids.reshape(num_tokens, -1)
    weights = is_valid.to(out.dtype).unsqueeze(1).expand_as(flat_ids)
    # Copy before masking: `.to()` is a no-op when the router already returns int64, and
    # an in-place clamp would then write through to the tensor the router just handed
    # back, breaking its read-only contract for any other caller.
    indices = flat_ids.reshape(-1).to(dtype=torch.int64, copy=True)
    in_range = (indices >= 0) & (indices < num_logical_experts)
    masked_weights = weights.reshape(-1) * in_range.to(out.dtype)
    indices.clamp_(0, num_logical_experts - 1)
    out.scatter_add_(0, indices, masked_weights)
    return out


@triton.jit
def _count_logical_experts_kernel(
    ids_ptr,
    unpadded_ptr,
    out_ptr,
    num_pairs,
    topk,
    num_logical_experts,
    BLOCK: tl.constexpr,
):
    """One atomic increment per valid, in-range token-expert pair.

    The elementwise path materialised a mask, a weight vector, a range mask, a product
    and
    a clamped index copy — five tensors sized `[num_tokens, topk]` — and issued about a
    dozen launches per source layer. Here the same arithmetic is per-element and stays
    in registers, so the layer costs one launch.

    The count is small, 128 to 256 integers, and the input large, tokens times topk, so
    atomics into the output are the right shape: contention is bounded by the expert
    count rather than the token count, and the skew this feature exists to find means
    the hot expert takes that contention either way.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offsets < num_pairs

    unpadded = tl.load(unpadded_ptr)
    # Integer division recovers the token a flattened pair belongs to, which is what the
    # padding boundary is expressed in.
    token = offsets // topk
    valid = in_bounds & (token < unpadded)

    ids = tl.load(ids_ptr + offsets, mask=in_bounds, other=-1)
    valid = valid & (ids >= 0) & (ids < num_logical_experts)

    tl.atomic_add(out_ptr + ids, 1, mask=valid)


def count_logical_experts_triton(
    logical_ids: torch.Tensor,
    num_unpadded: torch.Tensor,
    num_logical_experts: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Count predicted tokens per logical expert in one kernel.

    Replaces `count_logical_experts_reference` on the serving path. Prediction's own GPU
    work is small, about 2.2 ms per prefill window, but it added roughly 700 launches,
    and the token collectives then grew 39.9 ms with the same kernel count and the same
    byte volume: the launches desynchronise the DP ranks and the collectives absorb the
    skew.

    Measured, this kernel takes launches per source layer from 17.2 to 7.1 and recovers
    17% of that added waiting — which separated the per-layer cost into 2.21 ms that
    scales with launches and 5.28 ms that does not. The fixed part is the host
    synchronisation, so it is ticket 06 rather than this kernel that carries the rest.
    Ticket 03.

    Args and semantics are identical to the reference, which the tests assert by
    equality.
    """
    out.zero_()
    num_tokens = logical_ids.shape[0]
    if num_tokens == 0:
        return out

    flat = logical_ids.reshape(-1)
    topk = flat.numel() // num_tokens
    # Contiguous so the flattened index arithmetic matches the kernel's, whatever view
    # the router returned.
    if not flat.is_contiguous():
        flat = flat.contiguous()

    BLOCK = 1024
    grid = (triton.cdiv(flat.numel(), BLOCK),)
    _count_logical_experts_kernel[grid](
        flat,
        num_unpadded,
        out,
        flat.numel(),
        topk,
        num_logical_experts,
        BLOCK=BLOCK,
    )
    return out


class CrossLayerLoadPredictor:
    """Predicts one target MoE's logical-expert load from the current MoE.

    One instance is bound per non-final sparse MoE. It holds the target gate *module*
    rather than a snapshot of its weights, so ordinary weight loading stays
    authoritative.
    """

    def __init__(
        self,
        target_gate: torch.nn.Module,
        target_router: "FusedMoERouter",
        num_logical_experts: int,
        eplb_layer_state: "EplbLayerState",
    ):
        self.target_gate = target_gate
        self.target_router = target_router
        self.num_logical_experts = num_logical_experts
        self.eplb_layer_state = eplb_layer_state

        self._local_counts: torch.Tensor | None = None
        self._snapshot_flat: torch.Tensor | None = None
        self._work: torch.distributed.Work | None = None

    def predict_local_counts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Count predicted target-layer tokens per logical expert on this rank.

        Padding rows are excluded, so a dummy or padding-only forward contributes
        all-zero counts. Nothing here synchronizes with the host.

        Args:
            hidden_states: Source-local current-MoE hidden states, before token
                dispatch, shaped `[num_tokens, hidden_size]`.

        Returns:
            An int32 `[num_logical_experts]` count tensor, valid until this
            predictor's next `predict_local_counts` call.
        """
        target_logits, _ = self.target_gate(hidden_states)
        logical_ids = self.target_router.select_logical_experts(
            hidden_states, target_logits
        )

        counts = self._counts_buffer(hidden_states.device)
        num_tokens = hidden_states.shape[0]
        if num_tokens == 0:
            # Both counting implementations zero their output, so the only path that has
            # to do it here is the one that returns before calling either.
            counts.zero_()
            return counts

        # Real tokens occupy the leading rows; everything past the unpadded count is
        # padding and must not reach the predicted load.
        num_unpadded = self.eplb_layer_state.num_unpadded_tokens_tensors
        if num_unpadded is None:
            raise RuntimeError(
                "Predictive expert replication requires EPLB per-forward state; "
                "EplbState.prepare_forward must run before the model forward."
            )
        # `num_unpadded_tokens_tensors` is a **list** of per-ubatch scalars, so this
        # indexes rather than slices — a list slice would hand the kernel a Python list
        # and Triton would refuse to specialize it. The ubatch id matters even though
        # DBO is rejected by configuration validation: reading ubatch 0's count
        # unconditionally is the kind of wrong that shows up only once someone enables
        # the thing. Triton where there is a GPU, the reference otherwise. This branches
        # on a *device capability*, which is a static property identical on every rank,
        # not on per-rank state — so it is not the class of decision that has deadlocked
        # this branch. The CPU path exists because the deterministic behaviour of this
        # counting is asserted by CPU-only unit tests, which is where padding boundaries
        # and out-of-range ids are cheapest to pin.
        count = (
            count_logical_experts_triton
            if logical_ids.is_cuda
            else count_logical_experts_reference
        )
        return count(
            logical_ids,
            num_unpadded[dbo_current_ubatch_id()],
            self.num_logical_experts,
            counts,
        )

    def start_snapshot(self, local_counts: torch.Tensor) -> None:
        """Begin the predicted-count AllGather on the EPLB group.

        Call this only after the current layer's token dispatch, so the small collective
        overlaps this layer's local expert GEMM.
        """
        group = get_eplb_group().device_group
        # The gather buffer stays flat: ProcessGroupGloo rejects a pre-shaped `[ep_size,
        # num_logical_experts]` output that NCCL would accept, and the tests exercise
        # the gloo path.
        self._work = torch.distributed.all_gather_into_tensor(
            self._snapshot_buffer(local_counts, group.size()),
            local_counts,
            group=group,
            async_op=True,
        )

    def finish_snapshot(self) -> torch.Tensor | None:
        """Wait for the AllGather and return the Global predicted-load snapshot.

        Returns:
            A `[ep_size, num_logical_experts]` count matrix, identical on every
            EP rank, or None when no AllGather is in flight.
        """
        if self._work is None:
            return None
        self._work.wait()
        self._work = None
        assert self._snapshot_flat is not None
        return self._snapshot_flat.view(-1, self.num_logical_experts)

    def _counts_buffer(self, device: torch.device) -> torch.Tensor:
        if self._local_counts is None or self._local_counts.device != device:
            self._local_counts = torch.zeros(
                self.num_logical_experts, dtype=torch.int32, device=device
            )
        return self._local_counts

    def _snapshot_buffer(self, counts: torch.Tensor, ep_size: int) -> torch.Tensor:
        if self._snapshot_flat is None or self._snapshot_flat.device != counts.device:
            self._snapshot_flat = torch.zeros(
                ep_size * self.num_logical_experts,
                dtype=counts.dtype,
                device=counts.device,
            )
        return self._snapshot_flat


_PREDICTION_PAIRS: list[tuple[int, int, "MoERunner"]] = []
_BOUND_LAYER_COUNT: list[int] = []


def bound_layer_count() -> int | None:
    """Sparse MoE layers in the model the registry was last bound for.

    Lets a reader tell whether a given model is the one the pairs describe, since the
    registry holds runners rather than a model identity and the pairs of one model
    scored against another's load would look like poor accuracy.
    """
    return _BOUND_LAYER_COUNT[0] if _BOUND_LAYER_COUNT else None


def registered_prediction_pairs() -> list[tuple[int, int, "MoERunner"]]:
    """Source layer, target layer, and source runner for every bound prediction.

    A diagnostic registry filled by :func:`bind_moe_prediction_targets` so the
    prediction-accuracy study can pair a source layer's prediction with its target
    layer's recorded load. Nothing on the serving path reads it.

    Only one model's bindings are held, which is all the PoC supports: it rejects
    a model whose decoder layers do not all carry a sparse MoE, and pipeline parallelism
    is out of scope.

    Returns:
        `(source layer index, target layer index, source runner)` triples, in
        source-layer order.
    """
    return list(_PREDICTION_PAIRS)


def bind_moe_prediction_targets(
    moe_runners_in_layer_order: "Sequence[MoERunner | None]",
    lookahead: int,
    skip_first_layers: int = 0,
) -> list[int]:
    """Bind each source sparse MoE to the MoE `lookahead` layers ahead.

    This is the cross-layer gate registry. Source layers are the index range
    `[skip_first_layers, num_layers - lookahead)`. Layers outside it bind no
    target, create no plan, and run no predicted-count collective: the leading
    layers because their cross-layer prediction is unreliable, the trailing ones because
    they have no target.

    The PoC only supports a topology where every decoder layer holds a sparse MoE. A
    model that interleaves dense layers is rejected rather than silently predicting
    across a different distance than `lookahead` names.

    Args:
        moe_runners_in_layer_order: One entry per decoder layer in order, holding
            that layer's sparse MoE runner, or None if the layer has none. Layers absent
            from this pipeline-parallel rank must be omitted.
        lookahead: Distance in layers from a source to its target, so `lookahead`
            of `n` binds layer `i` to layer `i + n` with `n - 1` layers between.
        skip_first_layers: Leading layers excluded from prediction.

    Returns:
        The bound source layer indices, in order.

    Raises:
        ValueError: If any decoder layer lacks a sparse MoE, or if the lookahead
            and skip leave no valid source layer.
    """
    # Cleared before validation, not after binding: a second model that fails validation
    # would otherwise leave the first model's runners registered, and the accuracy dump
    # would score its predictions against the wrong model's load.
    _PREDICTION_PAIRS.clear()
    _BOUND_LAYER_COUNT.clear()

    runners: list[MoERunner] = []
    missing: list[int] = []
    for index, runner in enumerate(moe_runners_in_layer_order):
        if runner is None:
            missing.append(index)
        else:
            runners.append(runner)
    if missing:
        raise ValueError(
            "Predictive expert replication requires a sparse MoE in every "
            f"decoder layer, but layers {missing} have none."
        )
    source_indices = list(range(skip_first_layers, len(runners) - lookahead))
    if not source_indices:
        raise ValueError(
            f"Predictive expert replication has no source layer: "
            f"{len(runners)} sparse MoE layers with lookahead {lookahead} and "
            f"skip_first_layers {skip_first_layers} leaves nothing to predict "
            f"from."
        )
    for index, runner in enumerate(runners):
        # Every runner needs its own index, targets included: the target is where a
        # pending activation is applied, and it is not a source.
        runner.moe_layer_index = index
    for index in source_indices:
        runners[index].bind_prediction_target(runners[index + lookahead])
    _PREDICTION_PAIRS.extend(
        (index, index + lookahead, runners[index]) for index in source_indices
    )
    _BOUND_LAYER_COUNT.append(len(runners))
    return source_indices
