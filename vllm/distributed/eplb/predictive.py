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

from vllm import envs
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


class PredictionWindow:
    """One snapshot collective shared by a window of consecutive source layers.

    Ticket 13. Each source writes the predicted load of **its own** target into its own
    row, and the window's last source issues a single AllGather for all of them. That is
    what cuts the 44 per-layer collectives ticket 11 priced at 0.22 ms of barrier
    coupling
    each, without the defect the first design had: predictions come from `size`
    different
    source layers, so they are distinguishable.

    Why not one source predicting several targets, which is cheaper still: measured on
    the
    real model, four layers' gates applied to the *same* hidden states select almost the
    same experts — L1 of 16 to 36 out of 7896 assignments between the four predictions,
    against 5412 to 7660 between the four targets' actual loads. Three of every four
    targets were then planned from a distribution that was not theirs and placement's
    benefit fell from 26.9% of critical-path excess to 3.7%.

    Attributes:
        size: Source layers sharing this collective. The final window of a model may be
            shorter than the configured group, which costs nothing: it simply issues its
            collective one source early.
    """

    def __init__(self, size: int, num_logical_experts: int | None = None):
        if size < 1:
            raise ValueError(
                f"a prediction window needs at least one source, got {size}."
            )
        self.size = size
        # Learned from the first source that writes a row, so the binder does not have
        # to
        # reach into a target layer's config to build a window.
        self.num_logical_experts = num_logical_experts
        self.counts: torch.Tensor | None = None
        self._snapshot_flat: torch.Tensor | None = None
        self._work: torch.distributed.Work | None = None
        self._ep_size: int | None = None

    def issues_at(self, position: int) -> bool:
        """Whether the source at `position` is the one that starts the collective."""
        return position == self.size - 1

    def row(
        self,
        position: int,
        device: torch.device,
        num_logical_experts: int | None = None,
    ) -> torch.Tensor:
        """This source's `[num_logical_experts]` destination inside the window buffer.

        Returned as a view so the counting kernel writes straight into the tensor the
        collective sends, with no per-layer copy.
        """
        if not 0 <= position < self.size:
            raise ValueError(
                f"position {position} is outside a window of {self.size} sources."
            )
        if num_logical_experts is not None:
            if (
                self.num_logical_experts is not None
                and self.num_logical_experts != num_logical_experts
            ):
                # One buffer covers the window, so a window spanning two expert
                # geometries would index one source's row with another's width.
                raise ValueError(
                    f"a prediction window covers one expert geometry: "
                    f"{self.num_logical_experts} against {num_logical_experts}."
                )
            self.num_logical_experts = num_logical_experts
        if self.num_logical_experts is None:
            raise ValueError("the window does not know its logical expert count yet.")
        if self.counts is None or self.counts.device != device:
            self.counts = torch.zeros(
                (self.size, self.num_logical_experts), dtype=torch.int32, device=device
            )
        return self.counts[position]

    def start_snapshot(self) -> None:
        """Begin the window's one AllGather over the EPLB group."""
        assert self.counts is not None, "no source wrote a row before the collective"
        if envs.VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER:
            # Cost probe; see `CrossLayerLoadPredictor.start_snapshot` for why this may
            # not be combined with placement.
            self._work = None
            return
        group = get_eplb_group().device_group
        self._ep_size = group.size()
        # Both sides flat: ProcessGroupGloo rejects a pre-shaped output that NCCL
        # accepts,
        # and the group axis is recovered by `view` in `finish_snapshot`.
        self._work = torch.distributed.all_gather_into_tensor(
            self._snapshot_buffer(group.size()),
            self.counts.reshape(-1),
            group=group,
            async_op=True,
        )

    def finish_snapshot(self) -> torch.Tensor | None:
        """Wait for the collective and return `[ep_size, size, num_logical_experts]`.

        A source at window position `p` reads `snapshot[:, p, :]`, which is the
        `[ep_size, num_logical_experts]` shape the planner has always consumed.
        """
        if envs.VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER:
            if self.counts is None:
                return None
            ep_size = self._probe_ep_size()
            return self.counts.unsqueeze(0).expand(ep_size, -1, -1)
        if self._work is None:
            return None
        self._work.wait()
        self._work = None
        assert self._snapshot_flat is not None
        return self._snapshot_flat.view(-1, self.size, self.num_logical_experts)

    def _probe_ep_size(self) -> int:
        if self._ep_size is None:
            self._ep_size = get_eplb_group().device_group.size()
        return self._ep_size

    def _snapshot_buffer(self, ep_size: int) -> torch.Tensor:
        assert self.counts is not None
        if (
            self._snapshot_flat is None
            or self._snapshot_flat.device != self.counts.device
        ):
            # Known by construction here: the buffer is only sized once a source has
            # written a row, and `row` is what learns the width. Asserted rather than
            # cast, because multiplying by None would size the gather buffer from a
            # TypeError's ashes rather than from a wrong number — and mypy found this
            # where the tests could not, since they always supply the width.
            assert self.num_logical_experts is not None, (
                "the window's logical expert count is learned from its first written "
                "row; sizing the gather buffer before that is a wiring error."
            )
            self._snapshot_flat = torch.zeros(
                ep_size * self.size * self.num_logical_experts,
                dtype=torch.int32,
                device=self.counts.device,
            )
        return self._snapshot_flat


class CrossLayerLoadPredictor:
    """Predicts a group of target MoEs' logical-expert load from the current MoE.

    One instance is bound per source sparse MoE. It holds the target gate *modules*
    rather than a snapshot of their weights, so ordinary weight loading stays
    authoritative.

    **A group of one is the path that shipped.** Ticket 12 added the group axis so that
    ticket 11's measured barrier cost — 0.22 ms for each of 44 per-layer collectives —
    could be cut by covering several targets with one collective, and it lands at a
    group
    of 1 where every value is what the single-target predictor produced.

    **Read ticket 13 before raising the group above 1.** Measured on this model, four
    different layers' gates applied to the *same* hidden states select almost the same
    experts: within a group the four predicted distributions differ by an L1 of 16 to 36
    out of 7896 assignments, while the four target layers' actual loads differ by 5412
    to 7660. So three of every four targets are planned from a distribution that is not
    theirs, and placement's benefit collapses from 26.9% of critical-path excess to
    3.7%.
    """

    def __init__(
        self,
        target_gate: torch.nn.Module,
        target_router: "FusedMoERouter",
        num_logical_experts: int,
        eplb_layer_state: "EplbLayerState",
        window: "PredictionWindow | None" = None,
        window_position: int = 0,
    ):
        self.target_gate = target_gate
        self.target_router = target_router
        self.num_logical_experts = num_logical_experts
        self.eplb_layer_state = eplb_layer_state
        # A window of one is its own collective, which is the shipping configuration.
        self.window = window if window is not None else PredictionWindow(1)
        self.window_position = window_position

        self._local_counts: torch.Tensor | None = None
        self._snapshot_flat: torch.Tensor | None = None
        self._work: torch.distributed.Work | None = None
        # Only the cost probe uses this; see `start_snapshot`.
        self._probe_counts: torch.Tensor | None = None
        self._ep_size_for_probe: int | None = None

    @property
    def issues_snapshot(self) -> bool:
        """Whether this source is the one that starts its window's collective."""
        return self.window.issues_at(self.window_position)

    def predict_local_counts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Count predicted target-layer tokens per logical expert on this rank.

        Padding rows are excluded, so a dummy or padding-only forward contributes
        all-zero counts. Nothing here synchronizes with the host.

        Args:
            hidden_states: Source-local current-MoE hidden states, before token
                dispatch, shaped `[num_tokens, hidden_size]`.

        Returns:
            An int32 `[group_size, num_logical_experts]` count tensor, valid until this
            predictor's next `predict_local_counts` call.
        """
        target_logits, _ = self.target_gate(hidden_states)
        logical_ids = self.target_router.select_logical_experts(
            hidden_states, target_logits
        )

        # Straight into the window's row, so the collective sends what the kernel wrote
        # with no per-layer copy.
        counts = self.window.row(
            self.window_position, hidden_states.device, self.num_logical_experts
        )
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

    def start_snapshot(self, local_counts: torch.Tensor | None = None) -> None:
        """Begin this source's window's snapshot collective, if this source issues it.

        Only the window's last source starts it, which is what makes one collective
        serve
        the whole window. Call this after the current layer's token dispatch so the
        small
        collective overlaps this layer's local expert GEMM.

        Args:
            local_counts: Ignored, and accepted so callers written against the
                single-target predictor keep working. The counts are already in the
                window's row: `predict_local_counts` writes them there directly.
        """
        if self.issues_snapshot:
            self.window.start_snapshot()

    def finish_snapshot(self) -> torch.Tensor | None:
        """Wait for the window's collective and return its snapshot.

        Returns:
            An `[ep_size, window size, num_logical_experts]` count matrix, identical on
            every EP rank, or None on a source that does not issue the collective. A
            source at window position `p` reads `snapshot[:, p, :]`, which is the
            `[ep_size, num_logical_experts]` shape the planner has always consumed.
        """
        if not self.issues_snapshot:
            return None
        return self.window.finish_snapshot()


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
    group: int = 1,
) -> list[int]:
    """Bind each source sparse MoE to the `group` MoEs `lookahead` layers ahead.

    At `group = 1` this is the binding that shipped: source `i` binds target
    `i + lookahead`, sources stepping by one. **Read ticket 13 before raising it**: a
    group larger than one plans three of every four targets from another target's
    predicted load, because four layers' gates on one layer's hidden states select
    almost
    the same experts.

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
        lookahead: Distance from a source to the **first** target of its group, so
        within
            a group the distances are `lookahead .. lookahead + group - 1`.
        skip_first_layers: Leading layers excluded from prediction.
        group: Target layers per source, and therefore per snapshot collective.

    Returns:
        The bound source layer indices, in order.

    Raises:
        ValueError: If any decoder layer lacks a sparse MoE, if the lookahead and skip
            leave no valid source layer, or if the reachable targets do not divide into
            whole groups — dropping the remainder would leave trailing layers
            permanently
            unplaced, which is the coverage defect this project has already paid for and
            which is invisible in any aggregate.
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
    if group < 1:
        raise ValueError(
            f"Predictive expert replication needs a window of at least one source, "
            f"got {group}."
        )
    if group > lookahead:
        # The window's collective is issued at its **last** source, so the first target
        # must still be ahead of that source. With lookahead 1 and a window of 4, target
        # `L+1` has already run by the time source `L+3` gathers, and its plan would
        # name a layer in the past.
        raise ValueError(
            f"Predictive expert replication needs prediction_lookahead_layers "
            f"({lookahead}) to be at least the window size ({group}): the window's "
            f"snapshot is gathered at its last source, so a shorter distance would "
            f"plan a target layer that has already run."
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
    # Consecutive sources share a window and therefore one collective. The final window
    # may be short, which costs nothing: it gathers one source early. The first design
    # rejected a group that did not divide the span because the remainder was *targets*
    # that never got a prediction; here the remainder is sources, and every target is
    # still covered by its own.
    for position, index in enumerate(source_indices):
        if position % group == 0:
            remaining = len(source_indices) - position
            window = PredictionWindow(min(group, remaining))
        runners[index].bind_prediction_target(
            runners[index + lookahead], window, position % group
        )
    _PREDICTION_PAIRS.extend(
        (index, index + lookahead, runners[index]) for index in source_indices
    )
    _BOUND_LAYER_COUNT.append(len(runners))
    return source_indices
