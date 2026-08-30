# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Expert parallelism load balancer (EPLB) metrics and states.

# Glossary

- **Logical Expert**: An expert that is part of the model's logical structure.
  It holds a set of weights and is replicated across multiple physical
  experts.
- **Redundant Expert**: To achieve load balancing, for some popular logical
  experts, we create additional copies of the expert weights. During inference,
  each of these copies can be routed to by the same set of tokens.
- **Physical Expert**: An expert that is instantiated on a specific device.
  It is a replica of a logical expert and can be rearranged across devices.
  I.e., one logical expert may have multiple sets of weights initialized on
  different devices, and each of these sets is a physical expert.
- **Local Physical Expert**: A physical expert that is instantiated on the
  current device.

For example: DeepSeek-R1 has 256 logical experts, so each MoE layer
has 256 sets of linear layer weights in the model parameters. If we add 32
redundant experts, DeepSeek-R1 will have 256 + 32 = 288 physical experts in
total. And when deploying, we'll have 288 sets of linear layer weights for each
MoE layer. If we have 32 EP ranks, then each GPU will hold 288 / 32 = 9 local
physical experts.
"""

import json
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.distributed import ProcessGroup, all_reduce

import vllm.envs as envs
from vllm.config import ModelConfig, ParallelConfig
from vllm.config.utils import compute_hash_cached
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_ep_group,
    get_eplb_group,
    get_node_count,
    in_the_same_node_as,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.platforms import current_platform
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from .async_worker import start_async_worker
from .eplb_communicator import EplbCommunicator, create_eplb_communicator
from .eplb_utils import CpuGpuEvent
from .policy import EPLB_POLICIES, AbstractEplbPolicy, DefaultEplbPolicy
from .predictive import (
    bound_layer_count,
    build_predictive_physical_map,
    build_source_local_physical_map,
    registered_prediction_pairs,
)
from .predictive_coordinator import PlacementCoordinator
from .predictive_planner import Placement, apply_replica_maps

if TYPE_CHECKING:
    from .nvshmem_transfer import OneSidedExpertTransfer

from .rebalance_execute import (
    AsyncEplbLayerResult,
    move_from_buffer,
    rearrange_expert_weights_inplace,
)

logger = init_logger(__name__)

# The MoE kernel pads each expert's token list to a multiple of `BLOCK_SIZE_M`, so a
# replica taking fewer tokens saves no block and no time. Used only as the fallback when
# the kernel's own configuration cannot be resolved; `resolve_moe_block_size_m` is the
# source of truth, because this number gates placement suppression *and* floors the
# planner's minimum move, so a wrong value either reopens a regime that is settled
# negative or rejects placements that would have paid.
_MOE_BLOCK_SIZE_M_FALLBACK = 128


def resolve_moe_block_size_m(
    model: MixtureOfExperts,
    top_k: int,
    num_batched_tokens: int,
    dtype: str | None = None,
) -> int:
    """The `BLOCK_SIZE_M` the fused MoE kernel will use, at a stated batch size.

    Asked of the kernel rather than assumed. The value is not a constant: it is selected
    per `M` from the tuned configuration for this expert geometry, and the tuned files
    disagree
    across devices — the H200 entry for one shape uses 128 only from `M >= 1024`.
    Hardcoding it meant that on a device with no tuned configuration for `E=128,N=768`
    the bar was a guess in both directions.

    `M` matters, so the caller states it. The decision this feeds is "does a prefill
    step put more than one block of tokens on each expert", so the batch size to ask
    about is the largest a prefill step can present, not a decode-sized one.

    Args:
        model: The registered mixture-of-experts model. The expert geometry comes
            from its first MoE layer's config, because the `expert_weights` EPLB
            registers are flattened per-row views and the lookup needs `(E, K, N)`.
        top_k: Experts per token. Not on the `MixtureOfExperts` interface, so the caller
            takes it from a layer's MoE config.
        num_batched_tokens: The `M` to resolve at, post-allgather.
        dtype: The fused-MoE dtype token, or None for an unquantised model.

    Returns:
        The selected `BLOCK_SIZE_M`, or the fallback if the kernel cannot be consulted —
        which is logged, because silently guessing is what this function replaces.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        try_get_optimal_moe_config,
    )

    shapes: tuple[tuple[int, ...], ...] = ()
    try:
        # From the layer's MoE config, not from `model.expert_weights`. EPLB registers
        # those as flattened `[rows, numel]` views — measured `(65, 3145728)` and
        # `(65, 1572864)` on this model — and `try_get_optimal_moe_config` unpacks
        # `w2_shape` as `(E, K, N)`, so passing them raises "not enough values to
        # unpack" and every server silently took the fallback. The config is what the
        # kernel's own `w1` and `w2` were built from, so these are the shapes it will
        # look its own tuning up with.
        moe_config = model.moe_layers[0].moe_config
        experts = moe_config.num_local_experts
        inter = moe_config.intermediate_size_per_partition
        hidden = moe_config.hidden_dim
        shapes = ((experts, 2 * inter, hidden), (experts, hidden, inter))
        config = try_get_optimal_moe_config(
            w1_shape=shapes[0],
            w2_shape=shapes[1],
            top_k=top_k,
            dtype=dtype,
            M=num_batched_tokens,
        )
        block = int(config["BLOCK_SIZE_M"])
    except Exception:
        # What it tried is in the message: without it the warning cannot be acted on,
        # and this failure has already been shipped once as a silent fallback.
        logger.warning(
            "Predictive expert replication could not resolve the MoE BLOCK_SIZE_M from "
            "the kernel configuration; falling back to %d. It asked about w1%s and "
            "w2%s at M=%d. The suppression bar and the planner's minimum move both "
            "use this number, so verify it against this device's tuned configuration "
            "before trusting a result.",
            _MOE_BLOCK_SIZE_M_FALLBACK,
            shapes[0] if shapes else "(unknown)",
            shapes[1] if shapes else "(unknown)",
            num_batched_tokens,
            exc_info=True,
        )
        return _MOE_BLOCK_SIZE_M_FALLBACK
    logger.info(
        "Predictive expert replication resolved MoE BLOCK_SIZE_M=%d at M=%d.",
        block,
        num_batched_tokens,
    )
    return block


def _compute_eplb_load_stats(
    num_tokens_per_rank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    avg_tokens = num_tokens_per_rank.mean(dim=1).sum()
    max_tokens = num_tokens_per_rank.max(dim=1).values.sum()
    return avg_tokens, max_tokens


@dataclass
class EplbStats:
    """
    Model stats used in EPLB rebalancing algorithm.
    """

    global_expert_load_window: torch.Tensor
    """
    Experts load window.
    Shape: (window_size, num_moe_layers, num_physical_experts)
    """
    num_replicas: int
    """
    Number of physical experts.
    """
    num_groups: int
    """
    Number of expert groups.
    """
    num_nodes: int
    """
    Number of nodes.
    """
    num_gpus: int
    """
    Number of GPUs.
    """


@dataclass
class EplbModelState:
    """EPLB metrics."""

    physical_to_logical_map: torch.Tensor
    """
    Mapping from physical experts to logical experts.

    Shape: (num_moe_layers, num_physical_experts)

    # Example

    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the mapping could look like this:

    ```
    [[0, 1, 2, 3, 0, 1],
     [0, 2, 0, 1, 0, 3]]
    ```
    """
    logical_to_physical_map: torch.Tensor
    """
    Mapping from logical experts to physical experts.

    This is a sparse matrix, where -1 indicates no mapping.

    Shape: (num_moe_layers, num_logical_experts, num_redundant_experts + 1)

    # Example

    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the mapping could look like this:

    ```
    [[[0, 4, -1],
      [1, 5, -1],
      [2, -1, -1],
      [3, -1, -1]],
     [[0, 2, 4],
      [3, -1, -1],
      [1, -1, -1],
      [5, -1, -1]]]
    ```
    """
    logical_replica_count: torch.Tensor
    """
    Number of replicas for each logical expert.
    This is exactly the non-`-1` count in the `logical_to_physical_map`.

    Shape: (num_moe_layers, num_logical_experts)

    # Example
    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the count could look like this:

    ```
    [[2, 2, 1, 1],
     [3, 1, 1, 1]]
    """

    expert_load_pass: torch.Tensor
    """
    Expert load during this forward pass. 
    We use the token count each expert processes as the load.

    Shape: (num_moe_layers, num_physical_experts)
    """
    expert_load_window: torch.Tensor
    """
    A sliding window of expert load.

    Shape: (window_size, num_moe_layers, num_physical_experts)

    NOTE: The expert_load_view now records load for all physical experts
    rather than just local experts. This ensures consistent load statistics
    across different dispatch methods (naive all-to-all, DeepEP).
    The recorded load will be multiplied by dp_size when using naive all-to-all
    due to each DP rank contributing the same token set to the calculation.
    See:
    https://github.com/vllm-project/vllm/pull/22167#pullrequestreview-3086143856
    """
    model_name: str
    model: MixtureOfExperts
    expert_buffer: list[torch.Tensor]
    """
    The buffer to store the expert weights during transfer.
    """
    rebalanced: bool
    """
    This flag is only used when running Async EPLB. It is set to True by the main thread
    after the new expert maps have been computed. This indicates that the async worker
    should start transferring weights. move_to_workspace sets this flag to False when
    all weights have been transferred and the new map has been successfully committed.

    rebalanced relies on the GIL to synchronize access between the main thread and
    the async worker.
    """
    eplb_stats: EplbStats | None
    """
    EPLB stats for the model.
    """
    cuda_device_index: int | None
    """
    CUDA device index for the async EPLB worker thread.
    """
    communicator: EplbCommunicator
    """
    The communicator for expert weight transfers.
    """
    pending_result: AsyncEplbLayerResult | None = None
    """
    Set by the async worker after all writes to expert_buffer are done. Consumed
    and reset to None by the main thread in move_to_workspace() after the contents of
    expert_buffer have been transferred out. At most one result is pending at a time.

    pending_result relies on the GIL to synchronize access between the main thread and
    the async worker.
    """
    num_unpadded_tokens_tensors: list[torch.Tensor] | None = None
    """
    Per-ubatch scalar int32 tensors holding the number of real (non-padding)
    tokens.  Allocated once in :meth:`EplbState.add_model` so that device
    pointers remain stable across CUDA-graph replays.  The router kernel
    indexes this list with ``dbo_current_ubatch_id()``.
    """
    model_config: "ModelConfig | None" = None
    """
    The config this state was registered under. Held because the runtime placement
    path is driven from `step`, which has no config in hand, while `update_mapping`
    and `publish_source_local_maps` both require one.
    """


class EplbState:
    """
    EplbState of each expert parallel model. Key is the model config hash.
    """

    def __init__(self, parallel_config: ParallelConfig, device: torch.device):
        self.parallel_config = parallel_config
        self.device = device
        self.model_states: dict[str, EplbModelState] = {}
        self._logged_layers: set[int] = set()
        self._replica_slot_occupant: dict[int, dict[int, int]] = {}
        self.policy: type[AbstractEplbPolicy] = DefaultEplbPolicy
        """
        Selected EPLB algorithm class
        """
        self.expert_load_window_step: int = 0
        """
        Current step in the sliding window.

        Different from `expert_rearrangement_step`, 
        each EP rank may have its own `expert_load_window_step`.
        """
        self.expert_load_window_size: int = 0
        """
        Size of the expert load sliding window.
        This is a constant and is taken from the config.
        """
        self.expert_rearrangement_step: int = 0
        """
        Steps after last rearrangement.
        Will trigger a rearrangement if it exceeds the threshold.

        NOTE: Keep in mind that all EP ranks need to have the same
        `expert_rearrangement_step` value to ensure synchronization.
        Otherwise, the rearrangement will hang at collective
        communication calls.
        """
        self.expert_rearrangement_step_interval: int = 0
        """
        Interval for expert rearrangement steps.
        This is a constant and is taken from the config.
        """
        self.should_record_tensor: torch.Tensor | None = None
        """
        Shared scalar bool tensor for all layers.  Every
        :class:`EplbLayerState` holds a reference to the **same** object so
        a single ``.fill_()`` updates all layers at once.  Allocated on the
        first call to :meth:`_propagate_shared_tensors`.
        """
        self.is_async: bool = False
        """
        The flag indicates whether the EPLB is running in async mode.
        """
        self.rearrange_event: CpuGpuEvent = CpuGpuEvent()
        """
        Event to signal when a new rearrangement is needed for the async thread.
        """
        self.async_worker: threading.Thread | None = None
        """
        Background thread handling async transfers.
        """
        self.cuda_device_index: int | None = None
        """
        CUDA device index for the async EPLB worker thread.
        """
        self.startup_normalization_ms: float = 0.0
        """
        Total time spent installing the predictive fixed layout at startup.
        Reported separately from serving latency; zero outside predictive mode.
        """
        if self.device.type == "cuda":
            self.cuda_device_index = self.device.index
            if self.cuda_device_index is None and torch.cuda.is_available():
                self.cuda_device_index = torch.accelerator.current_device_index()

    @staticmethod
    def build_initial_global_physical_to_logical_map(
        num_routed_experts: int,
        num_redundant_experts: int,
    ) -> Sequence[int]:
        """
        Build an initial expert arrangement using the following structure:
        [original routed experts, redundant experts]

        Returns:
            physical_to_logical_map (Sequence[int]): A list of integers,
                where each integer is the index of the logical expert
                that the corresponding physical expert maps to.
        """
        global_physical_to_logical_map = list(range(num_routed_experts))
        global_physical_to_logical_map += [
            i % num_routed_experts for i in range(num_redundant_experts)
        ]
        return global_physical_to_logical_map

    def validate_ep_configuration(self, new_model: MixtureOfExperts):
        """
        Validate that the expert parallel configuration of
        the new model is the same as the existing models.
        """
        if len(self.model_states) > 0:
            model = next(iter(self.model_states.values())).model
            if (
                model.num_routed_experts != new_model.num_routed_experts
                or model.num_redundant_experts != new_model.num_redundant_experts
                or model.num_physical_experts != new_model.num_physical_experts
                or model.num_logical_experts != new_model.num_logical_experts
                or model.num_expert_groups != new_model.num_expert_groups
            ):
                raise RuntimeError(
                    "Model: {} "
                    "with config {} "
                    "{} {} {} {} "
                    "mismatch with new model {} "
                    "with config {} "
                    "{} {} {} {}".format(
                        type(model),
                        model.num_routed_experts,
                        model.num_redundant_experts,
                        model.num_physical_experts,
                        model.num_logical_experts,
                        model.num_expert_groups,
                        type(new_model),
                        new_model.num_routed_experts,
                        new_model.num_redundant_experts,
                        new_model.num_physical_experts,
                        new_model.num_logical_experts,
                        new_model.num_expert_groups,
                    )
                )

    def add_model(
        self,
        model: MixtureOfExperts,
        model_config: ModelConfig,
    ):
        """
        Build the initial EPLB state.
        """
        self.validate_ep_configuration(model)
        self.is_async = self.parallel_config.eplb_config.use_async

        physical_to_logical_map_list = (
            EplbState.build_initial_global_physical_to_logical_map(
                model.num_routed_experts,
                model.num_redundant_experts,
            )
        )
        physical_to_logical_map = torch.tensor(
            physical_to_logical_map_list,
            device=self.device,
        )
        # Assuming 8 GPUs per node, this supports up to
        # (1023 + 1) / 8 = 128 nodes for now.
        # TODO(rui): make this configurable
        MAX_EXPERT_REDUNDANCY = 1023
        assert model.num_redundant_experts <= MAX_EXPERT_REDUNDANCY, (
            f"num_redundant_experts {model.num_redundant_experts} "
            f"must be less than or equal to {MAX_EXPERT_REDUNDANCY}"
        )
        max_slots_per_logical_expert = MAX_EXPERT_REDUNDANCY + 1
        logical_to_physical_map = torch.full(
            (model.num_logical_experts, max_slots_per_logical_expert),
            -1,
            device=self.device,
        )
        logical_replica_count = torch.zeros(
            (model.num_logical_experts,),
            device=self.device,
            dtype=torch.long,
        )

        for i in range(model.num_physical_experts):
            logical_idx = physical_to_logical_map[i]
            logical_to_physical_map[logical_idx, logical_replica_count[logical_idx]] = i
            logical_replica_count[logical_idx] += 1

        # Duplicate initial mapping for all layers
        physical_to_logical_map = (
            physical_to_logical_map.unsqueeze(0)
            .expand(
                model.num_moe_layers,
                -1,
            )
            .contiguous()
        )
        logical_to_physical_map = (
            logical_to_physical_map.unsqueeze(0)
            .expand(
                model.num_moe_layers,
                -1,
                -1,
            )
            .contiguous()
        )
        logical_replica_count = (
            logical_replica_count.unsqueeze(0)
            .expand(
                model.num_moe_layers,
                -1,
            )
            .contiguous()
        )

        expert_load_pass = torch.zeros(
            (model.num_moe_layers, model.num_physical_experts),
            dtype=torch.int32,
            device=self.device,
        )
        self.expert_load_window_size = self.parallel_config.eplb_config.window_size
        expert_load_window = torch.zeros(
            (
                self.expert_load_window_size,
                model.num_moe_layers,
                model.num_physical_experts,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        # Set the initial progress of rearrangement to 3/4
        eplb_step_interval = self.parallel_config.eplb_config.step_interval
        self.expert_rearrangement_step = max(
            0, eplb_step_interval - eplb_step_interval // 4
        )
        self.expert_rearrangement_step_interval = eplb_step_interval

        policy_type = self.parallel_config.eplb_config.policy
        self.policy = EPLB_POLICIES[policy_type]
        logger.debug("Selected EPLB policy: %s", policy_type)

        # num_ubatches is 0 when DBO is disabled.
        num_ubatches = max(1, self.parallel_config.num_ubatches)
        num_unpadded_tokens_tensors = [
            torch.tensor(0, dtype=torch.int32, device=self.device)
            for _ in range(num_ubatches)
        ]

        model.set_eplb_state(
            expert_load_pass,
            logical_to_physical_map,
            logical_replica_count,
        )
        self._propagate_shared_tensors(model, num_unpadded_tokens_tensors)
        expert_buffer = [torch.empty_like(w) for w in model.expert_weights[0]]

        assert self.parallel_config.eplb_config.communicator is not None, (
            "EPLB communicator backend must be set by ParallelConfig"
        )
        communicator = create_eplb_communicator(
            group_coordinator=get_eplb_group(),
            backend=self.parallel_config.eplb_config.communicator,
            expert_weights=model.expert_weights,
            expert_buffer=expert_buffer,
        )

        model_state = EplbModelState(
            physical_to_logical_map=physical_to_logical_map,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            expert_load_pass=expert_load_pass,
            expert_load_window=expert_load_window,
            model_name=model_config.model,
            model=model,
            expert_buffer=expert_buffer,
            rebalanced=False,
            eplb_stats=None,
            cuda_device_index=self.cuda_device_index,
            communicator=communicator,
            num_unpadded_tokens_tensors=num_unpadded_tokens_tensors,
        )
        model_state.model_config = model_config
        self.model_states[compute_hash_cached(model_config)] = model_state

        if self.predictive_enabled:
            # The configuration validator checks the model-shape fingerprint;
            # the device can only be checked where a device is actually bound.
            self.parallel_config.predictive_expert_replication_config.validate_fingerprint(
                device_name=current_platform.get_device_name(
                    self.device.index if self.device.index is not None else 0
                )
            )
            self.startup_normalization_ms += self.normalize_predictive_layout(
                model_config
            )

    @property
    def predictive_enabled(self) -> bool:
        """Whether the predictive controller owns placement instead of Native EPLB."""
        return self.parallel_config.predictive_expert_replication_config.enabled

    def build_predictive_layout(self, model: MixtureOfExperts) -> torch.Tensor:
        """Build the fixed predictive layout for a registered model.

        Args:
            model: The registered mixture-of-experts model.

        Returns:
            A `[num_moe_layers, num_physical_experts]` physical-to-logical map.

        Raises:
            ValueError: If the model's physical layout cannot hold the canonical
                rows plus the configured replica slots.
        """
        ep_size = get_ep_group().device_group.size()
        predictive_config = self.parallel_config.predictive_expert_replication_config
        replica_slots = predictive_config.replica_slots_per_rank
        canonical_per_rank = model.num_logical_experts // ep_size
        expected_local = canonical_per_rank + replica_slots
        if model.num_local_physical_experts != expected_local:
            raise ValueError(
                f"Predictive expert replication expects {canonical_per_rank} "
                f"canonical + {replica_slots} inactive physical rows per rank, "
                f"but the model has {model.num_local_physical_experts}."
            )
        layout = build_predictive_physical_map(
            num_layers=len(model.expert_weights),
            num_logical_experts=model.num_logical_experts,
            ep_size=ep_size,
            replica_slots_per_rank=replica_slots,
            device=self.device,
        )
        placement = predictive_config.parsed_static_replica_placement
        if placement is not None:
            # A validation aid: pre-place one replica so source-rank routing can
            # be exercised before any transfer machinery exists.
            logical, target_rank = placement
            if not 0 <= logical < model.num_logical_experts:
                raise ValueError(
                    f"static_replica_placement expert {logical} is outside "
                    f"[0, {model.num_logical_experts})."
                )
            if not 0 <= target_rank < ep_size:
                raise ValueError(
                    f"static_replica_placement rank {target_rank} is outside "
                    f"[0, {ep_size})."
                )
            if logical // canonical_per_rank == target_rank:
                raise ValueError(
                    f"static_replica_placement puts expert {logical} on rank "
                    f"{target_rank}, which already owns it canonically. Both "
                    f"copies would live on one rank, so the routing assertions "
                    f"would pass without any cross-rank replica being exercised."
                )
            layout.view(len(model.expert_weights), ep_size, expected_local)[
                :, target_rank, canonical_per_rank
            ] = logical
        return layout

    def normalize_predictive_layout(self, model_config: ModelConfig) -> float:
        """Install the fixed predictive layout before the server reports ready.

        Rearranges the natively loaded rows into canonical ownership, clears the
        inactive replica rows, and republishes the routing maps. This is startup
        work and is deliberately excluded from request-path accounting.

        Args:
            model_config: Identifies which `EplbModelState` to normalize.

        Returns:
            Wall-clock duration in milliseconds.
        """
        model_state = self.model_states[compute_hash_cached(model_config)]
        model = model_state.model
        ep_group = get_ep_group().device_group
        layout = self.build_predictive_layout(model)

        start_time = time.perf_counter()
        rearrange_expert_weights_inplace(
            model_state.physical_to_logical_map,
            layout,
            model.expert_weights,
            model_state.expert_buffer,
            ep_group,
            model_state.communicator,
        )
        # Inactive rows must not hold stale canonical weights: a later replica
        # transfer is the only thing allowed to make them readable.
        local_layout = layout.view(len(model.expert_weights), ep_group.size(), -1)
        for layer_index, layer_weights in enumerate(model.expert_weights):
            # Per layer, not layer 0 applied to all: tickets 07 and 08 place
            # replicas per layer, and a row inactive in layer 0 but placed in
            # layer 3 would otherwise keep layer 3's stale canonical weights
            # readable, violating the invariant this loop exists to hold.
            inactive_rows = (
                (local_layout[layer_index, ep_group.rank()] < 0).nonzero().flatten()
            )
            for weight in layer_weights:
                weight[inactive_rows] = 0
        self.update_mapping(model_config, layout)
        self.publish_source_local_maps(model_config)
        self.verify_replica_weight_equality(model_config)
        if envs.VLLM_PREDICTIVE_PLACE_PER_FORWARD:
            self.attach_placement_coordinator(model_config)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if ep_group.rank() == 0:
            logger.info(
                "Predictive expert replication normalized %s to %d canonical + "
                "%d inactive rows per rank in %.2f ms.",
                model_state.model_name,
                model.num_logical_experts // ep_group.size(),
                model.num_local_physical_experts
                - model.num_logical_experts // ep_group.size(),
                elapsed_ms,
            )
        return elapsed_ms

    def verify_replica_weight_equality(self, model_config: ModelConfig) -> None:
        """Assert every physical copy of a logical expert holds the same weights.

        This is what makes an output comparison interpretable. Routing a logical
        expert to a second copy regroups tokens inside the expert GEMM, which
        perturbs reduction order, so logprobs move slightly even when nothing is
        wrong. Arguing about whether a given shift is noise or a defect is
        unnecessary if the copies are provably identical: the arithmetic is then
        the same and only its order differs.

        Compares position-weighted checksums rather than shipping weights around,
        so the cost is a couple of small collectives at startup. Position
        weighting catches a permuted copy, which a plain sum would not.

        Args:
            model_config: Identifies which `EplbModelState` to verify.

        Raises:
            RuntimeError: If two copies of one logical expert disagree, which
                means a transfer or the layout is wrong.
        """
        model_state = self.model_states[compute_hash_cached(model_config)]
        model = model_state.model
        ep_group = get_eplb_group().device_group
        ep_size = ep_group.size()
        num_layers = len(model.expert_weights)
        local_rows = model.num_local_physical_experts

        local = torch.zeros(
            (num_layers, local_rows), dtype=torch.float64, device=self.device
        )
        for layer_index, layer_weights in enumerate(model.expert_weights):
            for weight in layer_weights:
                flat = weight.reshape(local_rows, -1).to(torch.float64)
                position = torch.arange(
                    1, flat.shape[1] + 1, dtype=torch.float64, device=flat.device
                )
                local[layer_index] += (flat * position).sum(dim=1)

        gathered = torch.zeros(
            (ep_size, num_layers, local_rows), dtype=torch.float64, device=self.device
        )
        torch.distributed.all_gather_into_tensor(gathered, local, group=ep_group)
        # [ep_size, layers, local] -> [layers, ep_size * local], matching the
        # physical row numbering the maps use.
        checksums = gathered.permute(1, 0, 2).reshape(num_layers, -1).cpu()

        physical_to_logical = model_state.physical_to_logical_map.cpu()
        mismatches: list[str] = []
        compared = 0
        for layer_index in range(num_layers):
            rows_by_logical: dict[int, list[int]] = {}
            for row, logical in enumerate(physical_to_logical[layer_index].tolist()):
                if logical >= 0:
                    rows_by_logical.setdefault(logical, []).append(row)
            for logical, rows in rows_by_logical.items():
                if len(rows) < 2:
                    continue
                compared += 1
                values = [checksums[layer_index, row].item() for row in rows]
                spread = max(values) - min(values)
                if spread != 0.0:
                    mismatches.append(
                        f"layer {layer_index} expert {logical} rows {rows} "
                        f"checksum spread {spread:.6e}"
                    )
        if mismatches:
            raise RuntimeError(
                "Physical copies of a logical expert are not identical, so an "
                "output comparison could not distinguish reduction-order noise "
                f"from a real defect: {mismatches[:5]}"
            )
        if ep_group.rank() == 0:
            # Report how much was actually compared. With no replica placed there
            # is nothing to compare and this check passes without checking
            # anything, which would be worse than not having it: a vacuous pass
            # reads exactly like a real one.
            logger.info(
                "Predictive expert replication compared %d replicated "
                "(layer, expert) pairs across %d layers; all copies identical. "
                "%s",
                compared,
                num_layers,
                "No replicas are placed, so this check is vacuous."
                if compared == 0
                else "",
            )

    _predictive_stream: torch.cuda.Stream | None = None
    """Ordered predictive communication stream, spec section 9."""

    # Quoted, and imported only for type checking: `nvshmem_transfer` imports NVSHMEM at
    # module scope, and this module has to load on a host without it.
    _one_sided: "OneSidedExpertTransfer | None" = None
    """The worker's NVSHMEM transport, initialised once and shared by every layer.

    One symmetric staging buffer per worker, not per layer: at most one expert is in
    flight, so sizing it per layer would multiply it by the row count for nothing.
    """

    _logged_inactive_slot_check: bool = False
    """Set once the inactive-slot check has reported, so it logs a single line."""

    def verify_inactive_slots_unused(
        self,
        model_config: ModelConfig | None = None,
        reduce_across_ranks: bool = True,
    ) -> int:
        """Assert no token was routed to a physical slot holding no logical expert.

        An inactive slot's weights are never written, so a token routed there
        reads uninitialised memory and produces plausible-looking output instead
        of an error. The source-local physical map is supposed to make that
        structurally impossible; this checks that it does.

        A violation raises. Everything else is reported through the return value
        rather than by raising, because this runs on every forward and two
        conditions are ordinary rather than wrong: a dummy step's load is zeroed
        before this point, and a model may carry no inactive slot at all. Raising
        on either would kill serving over a non-problem.

        Args:
            model_config: Which model to check. All of them when None.
            reduce_across_ranks: All-reduce the load over the EP group first, so a
                slot busy on any rank is caught on every rank. Disable only in
                tests, where there is no process group.

        Returns:
            How many inactive slots were verified **against observed traffic** -
            zero when nothing was recorded, or when no slot is inactive. A caller
            reporting a pass must require a non-zero count: an all-zero load table
            makes every slot look idle, and `log_balancedness` (the only thing
            enabling recording in predictive mode) defaults to False, so counting
            slots alone would report an authoritative pass from a run that
            observed no tokens.

        Raises:
            RuntimeError: If an inactive slot carries load.
        """
        if model_config is None:
            states = list(self.model_states.values())
        else:
            states = [self.model_states[compute_hash_cached(model_config)]]

        verified = 0
        for model_state in states:
            load = model_state.expert_load_pass.clone()
            if reduce_across_ranks:
                torch.distributed.all_reduce(load, group=get_ep_group().device_group)
            inactive = model_state.physical_to_logical_map < 0
            count = int(inactive.sum().item())
            if count == 0:
                # Nothing to observe for this model. Ordinary for a second model or
                # a native-EPLB map, so it is not this check's business to object.
                continue
            busy = load * inactive.to(load.dtype)
            total = float(busy.sum().item())
            if total > 0:
                layers = torch.nonzero(busy.sum(dim=1) > 0).flatten().tolist()
                raise RuntimeError(
                    f"model {model_state.model_name}: inactive physical slots "
                    f"received {total:.0f} routed tokens across layers {layers}. "
                    "Their expert weights are uninitialised, so any output that "
                    "reached them is invalid."
                )
            if float(load.sum().item()) <= 0:
                # No traffic recorded, so an idle inactive slot proves nothing.
                continue
            verified += count
        return verified

    def attach_placement_coordinator(self, model_config: ModelConfig) -> None:
        """Give every MoE runner the coordinator that drives in-forward placement.

        Built here rather than at model construction because it needs the transfer
        buffers, the EPLB communicator and the EP group, none of which exist when the
        layers are built.
        """
        model_state = self.model_states[compute_hash_cached(model_config)]
        predictive = self.parallel_config.predictive_expert_replication_config
        ep_group = get_ep_group().device_group
        model = model_state.model
        # Resolved once, at the largest M a prefill step can present after the
        # allgather: every rank sees all DP ranks' tokens, so that is the scheduler's
        # batch limit times the EP size. The block size is selected per M, and the
        # decision it feeds is about the prefill regime, so asking at a decode-sized M
        # would answer the wrong question.
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        batched = (
            vllm_config.scheduler_config.max_num_batched_tokens
            if vllm_config is not None and vllm_config.scheduler_config is not None
            else 8192
        )
        block_size_m = resolve_moe_block_size_m(
            model,
            top_k=model.moe_layers[0].moe_config.experts_per_token,
            num_batched_tokens=batched * ep_group.size(),
        )
        if predictive.device_issued_transfer:
            coordinator = self._build_device_coordinator(
                model_config, model_state, ep_group, float(block_size_m)
            )
            if coordinator is not None:
                self._attach(model, coordinator, float(block_size_m))
                return
        # The host path launches the transfer at the head of the *following* layer's MoE
        # and waits for it on the next statement, so at a lookahead of 1 its overlap
        # window is zero and the whole transfer is exposed on top of a host
        # synchronisation. Configuration validation rejects that combination — but it
        # cannot see this fallback, reached when the device path was asked for and
        # NVSHMEM turned out to be missing. Failing here rather than warning: the
        # alternative is a server running the one configuration the validator forbids.
        if predictive.prediction_lookahead_layers == 1:
            raise RuntimeError(
                "Predictive expert replication fell back to the host-issued transfer "
                "with prediction_lookahead_layers=1, where the launch and the wait are "
                "adjacent statements and the overlap window is zero. Install NVSHMEM "
                "to use the device-issued path, set prediction_lookahead_layers=2, or "
                "disable the feature."
            )
        coordinator = PlacementCoordinator(
            ep_size=ep_group.size(),
            ep_rank=ep_group.rank(),
            canonical_per_rank=model.num_logical_experts // ep_group.size(),
            replica_slots_per_rank=predictive.replica_slots_per_rank,
            num_layers=len(model.expert_weights),
            lookahead=predictive.prediction_lookahead_layers,
            budget=predictive.max_transfers_per_forward,
            max_per_layer=predictive.max_replicas_per_layer,
            min_tokens_per_expert=float(block_size_m),
            min_tokens=float(block_size_m),
            expert_weights=model.expert_weights,
            expert_buffer=model_state.expert_buffer,
            communicator=model_state.communicator,
            stream=self._placement_stream(),
            publish=lambda layer, placements: self.activate_layer_replicas(
                model_config, layer, placements
            ),
        )
        self._attach(model, coordinator, float(block_size_m))

    def _attach(self, model, coordinator, block_size_m: float) -> None:
        """Give every MoE layer the coordinator and the shared token bar."""
        for layer in model.moe_layers:
            layer.placement_coordinator = coordinator
            # Prediction and placement share one bar. Predicting a forward the
            # placement gate is certain to reject costs a gate matmul and a small
            # AllGather per layer for nothing: measured at 7.3% of TPOT.
            layer.prediction_min_tokens_per_expert = block_size_m

    def _build_device_coordinator(
        self, model_config: ModelConfig, model_state, ep_group, block_size_m: float
    ):
        """The device-issued path, or None with a warning saying what was missing.

        Returning None rather than raising, because the host path still works and still
        serves — but *loudly*, because a silent fallback here produces a run that looks
        correct while measuring the very cost it was meant to remove. That has happened
        on this branch more than once, and it is why `analyse_e2e.py` treats an
        activation log line as the evidence a path was reached at all.
        """
        from vllm.distributed.eplb.device_coordinator import (
            DevicePlacementCoordinator,
            LayerMaps,
        )
        from vllm.distributed.eplb.device_transfer import (
            DeviceExpertTransfer,
            WeightPointers,
        )
        from vllm.distributed.eplb.nvshmem_transfer import (
            nvshmem_unavailable_reason,
        )

        predictive = self.parallel_config.predictive_expert_replication_config
        model = model_state.model
        reason = nvshmem_unavailable_reason()
        if reason is not None:
            logger.warning(
                "Predictive expert replication: falling back to the host-issued "
                "transfer because %s. The host synchronisation this path removes is "
                "5.28 ms per predicted layer, so the feature is expected to cost more "
                "than it returns in this configuration.",
                reason,
            )
            return None

        # Publishing needs the source-local maps, which the placement path writes into
        # rather than rebuilding. They are allocated here so a layer cannot be reached
        # with them unset.
        self.publish_source_local_maps(model_config)

        canonical_per_rank = model.num_logical_experts // ep_group.size()
        per_local = canonical_per_rank + predictive.replica_slots_per_rank
        layout = model_state.physical_to_logical_map.view(
            -1, ep_group.size(), per_local
        )
        try:
            pointers = [
                WeightPointers.build(tensors) for tensors in model.expert_weights
            ]
            maps = []
            for index, layer_module in enumerate(model.moe_layers):
                layer_state = layer_module.eplb_state
                maps.append(
                    LayerMaps(
                        logical_to_physical=model_state.logical_to_physical_map[index],
                        logical_replica_count=model_state.logical_replica_count[index],
                        source_local=layer_state.source_local_physical_map,
                        source_local_replica_count=(
                            layer_state.source_local_replica_count
                        ),
                        layout=layout[index],
                    )
                )
            staging = self._symmetric_staging(
                ep_group, max(p.total_bytes for p in pointers)
            )
            transfer = DeviceExpertTransfer(
                staging=staging,
                ep_rank=ep_group.rank(),
                per_rank_experts=canonical_per_rank,
            )
        except Exception:
            logger.exception(
                "Predictive expert replication: the device-issued transfer could not "
                "be set up, so the host-issued path is used. This is not a silent "
                "fallback: the run below measures the host synchronisation."
            )
            return None

        # Two per-layer lists, built from two different attributes, and indexed by the
        # same number. A length mismatch would have the weights and the routing maps
        # describing different layers, which routes tokens to a row holding another
        # expert's weights and raises nothing.
        if len(pointers) != len(maps):
            logger.error(
                "Predictive expert replication: %d weight layers against %d MoE "
                "layers, so the device path cannot index them together. Falling back "
                "to the host-issued transfer.",
                len(pointers),
                len(maps),
            )
            return None

        logger.info(
            "Predictive expert replication: device-issued transfer active on %d "
            "layers, %d ranks.",
            len(pointers),
            ep_group.size(),
        )
        return DevicePlacementCoordinator(
            ep_size=ep_group.size(),
            ep_rank=ep_group.rank(),
            canonical_per_rank=canonical_per_rank,
            replica_slots_per_rank=predictive.replica_slots_per_rank,
            num_layers=len(model.expert_weights),
            lookahead=predictive.prediction_lookahead_layers,
            budget=predictive.max_transfers_per_forward,
            min_tokens=block_size_m,
            min_tokens_per_expert=block_size_m,
            pointers=pointers,
            maps=maps,
            transfer=transfer,
            device=self.device,
            stream=self._placement_stream(),
        )

    def _symmetric_staging(self, ep_group, expert_bytes: int) -> torch.Tensor:
        """One symmetric staging buffer for this worker, allocated once.

        NVSHMEM is initialised here rather than earlier because it has to come after
        torch and NCCL, which vLLM brings up first and this code does not get to
        precede.
        """
        if self._one_sided is None:
            from vllm.distributed.eplb.nvshmem_transfer import OneSidedExpertTransfer

            def broadcast_uid(local):
                holder = [local]
                torch.distributed.broadcast_object_list(
                    holder,
                    src=torch.distributed.get_global_rank(ep_group, 0),
                    group=ep_group,
                )
                return holder[0]

            self._one_sided = OneSidedExpertTransfer(
                rank=ep_group.rank(),
                world_size=ep_group.size(),
                expert_bytes=expert_bytes,
                device=self.device,
                broadcast_uid=broadcast_uid,
            )
        return self._one_sided._staging

    def _placement_stream(self) -> torch.cuda.Stream:
        """The ordered predictive communication stream of spec section 9."""
        if self._predictive_stream is None:
            self._predictive_stream = torch.cuda.Stream(device=self.device)
        return self._predictive_stream

    def activate_layer_replicas(
        self, model_config: ModelConfig, layer: int, placements: list[Placement]
    ) -> None:
        """Make one layer's active replicas exactly `placements`.

        `placements` is the layer's complete desired set, so a replica the current plan
        no longer wants is reverted here. Reversion is a map edit with no transfer, and
        keeping an unwanted replica is not neutral: it goes on shedding half of an
        expert that may no longer be hot, onto a rank that may now be the peak. Ticket
        00 measured cross-forward residency at -20.0%, and with nothing reverting,
        active replicas accumulated to 47-76 per forward against a transfer budget of
        43.

        Incremental by construction: only this layer's rows are touched. A full
        republish rebuilds every logical map and walks all 48 layers, on the prefill
        step's critical path.
        """
        model_state = self.model_states[compute_hash_cached(model_config)]
        ep_group = get_ep_group().device_group
        ep_size = ep_group.size()
        canonical_per_rank = model_state.model.num_logical_experts // ep_size
        slots = (
            self.parallel_config.predictive_expert_replication_config
        ).replica_slots_per_rank
        view = model_state.physical_to_logical_map.view(
            -1, ep_size, canonical_per_rank + slots
        )
        reverted = self._republish_layer(
            model_state,
            layer,
            ep_group.rank(),
            placements,
            canonical_per_rank,
            slots,
        )
        # The layout is what `active_replicas` reads, so a reverted row has to read -1
        # there too or the layout and the routing maps disagree about what is live.
        # Clear before setting: a slot handed from one expert to another appears in
        # both sets, and the placement must win.
        for target_rank, _expert in reverted:
            view[layer, target_rank, canonical_per_rank] = -1
        for placement in placements:
            view[layer, placement.target_rank, canonical_per_rank] = (
                placement.logical_expert
            )
        # Bounded to the first activation of each layer: 48 lines, not one per forward.
        # `analyse_e2e.py` treats the presence of this line as the evidence that the
        # placement path was reached at all, so a run without it cannot be told apart
        # from a run whose arm was silently inert.
        if placements and layer not in self._logged_layers:
            self._logged_layers.add(layer)
            logger.info(
                "Predictive expert replication: activated %d replica(s) on layer %d",
                len(placements),
                layer,
            )

    def _republish_layer(
        self,
        model_state: EplbModelState,
        layer: int,
        source_rank: int,
        placements: list[Placement],
        canonical_per_rank: int,
        replica_slots_per_rank: int,
    ) -> set[tuple[int, int]]:
        """Update one layer's routing maps for `placements`, in place.

        Deliberately not a rebuild. `compute_logical_maps` inverts the whole physical
        layout with a 136-iteration Python loop and an `.item()` that makes its output
        shape data-dependent — the reason it asserts CPU — and it measured 4.707 ms
        per layer, 202 ms for a forward touching 43 layers. That ran here, on the
        activation path, inside the forward. Upstream calls it from `rearrange`, which
        fires every `step_interval` steps; this path fires per activation, so the two
        differ in frequency by orders of magnitude and the CPU round trip that is free
        there is not free here.

        Nothing about a placement needs an inversion: one expert gained one copy at a
        row that follows from the layout. `apply_replica_maps` writes exactly that,
        with host-side scalars only, so no device read and no synchronization. A suite
        of tests asserts it agrees with the inversion, including over 200 randomised
        placement sequences.
        """
        layer_module = model_state.model.moe_layers[layer]
        layer_state = getattr(layer_module, "eplb_state", None)
        if not isinstance(layer_state, EplbLayerState):
            raise RuntimeError(
                f"MoE layer {layer} has no EPLB layer state, so an activated replica "
                f"could not be routed to and tokens would keep going to the canonical "
                f"copy only."
            )
        # The **source-local** pair is what `_apply_eplb_mapping` reads:
        # it prefers `source_local_physical_map` over `logical_to_physical_map`
        # whenever the former is set, and under this feature it always is. Writing
        # the other pair transfers the replica, describes it correctly, and publishes
        # it where nothing reads — replicas activated, zero tokens routed to them,
        # physical per-rank load equal to canonical ownership to 0.00%. That is a
        # measured run: 131 replicas per forward removed 0.6% of prefill excess
        # against an oracle of 35.1%, at 2.2x the baseline TTFT.
        target_map = layer_state.source_local_physical_map
        target_count = layer_state.source_local_replica_count
        if target_map is None or target_count is None:
            raise RuntimeError(
                f"MoE layer {layer} has no source-local routing maps to publish into, "
                f"so an activated replica would never be routed to. "
                f"`publish_source_local_maps` allocates them before the first forward; "
                f"reaching here means it did not run."
            )
        reverted = apply_replica_maps(
            logical_to_physical=model_state.logical_to_physical_map[layer],
            logical_replica_count=model_state.logical_replica_count[layer],
            source_local=target_map,
            placements=placements,
            per_rank_experts=canonical_per_rank,
            replica_slots_per_rank=replica_slots_per_rank,
            source_rank=source_rank,
            slot_occupant=self._replica_slot_occupant.setdefault(layer, {}),
        )
        # All-ones, so the shared routing path's per-token replica choice stays a
        # lookup: with one copy on offer a rank's chunk cannot be split.
        target_count.fill_(1)
        return reverted

    def publish_source_local_maps(self, model_config: ModelConfig) -> None:
        """Give every MoE layer this rank's Source-local physical map.

        Routing reads these instead of the global map, which is what makes it
        source-rank rather than per-token. Must be called after any placement
        change, since a stale map would route to a slot that no longer holds the
        expert it was chosen for.

        Args:
            model_config: Identifies which `EplbModelState` to publish for.
        """
        model_state = self.model_states[compute_hash_cached(model_config)]
        source_rank = get_ep_group().device_group.rank()
        for layer_index, layer in enumerate(model_state.model.moe_layers):
            layer_state = getattr(layer, "eplb_state", None)
            if not isinstance(layer_state, EplbLayerState):
                # Skipping would leave this layer on the global map and the shared
                # path's per-token replica choice, which is the token-level
                # routing this feature exists to remove, with nothing to show it.
                raise RuntimeError(
                    f"MoE layer {layer_index} has no EPLB layer state, so it "
                    f"cannot be given a source-local physical map and would fall "
                    f"back to per-token replica selection."
                )
            source_map, replica_count = build_source_local_physical_map(
                model_state.logical_to_physical_map[layer_index],
                model_state.logical_replica_count[layer_index],
                source_rank,
            )
            # Copy into the existing buffers rather than rebinding. Callers
            # re-publish after a placement change, and a rebind would leave any
            # captured graph or cached reference reading the previous allocation,
            # routing to a slot whose expert has since moved.
            if layer_state.source_local_physical_map is None:
                layer_state.source_local_physical_map = source_map
                layer_state.source_local_replica_count = replica_count
            else:
                layer_state.source_local_physical_map.copy_(source_map)
                assert layer_state.source_local_replica_count is not None
                layer_state.source_local_replica_count.copy_(replica_count)

    def prepare_forward(
        self,
        model_config: ModelConfig,
        num_unpadded_tokens: int,
        ubatch_slices: list | None = None,
    ) -> None:
        """Fill the per-[u]batch ``num_unpadded_tokens`` tensors before a
        forward pass.

        Args:
            model_config: Identifies which ``EplbModelState`` to update.
            num_unpadded_tokens: Total number of real (non-padding) tokens
                in the batch.
            ubatch_slices: When DBO is active, a list of
                ``UBatchSlice`` objects describing each micro-batch's
                token range.  When ``None``, only ``tensors[0]`` is filled.
        """
        model_state = self.model_states.get(compute_hash_cached(model_config))
        if model_state is None or model_state.num_unpadded_tokens_tensors is None:
            return
        if self.predictive_enabled:
            # Predictive mode never rolls the pass into the sliding window, so it
            # is the only actual-load record there is. Clear it here, before the
            # forward, so that after the forward it holds exactly that forward's
            # load for prediction-accuracy scoring to read.
            model_state.expert_load_pass.zero_()
        tensors = model_state.num_unpadded_tokens_tensors
        if ubatch_slices is None:
            tensors[0].fill_(num_unpadded_tokens)
        else:
            for i, ubatch_slice in enumerate(ubatch_slices):
                ts = ubatch_slice.token_slice
                # Real tokens in this ubatch: clamp the global count into
                # the slice range so partially-filled ubatches get the
                # correct count.
                val = max(0, min(num_unpadded_tokens, ts.stop) - ts.start)
                tensors[i].fill_(val)

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> None:
        """
        Step the EPLB state.

        Args:
            is_dummy (bool): If `True`, this is a dummy step and the load
                metrics recorded in this forward pass will not count.
                Defaults to `False`.
            is_profile (bool): If `True`, perform a dummy rearrangement
                with maximum communication cost. This is used in
                `profile_run` to reserve enough memory
                for the communication buffer.
            log_stats (bool): If `True`, log the expert load metrics.

        # Stats
            The metrics are all summed up across layers.
            - `avg_tokens`: The average load across ranks.
            - `max_tokens`: The maximum load across ranks.
            - `balancedness`: The ratio of average load to maximum load.
        """
        ep_group = get_ep_group().device_group
        if is_profile:
            # Predictive mode also needs the transfer buffers reserved, so the
            # profile rearrangement runs for both controllers.
            self.rearrange(is_profile=True)
            return

        if is_dummy:
            # Do not record load metrics for dummy steps
            for eplb_model_state in self.model_states.values():
                eplb_model_state.expert_load_pass.zero_()

        self._run_step_diagnostics()

        if self.predictive_enabled:
            # Predictive expert replication owns placement. Native EPLB's
            # historical-load window and periodic rearrangement must not run, or
            # the two controllers would race on the same physical slots.
            #
            # Recording still has to be managed here. This return precedes the
            # only writer of `should_record_tensor`, which is allocated True, so
            # leaving it alone would run the per-layer record atomics on every
            # forward for the whole run with nothing reading the result, and
            # would inflate the very comparison the feature is measured by.
            # Recording is therefore enabled only when benchmark mode asked for
            # it, and the pass is cleared every step so a reader sees one
            # forward's load rather than everything since startup.
            self.configure_predictive_recording(log_stats, is_dummy)
            return

        if (
            log_stats
            and self.expert_rearrangement_step
            % self.parallel_config.eplb_config.log_balancedness_interval
            == 0
        ):
            # Sync the expert load pass for each model (main and drafter).
            # expert_load_pass: (num_moe_layers, num_physical_experts)
            expert_load_pass_list = self._sync_load_pass()
            ep_group = get_ep_group().device_group
            for expert_load_pass, eplb_model_state in zip(
                expert_load_pass_list, self.model_states.values()
            ):
                # num_tokens_per_rank: (num_moe_layers, num_ranks)
                num_tokens_per_rank = (
                    expert_load_pass.reshape(
                        expert_load_pass.shape[0], ep_group.size(), -1
                    )
                    .sum(dim=-1)
                    .float()
                )

                # Compute balancedness ratio:
                # for each layer:
                #   (mean load across ranks) / (max load across ranks)
                avg_tokens_tensor, max_tokens_tensor = _compute_eplb_load_stats(
                    num_tokens_per_rank
                )

                # This gpu/cpu sync only happens with expert stat logging enabled.
                with gpu_sync_allowed():
                    tokens_tensors: list[float] = torch.stack(
                        [avg_tokens_tensor, max_tokens_tensor]
                    ).tolist()
                avg_tokens, max_tokens = tokens_tensors
                balancedness = avg_tokens / max_tokens if max_tokens > 0 else 0.0

                if ep_group.rank() == 0:
                    logger.info(
                        "EPLB step: %d for model %s: avg_tokens=%.2f, "
                        "max_tokens=%d, balancedness=%.4f, "
                        "steps until the next rearrangement: %d",
                        self.expert_rearrangement_step,
                        eplb_model_state.model_name,
                        avg_tokens,
                        max_tokens,
                        balancedness,
                        self.expert_rearrangement_step_interval
                        - self.expert_rearrangement_step,
                    )

        # Update the expert load sliding window
        if not is_dummy:
            should_record = self._should_record_current_step(log_stats=log_stats)
            for eplb_model_state in self.model_states.values():
                if should_record:
                    eplb_model_state.expert_load_window[
                        self.expert_load_window_step
                    ].copy_(eplb_model_state.expert_load_pass)
                    eplb_model_state.expert_load_pass.zero_()

            if should_record:
                self.expert_load_window_step += 1
                if self.expert_load_window_step >= self.expert_load_window_size:
                    self.expert_load_window_step = 0

        # Step the expert rearrangement step
        # Note that even if this is a dummy step, we still increment the
        # rearrangement step and perform rearrangement to ensure all ranks are
        # performing collective communication.
        self.expert_rearrangement_step += 1

        if self.is_async:
            # Run _move_to_workspace if all ranks have finished transferring the
            # new weights to the intermediate buffer
            for eplb_model_state in self.model_states.values():
                # rebalanced must remain consistent amongst all ranks otherwise the
                # all_reduce in _all_ranks_result_ready will hang
                if eplb_model_state.rebalanced and self._all_ranks_result_ready(
                    eplb_model_state
                ):
                    _move_to_workspace(
                        model_state=eplb_model_state,
                        ep_rank=ep_group.rank(),
                    )

        if self.expert_rearrangement_step >= self.expert_rearrangement_step_interval:
            if self.is_async and any(
                eplb_model_state.rebalanced
                for eplb_model_state in self.model_states.values()
            ):
                # Still performing asynchronous rearrangement; update
                # should_record (step > step_interval, so always True) and
                # bail out before the step counter is reset.
                self._update_layer_should_record(log_stats=log_stats)
                return
            self.expert_rearrangement_step = 0
            self.rearrange()

        self._update_layer_should_record(log_stats=log_stats)

    def configure_predictive_recording(self, log_stats: bool, is_dummy: bool) -> None:
        """Manage expert-load recording for a predictive step.

        The predictive branch of `step` returns before the native scheduler that
        normally maintains the record flag, and that flag is allocated enabled.
        Left alone it would run the per-layer record atomics on every forward for
        the whole run with no predictive consumer reading them, inflating the
        latency comparison the feature is judged by, while `expert_load_pass`
        grew without bound.

        Recording is therefore enabled only for benchmark mode, which is what
        prediction-accuracy scoring needs. Clearing the accumulator is *not* done
        here: `step` runs after the forward, so zeroing it here would wipe the
        load the forward just recorded and every reader would see zero. The clear
        belongs before the forward, in `prepare_forward`.

        Args:
            log_stats: Whether benchmark-mode statistics were requested.
            is_dummy: Unused; kept so the call site reads symmetrically with the
                native path, which does distinguish dummy steps.
        """
        del is_dummy
        if self.should_record_tensor is not None:
            self.should_record_tensor.fill_(log_stats)

    def _run_step_diagnostics(self) -> None:
        """Dispatch the opt-in per-forward diagnostics.

        Every one of these performs a collective, so **none of them may be
        conditional on per-rank state**. `is_dummy` in particular differs across
        DP ranks - an idle rank runs a dummy batch to stay in lockstep - so
        gating on it makes some ranks enter the collective and others skip it,
        and the engine deadlocks with every rank waiting on shared memory. A
        dummy step's load is already zeroed above, so running the checks anyway
        is correct as well as safe.

        Extracted from `step` so that collective-safety can be tested rather than
        left as a comment.
        """
        if envs.VLLM_EPLB_DUMP_LOAD_PATH:
            self._dump_logical_expert_load(envs.VLLM_EPLB_DUMP_LOAD_PATH)
        if envs.VLLM_PREDICTIVE_ACCURACY_DUMP_PATH:
            self._dump_prediction_accuracy(envs.VLLM_PREDICTIVE_ACCURACY_DUMP_PATH)
        if envs.VLLM_PREDICTIVE_VERIFY_INACTIVE_SLOTS:
            # Raise on the forward that did it, so the log points at the step
            # rather than at whatever fails downstream of invalid output.
            checked = self.verify_inactive_slots_unused()
            if checked and not self._logged_inactive_slot_check:
                self._logged_inactive_slot_check = True
                # A silent pass is not evidence. Say how many slots were examined
                # so a run can be cited as having observed this.
                logger.info(
                    "Predictive expert replication: verified %d inactive physical "
                    "slots carry no routed load, against recorded traffic.",
                    checked,
                )

    def _dump_logical_expert_load(self, path: str) -> None:
        """Append this forward's per-logical-expert load, for offline analysis.

        A diagnostic, opt-in through an environment variable. The balancedness log
        only reports per-rank aggregates, which cannot answer how concentrated the
        skew is: whether a rank is hot because one expert dominates, or because all
        of its experts are mildly above average. Those two cases differ completely
        in how much replicating a single expert can achieve.
        """
        ep_size = get_ep_group().device_group.size()
        for model_state in self.model_states.values():
            reduced = self._reduced_load_this_forward(model_state)
            if get_ep_group().device_group.rank() != 0:
                continue
            per_logical = self._logical_from_reduced(model_state, reduced)
            # Per-rank load needs no logical-to-rank mapping: physical slots are
            # laid out rank-major, so reshaping is exact where assuming a
            # contiguous expert-to-rank assignment would not be. It must come from
            # the **reduced** tensor: the raw one holds only this rank's own tokens.
            per_rank = reduced.reshape(reduced.shape[0], ep_size, -1).sum(dim=-1)
            # Summed over experts, a layer's load is its token-expert assignment
            # count. Recorded so the analysis can place each forward on a batch-size
            # axis instead of inferring prefill from decode by a magnitude
            # threshold, which is a guess the data does not have to leave open.
            unpadded = model_state.num_unpadded_tokens_tensors
            record = {
                "rank_load": per_rank.tolist(),
                "logical_load": per_logical.tolist(),
                "assignments_per_layer": per_logical.sum(dim=1).tolist(),
                "local_unpadded_tokens": (
                    [int(t.item()) for t in unpadded] if unpadded else None
                ),
                "ep_size": ep_size,
            }
            # Keep the layer axis. Summing it first lets each layer's peak rank
            # cancel against the others, which understates the imbalance that
            # matters: every layer is its own collective and waits for its own
            # slowest rank, so the critical path is the sum of per-layer peaks.
            with open(path, "a") as handle:
                handle.write(json.dumps(record) + "\n")

    def _reduced_load_this_forward(self, model_state: EplbModelState) -> torch.Tensor:
        """All-reduce this forward's physical expert load over the EP group.

        Returns:
            A `[num_moe_layers, num_physical_experts]` tensor of global load, the
            same on **every** rank — the planner needs it everywhere, because a plan
            that differs by rank pairs a sender with no receiver. Both the per-rank
            and the per-logical views must be taken from this tensor: recording
            happens in the router on each rank's own tokens, so an un-reduced copy
            is "how one rank's tokens spread across the ranks", which is a different
            quantity and differs by a factor of the EP size. Deriving one view from
            the reduced tensor and the other from the raw one produced that error.
        """
        load = model_state.expert_load_pass.clone()
        torch.distributed.all_reduce(load, group=get_ep_group().device_group)
        return load

    def _logical_load_this_forward(
        self, model_state: EplbModelState
    ) -> torch.Tensor | None:
        """Reduce this forward's physical expert load to per-logical counts.

        The load is all-reduced over the EP group first, so the result is the
        global count for each logical expert rather than this rank's share.
        Inactive physical rows, marked `-1`, contribute nothing.

        Args:
            model_state: The model whose `expert_load_pass` to reduce.

        Returns:
            A `[num_moe_layers, num_logical_experts]` tensor on EP rank 0, or None
            on every other rank, which has nothing to report.
        """
        load = self._reduced_load_this_forward(model_state)
        if get_ep_group().device_group.rank() != 0:
            return None  # only rank 0 writes the diagnostic dumps
        return self._logical_from_reduced(model_state, load)

    @staticmethod
    def _logical_from_reduced(
        model_state: EplbModelState, reduced: torch.Tensor
    ) -> torch.Tensor:
        """Fold an already-reduced physical load onto logical experts.

        Pure: performs no collective, so a caller that already reduced does not
        reduce again. Inactive physical rows, marked `-1`, contribute nothing.
        """
        physical_to_logical = model_state.physical_to_logical_map
        num_logical = model_state.model.num_logical_experts
        per_logical = torch.zeros(
            reduced.shape[0], num_logical, dtype=reduced.dtype, device=reduced.device
        )
        valid = physical_to_logical.clamp(min=0)
        per_logical.scatter_add_(
            1, valid, reduced * (physical_to_logical >= 0).to(reduced.dtype)
        )
        return per_logical

    def _dump_prediction_accuracy(self, path: str) -> None:
        """Append each cross-layer prediction beside its target layer's load.

        The accuracy question is a comparison of two tensors that both already
        exist: the predicted logical-expert counts a source layer produced during
        this forward, and the target layer's own recorded load from the same
        forward. Both are written raw and every metric is computed offline, so no
        accuracy policy runs on the serving path.

        Diagnostic and opt-in. It synchronizes with the host, which is acceptable
        only because it is off unless the environment variable is set.
        """
        pairs = registered_prediction_pairs()
        if not pairs:
            return
        expected_layers = bound_layer_count()
        for model_state in self.model_states.values():
            per_logical = self._logical_load_this_forward(model_state)
            if per_logical is None:
                continue
            # The registry holds one model's runners. Scoring them against another
            # model's load - a MoE drafter under speculative decoding, say - would
            # silently mix the two into one average that reads as poor accuracy.
            # The layer count is the discriminator, taken from the tensor in hand.
            if expected_layers is not None and per_logical.shape[0] != expected_layers:
                continue
            actual = per_logical.tolist()
            records = []
            for source_index, target_index, runner in pairs:
                snapshot = runner.predicted_load_snapshot
                if snapshot is None:
                    # This layer ran no prediction in this forward, for instance
                    # a padding-only or dummy pass.
                    continue
                if target_index >= len(actual):
                    continue
                records.append(
                    {
                        "model": model_state.model_name,
                        "source": source_index,
                        "target": target_index,
                        # Summed over source ranks so both sides count the same
                        # population: the actual load is likewise a global count.
                        "predicted": snapshot.sum(dim=0).tolist(),
                        "actual": actual[target_index],
                    }
                )
            if records:
                with open(path, "a") as handle:
                    handle.write(json.dumps({"pairs": records}) + "\n")

    def _should_record_current_step(self, log_stats: bool = False) -> bool:
        """Return whether expert-load recording should be enabled this step.

        Recording is enabled when we are close to either:
        1) The next rearrangement step, so the sliding window is ready.
        2) The next balancedness logging step, when log_stats is enabled.
        """
        steps_remaining = (
            self.expert_rearrangement_step_interval - self.expert_rearrangement_step
        )
        should_record_for_rearrange = steps_remaining <= self.expert_load_window_size

        if not log_stats:
            return should_record_for_rearrange

        log_interval = self.parallel_config.eplb_config.log_balancedness_interval
        steps_until_next_log = (
            log_interval - (self.expert_rearrangement_step % log_interval)
        ) % log_interval
        should_record_for_log = steps_until_next_log <= self.expert_load_window_size
        return should_record_for_rearrange or should_record_for_log

    def _update_layer_should_record(self, log_stats: bool = False) -> None:
        """Update the shared ``should_record_tensor`` for all layers."""
        if self.should_record_tensor is not None:
            self.should_record_tensor.fill_(
                self._should_record_current_step(log_stats=log_stats)
            )

    def _propagate_shared_tensors(
        self,
        model: "MixtureOfExperts",  # type: ignore[name-defined]
        num_unpadded_tokens_tensors: list[torch.Tensor],
    ) -> None:
        """Propagate shared tensors to every :class:`EplbLayerState`.

        Allocates ``should_record_tensor`` on the first call and then
        assigns both it and ``num_unpadded_tokens_tensors`` to every
        MoE layer's :class:`EplbLayerState`.  All layers reference the
        **same** objects so a single update is visible everywhere.

        Must be called after :meth:`model.set_eplb_state` so that each
        layer's ``eplb_state`` is already populated.
        """
        layer_states = [
            layer.eplb_state
            for layer in model.moe_layers
            if hasattr(layer, "eplb_state")
            and isinstance(layer.eplb_state, EplbLayerState)
        ]

        if self.should_record_tensor is None and layer_states:
            self.should_record_tensor = torch.ones(
                (), dtype=torch.bool, device=self.device
            )

        for ls in layer_states:
            if ls is not None:
                ls.should_record_tensor = self.should_record_tensor
                ls.num_unpadded_tokens_tensors = num_unpadded_tokens_tensors

    def rearrange(
        self,
        is_profile: bool = False,
        rank_mapping: dict[int, int] | None = None,
    ) -> torch.Tensor | None:
        """
        Rearrange the experts according to the current load.

        Args:
            is_profile (bool): If `True`, perform a dummy rearrangement.
                This is used in `profile_run` to reserve enough memory,
                no memory movement will be performed. Default is False.
            rank_mapping (dict[int, int] | None): The rank mapping
                when scaling is done in EEP.
        """

        if self.predictive_enabled and not is_profile:
            # Predictive expert replication reuses the leading row of
            # `expert_buffer` as its one-expert staging workspace. That reuse is
            # only safe because nothing else touches the buffer during serving,
            # so a native rearrangement on the request path is an invariant
            # violation rather than a slow path.
            raise RuntimeError(
                "Native expert rearrangement must not run while Predictive "
                "expert replication is enabled: the predictive staging "
                "workspace shares the expert transfer buffer."
            )

        ep_group = get_ep_group().device_group
        ep_rank = ep_group.rank()

        start_event = None
        end_event = None
        is_main_rank = ep_rank == 0
        if is_main_rank:
            if not self.is_async or is_profile:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            logger.info(
                "Rearranging experts %s %s...",
                "(async mode)" if self.is_async else "sync mode",
                "(profile)" if is_profile else "",
            )

        # Map the physical expert load to global logical experts
        global_expert_load_windows = []
        for eplb_model_state in self.model_states.values():
            expert_load_window = eplb_model_state.expert_load_window
            physical_to_logical = eplb_model_state.physical_to_logical_map
            invalid_idx = eplb_model_state.model.num_logical_experts
            logical_expert_load_window = torch.zeros(
                self.expert_load_window_size,
                eplb_model_state.model.num_moe_layers,
                invalid_idx + 1,
                dtype=eplb_model_state.expert_load_window.dtype,
                device=eplb_model_state.expert_load_window.device,
            )
            logical_expert_load_window.scatter_add_(
                dim=-1,
                index=physical_to_logical.masked_fill(
                    physical_to_logical < 0, invalid_idx
                )
                .unsqueeze(0)
                .expand_as(expert_load_window)
                .long(),
                src=expert_load_window,
            )

            global_expert_load_window = logical_expert_load_window[..., :-1].sum(dim=0)
            global_expert_load_windows.append(global_expert_load_window)
        # Perform all-reduce to get the expert load across all ranks for each model
        global_expert_load_windows = self._allreduce_list(global_expert_load_windows)

        # TODO(bowen): Treat differently for prefill and decode nodes
        eplb_model_state = next(iter(self.model_states.values()))
        model = eplb_model_state.model
        num_replicas = model.num_physical_experts
        num_groups = model.num_expert_groups

        if rank_mapping is not None and len(rank_mapping) == ep_group.size():
            # NOTE(yongji): scale down, we need to rebalance the experts on
            # remaining GPUs, transfer the experts while we haven't shutdown
            # the GPUs to be released.
            coordinator = get_ep_group()
            assert isinstance(coordinator, StatelessGroupCoordinator)
            tcp_store_group = coordinator.tcp_store_group
            num_nodes = _node_count_with_rank_mapping(tcp_store_group, rank_mapping)
            num_gpus = sum(new_rank != -1 for new_rank in rank_mapping.values())
            num_replicas = (
                num_replicas // ep_group.size() * num_gpus
            )  # handle num replicas change
        else:
            num_nodes = get_node_count()
            num_gpus = ep_group.size()

        if num_gpus % num_nodes != 0:
            num_nodes = 1
            logger.warning_once(
                f"num_gpus % num_nodes != 0, "
                "not using hierarchical rearrangement algorithm.\n"
                f"{num_gpus=}, {num_nodes=}"
            )

        # Get new expert mappings
        for eplb_model_state, global_expert_load_window in zip(
            self.model_states.values(), global_expert_load_windows
        ):
            if not self.is_async or is_profile:
                # Get new expert mappings for the model. The policy runs on the
                # host, so the load window and current map have to come back.
                with gpu_sync_allowed():
                    new_physical_to_logical_map = self.policy.rebalance_experts(
                        global_expert_load_window.cpu(),
                        num_replicas,
                        num_groups,
                        num_nodes,
                        num_gpus,
                        eplb_model_state.physical_to_logical_map.cpu(),
                    )

                skip_rearrange = False
                if (
                    current_platform.is_rocm()
                    and not is_profile
                    and rank_mapping is None
                    and bool((eplb_model_state.physical_to_logical_map >= 0).all())
                ):
                    logical_loads = global_expert_load_window.float()
                    ep_size = ep_group.size()

                    def rank_load_imbalance(
                        mapping: torch.Tensor,
                        logical_loads: torch.Tensor = logical_loads,
                        ep_size: int = ep_size,
                    ) -> float:
                        mapping = mapping.to(
                            device=logical_loads.device,
                            dtype=torch.long,
                        )
                        replica_counts = torch.zeros_like(logical_loads)
                        replica_counts.scatter_add_(
                            dim=1,
                            index=mapping,
                            src=torch.ones_like(mapping, dtype=logical_loads.dtype),
                        )
                        loads_per_replica = torch.gather(
                            logical_loads / replica_counts.clamp_min(1),
                            dim=1,
                            index=mapping,
                        )
                        loads_per_rank = loads_per_replica.reshape(
                            logical_loads.shape[0], ep_size, -1
                        ).sum(dim=(0, 2))
                        mean_load = loads_per_rank.mean()
                        if mean_load == 0:
                            return 1.0
                        return (loads_per_rank.max() / mean_load).item()

                    current_imbalance = rank_load_imbalance(
                        eplb_model_state.physical_to_logical_map
                    )
                    proposed_imbalance = rank_load_imbalance(
                        new_physical_to_logical_map
                    )
                    relative_improvement = (
                        current_imbalance - proposed_imbalance
                    ) / current_imbalance
                    skip_rearrange = relative_improvement < 0.05
                    if skip_rearrange and is_main_rank:
                        logger.info(
                            "[EPLB] Skip rearrange: imbalance %.4f -> "
                            "%.4f (no material gain)",
                            current_imbalance,
                            proposed_imbalance,
                        )

                if not skip_rearrange:
                    # Update expert weights
                    rearrange_expert_weights_inplace(
                        eplb_model_state.physical_to_logical_map,
                        new_physical_to_logical_map,
                        eplb_model_state.model.expert_weights,
                        eplb_model_state.expert_buffer,
                        ep_group,
                        eplb_model_state.communicator,
                        is_profile,
                        rank_mapping,
                    )

                    if not is_profile:
                        _commit_eplb_maps(
                            eplb_model_state,
                            new_physical_to_logical_map=new_physical_to_logical_map,
                        )

                if is_main_rank:
                    assert start_event is not None
                    assert end_event is not None
                    end_event.record()
                    end_event.synchronize()
                    gpu_elapsed = start_event.elapsed_time(end_event) / 1000.0
                    logger.info(
                        "Rearranged experts %s in %.2f s.",
                        " (profile) " if is_profile else " ",
                        gpu_elapsed,
                    )
            else:
                eplb_model_state.eplb_stats = EplbStats(
                    # We copy the tensor to snapshot the global_expert_load_window
                    # on the main thread so that async worker can access it safely
                    # while the main thread is running.
                    global_expert_load_window=global_expert_load_window.clone(),
                    num_replicas=num_replicas,
                    num_groups=num_groups,
                    num_nodes=num_nodes,
                    num_gpus=num_gpus,
                )
                eplb_model_state.rebalanced = True
        # Signal async thread to start transferring layers
        if self.is_async and (not is_profile):
            self.rearrange_event.record()
        return None

    def start_async_loop(
        self,
        rank_mapping: dict[int, int] | None = None,
        is_profile: bool = False,
    ):
        if not self.is_async:
            return
        if self.async_worker is None:
            self.async_worker = start_async_worker(
                self,
                is_profile=is_profile,
            )

    def drain_async(self) -> None:
        """Drain in-flight async EPLB by consuming all remaining layer results.

        Each pending result is acknowledged (consumed_event recorded) so the
        async worker can proceed, but the transferred weights are intentionally
        NOT applied — a full rearrange is expected to follow.

        Ranks are kept in lockstep via _all_ranks_result_ready (all_reduce
        on the EP CPU group).  The async worker's coordinated-stop collectives
        use the separate EPLB group, so the two sets of collectives do not
        interfere.

        No-op when no async cycle is in progress (rebalanced=False).
        """
        if not self.is_async:
            return
        for model_key, ms in self.model_states.items():
            needs_drain = ms.rebalanced
            if needs_drain:
                logger.info(
                    "Draining async EPLB worker for model %s",
                    model_key,
                )
            while ms.rebalanced:
                if self._all_ranks_result_ready(ms):
                    result = ms.pending_result
                    assert result is not None
                    if result.layer_idx == ms.model.num_moe_layers - 1:
                        ms.rebalanced = False
                    ms.pending_result = None
                    result.consumed_event.record()
                else:
                    time.sleep(0.001)
            if needs_drain:
                logger.info(
                    "Async EPLB worker drained for model %s",
                    model_key,
                )

    def _all_ranks_result_ready(self, model_state: EplbModelState) -> bool:
        parallel_state = get_ep_group()
        has_result = int(model_state.pending_result is not None)

        cpu_group = getattr(parallel_state, "cpu_group", None)
        if cpu_group is not None and cpu_group.size() > 1:
            flag = torch.tensor((has_result,), dtype=torch.int32, device="cpu")
            all_reduce(flag, group=cpu_group)
            return int(flag.item()) == cpu_group.size()

        device_group = parallel_state.device_group
        if device_group.size() <= 1:
            return bool(has_result)

        device = getattr(
            parallel_state, "device", model_state.physical_to_logical_map.device
        )
        flag = torch.tensor((has_result,), dtype=torch.int32, device=device)
        all_reduce(flag, group=device_group)
        return int(flag.item()) == device_group.size()

    def _allreduce_list(self, tensor_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        All-reduce a list of tensors.
        """
        ep_group = get_ep_group().device_group
        if len(tensor_list) == 1:
            all_reduce(tensor_list[0], group=ep_group)
            return tensor_list
        assert all(t.dim() == 2 for t in tensor_list), "All tensors must be 2D."
        assert all(t.shape[1] == tensor_list[0].shape[1] for t in tensor_list), (
            "All tensors must have the same shape[1]."
        )
        # Concatenate, all_reduce, then unpack to original shapes.
        # We assume all tensors are 2D and shape[1] (num_physical_experts)
        # is the same across all models.
        shapes = [t.shape for t in tensor_list]
        concat_tensor = torch.cat(tensor_list, dim=0)

        all_reduce(concat_tensor, group=ep_group)

        all_reduce_list = []
        offset = 0
        for shape in shapes:
            all_reduce_list.append(concat_tensor[offset : offset + shape[0], :])
            offset += shape[0]
        return all_reduce_list

    def _sync_load_pass(self) -> list[torch.Tensor]:
        """
        Sync the expert load pass across all ranks for log stats.
        Doesn't update the expert load pass in eplb_model_state.
        """
        load_pass_list = []
        for eplb_model_state in self.model_states.values():
            load_pass_list.append(eplb_model_state.expert_load_pass.clone())
        return self._allreduce_list(load_pass_list)

    @classmethod
    def from_mapping(
        cls,
        model: MixtureOfExperts,
        model_config: ModelConfig,
        device: torch.device,
        parallel_config: ParallelConfig,
        expanded_physical_to_logical: torch.Tensor,
    ) -> "EplbState":
        eplb_state = cls(
            parallel_config=parallel_config,
            device=device,
        )
        eplb_state.add_model(
            model=model,
            model_config=model_config,
        )
        eplb_state.update_mapping(
            model_config,
            expanded_physical_to_logical,
        )

        return eplb_state

    def update_mapping(
        self,
        model_config: ModelConfig,
        expanded_physical_to_logical: torch.Tensor,
    ) -> None:
        eplb_model_state = self.model_states[model_config.compute_hash()]
        eplb_model_state.physical_to_logical_map.copy_(expanded_physical_to_logical)

        (logical_to_physical_map_cpu, logical_replica_count_cpu) = compute_logical_maps(
            expanded_physical_to_logical.cpu(),
            eplb_model_state.model.num_logical_experts,
        )

        max_num_replicas = eplb_model_state.logical_to_physical_map.shape[-1]
        num_replicas = logical_to_physical_map_cpu.shape[-1]
        logical_to_physical_map = torch.nn.functional.pad(
            logical_to_physical_map_cpu,
            (
                0,
                max_num_replicas - num_replicas,
            ),
            value=-1,
        ).to(self.device)
        logical_replica_count = logical_replica_count_cpu.to(self.device)

        eplb_model_state.logical_to_physical_map.copy_(logical_to_physical_map)
        eplb_model_state.logical_replica_count.copy_(logical_replica_count)

    def create_communicator(
        self, model_config: ModelConfig, group_coordinator: GroupCoordinator
    ) -> EplbCommunicator:
        model_state = self.model_states[model_config.compute_hash()]
        backend = self.parallel_config.eplb_config.communicator
        assert backend is not None
        return create_eplb_communicator(
            group_coordinator,
            backend,
            model_state.model.expert_weights,
            model_state.expert_buffer,
        )

    def update_communicator(
        self,
        model_config: ModelConfig,
        communicator: EplbCommunicator,
    ) -> None:
        self.model_states[model_config.compute_hash()].communicator = communicator


@dataclass
class EplbLayerState:
    """Runtime EPLB data stored in the MoE layer."""

    expert_load_view: torch.Tensor | None = None
    logical_to_physical_map: torch.Tensor | None = None
    logical_replica_count: torch.Tensor | None = None
    should_record_tensor: torch.Tensor | None = None
    """
    Shared scalar bool tensor controlling whether to accumulate expert load
    metrics during this forward pass.  All layers reference the **same**
    tensor object, which is owned and updated by :class:`EplbState`.

    Set to ``False`` for the first ``step_interval - window_size`` steps of
    each rearrangement period: those steps would be overwritten in the
    sliding window before the next rearrangement, so recording them wastes
    GPU work.
    """
    num_unpadded_tokens_tensors: list[torch.Tensor] | None = None
    """
    Reference to the parent :class:`EplbModelState`'s tensor list so the
    router can read the correct per-[u]batch unpadded token count.
    """
    source_local_physical_map: torch.Tensor | None = None
    """
    Source-local physical map for this rank: ``[num_logical_experts, 1]``, the one
    physical row this source rank routes each logical expert to.

    Set only under Predictive expert replication. Its presence is what switches
    routing from the shared path's per-token replica choice to source-rank
    routing: offering a single copy leaves nothing for a per-token decision to
    split, so a source rank's chunk cannot be divided across copies.
    """
    source_local_replica_count: torch.Tensor | None = None
    """
    All-ones counterpart to :attr:`source_local_physical_map`, which pins the
    single offered copy. Kept beside the map so both are swapped together.
    """

    def set_layer_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.expert_load_view = expert_load_view[moe_layer_idx]
        self.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
        self.logical_replica_count = logical_replica_count[moe_layer_idx]


def _node_count_with_rank_mapping(
    pg: ProcessGroup | StatelessProcessGroup,
    rank_mapping: dict[int, int],
) -> int:
    if isinstance(pg, ProcessGroup):
        world_size = torch.distributed.get_world_size(group=pg)
    else:
        world_size = pg.world_size

    if world_size == 1:
        return 1

    # Build node assignment map
    node_assignment = [0] * world_size  # rank -> node_id
    next_node_id = 0

    for current_rank in range(world_size):
        if node_assignment[current_rank] != 0:
            continue  # Already assigned to a node

        assert current_rank in rank_mapping
        if rank_mapping[current_rank] == -1:
            continue  # Pending shutdown

        # Assign current rank to a new node
        next_node_id += 1
        node_assignment[current_rank] = next_node_id

        # Find all ranks on the same node as current_rank
        same_node_flags = in_the_same_node_as(pg, current_rank)
        for other_rank, is_same_node in enumerate(same_node_flags):
            if is_same_node and node_assignment[other_rank] == 0:
                node_assignment[other_rank] = next_node_id

    return next_node_id


def compute_logical_maps(
    physical_to_logical_map: torch.Tensor,
    num_logical_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Derive logical_to_physical_map and logical_replica_count from
    physical_to_logical_map.

    Args:
        physical_to_logical_map: [num_layers, num_physical_experts], logical
            expert index for each physical expert slot
        num_logical_experts: total number of logical experts

    Returns:
        logical_to_physical_map: [num_layers, num_logical_experts, max_replicas],
            physical slots per logical expert; -1 where unused
        logical_replica_count: [num_layers, num_logical_experts], number of
            physical replicas per logical expert
    """
    device = physical_to_logical_map.device
    assert physical_to_logical_map.device.type == "cpu"

    dtype = physical_to_logical_map.dtype

    # If computing maps for a single layer, unsqueeze a single element layer dimension
    per_layer = physical_to_logical_map.dim() == 1
    physical_to_logical_map_view = physical_to_logical_map
    if per_layer:
        physical_to_logical_map_view = physical_to_logical_map.unsqueeze(0)
    assert len(physical_to_logical_map_view.shape) == 2
    num_layers, num_physical = physical_to_logical_map_view.shape

    valid_mask = physical_to_logical_map_view >= 0
    logical_replica_count = torch.zeros(
        num_layers,
        num_logical_experts,
        dtype=dtype,
        device=device,
    )
    logical_replica_count.scatter_add_(
        1,
        physical_to_logical_map_view.clamp(min=0),
        valid_mask.to(dtype),
    )

    max_replicas = int(logical_replica_count.max().item())
    logical_to_physical_map_out = torch.full(
        (num_layers, num_logical_experts, max_replicas),
        -1,
        dtype=dtype,
        device=device,
    )

    running_count = torch.zeros_like(logical_replica_count)
    layer_indices = torch.arange(num_layers, device=device)
    for phys_idx in range(num_physical):
        # Logical expert at physical slot phys_idx for each layer
        logical_expert_ids = physical_to_logical_map_view[:, phys_idx]  # [num_layers]

        # Scale up will set the logical expert ids to -1 for all new physical experts.
        # Only consider "valid" experts when setting up the logical_to_physical map.
        valid_expert_mask = logical_expert_ids >= 0
        if not valid_expert_mask.any():
            continue
        valid_layers = layer_indices[valid_expert_mask]
        valid_experts = logical_expert_ids[valid_expert_mask]

        # Use the current running count as the replica index, then increment it.
        replica_idx = running_count[valid_layers, valid_experts]
        logical_to_physical_map_out[valid_layers, valid_experts, replica_idx] = phys_idx
        running_count[valid_layers, valid_experts] += 1

    # If computing maps for a single layer, squeeze out the extra layer dimension
    # before returning
    if per_layer:
        return logical_to_physical_map_out.squeeze(0), logical_replica_count.squeeze(0)
    return logical_to_physical_map_out, logical_replica_count


def _pad_out_tensor(src: torch.Tensor, dst: torch.Tensor) -> None:
    src_padding = dst.shape[-1] - src.shape[-1]
    assert src_padding >= 0
    new_src = torch.nn.functional.pad(src, (0, src_padding), value=-1)
    # The map is committed from the host once per layer per rearrangement.
    with gpu_sync_allowed():
        dst.copy_(new_src)


def _commit_eplb_maps_for_layer(
    model_state: EplbModelState,
    new_physical_to_logical_map: torch.Tensor,
    layer: int,
) -> None:
    """
    Per-layer version of _commit_eplb_maps that's used by the sync portion of EPLB
    when running async EPLB. Copies all of the new_* maps into model_state. After this
    function completes, the new mappings will become the current mappings and will be
    visible to the model.
    """

    # Commit physical_to_logical_map
    src = new_physical_to_logical_map
    dst = model_state.physical_to_logical_map[layer]
    assert src.shape == dst.shape, (
        "The number of physical experts must stay the same while running Async EPLB. "
        f"Current number of physical experts: {dst.shape[0]}. New number of physical "
        f"experts {src.shape[0]}."
    )
    dst.copy_(src, non_blocking=True)

    num_logical_experts = model_state.logical_to_physical_map.shape[1]
    new_logical, new_replica_count = compute_logical_maps(src, num_logical_experts)
    # Commit logical_to_physical_map
    _pad_out_tensor(src=new_logical, dst=model_state.logical_to_physical_map[layer])

    # Commit logical_replica_count
    src = new_replica_count
    dst = model_state.logical_replica_count[layer]
    assert src.shape == dst.shape
    dst.copy_(src, non_blocking=True)


def _commit_eplb_maps(
    model_state: EplbModelState,
    new_physical_to_logical_map: torch.Tensor,
) -> None:
    """
    Copies all of the new_* maps into model_state. After this function completes,
    the new mappings will become the current mappings and will be visible to the
    model.
    """

    # Commit physical_to_logical_map
    src = new_physical_to_logical_map
    dst = model_state.physical_to_logical_map

    # Rare Case: When the number of physical experts has changed, discard the old
    # physical to logical expert map and use the new one. This only happens when the
    # number of GPUs available to vLLM changes while vLLM is running. Otherwise copy the
    # new map into the old one.
    if src.shape[1] != dst.shape[1]:
        model_state.physical_to_logical_map = src.to(dst.device)
    else:
        dst.copy_(src, non_blocking=True)

    num_logical_experts = model_state.logical_to_physical_map.shape[1]
    new_logical, new_replica_count = compute_logical_maps(src, num_logical_experts)
    # Commit logical_to_physical_map
    _pad_out_tensor(
        src=new_logical,
        dst=model_state.logical_to_physical_map,
    )

    # Commit logical_replica_count
    src = new_replica_count
    dst = model_state.logical_replica_count
    dst.copy_(src, non_blocking=True)


def _move_to_workspace(
    model_state: EplbModelState,
    ep_rank: int,
) -> None:
    result = model_state.pending_result
    assert result is not None
    move_from_buffer(
        expert_weights=model_state.model.expert_weights[result.layer_idx],
        expert_weights_buffers=model_state.expert_buffer,
        transfer_metadata=result.transfer_metadata,
        new_indices=result.new_physical_to_logical_map.numpy(),
        ep_rank=ep_rank,
    )

    _commit_eplb_maps_for_layer(
        model_state,
        new_physical_to_logical_map=result.new_physical_to_logical_map,
        layer=result.layer_idx,
    )

    if result.layer_idx == model_state.model.num_moe_layers - 1:
        model_state.rebalanced = False

    # Reset pending_result before unblocking the async worker
    model_state.pending_result = None
    result.consumed_event.record()
