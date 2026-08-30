# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predictive expert replication: configuration, fixed layout, and prediction.

These are the CPU/single-GPU proofs for vLLM tickets 01 and 02. Distributed
replica transfer and output equivalence belong to the multi-GPU suites.
"""

import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from vllm.config import ParallelConfig, VllmConfig
from vllm.config.parallel import PredictiveExpertReplicationConfig
from vllm.distributed.eplb.eplb_state import (
    EplbLayerState,
    EplbState,
    compute_logical_maps,
)
from vllm.distributed.eplb.predictive import (
    CrossLayerLoadPredictor,
    bind_moe_prediction_targets,
    build_predictive_physical_map,
    build_source_local_physical_map,
)
from vllm.distributed.eplb.predictive_planner import Placement

EP_SIZE = 8
NUM_LOGICAL_EXPERTS = 128
CANONICAL_PER_RANK = NUM_LOGICAL_EXPERTS // EP_SIZE


def _valid_profile(**overrides) -> dict:
    profile = {
        "fingerprint": {
            "model": "Qwen/Qwen3-30B-A3B",
            "dtype": "bfloat16",
            "ep_size": EP_SIZE,
            "num_logical_experts": NUM_LOGICAL_EXPERTS,
            "device_name": "NVIDIA GeForce RTX 5090",
        },
        "expert_compute_us_per_token": 0.4,
        "attention_window_us": 900.0,
        "transfer_latency_us": 30.0,
        "usable_transfer_bandwidth_bytes_per_us": 20_000.0,
    }
    profile.update(overrides)
    return profile


def _write_profile(tmp_path, profile) -> str:
    path = tmp_path / "cost-profile.json"
    path.write_text(profile if isinstance(profile, str) else json.dumps(profile))
    return str(path)


def _parallel_config(**overrides) -> ParallelConfig:
    kwargs = dict(
        tensor_parallel_size=1,
        data_parallel_size=EP_SIZE,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
    )
    kwargs.update(overrides)
    return ParallelConfig(**kwargs)


def _predictive_additional_config(profile_path: str, **overrides) -> dict:
    predictive = {"enabled": True, "cost_profile_path": profile_path}
    predictive.update(overrides)
    return {"predictive_expert_replication": predictive}


def _build(tmp_path, parallel_config=None, profile=None, **predictive_overrides):
    if profile is None:
        profile = _valid_profile()
    return VllmConfig(
        parallel_config=parallel_config or _parallel_config(),
        additional_config=_predictive_additional_config(
            _write_profile(tmp_path, profile), **predictive_overrides
        ),
    )


# --------------------------------------------------------------------------
# Ticket 01: feature boundary and controller exclusivity
# --------------------------------------------------------------------------


def test_predictive_replication_is_off_and_inert_by_default():
    """An existing deployment must not gain EPLB machinery implicitly."""
    config = VllmConfig(parallel_config=_parallel_config())

    assert config.parallel_config.enable_eplb is False
    assert config.parallel_config.predictive_expert_replication_config.enabled is False
    assert config.parallel_config.eplb_config.num_redundant_experts == 0


def test_predictive_replication_provisions_infrastructure_without_native_controller(
    tmp_path,
):
    """Enabling predictive mode must reserve one replica slot per EP rank."""
    config = _build(tmp_path)
    eplb_config = config.parallel_config.eplb_config

    assert config.parallel_config.enable_eplb, "replica infrastructure must be on"
    assert eplb_config.num_redundant_experts == EP_SIZE, "one slot per rank"
    assert eplb_config.use_async is False, "async rebalance is Native EPLB's path"
    assert eplb_config.communicator == "pynccl", "separate EPLB NCCL communicator"


def test_predictive_replication_and_native_eplb_cannot_both_run(tmp_path):
    parallel_config = _parallel_config()
    parallel_config.enable_eplb = True

    with pytest.raises(ValueError, match="mutually exclusive"):
        _build(tmp_path, parallel_config=parallel_config)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"tensor_parallel_size": 2, "data_parallel_size": 4}, "TP=2"),
        # One rank has nowhere to put a replica. Sizes other than the measured 8 are
        # allowed and warned about instead: pinning it to 8 blocked every functional run
        # on a smaller node, and the reason for 8 was that the measurements were taken
        # there, which a warning states better than a rejection.
        ({"data_parallel_size": 1}, "DP=1"),
        ({"pipeline_parallel_size": 2}, "PP=2"),
        ({"enable_expert_parallel": False}, "expert parallelism disabled"),
        (
            {"all2all_backend": "deepep_high_throughput"},
            "all2all_backend=deepep_high_throughput",
        ),
    ],
)
def test_predictive_replication_rejects_out_of_scope_runtime(
    tmp_path, overrides, expected
):
    """Unsupported scope must fail startup rather than imply working support."""
    with pytest.raises(ValueError, match="does not support") as excinfo:
        _build(tmp_path, parallel_config=_parallel_config(**overrides))
    assert expected in str(excinfo.value)


def test_predictive_replication_cannot_be_combined_with_dbo(tmp_path):
    """DBO needs a deepep/nixl backend, so it can never pair with the PoC's."""
    with pytest.raises(ValueError, match="Microbatching|DBO"):
        _build(
            tmp_path,
            parallel_config=_parallel_config(enable_dbo=True, ubatch_size=2),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("replica_slots_per_rank", 2),
        ("hot_stable_steps", 3),
        ("min_residency_steps", 8),
    ],
)
def test_predictive_replication_rejects_unimplemented_policy_values(
    tmp_path, field, value
):
    with pytest.raises(ValueError, match=field):
        _build(tmp_path, **{field: value})


# --------------------------------------------------------------------------
# Ticket 01: cost profile must be valid before the server reports ready
# --------------------------------------------------------------------------


def test_missing_cost_profile_path_fails_startup(tmp_path):
    with pytest.raises(ValueError, match="requires cost_profile_path"):
        VllmConfig(
            parallel_config=_parallel_config(),
            additional_config={"predictive_expert_replication": {"enabled": True}},
        )


def test_unreadable_cost_profile_fails_startup(tmp_path):
    with pytest.raises(ValueError, match="must be readable JSON"):
        VllmConfig(
            parallel_config=_parallel_config(),
            additional_config=_predictive_additional_config(
                str(tmp_path / "absent.json")
            ),
        )


@pytest.mark.parametrize(
    "profile,expected",
    [
        ("[]", "must be a JSON object"),
        ("not json", "must be readable JSON"),
        ({"expert_compute_us_per_token": 1.0}, "`fingerprint` object"),
    ],
)
def test_malformed_cost_profile_fails_startup(tmp_path, profile, expected):
    with pytest.raises(ValueError, match=expected):
        _build(tmp_path, profile=profile)


@pytest.mark.parametrize(
    "dropped",
    ["model", "dtype", "ep_size", "num_logical_experts", "device_name"],
)
def test_cost_profile_without_full_fingerprint_fails_startup(tmp_path, dropped):
    """A profile that does not pin its runtime could be silently reused."""
    profile = _valid_profile()
    del profile["fingerprint"][dropped]

    with pytest.raises(ValueError, match=f"missing \\['{dropped}'\\]"):
        _build(tmp_path, profile=profile)


@pytest.mark.parametrize(
    "key",
    [
        "expert_compute_us_per_token",
        "attention_window_us",
        "transfer_latency_us",
        "usable_transfer_bandwidth_bytes_per_us",
    ],
)
@pytest.mark.parametrize("bad_value", [0, -1.0, "fast", None])
def test_cost_profile_with_unusable_cost_fails_startup(tmp_path, key, bad_value):
    with pytest.raises(ValueError, match=key):
        _build(tmp_path, profile=_valid_profile(**{key: bad_value}))


def test_cost_profile_measured_on_another_runtime_is_rejected(tmp_path):
    """A profile from a different EP size must not be reused silently."""
    config = PredictiveExpertReplicationConfig(
        enabled=True,
        cost_profile_path=_write_profile(tmp_path, _valid_profile()),
    )

    config.validate_fingerprint(ep_size=EP_SIZE, dtype="bfloat16")

    with pytest.raises(ValueError, match="does not match this runtime"):
        config.validate_fingerprint(ep_size=4)
    with pytest.raises(ValueError, match="does not match this runtime"):
        config.validate_fingerprint(dtype="float16")


# --------------------------------------------------------------------------
# Ticket 01: fixed physical layout
# --------------------------------------------------------------------------


def test_fixed_layout_gives_each_rank_canonical_experts_plus_one_inactive_slot():
    layout = build_predictive_physical_map(
        num_layers=2,
        num_logical_experts=NUM_LOGICAL_EXPERTS,
        ep_size=EP_SIZE,
        replica_slots_per_rank=1,
    )
    per_rank = layout.view(2, EP_SIZE, CANONICAL_PER_RANK + 1)

    assert layout.shape == (2, EP_SIZE * (CANONICAL_PER_RANK + 1))
    for rank in range(EP_SIZE):
        expected = torch.arange(
            rank * CANONICAL_PER_RANK, (rank + 1) * CANONICAL_PER_RANK
        )
        assert torch.equal(per_rank[0, rank, :CANONICAL_PER_RANK], expected)
    assert torch.equal(
        per_rank[:, :, -1], torch.full((2, EP_SIZE), -1, dtype=layout.dtype)
    ), "the trailing row of every rank must stay inactive"
    assert torch.equal(per_rank[0], per_rank[1]), "layout is identical per layer"


def test_fixed_layout_routes_every_logical_expert_to_exactly_one_canonical_row():
    """Canonical-only serving: no logical expert has a replica before a plan."""
    layout = build_predictive_physical_map(
        num_layers=1,
        num_logical_experts=NUM_LOGICAL_EXPERTS,
        ep_size=EP_SIZE,
        replica_slots_per_rank=1,
    )

    logical_to_physical, replica_count = compute_logical_maps(
        layout, NUM_LOGICAL_EXPERTS
    )

    assert torch.equal(replica_count, torch.ones_like(replica_count))
    owning_rank = logical_to_physical[0, :, 0] // (CANONICAL_PER_RANK + 1)
    expected_owner = torch.arange(NUM_LOGICAL_EXPERTS) // CANONICAL_PER_RANK
    assert torch.equal(owning_rank, expected_owner), "canonical ownership is fixed"


def test_fixed_layout_rejects_experts_that_do_not_divide_across_ranks():
    with pytest.raises(ValueError, match="divide EP size"):
        build_predictive_physical_map(
            num_layers=1,
            num_logical_experts=130,
            ep_size=EP_SIZE,
            replica_slots_per_rank=1,
        )


# --------------------------------------------------------------------------
# Ticket 02: cross-layer gate registry
# --------------------------------------------------------------------------


class _RecordingRunner:
    """Stands in for a MoERunner to observe only the binding it receives."""

    def __init__(self, name: str):
        self.name = name
        self.bound_target: _RecordingRunner | None = None

    def bind_prediction_target(self, target: "_RecordingRunner") -> None:
        self.bound_target = target

    @property
    def target_name(self) -> str:
        assert self.bound_target is not None, f"{self.name} is unbound"
        return self.bound_target.name


def test_registry_binds_each_source_to_the_layer_lookahead_ahead():
    runners = [_RecordingRunner(f"layer{i}") for i in range(8)]

    sources = bind_moe_prediction_targets(runners, lookahead=2)

    assert sources == [0, 1, 2, 3, 4, 5]
    assert [r.target_name for r in runners[:6]] == [
        "layer2",
        "layer3",
        "layer4",
        "layer5",
        "layer6",
        "layer7",
    ]


def test_registry_leaves_the_trailing_lookahead_layers_unbound():
    """The last `lookahead` layers have no target, so they create no plan."""
    runners = [_RecordingRunner(f"layer{i}") for i in range(8)]

    bind_moe_prediction_targets(runners, lookahead=2)

    assert [r.bound_target for r in runners[6:]] == [None, None]


def test_registry_skips_the_leading_layers_whose_prediction_is_unreliable():
    runners = [_RecordingRunner(f"layer{i}") for i in range(8)]

    sources = bind_moe_prediction_targets(runners, lookahead=2, skip_first_layers=3)

    assert sources == [3, 4, 5]
    assert [r.bound_target for r in runners[:3]] == [None, None, None], (
        "skipped layers must run no prediction at all"
    )
    assert runners[3].target_name == "layer5"


def test_registry_rejects_a_topology_with_a_dense_layer_gap():
    """Predicting across a dense layer would span a different distance."""
    runners = [_RecordingRunner("l0"), None, _RecordingRunner("l2")]

    with pytest.raises(ValueError, match=r"layers \[1\] have none"):
        bind_moe_prediction_targets(runners, lookahead=1)


def test_registry_rejects_a_lookahead_that_leaves_no_source_layer():
    runners = [_RecordingRunner(f"layer{i}") for i in range(4)]

    with pytest.raises(ValueError, match="no source layer"):
        bind_moe_prediction_targets(runners, lookahead=2, skip_first_layers=3)


def test_qwen3_layer_count_supports_the_default_lookahead_and_skip():
    """48 sparse MoE layers, skip 3, lookahead 2 leaves layers 3..45 as sources."""
    runners = [_RecordingRunner(f"layer{i}") for i in range(48)]

    sources = bind_moe_prediction_targets(runners, lookahead=2, skip_first_layers=3)

    assert sources == list(range(3, 46))
    assert runners[45].target_name == "layer47", "the last layer is a target"


# --------------------------------------------------------------------------
# Ticket 02: prediction reads logical experts and stays read-only
# --------------------------------------------------------------------------

TOP_K = 8
PHYSICAL_ROWS = NUM_LOGICAL_EXPERTS + EP_SIZE


def _eplb_layer_state(device: torch.device, num_unpadded: int) -> EplbLayerState:
    """An EPLB layer state whose physical rows are deliberately far from logical.

    Every logical expert maps to physical row `logical + NUM_LOGICAL_EXPERTS % ...`,
    so a predicted count computed from physical ids cannot masquerade as a
    logical one.
    """
    logical_to_physical = (
        (torch.arange(NUM_LOGICAL_EXPERTS, device=device) + NUM_LOGICAL_EXPERTS)
        % PHYSICAL_ROWS
    ).view(NUM_LOGICAL_EXPERTS, 1)
    state = EplbLayerState()
    state.expert_load_view = torch.zeros(
        PHYSICAL_ROWS, dtype=torch.int32, device=device
    )
    state.logical_to_physical_map = logical_to_physical
    state.logical_replica_count = torch.ones(
        NUM_LOGICAL_EXPERTS, dtype=torch.long, device=device
    )
    state.should_record_tensor = torch.tensor(True, device=device)
    state.num_unpadded_tokens_tensors = [
        torch.tensor(num_unpadded, dtype=torch.int32, device=device)
    ]
    return state


def _predictor(device: torch.device, num_unpadded: int, hidden_size: int = 32):
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        FusedTopKRouter,
    )

    eplb_layer_state = _eplb_layer_state(device, num_unpadded)
    target_router = FusedTopKRouter(
        top_k=TOP_K,
        global_num_experts=NUM_LOGICAL_EXPERTS,
        renormalize=True,
        eplb_state=eplb_layer_state,
    )
    gate = torch.nn.Linear(
        hidden_size, NUM_LOGICAL_EXPERTS, bias=False, device=device
    ).to(torch.float32)

    def target_gate(hidden_states):
        return gate(hidden_states), None

    predictor = CrossLayerLoadPredictor(
        target_gate=target_gate,
        target_router=target_router,
        num_logical_experts=NUM_LOGICAL_EXPERTS,
        eplb_layer_state=eplb_layer_state,
    )
    return predictor, target_router, eplb_layer_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU router kernel")
def test_prediction_counts_logical_experts_not_physical_slots():
    """The snapshot must be indexed by logical expert, per the planner contract."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    num_tokens = 16
    predictor, _, _ = _predictor(device, num_unpadded=num_tokens)
    hidden_states = torch.randn(num_tokens, 32, device=device)

    counts = predictor.predict_local_counts(hidden_states)

    assert counts.shape == (NUM_LOGICAL_EXPERTS,), "one entry per logical expert"
    assert counts.sum().item() == num_tokens * TOP_K, "every routed slot counted once"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU router kernel")
def test_prediction_does_not_apply_the_eplb_physical_mapping():
    """Prediction must read the router's logical decision, not a placed one."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    num_tokens = 16
    predictor, router, _ = _predictor(device, num_unpadded=num_tokens)
    hidden_states = torch.randn(num_tokens, 32, device=device)
    logits, _ = predictor.target_gate(hidden_states)

    logical_ids = router.select_logical_experts(hidden_states, logits)
    _, physical_ids = router._select_experts(hidden_states, logits)

    assert logical_ids.max().item() < NUM_LOGICAL_EXPERTS
    assert not torch.equal(logical_ids.to(torch.long), physical_ids.to(torch.long)), (
        "the placed path must differ, otherwise this test proves nothing"
    )
    counts = predictor.predict_local_counts(hidden_states)
    expected = torch.zeros(NUM_LOGICAL_EXPERTS, dtype=counts.dtype, device=device)
    expected.scatter_add_(
        0,
        logical_ids.reshape(-1).to(torch.int64),
        torch.ones_like(logical_ids.reshape(-1), dtype=counts.dtype),
    )
    assert torch.equal(counts, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU router kernel")
def test_prediction_never_records_actual_expert_load():
    """Prediction is read-only: it must not pollute the target layer's metrics."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    predictor, _, eplb_layer_state = _predictor(device, num_unpadded=16)
    hidden_states = torch.randn(16, 32, device=device)

    predictor.predict_local_counts(hidden_states)

    assert eplb_layer_state.expert_load_view.sum().item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU router kernel")
def test_prediction_excludes_padding_tokens():
    """Padded rows carry no request, so they must not shift predicted load."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    num_tokens, num_real = 16, 5
    predictor, _, _ = _predictor(device, num_unpadded=num_real)
    hidden_states = torch.randn(num_tokens, 32, device=device)

    counts = predictor.predict_local_counts(hidden_states)

    assert counts.sum().item() == num_real * TOP_K


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU router kernel")
def test_dummy_forward_predicts_zero_load():
    """A dummy or padding-only forward must leave placement untouched."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    predictor, _, _ = _predictor(device, num_unpadded=0)
    hidden_states = torch.randn(16, 32, device=device)

    counts = predictor.predict_local_counts(hidden_states)

    assert counts.sum().item() == 0
    assert predictor.finish_snapshot() is None, "no snapshot without an AllGather"


def _snapshot_worker(rank: int, world_size: int, tmp_dir: str) -> None:
    """Build a rank-distinguishable predicted load and AllGather it."""
    import torch.distributed as dist

    from vllm.distributed.eplb import predictive as predictive_module

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_dir}/store",
        rank=rank,
        world_size=world_size,
    )
    try:
        group = dist.group.WORLD

        class _FakeEplbGroup:
            device_group = group

        predictive_module.get_eplb_group = lambda: _FakeEplbGroup()

        num_logical = 4
        predictor = CrossLayerLoadPredictor(
            target_gate=lambda h: (h, None),
            target_router=None,
            num_logical_experts=num_logical,
            eplb_layer_state=None,
        )
        # Rank r predicts r + 1 tokens for logical expert r, so provenance is
        # visible in the gathered matrix.
        local_counts = torch.zeros(num_logical, dtype=torch.int32)
        local_counts[rank % num_logical] = rank + 1

        predictor.start_snapshot(local_counts)
        snapshot = predictor.finish_snapshot()

        assert snapshot.shape == (world_size, num_logical)
        for source_rank in range(world_size):
            expected = torch.zeros(num_logical, dtype=torch.int32)
            expected[source_rank % num_logical] = source_rank + 1
            assert torch.equal(snapshot[source_rank], expected), (
                f"rank {rank} saw wrong row for source rank {source_rank}"
            )

        # Every rank must hold a byte-identical snapshot for the deterministic
        # planner to reach the same plan without a broadcast.
        gathered = [torch.zeros_like(snapshot) for _ in range(world_size)]
        dist.all_gather(gathered, snapshot)
        for other in gathered:
            assert torch.equal(other, snapshot)
    finally:
        dist.destroy_process_group()


def test_all_ep_ranks_receive_the_same_global_predicted_load_snapshot(tmp_path):
    """The planner runs locally on every rank, so the snapshot must agree."""
    world_size = 4
    torch.multiprocessing.spawn(
        _snapshot_worker,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )


def test_native_rearrangement_is_refused_on_the_request_path(tmp_path):
    """Predictive staging reuses `expert_buffer`, so nothing else may touch it.

    This is the stated precondition of sharing the transfer buffer as the
    one-expert staging workspace; it must be enforced, not merely intended.
    """
    from vllm.distributed.eplb.eplb_state import EplbState

    config = _build(tmp_path)
    state = EplbState.__new__(EplbState)
    state.parallel_config = config.parallel_config

    assert state.predictive_enabled
    with pytest.raises(RuntimeError, match="must not run while Predictive"):
        EplbState.rearrange(state)


def test_profile_rearrangement_is_still_allowed_in_predictive_mode(tmp_path):
    """Predictive transfers need the same buffers reserved during profiling."""
    from vllm.distributed.eplb.eplb_state import EplbState

    config = _build(tmp_path)
    state = EplbState.__new__(EplbState)
    state.parallel_config = config.parallel_config

    # Reaching past the guard means the profile path is not refused; it then
    # fails on unrelated uninitialised state, which is enough to distinguish.
    with pytest.raises(Exception) as excinfo:
        EplbState.rearrange(state, is_profile=True)
    assert "must not run while Predictive" not in str(excinfo.value)


# --------------------------------------------------------------------------
# Regressions from code review
# --------------------------------------------------------------------------


def _predictive_state():
    """An `EplbState` with only what the predictive recording branch touches."""
    from types import SimpleNamespace

    from vllm.distributed.eplb.eplb_state import EplbState

    state = EplbState.__new__(EplbState)
    state.parallel_config = SimpleNamespace(
        predictive_expert_replication_config=SimpleNamespace(enabled=True)
    )
    state.should_record_tensor = torch.tensor(True)
    state.model_states = {
        "m": SimpleNamespace(expert_load_pass=torch.ones(4, 8), model_name="m")
    }
    return state


def test_predictive_mode_does_not_leave_expert_load_recording_enabled():
    """Recording is not consumed by the predictive controller.

    The predictive early return in `step` precedes the native scheduler that
    maintains the record flag, and that flag is allocated enabled. Leaving it
    would run the per-layer record atomics on every forward for the whole run
    with nothing reading them, inflating the very latency comparison the feature
    is judged by.
    """
    state = _predictive_state()

    state.configure_predictive_recording(log_stats=False, is_dummy=False)

    assert bool(state.should_record_tensor.item()) is False


def test_the_load_accumulator_is_not_cleared_after_the_forward_that_filled_it():
    """`step` runs after the forward, so clearing there erases what was recorded.

    Prediction-accuracy scoring reads this accumulator between forwards, and in
    predictive mode nothing rolls it into the sliding window, so it is the only
    actual-load record there is. Clearing belongs before the forward.
    """
    state = _predictive_state()
    state.model_states["m"].expert_load_pass.fill_(7)

    state.configure_predictive_recording(log_stats=True, is_dummy=False)

    assert state.model_states["m"].expert_load_pass.sum().item() > 0, (
        "the forward's recorded load must survive the post-forward step"
    )


def test_benchmark_mode_can_still_record_expert_load_in_predictive_mode():
    """Ticket 06 scores prediction accuracy against recorded actual load."""
    state = _predictive_state()
    state.should_record_tensor = torch.tensor(False)

    state.configure_predictive_recording(log_stats=True, is_dummy=False)

    assert bool(state.should_record_tensor.item()) is True


def test_cost_profile_device_mismatch_is_rejected(tmp_path):
    """The worker check exists because config time has no device bound."""
    config = PredictiveExpertReplicationConfig(
        enabled=True,
        cost_profile_path=_write_profile(tmp_path, _valid_profile()),
    )

    config.validate_fingerprint(device_name="NVIDIA GeForce RTX 5090")

    with pytest.raises(ValueError, match="does not match this runtime"):
        config.validate_fingerprint(device_name="NVIDIA H100 80GB HBM3")


def test_prediction_ignores_out_of_range_expert_ids_without_a_host_sync():
    """An out-of-range id must contribute nothing, not shift a real expert.

    Validating on the host would synchronize every layer of every forward, which
    is the cost this counting path exists to avoid, so the weight excludes the
    entry and the clamp only keeps the scatter in bounds.
    """
    num_logical, top_k, num_tokens = 8, 2, 3

    class _Router:
        def select_logical_experts(self, hidden_states, router_logits):
            # A dropped slot (-1) and an over-range id, alongside valid ones.
            return torch.tensor([[0, -1], [1, 99], [2, 3]])

    state = EplbLayerState()
    state.num_unpadded_tokens_tensors = [torch.tensor(num_tokens)]
    predictor = CrossLayerLoadPredictor(
        target_gate=lambda h: (h, None),
        target_router=_Router(),
        num_logical_experts=num_logical,
        eplb_layer_state=state,
    )

    counts = predictor.predict_local_counts(torch.zeros(num_tokens, 4))

    expected = torch.zeros(num_logical, dtype=counts.dtype)
    for valid_id in (0, 1, 2, 3):
        expected[valid_id] = 1
    assert torch.equal(counts, expected), "only the four in-range ids may be counted"
    assert counts.sum().item() == 4, f"{num_tokens * top_k - 4} ids were dropped"


# --------------------------------------------------------------------------
# Ticket 03: source-rank routing
# --------------------------------------------------------------------------


def _global_map_with_one_replica(replica_of: int, target_rank: int) -> torch.Tensor:
    """The fixed layout, plus one logical expert copied into a rank's slot."""
    layout = build_predictive_physical_map(
        num_layers=1,
        num_logical_experts=NUM_LOGICAL_EXPERTS,
        ep_size=EP_SIZE,
        replica_slots_per_rank=1,
    )
    local_rows = CANONICAL_PER_RANK + 1
    layout[0, target_rank * local_rows + CANONICAL_PER_RANK] = replica_of
    return layout


class TestSourceLocalPhysicalMap:
    """Each source rank must pick exactly one physical copy per logical expert.

    The map is `[num_logical, 1]` paired with an all-ones replica count, so the
    shared routing path's per-token replica choice degenerates to a plain lookup.
    That makes "a source rank's chunk is never split" true by construction rather
    than something a test has to sample for.
    """

    def _build(self, layout, source_rank):
        logical_to_physical, replica_count = compute_logical_maps(
            layout, NUM_LOGICAL_EXPERTS
        )
        return build_source_local_physical_map(
            logical_to_physical[0], replica_count[0], source_rank
        )

    def test_without_replicas_every_rank_routes_canonically(self):
        """No replica means routing must be byte-identical to today's."""
        layout = build_predictive_physical_map(
            num_layers=1,
            num_logical_experts=NUM_LOGICAL_EXPERTS,
            ep_size=EP_SIZE,
            replica_slots_per_rank=1,
        )
        logical_to_physical, _ = compute_logical_maps(layout, NUM_LOGICAL_EXPERTS)

        for source_rank in range(EP_SIZE):
            source_map, count = self._build(layout, source_rank)
            assert torch.equal(source_map[:, 0], logical_to_physical[0, :, 0])
            assert torch.equal(count, torch.ones_like(count))

    def test_the_map_offers_exactly_one_copy_per_logical_expert(self):
        """One column and an all-ones count is what makes the chunk unsplittable."""
        source_map, count = self._build(_global_map_with_one_replica(20, 5), 3)

        assert source_map.shape == (NUM_LOGICAL_EXPERTS, 1)
        assert torch.equal(count, torch.ones(NUM_LOGICAL_EXPERTS, dtype=count.dtype))

    def test_source_ranks_are_split_across_the_replicated_expert_copies(self):
        """The point of the feature: some source ranks leave the hot owner."""
        layout = _global_map_with_one_replica(replica_of=20, target_rank=5)
        canonical_row = 1 * (CANONICAL_PER_RANK + 1) + 4  # expert 20 lives on rank 1
        replica_row = 5 * (CANONICAL_PER_RANK + 1) + CANONICAL_PER_RANK

        chosen = [self._build(layout, r)[0][20, 0].item() for r in range(EP_SIZE)]

        assert set(chosen) == {canonical_row, replica_row}, (
            "both copies must be used, or replication buys nothing"
        )
        assert chosen.count(canonical_row) == EP_SIZE // 2, "an even split"

    def test_unreplicated_experts_are_untouched_by_a_replica_elsewhere(self):
        """Replicating expert 20 must not move any other expert's routing."""
        layout = _global_map_with_one_replica(replica_of=20, target_rank=5)
        plain, _ = compute_logical_maps(
            build_predictive_physical_map(
                num_layers=1,
                num_logical_experts=NUM_LOGICAL_EXPERTS,
                ep_size=EP_SIZE,
                replica_slots_per_rank=1,
            ),
            NUM_LOGICAL_EXPERTS,
        )

        source_map, _ = self._build(layout, 3)

        others = [e for e in range(NUM_LOGICAL_EXPERTS) if e != 20]
        assert torch.equal(source_map[others, 0], plain[0, others, 0])

    def test_the_same_source_rank_always_gets_the_same_copy(self):
        """Determinism is what lets every rank derive the plan without a broadcast."""
        layout = _global_map_with_one_replica(replica_of=20, target_rank=5)

        first, _ = self._build(layout, 3)
        second, _ = self._build(layout, 3)

        assert torch.equal(first, second)

    def test_an_inactive_slot_is_never_selected(self):
        """An inactive row holds no weights, so routing to it would be wrong."""
        layout = build_predictive_physical_map(
            num_layers=1,
            num_logical_experts=NUM_LOGICAL_EXPERTS,
            ep_size=EP_SIZE,
            replica_slots_per_rank=1,
        )
        inactive_rows = {
            r * (CANONICAL_PER_RANK + 1) + CANONICAL_PER_RANK for r in range(EP_SIZE)
        }

        for source_rank in range(EP_SIZE):
            source_map, _ = self._build(layout, source_rank)
            assert not (set(source_map[:, 0].tolist()) & inactive_rows)


@pytest.mark.parametrize("value", ["20", "20:5:1", "a:5", "-1:5", "20:-1", "", ":"])
def test_malformed_static_replica_placement_fails_startup(tmp_path, value):
    """A silently ignored placement would make a routing test pass for the wrong
    reason, so the format is validated rather than best-effort parsed."""
    with pytest.raises(ValueError, match="static_replica_placement"):
        _build(tmp_path, static_replica_placement=value)


def test_static_replica_placement_is_parsed_for_the_layout_builder(tmp_path):
    config = _build(tmp_path, static_replica_placement="20:5")

    predictive = config.parallel_config.predictive_expert_replication_config
    assert predictive.parsed_static_replica_placement == (20, 5)


def test_no_static_replica_placement_by_default(tmp_path):
    """Normal serving must not carry a validation aid."""
    config = _build(tmp_path)

    predictive = config.parallel_config.predictive_expert_replication_config
    assert predictive.static_replica_placement is None
    assert predictive.parsed_static_replica_placement is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the routing kernel")
class TestRoutingUsesTheSourceLocalMap:
    """Cover the selection in `_apply_eplb_mapping`, which nothing else does.

    The map math is tested in isolation elsewhere, but that leaves the wiring
    unproven: if routing kept reading the global map, tokens would still be split
    across copies per token and every isolated test would stay green.

    DeepSeek-V4 and Kimi-K3 are unaffected by construction rather than by luck.
    They call the shared routing helper directly with their own map tensors, and
    that helper's signature is unchanged; only this selection is new. Predictive
    mode also rejects every architecture but Qwen3 at configuration time.
    """

    NUM_LOGICAL = 8
    LOCAL_ROWS = 3
    EP = 4

    def _router(self, with_source_map: bool):
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
            FusedTopKRouter,
        )

        device = torch.device("cuda")
        num_physical = self.EP * self.LOCAL_ROWS
        # Expert 0 is replicated: canonical row 0 and a replica on rank 2's slot.
        replica_row = 2 * self.LOCAL_ROWS + 2
        logical_to_physical = torch.full(
            (self.NUM_LOGICAL, 2), -1, dtype=torch.long, device=device
        )
        for expert in range(self.NUM_LOGICAL):
            logical_to_physical[expert, 0] = expert
        logical_to_physical[0, 1] = replica_row
        replica_count = torch.ones(self.NUM_LOGICAL, dtype=torch.long, device=device)
        replica_count[0] = 2

        state = EplbLayerState()
        state.expert_load_view = torch.zeros(
            num_physical, dtype=torch.int32, device=device
        )
        state.logical_to_physical_map = logical_to_physical
        state.logical_replica_count = replica_count
        state.should_record_tensor = torch.tensor(False, device=device)
        state.num_unpadded_tokens_tensors = [torch.tensor(64, device=device)]
        if with_source_map:
            source_map, source_count = build_source_local_physical_map(
                logical_to_physical, replica_count, source_rank=1
            )
            state.source_local_physical_map = source_map
            state.source_local_replica_count = source_count
        router = FusedTopKRouter(
            top_k=2, global_num_experts=self.NUM_LOGICAL, eplb_state=state
        )
        return router, replica_row

    def _route_all_to_expert_zero(self, router):
        """Send many tokens to logical expert 0 and see which rows they land on."""
        topk_ids = torch.zeros((32, 2), dtype=torch.int32, device="cuda")
        return router._apply_eplb_mapping(topk_ids)

    def test_without_a_source_local_map_tokens_split_across_copies(self):
        """The pre-existing behaviour, kept for every other caller."""
        router, replica_row = self._router(with_source_map=False)

        physical = self._route_all_to_expert_zero(router)

        assert set(physical.reshape(-1).tolist()) == {0, replica_row}, (
            "the shared path chooses per token, so both copies must appear"
        )

    def test_with_a_source_local_map_one_rank_uses_exactly_one_copy(self):
        """Criterion: a source rank's chunk is never split across copies."""
        router, replica_row = self._router(with_source_map=True)

        physical = self._route_all_to_expert_zero(router)

        chosen = set(physical.reshape(-1).tolist())
        assert len(chosen) == 1, f"chunk was split across {chosen}"
        # source_rank 1 with 2 copies takes replica index 1.
        assert chosen == {replica_row}


def test_a_layer_cannot_replicate_onto_every_rank(tmp_path):
    """One rank must keep the canonical copy, so 8 targets is impossible at EP=8."""
    with pytest.raises(ValueError, match="canonical owner"):
        _build(tmp_path, max_replicas_per_layer=8)


def test_replicas_per_layer_defaults_to_one_so_the_budget_buys_coverage(tmp_path):
    """The cap is the online planner's allocation target, not a safety limit.

    The planner sees one layer at a time, so layers fill to this cap in index order
    until the forward's budget is gone: at cap `k` a budget of `b` reaches `b/k`
    layers and no more. Coverage is what drives benefit, so 1 is the default —
    offline at equal budget it removes 48.0% of critical-path excess against a
    global-ranking oracle's 48.3%, where 2 removes 23.5%.

    An earlier default of 2 came from per-layer concentration, which measured how
    many experts a layer would need to *equalize* it. That is the wrong question
    under a global budget, and it is why this test changed rather than the reasoning
    being reversed.
    """
    config = _build(tmp_path)

    predictive = config.parallel_config.predictive_expert_replication_config
    assert predictive.max_replicas_per_layer == 1
    assert predictive.max_transfers_per_forward >= predictive.max_replicas_per_layer, (
        "a single layer's placements must fit inside the per-forward budget, "
        "or no layer could ever be approved"
    )


# --------------------------------------------------------------------------
# Ticket 06: prediction-accuracy dump
# --------------------------------------------------------------------------


class TestPredictionPairRegistry:
    """The accuracy study needs to know which layer each prediction was about."""

    def test_records_the_source_target_distance(self):
        from vllm.distributed.eplb.predictive import registered_prediction_pairs

        runners = [_RecordingRunner(f"layer{i}") for i in range(8)]
        bind_moe_prediction_targets(runners, lookahead=2, skip_first_layers=3)

        pairs = registered_prediction_pairs()

        assert [(source, target) for source, target, _ in pairs] == [
            (3, 5),
            (4, 6),
            (5, 7),
        ]

    def test_rebinding_replaces_rather_than_accumulates(self):
        """A second bind must not leave the first model's pairs behind.

        Stale pairs would be scored against the wrong layer's load, and because
        the runner objects still exist the error would look like poor accuracy
        rather than a bug.
        """
        from vllm.distributed.eplb.predictive import registered_prediction_pairs

        bind_moe_prediction_targets(
            [_RecordingRunner(f"a{i}") for i in range(8)], lookahead=2
        )
        bind_moe_prediction_targets(
            [_RecordingRunner(f"b{i}") for i in range(5)], lookahead=1
        )

        pairs = registered_prediction_pairs()

        assert [(source, target) for source, target, _ in pairs] == [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ]

    def test_holds_the_source_runner_whose_prediction_is_read(self):
        from vllm.distributed.eplb.predictive import registered_prediction_pairs

        runners = [_RecordingRunner(f"layer{i}") for i in range(6)]
        bind_moe_prediction_targets(runners, lookahead=2)

        pairs = registered_prediction_pairs()

        assert [runner.name for _, _, runner in pairs] == [
            "layer0",
            "layer1",
            "layer2",
            "layer3",
        ]


class TestPredictionAccuracyDump:
    """The dump pairs a prediction with the *target* layer's recorded load.

    Pairing against the source layer instead would compare a layer's gate to its
    own hidden states, which is not a prediction at all and would report near
    perfect accuracy.
    """

    def _state(self, per_logical, pairs):
        from types import SimpleNamespace

        from vllm.distributed.eplb.eplb_state import EplbState

        state = EplbState.__new__(EplbState)
        state.model_states = {"m": SimpleNamespace(model_name="m")}
        state._logical_load_this_forward = lambda _model_state: per_logical
        self._pairs = pairs
        return state

    def _run(self, monkeypatch, state, pairs, path):
        monkeypatch.setattr(
            "vllm.distributed.eplb.eplb_state.registered_prediction_pairs",
            lambda: pairs,
        )
        # These tests exercise pairing, not model discrimination, and the registry
        # is module state that another test in the same process may have filled.
        # None is the documented "unknown, do not discriminate" value.
        monkeypatch.setattr(
            "vllm.distributed.eplb.eplb_state.bound_layer_count", lambda: None
        )
        state._dump_prediction_accuracy(str(path))
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def test_pairs_the_prediction_with_the_target_layers_actual_load(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace

        # Layer 5 is hot on expert 0, layer 3 on expert 1. The source is layer 3
        # and its target is layer 5, so the actual side must be layer 5's row.
        per_logical = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 9.0], [0.0, 0.0], [9.0, 0.0]]
        )
        runner = SimpleNamespace(predicted_load_snapshot=torch.tensor([[7.0, 1.0]]))
        state = self._state(per_logical, [(3, 5, runner)])

        (record,) = self._run(
            monkeypatch, state, [(3, 5, runner)], tmp_path / "d.jsonl"
        )

        (pair,) = record["pairs"]
        assert pair["source"] == 3 and pair["target"] == 5
        assert pair["actual"] == [9.0, 0.0]

    def test_sums_the_prediction_over_source_ranks(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        runner = SimpleNamespace(
            predicted_load_snapshot=torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        )
        state = self._state(torch.tensor([[0.0, 0.0], [5.0, 5.0]]), [(0, 1, runner)])

        (record,) = self._run(
            monkeypatch, state, [(0, 1, runner)], tmp_path / "d.jsonl"
        )

        assert record["pairs"][0]["predicted"] == [4.0, 6.0]

    def test_a_layer_that_did_not_predict_is_omitted_not_written_as_zeros(
        self, monkeypatch, tmp_path
    ):
        """A zero prediction would score as a total miss that never happened."""
        from types import SimpleNamespace

        quiet = SimpleNamespace(predicted_load_snapshot=None)
        active = SimpleNamespace(predicted_load_snapshot=torch.tensor([[1.0, 0.0]]))
        state = self._state(torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]), [])

        (record,) = self._run(
            monkeypatch,
            state,
            [(0, 1, quiet), (1, 2, active)],
            tmp_path / "d.jsonl",
        )

        assert [p["source"] for p in record["pairs"]] == [1]

    def test_writes_nothing_when_no_prediction_is_bound(self, monkeypatch, tmp_path):
        """A native-EPLB run with the variable set must not create a bogus file."""
        state = self._state(torch.tensor([[1.0, 1.0]]), [])

        path = tmp_path / "d.jsonl"
        assert self._run(monkeypatch, state, [], path) == []
        assert not path.exists()


# --------------------------------------------------------------------------
# Ticket 03: an inactive replica slot must attract no routed tokens
# --------------------------------------------------------------------------


class TestInactiveSlotsAttractNoTokens:
    """Routing a token to an inactive slot reads uninitialised expert weights.

    That produces plausible-looking garbage rather than an error, so the check
    fails loudly and reports how many slots it examined: a check that silently
    examined nothing would pass just as quietly as a correct one.
    """

    def _state(self, physical_to_logical, load):
        from types import SimpleNamespace

        from vllm.distributed.eplb.eplb_state import EplbState

        state = EplbState.__new__(EplbState)
        state.model_states = {
            "m": SimpleNamespace(
                model_name="m",
                expert_load_pass=load,
                physical_to_logical_map=physical_to_logical,
            )
        }
        return state

    def test_passes_when_every_inactive_slot_is_idle(self):
        # Two layers, three physical rows, the last of which is inactive.
        mapping = torch.tensor([[0, 1, -1], [0, 1, -1]])
        load = torch.tensor([[5.0, 7.0, 0.0], [4.0, 9.0, 0.0]])
        state = self._state(mapping, load)

        checked = state.verify_inactive_slots_unused(reduce_across_ranks=False)

        assert checked == 2, "one inactive slot per layer must have been examined"

    def test_an_all_zero_load_table_verifies_nothing(self):
        """Recording is off by default, and then every slot looks idle.

        `log_balancedness` defaults to False and is the only thing enabling
        recording in predictive mode, so the table is all zero and every slot
        looks idle. Counting slots would report an authoritative pass from a run
        that observed no tokens. It must not raise either: a dummy step's load is
        zeroed before this runs, and raising would kill serving over a non-problem.
        """
        mapping = torch.tensor([[0, 1, -1], [0, 1, -1]])
        state = self._state(mapping, torch.zeros(2, 3))

        assert state.verify_inactive_slots_unused(reduce_across_ranks=False) == 0

    def test_a_layer_that_routed_nothing_does_not_invalidate_the_run(self):
        """Only a wholly empty table is vacuous; one idle layer is ordinary."""
        mapping = torch.tensor([[0, 1, -1], [0, 1, -1]])
        load = torch.tensor([[5.0, 7.0, 0.0], [0.0, 0.0, 0.0]])
        state = self._state(mapping, load)

        assert state.verify_inactive_slots_unused(reduce_across_ranks=False) == 2

    def test_raises_when_an_inactive_slot_received_tokens(self):
        mapping = torch.tensor([[0, 1, -1], [0, 1, -1]])
        load = torch.tensor([[5.0, 7.0, 0.0], [4.0, 9.0, 3.0]])
        state = self._state(mapping, load)

        with pytest.raises(RuntimeError, match="inactive"):
            state.verify_inactive_slots_unused(reduce_across_ranks=False)

    def test_a_layout_with_no_inactive_slot_verifies_nothing(self):
        """A second model or a native-EPLB map has no inactive slot, and that is fine.

        Raising here would abort serving on the first armed forward because some
        *other* model carries no reserved row, even though the model under test is
        correct.
        """
        mapping = torch.tensor([[0, 1, 2]])
        state = self._state(mapping, torch.tensor([[5.0, 7.0, 1.0]]))

        assert state.verify_inactive_slots_unused(reduce_across_ranks=False) == 0

    def test_an_active_replica_row_is_allowed_to_be_busy(self):
        """A replica installed into a slot is active, and load there is correct."""
        mapping = torch.tensor([[0, 1, 0], [0, 1, -1]])
        load = torch.tensor([[5.0, 7.0, 6.0], [4.0, 9.0, 0.0]])
        state = self._state(mapping, load)

        assert state.verify_inactive_slots_unused(reduce_across_ranks=False) == 1


class TestStepDiagnosticsAreCollectiveSafe:
    """Every per-forward diagnostic performs a collective, so none may be skipped.

    `is_dummy` differs across DP ranks - an idle rank runs a dummy batch to stay in
    lockstep - so gating a collective on it makes some ranks enter and others skip,
    and the engine deadlocks with all eight ranks waiting on shared memory. That is
    exactly what an `is_dummy` guard here once caused.
    """

    def _state(self, calls):
        from vllm.distributed.eplb.eplb_state import EplbState

        state = EplbState.__new__(EplbState)
        state.verify_inactive_slots_unused = lambda *a, **k: (
            calls.append("verify") or 384
        )
        state._dump_logical_expert_load = lambda path: calls.append("load")
        state._dump_prediction_accuracy = lambda path: calls.append("accuracy")
        return state

    def test_the_dispatch_takes_no_per_rank_argument(self):
        """A parameterless signature is what makes divergence impossible."""
        import inspect

        from vllm.distributed.eplb.eplb_state import EplbState

        signature = inspect.signature(EplbState._run_step_diagnostics)
        assert list(signature.parameters) == ["self"]

    def test_the_inactive_slot_check_runs_when_armed(self, monkeypatch):
        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_VERIFY_INACTIVE_SLOTS", True, raising=False
        )
        monkeypatch.setattr("vllm.envs.VLLM_EPLB_DUMP_LOAD_PATH", None, raising=False)
        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_ACCURACY_DUMP_PATH", None, raising=False
        )
        calls: list[str] = []
        self._state(calls)._run_step_diagnostics()
        assert calls == ["verify"]

    def test_nothing_runs_when_no_diagnostic_is_armed(self, monkeypatch):
        """The serving path must pay nothing: these all synchronize with the host."""
        for name in (
            "VLLM_PREDICTIVE_VERIFY_INACTIVE_SLOTS",
            "VLLM_EPLB_DUMP_LOAD_PATH",
            "VLLM_PREDICTIVE_ACCURACY_DUMP_PATH",
        ):
            monkeypatch.setattr(f"vllm.envs.{name}", None, raising=False)
        calls: list[str] = []
        self._state(calls)._run_step_diagnostics()
        assert calls == []

    def test_nothing_is_logged_when_the_check_verified_nothing(self, monkeypatch):
        """A zero count means no traffic was observed; claiming a pass would lie."""
        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_VERIFY_INACTIVE_SLOTS", True, raising=False
        )
        monkeypatch.setattr("vllm.envs.VLLM_EPLB_DUMP_LOAD_PATH", None, raising=False)
        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_ACCURACY_DUMP_PATH", None, raising=False
        )
        state = self._state([])
        state.verify_inactive_slots_unused = lambda *a, **k: 0
        state._run_step_diagnostics()
        assert state._logged_inactive_slot_check is False

    def test_the_slot_count_is_logged_once_not_every_forward(self, monkeypatch):
        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_VERIFY_INACTIVE_SLOTS", True, raising=False
        )
        monkeypatch.setattr("vllm.envs.VLLM_EPLB_DUMP_LOAD_PATH", None, raising=False)
        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_ACCURACY_DUMP_PATH", None, raising=False
        )
        state = self._state([])
        assert state._logged_inactive_slot_check is False
        state._run_step_diagnostics()
        assert state._logged_inactive_slot_check is True


class TestAccuracyDumpDoesNotMixModels:
    """The registry holds one model's runners; another model's load must not be scored.

    Under speculative decoding with a MoE drafter, the drafter's construction
    rebinds the registry, and a dump looping over every registered model would
    write lines pairing one model's predictions against the other's actual load.
    The offline scorer averages them with no way to separate them.
    """

    def _state(self, per_logical_by_model):
        from types import SimpleNamespace

        from vllm.distributed.eplb.eplb_state import EplbState

        state = EplbState.__new__(EplbState)
        state.model_states = {
            name: SimpleNamespace(model_name=name) for name in per_logical_by_model
        }
        state._logical_load_this_forward = lambda ms: per_logical_by_model[
            ms.model_name
        ]
        return state

    def _run(self, monkeypatch, state, pairs, path, bound_layers):
        monkeypatch.setattr(
            "vllm.distributed.eplb.eplb_state.registered_prediction_pairs",
            lambda: pairs,
        )
        monkeypatch.setattr(
            "vllm.distributed.eplb.eplb_state.bound_layer_count",
            lambda: bound_layers,
        )
        state._dump_prediction_accuracy(str(path))
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def test_a_model_with_a_different_layer_count_is_skipped(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace

        runner = SimpleNamespace(predicted_load_snapshot=torch.tensor([[3.0, 1.0]]))
        state = self._state(
            {
                "target": torch.tensor([[0.0, 0.0], [5.0, 1.0], [4.0, 2.0]]),
                "drafter": torch.tensor([[9.0, 1.0], [8.0, 2.0]]),
            }
        )

        records = self._run(
            monkeypatch, state, [(0, 1, runner)], tmp_path / "d.jsonl", bound_layers=3
        )

        assert [r["pairs"][0]["model"] for r in records] == ["target"]

    def test_every_record_names_its_model(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        runner = SimpleNamespace(predicted_load_snapshot=torch.tensor([[3.0, 1.0]]))
        state = self._state({"only": torch.tensor([[0.0, 0.0], [5.0, 1.0]])})

        (record,) = self._run(
            monkeypatch, state, [(0, 1, runner)], tmp_path / "d.jsonl", bound_layers=2
        )

        assert record["pairs"][0]["model"] == "only"


class TestBothLoadViewsComeFromTheReducedTensor:
    """Per-rank and per-logical load must be derived from the *same* reduced tensor.

    Recording happens in the router on each rank's own tokens, so an un-reduced
    `expert_load_pass` is "how one rank's tokens spread across the ranks" — a
    different quantity, smaller by a factor of the EP size. A dump that reduced one
    view and not the other put the two on different scales, which made a replicated
    expert appear to carry 181% of its own rank's load and overstated one domain's
    measured imbalance by 0.30.
    """

    def _state(self, physical_to_logical, reduced):
        from types import SimpleNamespace

        from vllm.distributed.eplb.eplb_state import EplbState

        state = EplbState.__new__(EplbState)
        state.model_states = {
            "m": SimpleNamespace(
                model_name="m",
                physical_to_logical_map=physical_to_logical,
                expert_load_pass=reduced,
                model=SimpleNamespace(
                    num_logical_experts=int(physical_to_logical.max()) + 1
                ),
                num_unpadded_tokens_tensors=None,
            )
        }
        state._reduced_load_this_forward = lambda _ms: reduced
        return state

    def test_the_two_views_sum_to_the_same_total(self, monkeypatch, tmp_path):
        """The invariant that a vacuous check missed.

        Two ranks of two experts each, *both* carrying load — a single-rank forward
        would satisfy this trivially, which is how the original check passed on a
        warmup step while the scales were eight times apart.
        """
        mapping = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
        reduced = torch.tensor([[3.0, 5.0, 7.0, 9.0], [2.0, 4.0, 6.0, 8.0]])
        state = self._state(mapping, reduced)
        monkeypatch.setattr(
            "vllm.distributed.eplb.eplb_state.get_ep_group",
            lambda: type(
                "G",
                (),
                {
                    "device_group": type(
                        "D",
                        (),
                        {
                            "size": staticmethod(lambda: 2),
                            "rank": staticmethod(lambda: 0),
                        },
                    )()
                },
            )(),
        )
        path = tmp_path / "d.jsonl"
        state._dump_logical_expert_load(str(path))

        record = json.loads(path.read_text().splitlines()[0])
        for layer in range(2):
            assert sum(record["rank_load"][layer]) == pytest.approx(
                sum(record["logical_load"][layer])
            ), "the two views are on different scales"

    def test_more_than_one_rank_carries_load_in_the_fixture(self):
        """Guards the guard: the fixture above must not be single-rank."""
        reduced = torch.tensor([[3.0, 5.0, 7.0, 9.0]])
        per_rank = reduced.reshape(1, 2, -1).sum(dim=-1)
        assert (per_rank > 0).sum().item() >= 2

    def test_folding_a_reduced_tensor_performs_no_collective(self):
        """`_logical_from_reduced` is pure, so a caller that reduced does not again."""
        from vllm.distributed.eplb.eplb_state import EplbState

        assert isinstance(EplbState.__dict__["_logical_from_reduced"], staticmethod), (
            "a static method cannot reach a process group"
        )


def _republish(model_state, placements, layer=0, source_rank=1):
    """Drive `_republish_layer` without a process group.

    It takes the layout constants from its caller rather than reaching for
    `get_ep_group()`, which is what makes this callable on CPU.
    """
    state = EplbState.__new__(EplbState)
    state._replica_slot_occupant = {}
    return EplbState._republish_layer(
        state,
        model_state,
        layer=layer,
        source_rank=source_rank,
        placements=placements,
        canonical_per_rank=2,
        replica_slots_per_rank=1,
    )


class TestActivationPublishesWhereRoutingReads:
    """An activated replica must land in the maps `_apply_eplb_mapping` consults.

    That method prefers `source_local_physical_map` over `logical_to_physical_map`
    whenever the former is set, and under this feature it always is. Publishing into
    the latter therefore transfers the replica, describes it correctly, and puts it
    where nothing reads: a measured run activated 131 replicas per forward and
    removed 0.6% of prefill excess against an oracle of 35.1%, at 2.2x the baseline
    TTFT, with physical per-rank load equal to canonical ownership to 0.00%.

    The assertion is on the buffer routing reads, not on a buffer name, so moving the
    write back would fail this even if it still updated something plausible.
    """

    EP_SIZE = 2
    PER_RANK = 2
    SLOTS = 1
    NUM_LOGICAL = 4

    def _model_state(self):
        # Rank-major: rank 0 holds experts 0,1 then a free replica slot; rank 1
        # holds 2,3 then its own. -1 marks a slot no logical expert occupies.
        physical = torch.tensor([[0, 1, -1, 2, 3, -1]], dtype=torch.long)
        layer_state = EplbLayerState()
        layer_state.logical_to_physical_map = torch.tensor(
            [[0, -1], [1, -1], [2, -1], [3, -1]], dtype=torch.long
        )
        layer_state.logical_replica_count = torch.ones(
            self.NUM_LOGICAL, dtype=torch.long
        )
        # Routing reads this pair, so this is what has to change. These are physical
        # rows, not logical ids: with 2 experts and 1 replica row per rank the stride
        # is 3, so experts 2 and 3 sit at rows 3 and 4. Seeding ids here would be a
        # state `publish_source_local_maps` never produces, and the incremental path
        # deliberately touches only the experts a placement names, so it cannot
        # repair a wrong starting point the way a full rebuild would.
        layer_state.source_local_physical_map = torch.tensor(
            [[0], [1], [3], [4]], dtype=torch.long
        )
        layer_state.source_local_replica_count = torch.ones(
            self.NUM_LOGICAL, dtype=torch.long
        )
        model = SimpleNamespace(
            num_logical_experts=self.NUM_LOGICAL,
            moe_layers=[SimpleNamespace(eplb_state=layer_state)],
        )
        return SimpleNamespace(
            model=model,
            physical_to_logical_map=physical,
            logical_to_physical_map=torch.full(
                (1, self.NUM_LOGICAL, 2), -1, dtype=torch.long
            ),
            logical_replica_count=torch.ones((1, self.NUM_LOGICAL), dtype=torch.long),
        ), layer_state

    def test_the_source_local_map_gains_the_replica(self):
        model_state, layer_state = self._model_state()
        # Activate a replica of logical expert 0 in rank 1's replica slot (row 5).
        model_state.physical_to_logical_map[0, 5] = 0
        before = layer_state.source_local_physical_map.clone()

        # Source rank 1 of two copies takes copy 1 % 2 == 1, which is the replica.
        _republish(model_state, placements=[Placement(0, 0, 0, 1, 1.0)])

        after = layer_state.source_local_physical_map
        assert after[0].item() == 5, (
            f"logical expert 0 must route to its replica at physical row 5 for this "
            f"source rank, got {after[0].item()}; the replica was published somewhere "
            f"routing does not read"
        )
        assert not torch.equal(before, after), "the map routing reads did not change"

    def test_an_untouched_expert_keeps_its_canonical_row(self):
        model_state, layer_state = self._model_state()
        model_state.physical_to_logical_map[0, 5] = 0

        _republish(model_state, placements=[Placement(0, 0, 0, 1, 1.0)])

        rows = layer_state.source_local_physical_map.reshape(-1).tolist()
        # Physical rows, not logical ids: row 2 is rank 0's empty replica slot, so
        # experts 2 and 3 sit at rows 3 and 4. Asserting [1, 2, 3] here would be
        # asserting the ids and would pass only by coincidence on rank 0.
        assert rows[1:] == [1, 3, 4], (
            f"only expert 0 was replicated, so the rest must keep canonical rows, "
            f"got {rows}"
        )

    def test_missing_source_local_buffers_are_refused_loudly(self):
        model_state, layer_state = self._model_state()
        layer_state.source_local_physical_map = None

        with pytest.raises(RuntimeError, match="source-local routing maps"):
            _republish(model_state, placements=[Placement(0, 0, 0, 1, 1.0)])


class TestPredictionIsSkippedWhenNoPlacementCanFollow:
    """Predicting a forward the placement gate will reject is wasted work.

    Below one block per expert the MoE kernel pads every expert to the same block
    count, so no placement can save anything. The prediction still costs a gate matmul
    and a small AllGather per layer, measured at 7.3% of TPOT on a run where the
    placement gate rejected every forward — the CONC=1 control, which placed nothing
    and still cost that.

    The count must be one every rank agrees on: `start_snapshot` performs an AllGather,
    so a per-rank decision would hang the engine, which is the failure this branch has
    hit twice. `num_tokens_across_dp_cpu` is the DP coordination all-reduce's result,
    identical on every rank and already on the host.
    """

    def _runner(
        self, tokens_across_dp, threshold=128.0, experts_per_token=8, logical=128
    ):
        runner = SimpleNamespace(
            prediction_min_tokens_per_expert=threshold,
            moe_config=SimpleNamespace(
                experts_per_token=experts_per_token, num_logical_experts=logical
            ),
        )
        # The token count lives in its own method because the placement path needs the
        # number itself, not just the boolean: the coordinator must be told the load to
        # decide suppression before anything is recorded. Bound here so this fixture
        # keeps exercising the real arithmetic rather than a stub of it.
        mod = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.runner.moe_runner"
        )
        runner._forward_tokens_per_expert = lambda: (
            mod.MoERunner._forward_tokens_per_expert(runner)
        )
        dp = (
            None
            if tokens_across_dp is None
            else SimpleNamespace(
                num_tokens_across_dp_cpu=torch.tensor(tokens_across_dp)
            )
        )
        return runner, SimpleNamespace(dp_metadata=dp)

    def _call(self, runner, context):
        from vllm.model_executor.layers.fused_moe.runner import moe_runner as mod

        original = mod.get_forward_context
        mod.get_forward_context = lambda: context
        try:
            return mod.MoERunner._prediction_is_worth_it(runner)
        finally:
            mod.get_forward_context = original

    def test_a_decode_sized_forward_is_not_predicted(self):
        # 8 ranks x 64 tokens = 512; 512 * 8 / 128 = 32 per expert, under a block.
        runner, context = self._runner([64] * 8)
        assert not self._call(runner, context)

    def test_a_prefill_sized_forward_is_predicted(self):
        # 8 x 2048 = 16384; 16384 * 8 / 128 = 1024 per expert.
        runner, context = self._runner([2048] * 8)
        assert self._call(runner, context)

    def test_the_count_is_summed_across_dp_ranks_not_taken_locally(self):
        """M is the post-allgather count. Reading one rank's share put a real
        measurement at one block per expert instead of eight."""
        runner, context = self._runner([300] * 8)
        # One rank's 300 would be 18 per expert and gate off; the total 2400 is 150.
        assert self._call(runner, context), (
            "the threshold must be compared against the summed token count"
        )

    def test_a_threshold_of_zero_predicts_everything(self):
        runner, context = self._runner([1] * 8, threshold=0.0)
        assert self._call(runner, context)

    def test_missing_dp_metadata_still_predicts(self):
        """The diagnostic paths run without DP metadata; they must not go silent."""
        runner, context = self._runner(None)
        assert self._call(runner, context)


def test_lookahead_of_one_is_rejected_until_the_launch_point_moves(tmp_path):
    """A lookahead of 1 currently gives an overlap window of zero, not one Attention.

    `plan_and_launch` and `activate_and_publish` are adjacent statements at the head of
    the
    MoE forward, so at lookahead 1 the transfer aimed at *this* layer is issued on one
    line
    and awaited on the next, with no compute in between. The value reads like the
    obvious
    choice — it is the shortest prediction distance and the most accurate — which is
    exactly
    why it has to fail loudly rather than quietly cost the full transfer. Ticket 07
    lifts
    this once the launch moves to the predicting layer's MoE tail.
    """
    with pytest.raises(ValueError, match="overlap window is zero"):
        _build(tmp_path, prediction_lookahead_layers=1)


def test_lookahead_of_two_remains_accepted(tmp_path):
    """The guard must reject one value, not the feature."""
    config = _build(tmp_path, prediction_lookahead_layers=2)

    predictive = config.parallel_config.predictive_expert_replication_config
    assert predictive.prediction_lookahead_layers == 2


class TestTheBlockSizeComesFromTheKernel:
    """`BLOCK_SIZE_M` is asked of the kernel, not assumed.

    The same number gates suppression and floors the planner's minimum move, so a
    wrong value either reopens the decode regime settled negative, or rejects
    placements that would have paid. It is also not a constant: it is selected per `M`
    from the tuned configuration for an expert geometry, and the tuned files disagree
    across devices. Tested against `try_get_optimal_moe_config` itself rather than a
    hardcoded expectation, for the same reason the counting kernel is tested against its
    reference: a number invented here could be wrong in the way I happened to imagine.
    """

    def _model(self, num_rows=17, inter=768, hidden=2048, top_k=8):
        """A model shaped like the real one, which is where this went wrong.

        `expert_weights` are the **flattened** `[rows, numel]` views EPLB registers, not
        the `[E, 2N, K]` and `[E, K, N]` tensors the kernel holds. The earlier fake here
        supplied the three-dimensional ones, so the resolver passed them straight to
        `try_get_optimal_moe_config` and this test agreed with it — while every real
        server raised "not enough values to unpack" and took the fallback in silence.
        """
        return SimpleNamespace(
            expert_weights=[
                [
                    torch.zeros(num_rows, 2 * inter * hidden),
                    torch.zeros(num_rows, hidden * inter),
                ]
            ],
            moe_layers=[
                SimpleNamespace(
                    moe_config=SimpleNamespace(
                        experts_per_token=top_k,
                        num_local_experts=num_rows,
                        intermediate_size_per_partition=inter,
                        hidden_dim=hidden,
                    )
                )
            ],
        )

    @pytest.mark.parametrize("m", [8, 512, 8192, 65536])
    def test_it_agrees_with_the_kernel_at_every_batch_size(self, m):
        from vllm.distributed.eplb.eplb_state import resolve_moe_block_size_m
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            try_get_optimal_moe_config,
        )

        model = self._model()
        expected = try_get_optimal_moe_config(
            w1_shape=(17, 2 * 768, 2048),
            w2_shape=(17, 2048, 768),
            top_k=8,
            dtype=None,
            M=m,
        )["BLOCK_SIZE_M"]

        got = resolve_moe_block_size_m(model, top_k=8, num_batched_tokens=m)

        assert got == expected

    def test_a_model_it_cannot_read_falls_back_rather_than_raising(self):
        """Startup must not die because the config could not be consulted.

        The fallback is logged, because the mode it replaces is silently guessing.
        """
        from vllm.distributed.eplb.eplb_state import (
            _MOE_BLOCK_SIZE_M_FALLBACK,
            resolve_moe_block_size_m,
        )

        broken = SimpleNamespace(expert_weights=[])

        assert (
            resolve_moe_block_size_m(broken, top_k=8, num_batched_tokens=4096)
            == _MOE_BLOCK_SIZE_M_FALLBACK
        )
