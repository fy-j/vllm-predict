# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed.eplb.eplb_state import compute_logical_maps
from vllm.v1.worker.gpu.eplb_utils import EPLBController


def _predictive_config(profile_path: str) -> dict:
    return {
        "predictive_expert_replication": {
            "enabled": True,
            "cost_profile_path": profile_path,
        }
    }


def _parallel_config() -> ParallelConfig:
    return ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=8,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
    )


def test_predictive_config_is_disabled_by_default():
    config = VllmConfig(parallel_config=_parallel_config())

    assert config.parallel_config.enable_eplb is False
    assert config.parallel_config.predictive_expert_replication_config.enabled is False


def test_predictive_config_provisions_infrastructure(tmp_path):
    profile = tmp_path / "cost-profile.json"
    profile.write_text(json.dumps({"version": 1}))
    config = VllmConfig(
        parallel_config=_parallel_config(),
        additional_config=_predictive_config(str(profile)),
    )

    assert config.parallel_config.enable_eplb
    assert config.parallel_config.eplb_config.num_redundant_experts == 8
    assert config.parallel_config.eplb_config.use_async is False
    assert config.parallel_config.eplb_config.communicator == "pynccl"


def test_predictive_config_rejects_native_eplb(tmp_path):
    profile = tmp_path / "cost-profile.json"
    profile.write_text("{}")
    parallel_config = _parallel_config()
    parallel_config.enable_eplb = True

    with pytest.raises(ValueError, match="mutually exclusive"):
        VllmConfig(
            parallel_config=parallel_config,
            additional_config=_predictive_config(str(profile)),
        )


@pytest.mark.parametrize("profile", ["", "[]"])
def test_predictive_config_rejects_invalid_cost_profile(tmp_path, profile):
    profile_path = tmp_path / "cost-profile.json"
    if profile:
        profile_path.write_text(profile)

    with pytest.raises(ValueError, match="cost profile"):
        VllmConfig(
            parallel_config=_parallel_config(),
            additional_config=_predictive_config(str(profile_path)),
        )


def test_fixed_layout_has_one_inactive_slot_per_rank():
    ep_size = 8
    canonical_per_rank = 16
    physical_per_rank = canonical_per_rank + 1
    layout = torch.full((1, ep_size * physical_per_rank), -1, dtype=torch.long)
    for rank in range(ep_size):
        start = rank * physical_per_rank
        layout[0, start : start + canonical_per_rank] = torch.arange(
            rank * canonical_per_rank, (rank + 1) * canonical_per_rank
        )

    logical_to_physical, replica_count = compute_logical_maps(layout, 128)

    assert torch.equal(replica_count, torch.ones_like(replica_count))
    expected_physical_rows = torch.cat(
        [
            torch.arange(
                rank * physical_per_rank,
                rank * physical_per_rank + canonical_per_rank,
            )
            for rank in range(ep_size)
        ]
    )
    assert torch.equal(logical_to_physical[0, :, 0], expected_physical_rows)
    assert torch.equal(
        layout.view(ep_size, physical_per_rank)[:, -1],
        torch.full((ep_size,), -1, dtype=torch.long),
    )


def test_predictive_mode_does_not_step_native_eplb():
    parallel_config = SimpleNamespace(
        enable_eplb=True,
        predictive_expert_replication_config=SimpleNamespace(enabled=True),
    )
    controller = EPLBController(parallel_config, torch.device("cpu"))
    controller.state = type(
        "State",
        (),
        {"step": lambda self, *args, **kwargs: pytest.fail("native step ran")},
    )()
    controller._has_registered_models = True

    controller.step()
