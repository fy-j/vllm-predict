# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ticket 00 baseline analysis logic.

These live beside the harness rather than in `tests/` because this is PoC
measurement tooling, not product code.
"""

import math

import pytest
from analyze import (
    QWEN3_30B_A3B,
    build_cost_profile,
    expert_counts_to_rank_load,
    headroom_fraction,
    min_residency_steps,
    moe_boundness_ratio,
    reachable_decode_concurrency,
    recommend,
)


class TestExpertCountsToRankLoad:
    def test_perfectly_balanced_counts_give_no_imbalance(self):
        counts = [8] * 128

        load = expert_counts_to_rank_load(counts, ep_size=8)

        assert load == [128] * 8, "16 experts per rank, 8 tokens each"

    def test_one_hot_expert_concentrates_on_its_canonical_owner(self):
        """Expert 20 lives on rank 1, so rank 1 carries the excess."""
        counts = [1] * 128
        counts[20] = 101

        load = expert_counts_to_rank_load(counts, ep_size=8)

        assert load[1] == 16 + 100
        assert load[0] == 16
        assert sum(load) == sum(counts)

    def test_rejects_experts_that_do_not_divide_across_ranks(self):
        with pytest.raises(ValueError, match="divide"):
            expert_counts_to_rank_load([1] * 130, ep_size=8)


class TestHeadroomFraction:
    def test_balanced_load_has_no_headroom(self):
        assert headroom_fraction([100] * 8) == 0.0

    def test_headroom_is_the_share_of_peak_above_the_mean(self):
        # peak 200, mean 116.25 -> (200-116.25)/200
        load = [200, 100, 100, 100, 100, 100, 100, 130]
        assert headroom_fraction(load) == pytest.approx(1 - (sum(load) / 8) / 200)

    def test_a_single_loaded_rank_approaches_full_headroom(self):
        load = [800] + [0] * 7
        assert headroom_fraction(load) == pytest.approx(0.875)

    def test_idle_model_has_no_headroom_rather_than_dividing_by_zero(self):
        assert headroom_fraction([0] * 8) == 0.0


class TestMoeBoundnessRatio:
    def test_ratio_at_the_weight_read_floor_is_one(self):
        assert moe_boundness_ratio(measured_us=84.0, weight_read_floor_us=84.0) == 1.0

    def test_compute_bound_execution_exceeds_one(self):
        assert moe_boundness_ratio(191.0, 84.0) == pytest.approx(2.274, abs=1e-3)

    def test_a_ratio_near_one_means_balance_cannot_convert_to_time(self):
        """Guards the interpretation, which is the point of reporting it."""
        assert moe_boundness_ratio(88.0, 84.0) < 1.1


class TestMinResidencySteps:
    def test_a_replica_that_gains_nothing_can_never_amortize(self):
        assert min_residency_steps(exposed_us=150.0, gain_per_step_us=0.0) == math.inf

    def test_residency_is_the_steps_needed_to_cover_the_exposed_transfer(self):
        assert min_residency_steps(150.0, 20.0) == 8, "ceil(150/20)"

    def test_a_gain_larger_than_the_transfer_amortizes_in_one_step(self):
        assert min_residency_steps(150.0, 200.0) == 1


class TestReachableDecodeConcurrency:
    def test_concurrency_is_kv_capacity_divided_by_context_length(self):
        kv_bytes = 96 * 1024 * 1000  # 1000 tokens' worth
        assert reachable_decode_concurrency(kv_bytes, 96 * 1024, 100) == 10

    def test_qwen3_kv_per_token_matches_its_grouped_query_shape(self):
        # 48 layers * 4 kv heads * 128 head dim * (K+V) * 2 bytes
        assert QWEN3_30B_A3B.kv_bytes_per_token == 96 * 1024


class TestRecommend:
    def test_no_measurable_imbalance_recommends_stopping(self):
        verdict = recommend(
            headroom_fraction=0.01,
            moe_ratio=1.6,
            exposed_transfer_us=150.0,
            peak_moe_us=140.0,
        )

        assert verdict.proceed is False
        assert "headroom" in verdict.reason.lower()

    def test_large_imbalance_that_pays_for_the_transfer_recommends_proceeding(self):
        verdict = recommend(
            headroom_fraction=0.30,
            moe_ratio=2.3,
            exposed_transfer_us=150.0,
            peak_moe_us=191.0,
        )

        assert verdict.proceed is True
        assert verdict.min_residency_steps < 10

    def test_a_weight_bound_operating_point_is_called_out_not_hidden(self):
        """A null result here must be attributable to the operating point."""
        verdict = recommend(
            headroom_fraction=0.30,
            moe_ratio=1.05,
            exposed_transfer_us=150.0,
            peak_moe_us=88.0,
        )

        assert verdict.proceed is False
        assert "weight" in verdict.reason.lower()


class TestBuildCostProfile:
    def test_profile_is_accepted_by_the_running_configuration_validator(self):
        """The profile this harness emits must satisfy the feature's own schema."""
        from vllm.config.parallel import PredictiveExpertReplicationConfig

        profile = build_cost_profile(
            model="Qwen/Qwen3-30B-A3B",
            dtype="bfloat16",
            ep_size=8,
            num_logical_experts=128,
            device_name="NVIDIA GeForce RTX 5090",
            expert_compute_us_per_token=0.4,
            attention_window_us=28.0,
            transfer_latency_us=31.8,
            usable_transfer_bandwidth_bytes_per_us=40_000.0,
        )

        # Constructed through the public path, so a schema drift fails here.
        config = PredictiveExpertReplicationConfig(
            enabled=True, cost_profile_path=_write_tmp(profile)
        )

        assert config.cost_profile["fingerprint"]["ep_size"] == 8
        config.validate_fingerprint(ep_size=8, dtype="bfloat16")

    def test_a_profile_missing_a_measured_cost_is_refused(self):
        """The harness must not emit a profile the server would reject."""
        from vllm.config.parallel import PredictiveExpertReplicationConfig

        profile = build_cost_profile(
            model="Qwen/Qwen3-30B-A3B",
            dtype="bfloat16",
            ep_size=8,
            num_logical_experts=128,
            device_name="NVIDIA GeForce RTX 5090",
            expert_compute_us_per_token=0.4,
            attention_window_us=28.0,
            transfer_latency_us=31.8,
            usable_transfer_bandwidth_bytes_per_us=40_000.0,
        )
        del profile["attention_window_us"]

        with pytest.raises(ValueError, match="attention_window_us"):
            PredictiveExpertReplicationConfig(
                enabled=True, cost_profile_path=_write_tmp(profile)
            )


def _write_tmp(profile: dict) -> str:
    import json
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".json")
    with open(fd, "w") as handle:
        json.dump(profile, handle)
    return path


class TestIsUsableCase:
    """`vllm bench serve` is a client; an unreachable server yields zeros, not an
    error. A zero TPOT would read as a win, so it must never reach the report."""

    def _payload(self, **overrides):
        payload = {"completed": 100, "num_prompts": 100, "mean_tpot_ms": 80.0}
        payload.update(overrides)
        return payload

    def test_a_complete_run_is_usable(self):
        from report import is_usable_case

        usable, reason = is_usable_case(self._payload())

        assert usable is True
        assert reason == ""

    def test_zero_tpot_is_rejected_as_an_unreachable_server(self):
        from report import is_usable_case

        usable, reason = is_usable_case(self._payload(mean_tpot_ms=0.0))

        assert usable is False
        assert "unreachable" in reason

    def test_a_run_with_no_completed_requests_is_rejected(self):
        from report import is_usable_case

        usable, reason = is_usable_case(self._payload(completed=0))

        assert usable is False
        assert "no requests completed" in reason

    def test_a_truncated_run_is_rejected(self):
        """A run killed part way through under-reports latency."""
        from report import is_usable_case

        usable, reason = is_usable_case(self._payload(completed=15, num_prompts=512))

        assert usable is False
        assert "truncated" in reason and "15/512" in reason

    def test_request_failures_are_reported_as_failures_not_truncation(self):
        """They point at different causes: a shared or overloaded server versus a
        killed client. Naming the wrong one sends the next reader off course."""
        from report import is_usable_case

        usable, reason = is_usable_case(
            self._payload(completed=77, failed=43, num_prompts=120)
        )

        assert usable is False
        assert "43 requests failed" in reason
        assert "truncated" not in reason
