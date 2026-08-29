# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the decode-only profile attribution in `parse_profile`.

The bug these guard against is real and was hit twice: attributing GPU time
across a window that mixes prefill with decode, and dividing by a step count
inferred from a kernel that fires more than once per layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from parse_profile import (  # noqa: E402
    KERNEL_CLASSES,
    attribute_windows,
    classify_kernel,
    decode_windows,
    parse_annotation,
    summarize,
)


class TestParseAnnotation:
    """The annotation is the only reliable source of a step's batch shape."""

    def test_reads_all_four_counts(self):
        got = parse_annotation("execute_context_3(2047)_generation_1(1)")
        assert got == (3, 2047, 1, 1)

    def test_pure_decode_has_zero_context(self):
        assert parse_annotation("execute_context_0(0)_generation_8(8)") == (0, 0, 8, 8)

    def test_unrelated_name_is_not_an_annotation(self):
        assert parse_annotation("ncclDevKernel_AllGather_RING_LL") is None


class TestDecodeWindows:
    """A window that carries any prefill token must be excluded.

    A 2047-token prefill costs ~300 ms against a ~136 ms decode step, so one
    leaked prefill window doubles the per-step attribution.
    """

    def _event(self, name, ts, dur, cat="gpu_user_annotation"):
        return {"name": name, "ts": ts, "dur": dur, "cat": cat}

    def test_keeps_only_context_free_windows(self):
        events = [
            self._event("execute_context_0(0)_generation_8(8)", 100.0, 10.0),
            self._event("execute_context_3(2047)_generation_1(1)", 200.0, 300.0),
            self._event("execute_context_0(0)_generation_8(8)", 600.0, 12.0),
        ]
        windows = decode_windows(events)
        assert [w.gen_tokens for w in windows] == [8, 8]
        assert [w.start for w in windows] == [100.0, 600.0]

    def test_ignores_the_cpu_side_annotation(self):
        """Both a CPU and a GPU annotation exist per step; counting both doubles it."""
        events = [
            self._event("execute_context_0(0)_generation_8(8)", 100.0, 10.0),
            self._event(
                "execute_context_0(0)_generation_8(8)",
                90.0,
                30.0,
                cat="user_annotation",
            ),
        ]
        assert len(decode_windows(events)) == 1

    def test_no_decode_window_is_reported_not_silently_zero(self):
        events = [self._event("execute_context_3(2047)_generation_1(1)", 0.0, 5.0)]
        assert decode_windows(events) == []


class TestClassifyKernel:
    def test_nccl_collectives(self):
        assert classify_kernel("ncclDevKernel_AllGather_RING_LL(args)") == "nccl"
        assert classify_kernel("ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL") == "nccl"

    def test_expert_gemm(self):
        assert classify_kernel("fused_moe_kernel") == "moe_expert"

    def test_attention(self):
        name = "void flash::flash_fwd_splitkv_kernel<...>"
        assert classify_kernel(name) == "attention"

    def test_anything_else_is_other(self):
        assert classify_kernel("void vllm::rms_norm_kernel<...>") == "other"

    def test_every_class_is_declared(self):
        assert set(KERNEL_CLASSES) == {"nccl", "moe_expert", "attention", "other"}


class TestAttributeWindows:
    """Kernels are bucketed by the window their start falls inside."""

    def _win(self, ts, dur, gen=8):
        return {
            "name": f"execute_context_0(0)_generation_{gen}({gen})",
            "ts": ts,
            "dur": dur,
            "cat": "gpu_user_annotation",
        }

    def _k(self, name, ts, dur):
        return {"name": name, "ts": ts, "dur": dur, "cat": "kernel"}

    def test_sums_by_class_inside_the_window(self):
        windows = decode_windows([self._win(100.0, 50.0)])
        kernels = [
            self._k("fused_moe_kernel", 110.0, 5.0),
            self._k("fused_moe_kernel", 120.0, 7.0),
            self._k("ncclDevKernel_AllGather_RING_LL", 130.0, 20.0),
        ]
        (row,) = attribute_windows(windows, kernels)
        assert row.by_class["moe_expert"] == pytest.approx(12.0)
        assert row.by_class["nccl"] == pytest.approx(20.0)
        assert row.gen_tokens == 8

    def test_drops_kernels_outside_every_decode_window(self):
        """Prefill kernels sit outside the decode windows and must not be counted."""
        windows = decode_windows([self._win(100.0, 50.0)])
        kernels = [
            self._k("fused_moe_kernel", 10.0, 999.0),  # prefill, before the window
            self._k("fused_moe_kernel", 110.0, 5.0),
        ]
        (row,) = attribute_windows(windows, kernels)
        assert row.by_class["moe_expert"] == pytest.approx(5.0)

    def test_kernel_on_the_window_boundary_belongs_to_that_window(self):
        windows = decode_windows([self._win(100.0, 50.0), self._win(200.0, 50.0)])
        kernels = [self._k("fused_moe_kernel", 200.0, 3.0)]
        rows = attribute_windows(windows, kernels)
        assert rows[0].by_class["moe_expert"] == 0.0
        assert rows[1].by_class["moe_expert"] == pytest.approx(3.0)


class TestSummarize:
    """Per-step means, and the per-layer figure the boundness question needs."""

    def _rows(self, moe_us, nccl_us, n=4, gen=8):
        windows = [
            {
                "name": f"execute_context_0(0)_generation_{gen}({gen})",
                "ts": 1000.0 * i,
                "dur": 900.0,
                "cat": "gpu_user_annotation",
            }
            for i in range(n)
        ]
        kernels = []
        for i in range(n):
            kernels.append(
                {
                    "name": "fused_moe_kernel",
                    "ts": 1000.0 * i + 1,
                    "dur": moe_us,
                    "cat": "kernel",
                }
            )
            kernels.append(
                {
                    "name": "ncclDevKernel_AllGather_RING_LL",
                    "ts": 1000.0 * i + 2,
                    "dur": nccl_us,
                    "cat": "kernel",
                }
            )
        return attribute_windows(decode_windows(windows), kernels)

    def test_per_step_mean_uses_the_decode_step_count(self):
        got = summarize(self._rows(moe_us=480.0, nccl_us=5600.0, n=4), num_layers=48)
        assert got["decode_steps"] == 4
        assert got["per_step_ms"]["moe_expert"] == pytest.approx(0.48)
        assert got["per_step_ms"]["nccl"] == pytest.approx(5.6)

    def test_moe_per_layer_divides_by_layers_not_by_kernel_calls(self):
        """`fused_moe_kernel` fires twice per layer (w13 then w2).

        Dividing the MoE total by the call count reports half the per-layer cost,
        which is exactly the error this figure existed to avoid.
        """
        got = summarize(self._rows(moe_us=4800.0, nccl_us=0.0, n=1), num_layers=48)
        assert got["moe_us_per_layer"] == pytest.approx(100.0)

    def test_reports_the_moe_share_of_attributed_gpu_time(self):
        got = summarize(self._rows(moe_us=1000.0, nccl_us=9000.0, n=2), num_layers=48)
        assert got["moe_share_of_attributed"] == pytest.approx(0.10)

    def test_records_the_decode_batch_from_the_annotation(self):
        got = summarize(self._rows(moe_us=1.0, nccl_us=1.0, n=3, gen=48), num_layers=48)
        assert got["gen_tokens_per_step"] == pytest.approx(48.0)

    def test_no_decode_steps_raises_rather_than_returning_zeros(self):
        """A zero MoE share would read as a finding; it must fail loudly instead."""
        with pytest.raises(ValueError, match="no decode-only"):
            summarize([], num_layers=48)
