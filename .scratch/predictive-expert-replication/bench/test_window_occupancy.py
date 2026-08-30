# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `window_occupancy`, whose failure mode is a plausible wrong number.

Every test here guards a mistake this project has already made and recorded:

* Summing kernel durations across streams. Attributed time adds the compute, token
  collective and prediction streams, which overlap, so a sum reports more busy time
  than the window contains. That error made the expert GEMM look like 10.76% of a step
  when it is 14.29% of one, and it is why occupancy must be a *union*.
* Reading an 8-rank-summed total as a per-rank one, which concluded the window was 10%
  occupied when seven of eight ranks were at 84%.
* Attributing across a window that mixes prefill with decode.
* Reporting zero where nothing was captured, which reads as a finding rather than as a
  failed measurement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from window_occupancy import (  # noqa: E402
    gaps_over,
    summarize_windows,
    union_busy_us,
    window_occupancies,
)


def _annotation(name, ts, dur):
    return {"name": name, "ts": ts, "dur": dur, "cat": "gpu_user_annotation"}


def _kernel(name, ts, dur):
    return {"name": name, "ts": ts, "dur": dur, "cat": "kernel"}


PREFILL = "execute_context_3(2047)_generation_1(1)"
DECODE = "execute_context_0(0)_generation_8(8)"


class TestUnionBusyUs:
    """Busy time is the union of the intervals, never their sum."""

    def test_overlapping_intervals_are_counted_once(self):
        # The whole reason this module exists: an expert GEMM on the compute stream and
        # an AllGather on the collective stream run at the same time, and summing them
        # reports 200 us of work inside a 100 us span.
        assert union_busy_us([(0.0, 100.0), (20.0, 120.0)]) == 120.0

    def test_disjoint_intervals_add(self):
        assert union_busy_us([(0.0, 10.0), (50.0, 60.0)]) == 20.0

    def test_a_contained_interval_adds_nothing(self):
        assert union_busy_us([(0.0, 100.0), (10.0, 20.0)]) == 100.0

    def test_no_intervals_is_zero_busy(self):
        assert union_busy_us([]) == 0.0

    def test_unsorted_input_is_handled(self):
        # Trace events arrive in completion order, not start order, and per-stream
        # sections are interleaved.
        assert union_busy_us([(50.0, 60.0), (0.0, 10.0), (5.0, 55.0)]) == 60.0


class TestGapsOver:
    """Idle stretches are what a launch-bound engine leaves behind."""

    def test_reports_only_gaps_above_the_threshold(self):
        gaps = gaps_over([(0.0, 100.0), (200.0, 300.0)], 0.0, 1000.0, threshold=50.0)
        # The 100 us hole between the kernels and the 700 us tail; nothing else.
        assert gaps == [100.0, 700.0]

    def test_small_gaps_are_ignored(self):
        assert gaps_over([(0.0, 100.0), (110.0, 200.0)], 0.0, 200.0, 50.0) == []

    def test_leading_idle_counts(self):
        # The host arriving late is exactly the cost being measured, so a gap before the
        # first kernel is not an artifact to be trimmed.
        assert gaps_over([(500.0, 600.0)], 0.0, 600.0, 50.0) == [500.0]

    def test_a_window_with_no_kernels_is_one_whole_gap(self):
        assert gaps_over([], 0.0, 400.0, 50.0) == [400.0]


class TestWindowOccupancies:
    """One row per real forward window, with the other phase's work excluded."""

    def test_occupancy_is_busy_over_wall_clock(self):
        events = [
            _annotation(PREFILL, 1000.0, 100.0),
            _kernel("fused_moe_kernel", 1000.0, 40.0),
            _kernel("flash_fwd_kernel", 1050.0, 20.0),
        ]
        (row,) = window_occupancies(events)
        assert row.wall_us == 100.0
        assert row.busy_us == 60.0
        assert row.occupancy == pytest.approx(0.6)

    def test_concurrent_streams_cannot_push_occupancy_above_one(self):
        # 90 us of expert GEMM and 90 us of NCCL inside a 100 us window: a sum-based
        # implementation reports 180% occupancy, which is the bug.
        events = [
            _annotation(PREFILL, 0.0, 100.0),
            _kernel("fused_moe_kernel", 0.0, 90.0),
            _kernel("ncclDevKernel_AllGather", 5.0, 90.0),
        ]
        (row,) = window_occupancies(events)
        assert row.busy_us == 95.0
        assert row.occupancy <= 1.0

    def test_a_kernel_running_past_the_window_is_clipped(self):
        events = [
            _annotation(PREFILL, 0.0, 100.0),
            _kernel("fused_moe_kernel", 50.0, 500.0),
        ]
        (row,) = window_occupancies(events)
        assert row.busy_us == 50.0
        assert row.occupancy == pytest.approx(0.5)

    def test_decode_work_never_enters_a_prefill_window(self):
        events = [
            _annotation(PREFILL, 0.0, 100.0),
            _kernel("fused_moe_kernel", 10.0, 10.0),
            _annotation(DECODE, 1000.0, 100.0),
            _kernel("fused_moe_kernel", 1010.0, 90.0),
        ]
        (row,) = window_occupancies(events, phase="prefill")
        assert row.busy_us == 10.0

    def test_the_expert_gemm_share_uses_wall_clock_as_its_denominator(self):
        # The recorded correction: dividing by attributed time understated the expert
        # GEMM's share of a step, because attributed time double-counts streams.
        events = [
            _annotation(PREFILL, 0.0, 200.0),
            _kernel("fused_moe_kernel", 0.0, 20.0),
            _kernel("ncclDevKernel_AllGather", 0.0, 100.0),
        ]
        (row,) = window_occupancies(events)
        assert row.by_class["moe_expert"] == 20.0
        assert row.expert_gemm_share == pytest.approx(0.1)

    def test_occupancy_excluding_collectives_separates_work_from_waiting(self):
        """A rank parked inside `ncclDevKernel` is busy by kernel time and idle in fact.

        This is what makes raw occupancy the wrong instrument for "is the host feeding
        the GPU": measured at DP=8, per-rank occupancy tracks NCCL residency almost
        exactly, so the rank that *arrives first* and waits longest inside the
        collective scores as the busiest. Excluding the collectives leaves the work.
        """
        events = [
            _annotation(PREFILL, 0.0, 100.0),
            _kernel("fused_moe_kernel", 0.0, 20.0),
            _kernel("ncclDevKernel_AllGather", 20.0, 60.0),
        ]
        (row,) = window_occupancies(events)
        assert row.occupancy == pytest.approx(0.8)
        assert row.occupancy_ex_nccl == pytest.approx(0.2)

    def test_excluding_collectives_still_counts_overlap_once(self):
        # Compute overlapping a collective must not be double-subtracted: the union of
        # the non-NCCL intervals is the quantity, not busy minus NCCL time.
        events = [
            _annotation(PREFILL, 0.0, 100.0),
            _kernel("fused_moe_kernel", 0.0, 50.0),
            _kernel("ncclDevKernel_AllGather", 0.0, 90.0),
        ]
        (row,) = window_occupancies(events)
        assert row.occupancy == pytest.approx(0.9)
        assert row.occupancy_ex_nccl == pytest.approx(0.5)

    def test_gaps_over_the_threshold_are_carried_per_window(self):
        events = [
            _annotation(PREFILL, 0.0, 3000.0),
            _kernel("fused_moe_kernel", 0.0, 100.0),
            _kernel("fused_moe_kernel", 2000.0, 100.0),
        ]
        (row,) = window_occupancies(events, gap_threshold_us=500.0)
        assert row.gap_us_over_threshold == pytest.approx(1900.0 + 900.0)
        assert row.largest_gap_us == pytest.approx(1900.0)

    def test_prefill_tokens_are_carried_so_arms_can_be_compared(self):
        # The two arms' window shapes are not paired: the placed arm split the same work
        # into 18-22 windows per rank where the others took 11-13, so comparing expert
        # GEMM milliseconds *per window* across arms compares different amounts of work.
        # The annotation's context-token count is what normalises it.
        events = [
            _annotation("execute_context_3(2047)_generation_1(1)", 0.0, 200.0),
            _kernel("fused_moe_kernel", 0.0, 40.0),
        ]
        (row,) = window_occupancies(events)
        assert row.ctx_tokens == 2047
        # 40 us of expert GEMM over 2047 prefill tokens.
        assert row.expert_gemm_us_per_1k_ctx == pytest.approx(40.0 / 2.047)

    def test_a_window_with_no_prefill_token_has_no_per_token_rate(self):
        # Guards a divide-by-zero that would otherwise surface as `inf` in a report.
        events = [
            _annotation("execute_context_1(0)_generation_0(0)", 0.0, 100.0),
            _kernel("fused_moe_kernel", 0.0, 40.0),
        ]
        (row,) = window_occupancies(events)
        assert row.expert_gemm_us_per_1k_ctx == 0.0

    def test_no_window_of_the_asked_phase_raises(self):
        # Returning zeros here would present a failed capture as a 0% occupancy
        # finding, which is the direction that has cost this project runs.
        with pytest.raises(ValueError, match="no prefill window"):
            window_occupancies([_annotation(DECODE, 0.0, 100.0)], phase="prefill")


class TestSummarizeWindows:
    """One rank's report. Medians, because a first window carries JIT work."""

    def test_medians_across_windows(self):
        events = [
            _annotation(PREFILL, 0.0, 100.0),
            _kernel("fused_moe_kernel", 0.0, 50.0),
            _annotation(PREFILL, 1000.0, 100.0),
            _kernel("fused_moe_kernel", 1000.0, 90.0),
            _annotation(PREFILL, 2000.0, 100.0),
            _kernel("fused_moe_kernel", 2000.0, 70.0),
        ]
        got = summarize_windows(window_occupancies(events))
        assert got["windows"] == 3
        assert got["wall_ms"] == pytest.approx(0.1)
        assert got["occupancy"] == pytest.approx(0.7)

    def test_an_empty_row_list_raises_rather_than_reporting_zero(self):
        with pytest.raises(ValueError, match="no window"):
            summarize_windows([])
