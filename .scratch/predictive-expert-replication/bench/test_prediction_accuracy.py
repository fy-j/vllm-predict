# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ticket 06 prediction-accuracy metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from prediction_accuracy import (  # noqa: E402
    aggregate,
    count_error,
    hot_set_recall,
    normalize,
    peak_expert_hit,
    top_k_experts,
)


class TestNormalize:
    """Both sides must be fractions before comparison.

    Under the allgather/reduce-scatter backend the recorded load is multiplied by
    the DP size while the prediction is a plain per-rank count, so absolute counts
    are not comparable and only shares are.
    """

    def test_turns_counts_into_shares(self):
        assert normalize([1.0, 3.0]) == pytest.approx([0.25, 0.75])

    def test_all_zero_stays_all_zero_rather_than_dividing_by_zero(self):
        assert normalize([0.0, 0.0]) == [0.0, 0.0]


class TestTopKExperts:
    def test_picks_the_hottest(self):
        assert top_k_experts([5.0, 1.0, 9.0, 3.0], 2) == [2, 0]

    def test_ties_break_by_index_so_the_metric_is_deterministic(self):
        assert top_k_experts([4.0, 4.0, 4.0], 2) == [0, 1]

    def test_k_larger_than_the_expert_count_returns_all(self):
        assert sorted(top_k_experts([1.0, 2.0], 5)) == [0, 1]


class TestHotSetRecall:
    def test_perfect_prediction_recalls_everything(self):
        counts = [9.0, 1.0, 5.0, 3.0]
        assert hot_set_recall(counts, counts, k=2) == pytest.approx(1.0)

    def test_reversed_prediction_misses_the_hottest(self):
        assert hot_set_recall([1.0, 9.0], [9.0, 1.0], k=1) == pytest.approx(0.0)

    def test_half_overlap_scores_half(self):
        # actual top-2 is {0, 1}; predicted top-2 is {0, 2}
        got = hot_set_recall([9.0, 1.0, 5.0], [9.0, 6.0, 1.0], k=2)
        assert got == pytest.approx(0.5)

    def test_scale_does_not_matter(self):
        """A prediction scaled by any positive factor ranks identically."""
        actual = [9.0, 1.0, 5.0]
        scaled = [c * 8 for c in actual]
        assert hot_set_recall(scaled, actual, k=2) == pytest.approx(1.0)


class TestCountError:
    def test_identical_shares_have_no_error(self):
        assert count_error([2.0, 6.0], [1.0, 3.0]) == pytest.approx(0.0)

    def test_disjoint_shares_have_maximal_error(self):
        """Total-variation distance is 1.0 when the two distributions do not overlap."""
        assert count_error([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_partial_disagreement_is_between(self):
        got = count_error([0.5, 0.5], [1.0, 0.0])
        assert got == pytest.approx(0.5)


class TestPeakExpertHit:
    """The planner acts on the hottest expert, so predicting it is the operative bar."""

    def test_is_the_same_measure_as_recall_at_one(self):
        """Pinned so the two are never reported as independent evidence."""
        for predicted, actual in (
            ([1.0, 7.0, 2.0], [3.0, 9.0, 1.0]),
            ([7.0, 1.0], [1.0, 7.0]),
            ([4.0, 4.0, 1.0], [4.0, 4.0, 9.0]),
        ):
            expected = 1.0 if peak_expert_hit(predicted, actual) else 0.0
            assert hot_set_recall(predicted, actual, k=1) == expected

    def test_hit(self):
        assert peak_expert_hit([1.0, 7.0, 2.0], [3.0, 9.0, 1.0]) is True

    def test_miss(self):
        assert peak_expert_hit([7.0, 1.0], [1.0, 7.0]) is False


class TestAggregate:
    def _record(self, target, predicted, actual):
        return {
            "source": target - 2,
            "target": target,
            "predicted": predicted,
            "actual": actual,
        }

    def test_groups_by_target_layer(self):
        records = [
            self._record(10, [9.0, 1.0], [9.0, 1.0]),
            self._record(11, [1.0, 9.0], [9.0, 1.0]),
        ]
        got = aggregate(records, ks=(1,))
        assert got["by_layer"]["10"]["recall_at_1"] == pytest.approx(1.0)
        assert got["by_layer"]["11"]["recall_at_1"] == pytest.approx(0.0)

    def test_skips_records_whose_actual_load_is_empty(self):
        """A layer that saw no tokens has no hot set; scoring it zero invents a miss."""
        records = [
            self._record(10, [9.0, 1.0], [0.0, 0.0]),
            self._record(10, [9.0, 1.0], [9.0, 1.0]),
        ]
        got = aggregate(records, ks=(1,))
        assert got["scored"] == 1
        assert got["skipped_empty_actual"] == 1
        assert got["overall"]["recall_at_1"] == pytest.approx(1.0)

    def test_reports_the_sample_size_so_a_thin_result_is_visible(self):
        got = aggregate([self._record(10, [1.0, 2.0], [1.0, 2.0])], ks=(1,))
        assert got["scored"] == 1
        assert got["by_layer"]["10"]["samples"] == 1

    def test_no_scorable_record_raises_rather_than_reporting_perfect(self):
        """An empty mean would otherwise surface as flawless accuracy."""
        with pytest.raises(ValueError, match="no scorable"):
            aggregate([self._record(10, [1.0], [0.0])], ks=(1,))

    def test_reports_every_requested_k(self):
        got = aggregate([self._record(10, [3.0, 2.0, 1.0], [3.0, 2.0, 1.0])], ks=(1, 2))
        assert "recall_at_1" in got["overall"] and "recall_at_2" in got["overall"]

    def test_reports_count_error_and_peak_hit_rate(self):
        got = aggregate([self._record(10, [9.0, 1.0], [9.0, 1.0])], ks=(1,))
        assert got["overall"]["count_error"] == pytest.approx(0.0)
        assert got["overall"]["peak_hit_rate"] == pytest.approx(1.0)
