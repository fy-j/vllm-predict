# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ticket 10's prefill accuracy analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import prefill_accuracy as pa  # noqa: E402


class TestRankAttribution:
    """Per-rank load comes from canonical ownership, and it must be exact."""

    def test_experts_are_grouped_rank_major(self):
        # 8 experts, 2 ranks: rank 0 owns 0..3, rank 1 owns 4..7.
        assert pa.rank_loads([1, 2, 3, 4, 10, 20, 30, 40], ep_size=2) == [10.0, 100.0]

    def test_the_peak_rank_is_the_one_the_layer_waits_on(self):
        assert pa.peak_rank([1, 1, 50, 1], ep_size=2) == 1

    def test_a_layout_that_does_not_divide_is_refused(self):
        # Guessing a grouping here would silently misattribute load to ranks.
        with pytest.raises(ValueError, match="do not divide"):
            pa.rank_loads([1, 2, 3], ep_size=2)


class TestPrefillSelection:
    """The regime is read off the recorded assignment count, never a magnitude."""

    def test_a_pair_below_the_count_is_excluded(self):
        records = [
            {"pairs": [{"source": 1, "target": 3, "predicted": [1], "actual": [10]}]}
        ]
        assert pa.prefill_pairs(records, min_assignments=100) == []

    def test_a_pair_at_the_count_is_included(self):
        pair = {"source": 1, "target": 3, "predicted": [1], "actual": [100]}
        assert pa.prefill_pairs([{"pairs": [pair]}], min_assignments=100) == [pair]

    def test_forwards_keep_their_layers_together(self):
        # The budget is spent across a forward's layers, so scoring must see the
        # whole forward. Flattening would let every layer spend the whole budget.
        hot = {"source": 1, "target": 3, "predicted": [1], "actual": [100]}
        cold = {"source": 2, "target": 4, "predicted": [1], "actual": [1]}
        grouped = pa.group_by_forward(
            [{"pairs": [hot, cold, hot]}, {"pairs": [cold]}], min_assignments=100
        )
        assert [len(f) for f in grouped] == [2], (
            "one forward with two prefill layers; the all-cold forward drops out"
        )


class TestPeakRankAccuracy:
    """Accuracy is scored where the planner actually looks."""

    def test_recall_is_scored_on_the_predicted_peak_ranks_slice(self):
        # Rank 1 (experts 2,3) is predicted hottest at 14 against 2, and within that
        # slice both sides put expert 3 on top. Pooled over all four experts the
        # top-1 disagrees (3 predicted, 0 actual), so this distinguishes the two.
        predicted = [1.0, 1.0, 5.0, 9.0]
        actual = [20.0, 0.0, 1.0, 9.0]
        assert pa.peak_rank_recall(predicted, actual, ep_size=2, k=1) == 1.0
        from prediction_accuracy import hot_set_recall

        assert hot_set_recall(predicted, actual, 1) == 0.0, (
            "the pooled figure must disagree here, or this test proves nothing "
            "about restricting to the peak rank"
        )

    def test_a_wrong_peak_rank_is_reported_as_such(self):
        assert not pa.peak_rank_hit([9.0, 9.0, 0.0, 0.0], [0.0, 0.0, 9.0, 9.0], 2)

    def test_a_right_peak_rank_is_reported_as_such(self):
        assert pa.peak_rank_hit([0.0, 0.0, 9.0, 9.0], [1.0, 0.0, 9.0, 9.0], 2)


class TestTheDecidingNumber:
    """Placements chosen on predicted load, scored on actual load."""

    def _forward(self, predicted, actual):
        return [[{"source": 0, "target": 2, "predicted": predicted, "actual": actual}]]

    def test_a_perfect_prediction_reaches_the_oracle(self):
        load = [90.0, 10.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        out = pa.deciding_benefit(self._forward(load, load), ep_size=4, budget=1)
        assert out["excess_removed_predicted_pct"] == pytest.approx(
            out["excess_removed_oracle_pct"]
        ), "with predicted == actual the two must coincide by construction"

    def test_a_misleading_prediction_is_reported_as_harmful(self):
        # Predicts rank 1 hottest; actually rank 0 is. The replica lands on rank 0
        # and makes the peak worse, which must show up rather than be clipped to 0.
        predicted = [2.0, 2.0, 90.0, 10.0, 2.0, 2.0, 2.0, 2.0]
        actual = [90.0, 10.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        out = pa.deciding_benefit(self._forward(predicted, actual), ep_size=4, budget=1)
        assert out["forwards_made_worse"] == 1
        assert out["excess_removed_predicted_pct"] < 0, (
            "a placement that raises the peak removes negative excess; reporting it "
            "as zero would hide that bad prediction costs more than doing nothing"
        )

    def test_the_budget_is_spent_across_the_forwards_layers(self):
        load = [90.0, 10.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        pair = {"source": 0, "target": 2, "predicted": load, "actual": load}
        out = pa.deciding_benefit([[pair, pair, pair]], ep_size=4, budget=2)
        assert out["placements_per_forward"] <= 2, (
            "three layers must share the budget of two, not take two each"
        )

    def test_no_prefill_forwards_reports_nothing_rather_than_a_flat_one(self):
        assert pa.deciding_benefit([], ep_size=4, budget=1) == {}
