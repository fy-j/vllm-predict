# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ticket 06 summary: degradation axis and the skip recommendation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from accuracy_report import (  # noqa: E402
    by_layer_curve,
    degradation,
    domain_of,
    skip_recommendation,
)


def _report(label, lookahead, recall_at_2, count_error=0.1, by_layer=None):
    return {
        "label": label,
        "lookahead": lookahead,
        "overall": {"recall_at_2": recall_at_2, "count_error": count_error},
        "by_layer": by_layer or {},
    }


class TestDomainOf:
    def test_strips_the_lookahead_prefix(self):
        assert domain_of("L2-code-p1024") == "code-p1024"

    def test_a_label_without_a_prefix_is_returned_whole(self):
        assert domain_of("code") == "code"


class TestDegradation:
    def test_places_each_run_on_the_lookahead_axis(self):
        got = degradation(
            [
                _report("L1-code-p1024", 1, 0.83),
                _report("L2-code-p1024", 2, 0.78),
                _report("L1-text-p2048", 1, 0.79),
            ]
        )
        assert got["recall"]["code-p1024"] == {1: 0.83, 2: 0.78}
        assert got["recall"]["text-p2048"] == {1: 0.79}

    def test_a_mixed_distance_run_is_excluded_from_the_axis(self):
        """A dump spanning several distances has no single lookahead to plot."""
        got = degradation(
            [_report("L1-code", 1, 0.8), _report("Lx-code", [1, 2, 3], 0.5)]
        )
        assert got["recall"]["code"] == {1: 0.8}

    def test_a_missing_lookahead_one_run_is_reported_not_raised(self):
        """Refusing took the whole report down with it.

        The skip-first-layers evidence lives in a directory holding a single
        lookahead, and the per-layer curve needs no baseline at all, so raising
        here destroyed the only output that answered that question.
        """
        got = degradation([_report("L2-code", 2, 0.78)])

        assert got["has_lookahead_1_baseline"] is False
        assert got["recall"]["code"] == {2: 0.78}

    def test_a_present_lookahead_one_run_is_flagged_as_such(self):
        got = degradation([_report("L1-code", 1, 0.83)])
        assert got["has_lookahead_1_baseline"] is True


class TestByLayerCurve:
    def test_refuses_to_pool_across_lookaheads(self):
        """Pooling inflates exactly the leading layers a skip decision turns on.

        Recall falls monotonically with distance, and the earliest target layers
        are reachable only by the shortest-distance runs — target 4 by L=1 alone,
        target 5 by L=1 and 2 — so a pooled mean lifts them above the later layers
        and `skip_recommendation` then finds nothing to skip.
        """
        reports = [
            _report("L1-code", 1, 0.8, by_layer={"4": {"recall_at_2": 0.76}}),
            _report("L3-code", 3, 0.7, by_layer={"6": {"recall_at_2": 0.68}}),
        ]
        with pytest.raises(ValueError, match="cannot pool a per-layer curve"):
            by_layer_curve(reports)

    def test_pools_the_same_layer_across_runs(self):
        reports = [
            _report("L1-code", 1, 0.8, by_layer={"10": {"recall_at_2": 0.6}}),
            _report("L1-text", 1, 0.8, by_layer={"10": {"recall_at_2": 0.8}}),
        ]
        assert by_layer_curve(reports) == {10: 0.7}

    def test_sorted_by_layer_index_not_string_order(self):
        reports = [
            _report(
                "L1-code",
                1,
                0.8,
                by_layer={
                    "9": {"recall_at_2": 0.5},
                    "10": {"recall_at_2": 0.6},
                    "100": {"recall_at_2": 0.7},
                },
            )
        ]
        assert list(by_layer_curve(reports)) == [9, 10, 100]


class TestSkipRecommendation:
    """The skip is justified only by a *leading* run of unreliable layers."""

    def test_flags_leading_layers_that_predict_worse(self):
        curve = {3: 0.50, 4: 0.55, 5: 0.80, 6: 0.81, 7: 0.82, 8: 0.80}
        got = skip_recommendation(curve, tolerance=0.05)
        assert got["unreliable_layers"] == [3, 4]
        assert got["implied_skip_target_layers"] == 5

    def test_a_dip_in_the_middle_does_not_justify_a_skip(self):
        """Skipping leading layers cannot fix a late layer, so it must not count."""
        curve = {3: 0.80, 4: 0.81, 5: 0.40, 6: 0.82, 7: 0.81, 8: 0.80}
        got = skip_recommendation(curve, tolerance=0.05)
        assert got["unreliable_layers"] == []
        assert got["implied_skip_target_layers"] == 0

    def test_uniformly_good_layers_imply_no_skip(self):
        curve = {3: 0.80, 4: 0.81, 5: 0.79, 6: 0.82}
        assert skip_recommendation(curve)["unreliable_layers"] == []

    def test_an_empty_curve_reports_nothing_rather_than_zero(self):
        got = skip_recommendation({})
        assert got["stable_median"] is None
        assert got["implied_skip_target_layers"] is None
