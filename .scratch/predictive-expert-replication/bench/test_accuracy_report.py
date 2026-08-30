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
    lookahead_pair,
    skip_recommendation,
)


def _report(label, lookahead, recall, count_error=0.1, by_layer=None):
    """A scored run, with `recall` written at both k.

    So a test reads the same number whichever `PLANNER_K` is current: the default moved
    from 2 to 1 when the planner's per-layer cap did, and fixtures pinned to one k would
    have quietly stopped matching.
    """
    return {
        "label": label,
        "lookahead": lookahead,
        "overall": {
            "recall_at_1": recall,
            "recall_at_2": recall,
            "count_error": count_error,
        },
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
            _report(
                "L1-code",
                1,
                0.8,
                by_layer={"4": {"recall_at_1": 0.76, "recall_at_2": 0.76}},
            ),
            _report(
                "L3-code",
                3,
                0.7,
                by_layer={"6": {"recall_at_1": 0.68, "recall_at_2": 0.68}},
            ),
        ]
        with pytest.raises(ValueError, match="cannot pool a per-layer curve"):
            by_layer_curve(reports)

    def test_pools_the_same_layer_across_runs(self):
        reports = [
            _report(
                "L1-code",
                1,
                0.8,
                by_layer={"10": {"recall_at_1": 0.6, "recall_at_2": 0.6}},
            ),
            _report(
                "L1-text",
                1,
                0.8,
                by_layer={"10": {"recall_at_1": 0.8, "recall_at_2": 0.8}},
            ),
        ]
        assert by_layer_curve(reports) == {10: 0.7}

    def test_sorted_by_layer_index_not_string_order(self):
        reports = [
            _report(
                "L1-code",
                1,
                0.8,
                by_layer={
                    "9": {"recall_at_1": 0.5, "recall_at_2": 0.5},
                    "10": {"recall_at_1": 0.6, "recall_at_2": 0.6},
                    "100": {"recall_at_1": 0.7, "recall_at_2": 0.7},
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


class TestLookaheadPair:
    """Ticket 07 asks whether shortening the lookahead to 1 predicts better.

    The two arms do not cover the same target layers: at a lookahead of `n` the reachable
    targets start at `skip_first + n`, so lookahead 1 covers one extra early layer — and
    early layers predict worst, which is the whole reason
    `prediction_skip_first_layers` exists. Comparing pooled overalls therefore charges
    lookahead 1 for a layer lookahead 2 never had to predict.
    """

    def _pair(self, l1_layers, l2_layers, l1_overall=0.9, l2_overall=0.8):
        return [
            _report("L1-ko-p1024", 1, l1_overall, by_layer=l1_layers),
            _report("L2-ko-p1024", 2, l2_overall, by_layer=l2_layers),
        ]

    def test_compares_on_the_layers_both_arms_cover(self):
        got = lookahead_pair(
            self._pair(
                {
                    "4": {"recall_at_1": 0.30, "recall_at_2": 0.30, "count_error": 0.5},
                    "5": {"recall_at_1": 0.90, "recall_at_2": 0.90, "count_error": 0.1},
                    "6": {"recall_at_1": 0.90, "recall_at_2": 0.90, "count_error": 0.1},
                },
                {
                    "5": {"recall_at_1": 0.80, "recall_at_2": 0.80, "count_error": 0.2},
                    "6": {"recall_at_1": 0.80, "recall_at_2": 0.80, "count_error": 0.2},
                },
            ),
            k=2,
        )["ko-p1024"]
        assert got["common_layers"] == [5, 6]
        assert got["common"]["recall_at_2"] == {1: 0.9, 2: 0.8}
        assert got["common"]["count_error"] == {1: 0.1, 2: 0.2}
        assert got["better_at_1"] is True

    def test_reports_the_pooled_overall_too(self):
        """Both are useful; the point is not to quote only the flattering one."""
        got = lookahead_pair(
            self._pair(
                {
                    "4": {"recall_at_1": 0.30, "recall_at_2": 0.30, "count_error": 0.5},
                    "5": {"recall_at_1": 0.90, "recall_at_2": 0.90, "count_error": 0.1},
                },
                {
                    "5": {"recall_at_1": 0.80, "recall_at_2": 0.80, "count_error": 0.2},
                },
                l1_overall=0.60,
                l2_overall=0.80,
            ),
            k=2,
        )["ko-p1024"]
        assert got["overall"]["recall_at_2"] == {1: 0.6, 2: 0.8}
        assert got["common"]["recall_at_2"] == {1: 0.9, 2: 0.8}
        assert got["layers_only_at_1"] == [4]

    def test_no_common_layer_is_stated_rather_than_averaged_to_nothing(self):
        got = lookahead_pair(
            self._pair(
                {"4": {"recall_at_1": 0.9, "recall_at_2": 0.9, "count_error": 0.1}},
                {"9": {"recall_at_1": 0.8, "recall_at_2": 0.8, "count_error": 0.2}},
            ),
            k=2,
        )["ko-p1024"]
        assert got["common_layers"] == []
        assert got["common"] is None
        assert got["better_at_1"] is None
        assert got["reason"] == "the two arms share no target layer"

    def test_a_mixed_distance_report_is_left_off_the_axis(self):
        """A dump spanning several distances has no single lookahead to compare at."""
        got = lookahead_pair(
            [
                _report("L1-ko-p1024", 1, 0.9, by_layer={"5": {"recall_at_2": 0.9}}),
                _report("Lx-ko-p1024", [1, 2], 0.5, by_layer={"5": {"recall_at_2": 0.5}}),
            ],
            k=2,
        )["ko-p1024"]
        assert got["lookaheads"] == [1]
        assert got["reason"] == "only one lookahead measured"


class TestLookaheadPairAcrossDomains:
    """Two domains per lookahead is the runner's default, and it must not silently pool.

    Keying by lookahead alone made the last report win: with code at 0.9 and text at 0.1
    the comparison reported 0.1 for both arms and "not better at 1", presented in the
    report as the answer to ticket 07. Accuracy is a property of the content's routing, so
    the domains are separate results and the comparison is per domain.
    """

    def _layers(self, recall):
        return {
            "5": {"recall_at_1": recall, "recall_at_2": recall, "count_error": 0.1},
            "6": {"recall_at_1": recall, "recall_at_2": recall, "count_error": 0.1},
        }

    def test_each_domain_is_compared_against_its_own_other_arm(self):
        got = lookahead_pair(
            [
                _report("L1-code-p1024", 1, 0.90, by_layer=self._layers(0.90)),
                _report("L2-code-p1024", 2, 0.80, by_layer=self._layers(0.80)),
                _report("L1-text-p2048", 1, 0.10, by_layer=self._layers(0.10)),
                _report("L2-text-p2048", 2, 0.05, by_layer=self._layers(0.05)),
            ],
            k=1,
        )
        assert set(got) == {"code-p1024", "text-p2048"}
        assert got["code-p1024"]["common"]["recall_at_1"] == {1: 0.9, 2: 0.8}
        assert got["text-p2048"]["common"]["recall_at_1"] == {1: 0.1, 2: 0.05}
        assert got["code-p1024"]["better_at_1"] is True
        assert got["text-p2048"]["better_at_1"] is True

    def test_a_domain_with_only_one_arm_is_reported_as_such_not_dropped(self):
        """Silence would read as "the comparison says nothing", which is not the same."""
        got = lookahead_pair(
            [
                _report("L1-code-p1024", 1, 0.9, by_layer=self._layers(0.9)),
                _report("L2-code-p1024", 2, 0.8, by_layer=self._layers(0.8)),
                _report("L1-text-p2048", 1, 0.5, by_layer=self._layers(0.5)),
            ],
            k=1,
        )
        assert got["text-p2048"]["lookaheads"] == [1]
        assert got["text-p2048"]["common"] is None
        assert got["text-p2048"]["reason"] == "only one lookahead measured"

    def test_three_lookaheads_compare_the_two_shortest_and_say_so(self):
        """The runner's own default is `LOOKAHEADS="1 2 3"`, which used to raise and be
        swallowed by both callers, so the deliverable table vanished without a word."""
        got = lookahead_pair(
            [
                _report("L1-ko-p1024", 1, 0.9, by_layer=self._layers(0.9)),
                _report("L2-ko-p1024", 2, 0.8, by_layer=self._layers(0.8)),
                _report("L3-ko-p1024", 3, 0.7, by_layer=self._layers(0.7)),
            ],
            k=1,
        )
        assert got["ko-p1024"]["lookaheads"] == [1, 2]
        assert got["ko-p1024"]["ignored_lookaheads"] == [3]
        assert got["ko-p1024"]["common"]["recall_at_1"] == {1: 0.9, 2: 0.8}
