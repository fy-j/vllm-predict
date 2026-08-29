# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the rank-imbalance analysis.

The first two classes exist because both errors they pin were made for real in
this branch: an imbalance aggregated over layers before comparing ranks
(understates it about threefold), and a placement model that credited a replica
with an expert's whole load when source-rank routing only moves half of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import imbalance  # noqa: E402
from imbalance import (  # noqa: E402
    Layer,
    aggregated_imbalance,
    bin_by_assignments,
    critical_path_imbalance,
    place_by_threshold_search,
    place_globally,
    place_uniformly,
    shed_fraction,
)


def _layer(rank_load, experts=None):
    """A layer whose per-rank load is given; experts split evenly unless stated."""
    if experts is None:
        experts = [[v] for v in rank_load]
    return Layer(rank_load=list(rank_load), expert_load=[list(e) for e in experts])


class TestCriticalPathIsPerLayer:
    """Σ per-layer peak over Σ per-layer mean — never rank totals."""

    def test_peaks_on_different_ranks_do_not_cancel(self):
        """The exact case that understated this branch's first measurement.

        Two layers, each badly skewed, but skewed towards *different* ranks.
        Summing the layers first makes them look perfectly balanced.
        """
        layers = [_layer([9.0, 1.0]), _layer([1.0, 9.0])]

        assert aggregated_imbalance(layers) == pytest.approx(1.0)
        assert critical_path_imbalance(layers) == pytest.approx(1.8)

    def test_a_single_layer_agrees_with_the_aggregate(self):
        layers = [_layer([9.0, 1.0])]
        assert critical_path_imbalance(layers) == pytest.approx(1.8)
        assert aggregated_imbalance(layers) == pytest.approx(1.8)

    def test_a_balanced_model_is_one(self):
        layers = [_layer([5.0, 5.0]), _layer([5.0, 5.0])]
        assert critical_path_imbalance(layers) == pytest.approx(1.0)

    def test_empty_layers_are_excluded_not_counted_as_balanced(self):
        """A layer that routed nothing has no imbalance; averaging it in dilutes."""
        layers = [_layer([9.0, 1.0]), _layer([0.0, 0.0])]
        assert critical_path_imbalance(layers) == pytest.approx(1.8)

    def test_no_layers_raises_rather_than_returning_one(self):
        with pytest.raises(ValueError, match="no layer"):
            critical_path_imbalance([_layer([0.0, 0.0])])


class TestShedFraction:
    """Source-rank routing splits source ranks across copies, so a replica takes
    a fraction of the expert's load — never all of it."""

    def test_one_replica_takes_half(self):
        assert shed_fraction(1) == pytest.approx(0.5)

    def test_two_replicas_of_the_same_expert_take_two_thirds(self):
        """Fan-out of K sheds K/(K+1); the glossary's arithmetic."""
        assert shed_fraction(2) == pytest.approx(2 / 3)

    def test_it_never_reaches_one(self):
        assert shed_fraction(7) < 1.0


class TestPlacement:
    def test_a_replica_moves_half_the_expert_from_peak_to_target(self):
        layer = _layer([10.0, 2.0], experts=[[6.0, 4.0], [1.0, 1.0]])
        after = place_uniformly([layer], per_layer=1)[0]
        # Peak rank 0's hottest expert is 6.0; half of it moves to rank 1.
        assert after.rank_load == pytest.approx([7.0, 5.0])

    def test_the_target_is_the_lightest_rank(self):
        layer = _layer([10.0, 5.0, 1.0], experts=[[10.0], [5.0], [1.0]])
        after = place_uniformly([layer], per_layer=1)[0]
        assert after.rank_load == pytest.approx([5.0, 5.0, 6.0])

    def test_an_expert_is_never_moved_twice(self):
        layer = _layer([10.0, 0.0], experts=[[10.0], [0.0]])
        after = place_uniformly([layer], per_layer=3)[0]
        # Only one expert exists on the peak rank, so only one move is possible.
        assert after.rank_load == pytest.approx([5.0, 5.0])

    def test_a_move_that_would_not_lower_the_peak_is_not_made(self):
        """Overshooting makes the target the new peak; that is not an improvement."""
        layer = _layer([6.0, 5.0], experts=[[6.0], [5.0]])
        before = list(layer.rank_load)
        after = place_uniformly([layer], per_layer=1)[0]
        # Moving 3.0 gives [3, 8] — peak rose from 6 to 8, so refuse it.
        assert after.rank_load == pytest.approx(before)


class TestGlobalVersusUniform:
    """A global transfer budget cannot be spent well one layer at a time."""

    def test_global_beats_uniform_at_the_same_budget(self):
        """Four ranks, so the skewed layer can absorb more than one placement."""
        skewed = _layer(
            [24.0, 0.0, 0.0, 0.0],
            experts=[[8.0, 8.0, 8.0], [0.0] * 3, [0.0] * 3, [0.0] * 3],
        )
        even = _layer([6.0] * 4, experts=[[6.0]] * 4)
        layers = [skewed, even, even]

        uniform = critical_path_imbalance(place_uniformly(layers, per_layer=1))
        globl = critical_path_imbalance(place_globally(layers, budget=3))

        assert globl < uniform

    def test_a_budget_of_zero_changes_nothing(self):
        layers = [_layer([9.0, 1.0])]
        assert place_globally(layers, budget=0)[0].rank_load == pytest.approx(
            [9.0, 1.0]
        )

    def test_the_budget_is_never_exceeded(self):
        """Three ranks can host two replicas, so a budget of two spends exactly two."""
        layers = [
            _layer(
                [20.0, 0.0, 0.0],
                experts=[[5.0, 5.0, 5.0, 5.0], [0.0] * 4, [0.0] * 4],
            )
        ]
        after = place_globally(layers, budget=2)
        assert after[0].moves == 2


class TestBinByAssignments:
    """Imbalance is reported against batch size, not a prefill/decode guess."""

    def test_groups_records_by_assignment_magnitude(self):
        recs = [
            {"assignments": 100.0, "imbalance": 1.5},
            {"assignments": 120.0, "imbalance": 1.7},
            {"assignments": 10000.0, "imbalance": 1.2},
        ]
        got = bin_by_assignments(recs, edges=(0, 1000, 100000))
        assert got[(0, 1000)]["n"] == 2
        assert got[(0, 1000)]["imbalance"] == pytest.approx(1.6)
        assert got[(1000, 100000)]["n"] == 1

    def test_an_empty_bin_is_absent_not_zero(self):
        got = bin_by_assignments(
            [{"assignments": 5.0, "imbalance": 1.1}], edges=(0, 10, 20)
        )
        assert (10, 20) not in got


class TestLoadDumpDetectsMixedScales:
    """A dump whose two views disagree in scale must not be read as-is.

    Dumps written before 2026-08-25 reduced the logical view over the EP group and
    left the per-rank view raw, so they differ by a factor of the EP size — and the
    raw one measures "how one rank's tokens spread across the ranks", not what each
    rank does. Reading them together made a replicated expert appear to carry 181%
    of its rank's load.
    """

    def _write(self, tmp_path, rank_load, logical_load):
        import json

        p = tmp_path / "d.jsonl"
        p.write_text(
            json.dumps(
                {
                    "rank_load": rank_load,
                    "logical_load": logical_load,
                    "assignments_per_layer": [sum(logical_load[0])],
                    "ep_size": len(rank_load[0]),
                }
            )
            + "\n"
        )
        return p

    def test_a_mismatched_per_rank_view_is_replaced_by_the_logical_grouping(
        self, tmp_path
    ):
        from imbalance import load_dump

        # Per-rank totals 6; the logical view totals 56 — a mismatch in scale.
        # Eight logical experts over two ranks means four each, so the grouping is
        # [8+8+8+8, 8+4+6+6].
        path = self._write(
            tmp_path, [[2.0, 4.0]], [[8.0, 8.0, 8.0, 8.0, 8.0, 4.0, 6.0, 6.0]]
        )
        (forward,) = load_dump(path)

        assert forward[0].rank_load == pytest.approx([32.0, 24.0])

    def test_a_consistent_dump_is_used_unchanged(self, tmp_path):
        from imbalance import load_dump

        # One layer, two ranks, one expert each: the grouping equals the recorded
        # per-rank view, so it is kept.
        path = self._write(tmp_path, [[3.0, 7.0]], [[3.0, 7.0]])
        (forward,) = load_dump(path)

        assert forward[0].rank_load == pytest.approx([3.0, 7.0])

    def test_an_all_zero_forward_is_not_treated_as_a_mismatch(self, tmp_path):
        from imbalance import load_dump

        path = self._write(tmp_path, [[0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]])
        (forward,) = load_dump(path)
        assert forward[0].rank_load == pytest.approx([0.0, 0.0])


class TestThresholdSearch:
    """A global budget wants a global target, not a per-layer allowance.

    Borrowed from UltraEP: binary-search the achievable imbalance ratio and let the
    placement count fall out, instead of fixing the count and letting the ratio fall
    out. Layers that are worse then draw more replicas on their own, which is what
    makes the per-layer replica count a result rather than a constant.
    """

    def _layer(self, rank_load, experts=None):
        if experts is None:
            experts = [[v] for v in rank_load]
        return Layer(rank_load=list(rank_load), expert_load=[list(e) for e in experts])

    def test_an_already_balanced_set_spends_nothing(self):
        layers = [self._layer([5.0, 5.0]), self._layer([5.0, 5.0])]
        out = place_by_threshold_search(layers, budget=8)
        assert sum(x.moves for x in out) == 0

    def test_it_spends_where_the_skew_is(self):
        """The skewed layer draws the budget; the even one draws none."""
        skewed = self._layer([20.0, 0.0], experts=[[10.0, 10.0], [0.0, 0.0]])
        even = self._layer([5.0, 5.0], experts=[[5.0], [5.0]])
        out = place_by_threshold_search([skewed, even], budget=2)
        assert out[0].moves >= 1
        assert out[1].moves == 0

    def test_the_budget_is_never_exceeded(self):
        layers = [
            self._layer([40.0, 0.0], experts=[[10.0] * 4, [0.0] * 4]) for _ in range(3)
        ]
        out = place_by_threshold_search(layers, budget=5)
        assert sum(x.moves for x in out) <= 5

    def test_a_bigger_budget_never_gives_a_worse_result(self):
        """Monotone in budget — a search that is not would be a bug, not a trade."""
        layers = [self._layer([20.0, 4.0], experts=[[8.0, 7.0, 5.0], [4.0, 0.0, 0.0]])]
        small = critical_path_imbalance(place_by_threshold_search(layers, budget=1))
        large = critical_path_imbalance(place_by_threshold_search(layers, budget=3))
        assert large <= small + 1e-9

    def test_a_placement_below_the_token_floor_is_refused(self):
        """Under one BLOCK_SIZE_M a replica saves no block, so it saves no time."""
        layers = [self._layer([100.0, 40.0], experts=[[100.0], [40.0]])]
        # Moving half of 100 is 50, above a floor of 10 but below a floor of 80.
        assert place_by_threshold_search(layers, budget=1, min_tokens=10)[0].moves == 1
        assert place_by_threshold_search(layers, budget=1, min_tokens=80)[0].moves == 0

    def test_it_never_pushes_a_target_past_the_peak_it_is_relieving(self):
        layers = [self._layer([6.0, 5.0], experts=[[6.0], [5.0]])]
        out = place_by_threshold_search(layers, budget=1)
        assert out[0].rank_load == pytest.approx([6.0, 5.0])


class TestOneSlotPerRankPerLayer:
    """A rank holds one replica slot per layer, so it can host one replica.

    Ignoring this let a layer place two replicas on the same rank — not a placement
    the hardware can express. Measured at 6.6% of placements at one per layer and
    27.3% at three, so the numbers it inflated were not marginal.
    """

    def test_a_target_rank_is_not_reused_within_a_layer(self):
        layer = Layer(
            rank_load=[40.0, 1.0, 1.0],
            expert_load=[
                [10.0, 10.0, 10.0, 10.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
        )
        (out,) = place_globally([layer], budget=4)
        # Only two ranks can host, so at most two placements are possible.
        assert out.moves <= 2

    def test_the_search_respects_it_too(self):
        layer = Layer(
            rank_load=[40.0, 1.0, 1.0],
            expert_load=[
                [10.0, 10.0, 10.0, 10.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
        )
        (out,) = place_by_threshold_search([layer], budget=4)
        assert out.moves <= 2

    def test_placements_stop_when_every_slot_is_taken(self):
        """Two ranks means two slots — each can host a replica of the other's expert.

        A third placement is impossible however much budget remains.
        """
        layer = Layer(rank_load=[9.0, 1.0], expert_load=[[9.0], [1.0]])
        (out,) = place_globally([layer], budget=9)
        assert out.moves == 2


class TestPlanningIsSeparableFromEvaluation:
    """Ticket 10 needs to plan on one load and score the result on another.

    The number that decides the prefill build is the benefit of placements chosen
    from *predicted* load and scored against the *actual* load of the same forward.
    That requires the move list, not just the placed layers.
    """

    def _skewed(self):
        return imbalance.Layer(
            rank_load=[100.0, 20.0, 20.0, 20.0],
            expert_load=[[60.0, 40.0], [10.0, 10.0], [10.0, 10.0], [10.0, 10.0]],
        )

    def test_replaying_the_moves_reproduces_place_globally(self):
        direct = imbalance.place_globally([self._skewed()], budget=2)
        moves = imbalance.plan_moves([self._skewed()], budget=2)
        replayed = imbalance.apply_moves([self._skewed()], moves)
        assert [x.rank_load for x in replayed] == [x.rank_load for x in direct], (
            "replaying the planned moves must land exactly where place_globally did, "
            "or planning and evaluation are not the same policy"
        )

    def test_a_perfect_prediction_matches_the_oracle(self):
        oracle = imbalance.place_globally([self._skewed()], budget=2)
        moves = imbalance.plan_moves([self._skewed()], budget=2)
        scored = imbalance.apply_moves([self._skewed()], moves)
        assert imbalance.critical_path_imbalance(scored) == pytest.approx(
            imbalance.critical_path_imbalance(oracle)
        ), "planning on a load identical to the scoring load is the oracle case"

    def test_a_wrong_prediction_can_be_worse_than_placing_nothing(self):
        actual = self._skewed()
        # Predicts the peak on the wrong rank entirely.
        predicted = imbalance.Layer(
            rank_load=[20.0, 100.0, 20.0, 20.0],
            expert_load=[[10.0, 10.0], [60.0, 40.0], [10.0, 10.0], [10.0, 10.0]],
        )
        oracle = imbalance.critical_path_imbalance(
            imbalance.place_globally([self._skewed()], budget=1)
        )
        misled = imbalance.critical_path_imbalance(
            imbalance.apply_moves(
                [self._skewed()], imbalance.plan_moves([predicted], budget=1)
            )
        )
        base = imbalance.critical_path_imbalance([actual])
        # A misled placement adds load to the rank the layer is actually waiting
        # on, so it overshoots the baseline rather than merely failing to help.
        # The realistic benefit is therefore *not* the oracle bound times a recall
        # figure: that product cannot go negative and this quantity can. Ticket 10
        # says otherwise and is wrong on this point; the benefit has to be measured
        # by replaying predicted-chosen moves against actual load, which is what
        # this seam is for.
        assert oracle < base < misled, (
            f"a prediction pointing at the wrong rank moves load onto the true "
            f"peak, so it should end up worse than the {base:.4f} baseline, not "
            f"merely short of the {oracle:.4f} oracle; got {misled:.4f}"
        )

    def test_a_move_onto_a_rank_that_is_no_longer_light_can_be_replayed(self):
        # Replay must not silently drop a move that is legal in the plan but
        # unhelpful in the scored load; the benefit measured must include its cost.
        predicted = imbalance.Layer(
            rank_load=[100.0, 10.0, 20.0, 20.0],
            expert_load=[[60.0, 40.0], [5.0, 5.0], [10.0, 10.0], [10.0, 10.0]],
        )
        moves = imbalance.plan_moves([predicted], budget=1)
        assert len(moves) == 1
        scored = imbalance.apply_moves([self._skewed()], moves)
        assert scored[0].moves == 1, "the move must be applied, not skipped"
