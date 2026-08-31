# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Each target in a group must be planned from its **own** predicted load.

Ticket 12 asserted equality at a group of 1 and nothing about which row a target reads
at
a group of 4. That gap is why the first K=4 measurement placed on 44 layers and removed
3.7% of critical-path excess where a group of 1 removes 26.5%: placements happened,
`connected` was true, nothing raised, and most layers were shedding an expert that was
never theirs. A layer whose plan comes from a neighbour's prediction helps only when the
two layers' hot experts happen to coincide — measured, 13 of 44 layers improved and 4
got
worse, which is what coincidence looks like.

The check is differential and needs no GPU: drive the real coordinator with a snapshot
whose rows are deliberately different, and assert each target's plan is the one that
target's row implies.
"""

from __future__ import annotations

import pytest
import torch

from vllm.distributed.eplb.predictive_planner import plan_replicas

EP_SIZE = 2
NUM_LOGICAL = 8
PER_RANK = NUM_LOGICAL // EP_SIZE


def _snapshot_with_distinct_rows(group: int) -> torch.Tensor:
    """`[ep_size, group, num_logical]` where each target's peak expert is different.

    Target `j` makes expert `j` the hot one on rank 0, so a plan naming a different
    expert
    is proof that the target read another target's row.
    """
    snapshot = torch.ones((EP_SIZE, group, NUM_LOGICAL), dtype=torch.int32) * 10
    for target in range(group):
        snapshot[0, target, target] = 1000
    return snapshot


def _expected_expert(snapshot: torch.Tensor, target: int) -> int:
    """What the retained host planner chooses from this target's row alone."""
    summed = snapshot[:, target, :].sum(dim=0).to(torch.float64)
    placements = plan_replicas(summed.unsqueeze(0), ep_size=EP_SIZE, budget=1)
    assert placements, "the fixture must produce a placement to compare against"
    return placements[0].logical_expert


class TestEachTargetPlansFromItsOwnRow:
    @pytest.mark.parametrize("group", [1, 2, 4])
    def test_the_plan_names_the_experts_the_rows_imply(self, group):
        snapshot = _snapshot_with_distinct_rows(group)
        planned: dict[int, int] = {}

        # A stand-in for the coordinator that records what each target was planned from,
        # using the retained host planner as the oracle. The runner's loop is the thing
        # under test, so it is driven for real.
        class _Coordinator:
            launch_at_predicting_layer_tail = True
            lookahead = 4

            def __init__(self):
                self._recorded = None

            def record_prediction(self, source_layer, predicted, target_offset=0):
                self._recorded = (
                    source_layer + self.lookahead + target_offset,
                    predicted,
                )

            def plan_and_launch(self):
                assert self._recorded is not None, "plan_and_launch without a record"
                target, predicted = self._recorded
                self._recorded = None
                summed = predicted.sum(dim=0).to(torch.float64)
                placements = plan_replicas(
                    summed.unsqueeze(0), ep_size=EP_SIZE, budget=1
                )
                planned[target] = placements[0].logical_expert if placements else -1
                return []

        from types import SimpleNamespace

        from vllm.model_executor.layers.fused_moe.runner import moe_runner as mod

        # Layer 3 is the window's last source at a lookahead of 4, so the window's
        # sources are 0..3 and its targets are 4..7 — one per source, each from its own.
        runner = SimpleNamespace(
            placement_coordinator=_Coordinator(),
            moe_layer_index=3,
            _forward_tokens_per_expert=lambda: 4096.0,
            load_predictor=SimpleNamespace(window=SimpleNamespace(size=group)),
        )
        mod.MoERunner._placement_after_snapshot(runner, snapshot)

        # Layer 3 is the window's last source at a lookahead of 4, so the window covers
        # sources `3 - (group - 1) .. 3` and their targets are `8 - group .. 7`.
        first_target = 8 - group
        assert sorted(planned) == [first_target + p for p in range(group)], (
            "every source in the window must be planned exactly once"
        )
        for offset in range(group):
            target = first_target + offset
            expected = _expected_expert(snapshot, offset)
            # Naming which row it *did* read is what turns a failure into a diagnosis.
            read_from = [
                other
                for other in range(group)
                if _expected_expert(snapshot, other) == planned[target]
            ]
            assert planned[target] == expected, (
                f"target layer {target} was planned from row {read_from} instead of "
                f"its own row {offset}"
            )

    def test_the_rows_really_are_distinguishable(self):
        """Otherwise the test above would pass on a path that reads any row."""
        snapshot = _snapshot_with_distinct_rows(4)
        chosen = {_expected_expert(snapshot, offset) for offset in range(4)}
        assert len(chosen) == 4, f"the fixture must separate the targets, got {chosen}"
