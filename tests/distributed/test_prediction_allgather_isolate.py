# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The measurement switch that separates prediction's collective from its compute.

Prediction costs +13.5% mean TTFT at EP=8 and the profile cannot say which half it is.
The 44 per-layer snapshot AllGathers are barriers: measured, the *last* rank to arrive
sees 9.4 us and the other seven see 130 to 748 us, so a single rank's view prices the
kernel and not the coupling. The window's length tracks the *number* of collectives —
192 at 88.9 ms and 236 at 106.9 ms, 0.463 against 0.453 ms each — but that is two points
and the AllGather count and the launch count move together.

`VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER` holds prediction's compute fixed and removes
only the collective. It is a **cost probe and nothing else**: the snapshot it produces
is this rank's own counts, so ranks no longer agree, so a plan from it is per-rank. That
is why it refuses to run when placement is armed rather than trusting the operator — a
per-rank plan pairs a send with no receive, which on this branch means a deadlock or a
replica row holding another expert's weights.
"""

from __future__ import annotations

import pytest
import torch

from vllm.distributed.eplb.predictive import PredictionWindow


class TestSkippingTheAllGatherKeepsTheShapeAndTheCounts:
    """The probe must change one thing: whether a collective runs.

    It lives on `PredictionWindow` since ticket 13's second design, because that is
    where
    the collective moved: a window's sources each write a row and only the last one
    gathers.
    """

    def _window(self, size, num_logical, ep_size):
        window = PredictionWindow(size=size, num_logical_experts=num_logical)
        window._ep_size = ep_size
        for position in range(size):
            window.row(position, torch.device("cpu"), num_logical)
        return window

    def test_the_snapshot_still_has_a_row_per_rank_and_per_source(self, monkeypatch):
        # The planner indexes `[source rank, window position, logical expert]`, so a
        # probe
        # returning a different shape would exercise a different code path downstream.
        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER", True, raising=False
        )
        window = self._window(size=2, num_logical=8, ep_size=4)
        window.counts[0, 3] = 7

        window.start_snapshot()
        snapshot = window.finish_snapshot()

        assert snapshot is not None
        assert snapshot.shape == (4, 2, 8)

    def test_every_rank_row_is_this_ranks_own_counts(self, monkeypatch):
        """Which is exactly why it may not be used to plan: rows are not per rank."""
        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER", True, raising=False
        )
        window = self._window(size=1, num_logical=4, ep_size=2)
        window.counts[0] = torch.tensor([7, 0, 0, 1], dtype=torch.int32)

        window.start_snapshot()
        snapshot = window.finish_snapshot()

        assert torch.equal(snapshot[0], snapshot[1])
        assert snapshot[0, 0].tolist() == [7, 0, 0, 1]

    def test_no_collective_is_started(self, monkeypatch):
        # The whole point. If `all_gather_into_tensor` still ran, the probe would
        # measure
        # the thing it exists to remove.
        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER", True, raising=False
        )
        called = []
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_into_tensor",
            lambda *a, **k: called.append(1),
        )
        window = self._window(size=2, num_logical=4, ep_size=2)

        window.start_snapshot()
        window.finish_snapshot()

        assert called == []

    def test_a_window_that_never_predicted_returns_nothing(self, monkeypatch):
        """The runner reaches layers whose window has not been written this forward."""
        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER", True, raising=False
        )
        window = PredictionWindow(size=1, num_logical_experts=4)
        assert window.finish_snapshot() is None


class TestThePlacementPathRefusesTheProbe:
    """A per-rank plan is a deadlock, so this is rejected rather than warned about."""

    def test_arming_placement_with_the_probe_raises(self, monkeypatch):
        from vllm.distributed.eplb.eplb_state import (
            reject_snapshot_probe_with_placement,
        )

        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER", True, raising=False
        )
        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_PLACE_PER_FORWARD", 43, raising=False
        )
        with pytest.raises(RuntimeError, match="cost probe"):
            reject_snapshot_probe_with_placement()

    def test_the_probe_alone_is_allowed(self, monkeypatch):
        from vllm.distributed.eplb.eplb_state import (
            reject_snapshot_probe_with_placement,
        )

        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER", True, raising=False
        )
        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_PLACE_PER_FORWARD", 0, raising=False
        )
        reject_snapshot_probe_with_placement()

    def test_placement_alone_is_allowed(self, monkeypatch):
        from vllm.distributed.eplb.eplb_state import (
            reject_snapshot_probe_with_placement,
        )

        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER", False, raising=False
        )
        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_PLACE_PER_FORWARD", 43, raising=False
        )
        reject_snapshot_probe_with_placement()
