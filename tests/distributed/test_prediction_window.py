# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One collective for a window of source layers, each predicting its own target.

Ticket 13, second design. The first one had a single source layer predict `K` targets
from
its own hidden states, and it failed for a reason no unit test had looked for: four
different layers' gates applied to the *same* hidden states select almost the same
experts. Measured on the real model, the four predicted distributions within a group
differed by an L1 of 16 to 36 out of 7896 assignments while the four target layers'
actual
loads differed by 5412 to 7660 — so three of every four targets were planned from a
distribution that was not theirs, and placement's benefit fell from 26.9% of
critical-path
excess to 3.7%.

So the predictions must come from `K` **different source layers**, which is what makes
them
distinguishable, and the batching moves to the collective: the window's sources each
write
their own row and the last one issues a single AllGather.

The hard constraint that follows is `lookahead >= group`: the window's collective is
issued
at its last source, and the first target must not have run yet.
"""

from __future__ import annotations

import pytest
import torch

from vllm.distributed.eplb.predictive import (
    PredictionWindow,
    bind_moe_prediction_targets,
)


class _Runner:
    def __init__(self):
        self.moe_layer_index = None
        self.target = None
        self.window = None
        self.window_position = None

    def bind_prediction_target(self, target, window, position):
        self.target = target.moe_layer_index
        self.window = window
        self.window_position = position


def _bind(num_layers, lookahead, skip, group):
    runners = [_Runner() for _ in range(num_layers)]
    sources = bind_moe_prediction_targets(runners, lookahead, skip, group)
    return runners, sources


class TestEverySourcePredictsItsOwnTarget:
    """The fix for the first design: one source, one target, always."""

    @pytest.mark.parametrize("group", [1, 2, 4])
    def test_each_source_keeps_a_single_target_at_the_configured_distance(self, group):
        runners, sources = _bind(48, lookahead=4, skip=3, group=group)
        for index in sources:
            assert runners[index].target == index + 4, (
                "a source must predict the layer `lookahead` ahead, from its own "
                "hidden "
                "states; sharing one source's prediction across a group is what failed"
            )

    def test_every_layer_is_still_a_source(self):
        """Coverage of *sources* is what keeps predictions distinguishable."""
        _, sources = _bind(48, lookahead=4, skip=3, group=4)
        assert sources == list(range(3, 44))
        assert len(sources) == 41

    @pytest.mark.parametrize("group", [1, 2, 4])
    def test_every_reachable_target_is_covered_exactly_once(self, group):
        runners, sources = _bind(48, lookahead=4, skip=3, group=group)
        covered = sorted(runners[i].target for i in sources)
        assert covered == list(range(7, 48))
        assert len(covered) == len(set(covered))


class TestOneCollectivePerWindow:
    def test_sources_are_grouped_into_windows_of_the_group_size(self):
        runners, sources = _bind(48, lookahead=4, skip=3, group=4)
        windows = [runners[i].window for i in sources]
        positions = [runners[i].window_position for i in sources]
        # 41 sources in windows of 4: ten full windows and a short final one.
        assert positions[:8] == [0, 1, 2, 3, 0, 1, 2, 3]
        assert windows[0] is windows[3]
        assert windows[0] is not windows[4]
        assert len({id(w) for w in windows}) == 11

    def test_a_short_final_window_is_allowed(self):
        """41 sources do not divide by 4, and rejecting that would cost coverage.

        The first design had to reject a group that did not divide the span, because a
        remainder there meant target layers that never received a prediction. Here the
        remainder is a *window*, and a short window simply issues its collective one
        source early.
        """
        runners, sources = _bind(48, lookahead=4, skip=3, group=4)
        last = [runners[i] for i in sources][-1]
        assert last.window_position == 0
        assert last.window.size == 1, (
            "the final window holds the single leftover source"
        )

    def test_only_the_last_source_of_a_window_issues_the_collective(self):
        runners, sources = _bind(48, lookahead=4, skip=3, group=4)
        issuing = [
            i
            for i in sources
            if runners[i].window.issues_at(runners[i].window_position)
        ]
        # One per window, and it is the window's last source.
        assert issuing == [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 43]
        assert len(issuing) == 11


class TestTheWindowHoldsARowPerSource:
    def test_each_source_writes_its_own_row(self):
        window = PredictionWindow(size=3, num_logical_experts=4)
        for position in range(3):
            row = window.row(position, torch.device("cpu"))
            row.fill_(position + 1)
        assert window.counts.tolist() == [[1] * 4, [2] * 4, [3] * 4]

    def test_the_snapshot_maps_a_position_back_to_its_row(self, monkeypatch):
        window = PredictionWindow(size=2, num_logical_experts=3)
        for position in range(2):
            window.row(position, torch.device("cpu")).fill_(position + 5)

        def fake_all_gather(out, local, group=None, async_op=False):
            out.view(2, -1)[0] = local
            out.view(2, -1)[1] = local

            class _Work:
                def wait(self_inner):
                    return None

            return _Work()

        monkeypatch.setattr(
            "vllm.distributed.eplb.predictive.get_eplb_group",
            lambda: type(
                "G",
                (),
                {"device_group": type("D", (), {"size": staticmethod(lambda: 2)})()},
            )(),
        )
        monkeypatch.setattr(
            torch.distributed, "all_gather_into_tensor", fake_all_gather
        )

        window.start_snapshot()
        snapshot = window.finish_snapshot()
        assert snapshot.shape == (2, 2, 3)
        assert snapshot[0, 0].tolist() == [5, 5, 5]
        assert snapshot[0, 1].tolist() == [6, 6, 6]


class TestTheDistanceMustCoverTheWindow:
    def test_a_group_larger_than_the_lookahead_is_rejected(self):
        # The collective is issued at the window's last source, so with lookahead 1 and
        # a
        # group of 4 the first target has already run by then. Rejected rather than
        # silently planning a layer that is in the past.
        with pytest.raises(ValueError, match="lookahead"):
            _bind(48, lookahead=1, skip=3, group=4)

    def test_a_group_equal_to_the_lookahead_is_allowed(self):
        _, sources = _bind(48, lookahead=4, skip=3, group=4)
        assert sources

    def test_a_group_of_one_needs_no_lookahead_at_all(self):
        """The shipping configuration: lookahead 1, group 1."""
        runners, sources = _bind(48, lookahead=1, skip=3, group=1)
        assert sources == list(range(3, 47))
        assert runners[3].target == 4
        assert runners[3].window.size == 1


class TestTheGroupKnob:
    """The knob defaults to the shipping value and refuses the path that cannot hold
    it."""

    def _config(self, tmp_path, **overrides):
        import json

        from vllm.config.parallel import PredictiveExpertReplicationConfig

        profile = {
            "fingerprint": {
                "model": "m",
                "dtype": "bfloat16",
                "ep_size": 8,
                "num_logical_experts": 128,
                "device_name": "NVIDIA H100 80GB HBM3",
            },
            "expert_compute_us_per_token": 0.4,
            "attention_window_us": 900.0,
            "transfer_latency_us": 30.0,
            "usable_transfer_bandwidth_bytes_per_us": 20_000.0,
        }
        path = tmp_path / "cp.json"
        path.write_text(json.dumps(profile))
        return PredictiveExpertReplicationConfig(
            enabled=True, cost_profile_path=str(path), **overrides
        )

    def test_the_default_is_the_shipping_value(self, tmp_path):
        assert self._config(tmp_path).prediction_target_group == 1

    def test_a_group_is_accepted_on_the_device_path(self, tmp_path):
        # With a lookahead that covers the window; the pairing is checked separately.
        config = self._config(
            tmp_path, prediction_target_group=4, prediction_lookahead_layers=4
        )
        assert config.prediction_target_group == 4

    def test_the_host_path_refuses_a_group(self, tmp_path):
        # It holds one recorded prediction at a time, so a larger group would plan only
        # the last target and leave the rest unplaced with nothing logged.
        with pytest.raises(ValueError, match="prediction_target_group=1 only"):
            self._config(
                tmp_path,
                prediction_target_group=4,
                prediction_lookahead_layers=4,
                device_issued_transfer=False,
            )

    def test_the_host_path_still_accepts_a_group_of_one(self, tmp_path):
        config = self._config(
            tmp_path,
            prediction_target_group=1,
            device_issued_transfer=False,
            prediction_lookahead_layers=2,
        )
        assert config.prediction_target_group == 1

    def test_a_group_larger_than_the_lookahead_is_rejected_at_startup(self, tmp_path):
        # The binder rejects this too, but by then a model is being built. Startup is
        # where a bad configuration should stop.
        with pytest.raises(ValueError, match="at least"):
            self._config(
                tmp_path, prediction_target_group=4, prediction_lookahead_layers=2
            )

    def test_a_group_equal_to_the_lookahead_is_accepted(self, tmp_path):
        config = self._config(
            tmp_path, prediction_target_group=4, prediction_lookahead_layers=4
        )
        assert config.prediction_target_group == 4


class TestStagingBuffersCoverAWholeWindow:
    """Two same-window targets must never share a staging workspace.

    The transfer is put, barrier, drain, and `barrier_all` orders arrival — not one
    rank's next put against another's drain. Two buffers sufficed while consecutive
    same-buffer layers were a layer of compute apart; a window launches all its
    transfers at its last source, microseconds apart, which removes exactly that
    margin. So the buffer index has to separate every position in a window, not just
    odd from even.
    """

    def _offsets(self, first_target, group, buffers, stride=1):
        return [
            (t % buffers) * stride for t in range(first_target, first_target + group)
        ]

    def test_a_window_of_four_uses_four_distinct_buffers(self):
        for first_target in range(8):
            offsets = self._offsets(first_target, group=4, buffers=4)
            assert len(set(offsets)) == 4, (
                f"targets {first_target}..{first_target + 3} collide: {offsets}"
            )

    def test_two_buffers_would_collide_inside_a_window_of_four(self):
        """The control: this is the hazard the change removes, not a hypothetical."""
        offsets = self._offsets(4, group=4, buffers=2)
        assert len(set(offsets)) == 2, offsets

    def test_a_window_of_one_keeps_the_two_buffer_alternation(self):
        """The shipping configuration must be untouched."""
        assert self._offsets(0, group=1, buffers=2) == [0]
        assert self._offsets(1, group=1, buffers=2) == [1]

    def test_buffers_a_window_apart_may_share(self):
        # Separated by a whole window's compute, which is the margin two buffers
        # relied on, and the reason this needs `group` buffers not `num_layers`.
        assert (4 % 4) == (8 % 4)

    def test_fewer_than_two_buffers_is_rejected(self):
        from vllm.distributed.eplb.device_coordinator import DevicePlacementCoordinator

        with pytest.raises(ValueError, match="at least 2"):
            DevicePlacementCoordinator(
                ep_size=2,
                ep_rank=0,
                canonical_per_rank=2,
                replica_slots_per_rank=1,
                num_layers=4,
                lookahead=1,
                budget=1,
                min_tokens=0.0,
                min_tokens_per_expert=0.0,
                pointers=[],
                maps=[],
                transfer=None,
                device=torch.device("cpu"),
                staging_stride=1,
                staging_buffers=1,
            )


class TestTheDeterministicSnapshotBoundsWhatAReductionCanReturn:
    """Ticket 14's ceiling: no synchronisation at all, with placement still armed.

    Ticket 11's probe removed the collective but produced a *per-rank* snapshot, so
    placement had to be withheld and the bound only covered the prediction arm. This one
    is rank-identical by construction, so the plans still agree and the whole placement
    machinery runs — which is what makes its number a bound on the real path.
    """

    def _window(self, size, num_logical, ep_size):
        window = PredictionWindow(size=size, num_logical_experts=num_logical)
        window._ep_size = ep_size
        for position in range(size):
            window.row(position, torch.device("cpu"), num_logical)
        return window

    def _armed(self, monkeypatch):
        monkeypatch.setattr(
            "vllm.envs.VLLM_PREDICTIVE_DETERMINISTIC_SNAPSHOT", True, raising=False
        )

    def test_no_collective_runs(self, monkeypatch):
        self._armed(monkeypatch)
        called = []
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_into_tensor",
            lambda *a, **k: called.append(1),
        )
        window = self._window(size=2, num_logical=8, ep_size=4)

        window.start_snapshot()
        assert window.finish_snapshot() is not None
        assert called == []

    def test_two_windows_at_the_same_count_agree_exactly(self, monkeypatch):
        """This is the property placement depends on: two ranks must see one snapshot.

        Two windows stepped the same number of times stand in for two ranks in lockstep.
        If they diverged, each rank would derive its own plan and a put would pair
        with a
        peer expecting nothing — the failure the other probe is rejected for.
        """
        self._armed(monkeypatch)
        left = self._window(size=2, num_logical=8, ep_size=4)
        right = self._window(size=2, num_logical=8, ep_size=4)
        for _ in range(3):
            left.start_snapshot()
            right.start_snapshot()
            assert torch.equal(left.finish_snapshot(), right.finish_snapshot())

    def _hot_over(self, window, forwards):
        hot = []
        for _ in range(forwards):
            window.start_snapshot()
            hot.append(int(window.finish_snapshot()[0, 0].argmax()))
        return hot

    def test_the_hot_expert_changes_eventually_but_not_every_forward(self, monkeypatch):
        """The probe's churn rate is the thing that decides whether it bounds anything.

        Both directions invalidate it. A fixed choice lets residency make every transfer
        free, so the probe measures an idle path. Rotating every forward re-plans every
        layer every forward: measured, that activated 1634 replicas over 70 forwards
        where the real arm activates 408, and the probe came in 46 ms *slower* than the
        arm it was meant to bound. So both are asserted here, not just the first.
        """
        self._armed(monkeypatch)
        hot = self._hot_over(self._window(size=1, num_logical=8, ep_size=2), 12)

        changes = sum(a != b for a, b in zip(hot, hot[1:]))
        assert changes > 0, f"never rotated, so residency would hide the cost: {hot}"
        assert changes < len(hot) - 1, f"rotated every forward, a transfer storm: {hot}"

    def test_windows_are_staggered_so_they_do_not_all_rotate_together(
        self, monkeypatch
    ):
        """A bursty probe and a steady one do not cost the same.

        44 layers rotating on the same forward spends the whole budget in one forward
        and leaves the next three idle; a real path's transfers arrive spread out, so
        windows at different source layers rotate on different forwards.
        """
        self._armed(monkeypatch)
        first = PredictionWindow(size=1, num_logical_experts=8, phase=0)
        second = PredictionWindow(size=1, num_logical_experts=8, phase=1)
        for window in (first, second):
            window._ep_size = 2
            window.row(0, torch.device("cpu"), 8)

        changed_on = [
            {index for index, (a, b) in enumerate(zip(hot, hot[1:])) if a != b}
            for hot in (self._hot_over(first, 12), self._hot_over(second, 12))
        ]
        assert changed_on[0] and changed_on[1]
        assert changed_on[0] != changed_on[1], changed_on

    def test_the_snapshot_always_admits_a_placement(self, monkeypatch):
        """A snapshot with no clear peak would leave the transfer path unexercised."""
        self._armed(monkeypatch)
        window = self._window(size=1, num_logical=8, ep_size=2)
        window.start_snapshot()
        snapshot = window.finish_snapshot()
        summed = snapshot[:, 0, :].sum(dim=0)
        assert summed.max() > 10 * summed.median()
