# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the end-to-end run's own self-check.

This verdict has been wrong twice, both times reading as an all-clear: once from a
placement-log pattern that never matched a real line, and once from a fatal marker
that fires at every clean teardown. An all-clear on a broken arm is what costs a run,
so the rules are pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyse_e2e import is_connected  # noqa: E402


def _placed(**overrides):
    base = {"placement_log_lines": 152, "failed": 0, "completed": 120, "fatal": 0}
    base.update(overrides)
    return base


def test_a_healthy_placed_arm_is_connected():
    assert is_connected(_placed())


def test_teardown_fatals_do_not_make_it_disconnected():
    """The regression this replaced: 152 activations, 0 failures, called inert."""
    assert is_connected(_placed(fatal=2))


def test_no_placement_lines_is_not_connected():
    assert not is_connected(_placed(placement_log_lines=0))


def test_failed_requests_are_not_connected():
    """The hung run: 64 succeeded, 56 failed, empty dump."""
    assert not is_connected(_placed(completed=64, failed=56))


def test_no_completed_requests_is_not_connected():
    assert not is_connected(_placed(completed=0, failed=0))
