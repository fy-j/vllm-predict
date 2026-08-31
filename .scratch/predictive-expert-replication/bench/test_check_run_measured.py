# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The measured-nothing guard, tested on the failures it exists to catch.

Ticket 01. A guard nobody tests is a guard that rots into a no-op, and this one guards
against the failure mode that has cost this project the most: a run that reports success
having served no request or captured no annotation. Each test below reconstructs one of
the three real incidents, or one of the two ways an arm can lie about placement.

The fixtures are deliberately minimal — a bench JSON, a server log, a dump — because
that is all the guard reads, and a fixture that mirrored a real run would hide which
field the guard actually depends on.
"""

import gzip
import json

import pytest
from check_run_measured import (
    _arm_records_load,
    _arm_should_place,
    _budget_of,
    check,
)
from check_trace_nonempty import count_step_annotations

PLACEMENT_LINE = "Predictive expert replication: activated 1 replica(s) on layer 5"


def _arm(
    results_dir, arm, *, ttft=120.0, completed=400, failed=0, placements=0, dump=True
):
    """Write the files one arm of a run leaves behind."""
    if ttft is not None:
        (results_dir / f"bench-b{arm}.json").write_text(
            json.dumps({"mean_ttft_ms": ttft, "completed": completed, "failed": failed})
        )
    (results_dir / f"server-b{arm}.log").write_text(
        "\n".join([PLACEMENT_LINE] * placements)
    )
    if dump:
        (results_dir / f"dump-b{arm}.jsonl").write_text(
            json.dumps(
                {"rank_load": [[1] * 8], "logical_load": [[1] * 8], "ep_size": 8}
            )
            + "\n"
        )


def test_a_sound_three_arm_run_passes(tmp_path):
    """The guard must not fire on a healthy run, or whoever hits it will remove it."""
    _arm(tmp_path, "off", placements=0, dump=False)
    _arm(tmp_path, "0", placements=0)
    _arm(tmp_path, "43", placements=344)

    assert check(tmp_path, ["off", "0", "43"], None) == []


def test_an_arm_that_served_no_request_is_caught(tmp_path):
    """The DSV4 incident: the endpoint raised before a single request went out.

    The runner still printed `done` and reported eight rank traces, which held 54
    events.
    """
    _arm(tmp_path, "off", ttft=None, dump=False)

    problems = check(tmp_path, ["off"], None)

    assert len(problems) == 1
    assert "no request was served" in problems[0]


def test_an_empty_dump_is_caught_where_load_is_recorded(tmp_path):
    """An arm whose instrumentation never ran cannot support any imbalance claim."""
    _arm(tmp_path, "0", dump=False)

    problems = check(tmp_path, ["0"], None)

    assert any("dump is empty" in p for p in problems)


def test_the_stock_arm_is_not_expected_to_dump(tmp_path):
    """`off` records no expert load by design, so an absent dump there is correct.

    Getting this backwards would make the honest baseline arm always fail, and the usual
    response to a guard that cries wolf is to remove it.
    """
    _arm(tmp_path, "off", dump=False)

    assert check(tmp_path, ["off"], None) == []


def test_an_inert_placed_arm_is_caught(tmp_path):
    """A non-zero budget that activated nothing is the silent-inert failure.

    Five separate defects in this project each produced exactly this and each looked
    healthy.
    """
    _arm(tmp_path, "43", placements=0)

    problems = check(tmp_path, ["43"], None)

    assert any("inert" in p for p in problems)


def test_a_placed_arm_that_is_not_connected_is_caught(tmp_path):
    """Replicas activated, requests failed: placed but nothing usable came of it.

    This is the case `is_connected` exists for, and it has read as an all-clear twice.
    """
    _arm(tmp_path, "43", placements=344, completed=120, failed=56)

    problems = check(tmp_path, ["43"], None)

    assert any("not connected" in p for p in problems)


def test_an_arm_that_must_not_place_but_did_is_caught(tmp_path):
    """`0` predicts and withholds placement, so any activation is a wiring bug.

    Caught in the other direction too, because an arm quietly placing would make the
    prediction-only baseline measure placement and understate the feature's cost.
    """
    _arm(tmp_path, "0", placements=12)

    problems = check(tmp_path, ["0"], None)

    assert any("must not place" in p for p in problems)


def test_several_problems_are_all_reported(tmp_path):
    """One message per failure, so a fix does not reveal the next one run at a time."""
    _arm(tmp_path, "0", ttft=None, dump=False, placements=7)

    problems = check(tmp_path, ["0"], None)

    assert len(problems) == 3


@pytest.mark.parametrize("events,expected", [([], 0), ([{"cat": "kernel"}] * 5, 0)])
def test_a_trace_without_step_annotations_counts_zero(tmp_path, events, expected):
    """The trace side of the same failure: eight files, none of them useful."""
    with gzip.open(tmp_path / "dp0_x.pt.trace.json.gz", "wt") as handle:
        json.dump({"traceEvents": events}, handle)

    assert count_step_annotations(tmp_path) == expected


def test_a_trace_with_step_annotations_counts_them(tmp_path):
    events = [
        {
            "cat": "gpu_user_annotation",
            "name": "execute_context_2(1024)_generation_0(0)",
        },
        {"cat": "gpu_user_annotation", "name": "something_else"},
        {"cat": "kernel", "name": "execute_context_2(1024)"},
    ]
    with gzip.open(tmp_path / "dp0_x.pt.trace.json.gz", "wt") as handle:
        json.dump({"traceEvents": events}, handle)

    assert count_step_annotations(tmp_path) == 1, (
        "only gpu_user_annotation counts; the CPU-side annotation and the kernels "
        "would double the step count"
    )


def test_a_missing_trace_directory_counts_zero(tmp_path):
    """No traces and empty traces are the same failure for the caller."""
    assert count_step_annotations(tmp_path / "absent") == 0


def test_repeat_suffixed_arms_are_classified_by_their_budget(tmp_path):
    """A repeated run labels arms `off-r1`, `0-r2`; the checks are about the budget.

    Comparing the whole label against "off" made every repeat look like a placing arm,
    so the guard faulted the stock baseline for having no dump and no activation,
    failing a run whose nine arms were healthy. Pinned, because a guard that cries wolf
    gets deleted.
    """
    _arm(tmp_path, "off-r1", placements=0, dump=False)
    _arm(tmp_path, "0-r1", placements=0)
    _arm(tmp_path, "43-r1", placements=344)
    _arm(tmp_path, "off-r2", placements=0, dump=False)
    _arm(tmp_path, "0-r2", placements=0)
    _arm(tmp_path, "43-r2", placements=344)

    problems = check(
        tmp_path, ["off-r1", "0-r1", "43-r1", "off-r2", "0-r2", "43-r2"], None
    )

    assert problems == []


def test_a_repeat_suffixed_placed_arm_is_still_checked(tmp_path):
    """Stripping the suffix must not stop the real checks applying to each pass."""
    _arm(tmp_path, "43-r2", placements=0)

    problems = check(tmp_path, ["43-r2"], None)

    assert any("inert" in p for p in problems)


class TestTheBudgetSurvivesEveryLabelDecoration:
    """An arm label carries transport and group infixes now, and a repeat suffix.

    This parser has cried wolf twice on a label change, and each time the run it
    failed was
    healthy: the repeat suffix made the stock arm look like a placing arm, and the
    transport/group infixes made a zero-budget arm look non-zero, so the guard called the
    prediction-only arm inert for correctly placing nothing.
    """

    @pytest.mark.parametrize(
        "label,budget",
        [
            ("off", "off"),
            ("off-r1", "off"),
            ("0", "0"),
            ("0-r2", "0"),
            ("43", "43"),
            ("43-host", "43"),
            ("43-host-r2", "43"),
            ("0-device-g1", "0"),
            ("0-device-g4-r3", "0"),
            ("43-device-g4-r1", "43"),
        ],
    )
    def test_the_leading_field_is_the_budget(self, label, budget):
        assert _budget_of(label) == budget

    def test_a_zero_budget_arm_is_never_expected_to_place(self):
        # The false alarm itself: `0-device-g1-r2` has budget 0 and correctly places
        # nothing, and the guard used to fail the whole run over it.
        assert _arm_should_place("0-device-g1-r2") is False
        assert _arm_should_place("43-device-g4-r1") is True

    def test_the_stock_arm_still_records_no_load(self):
        assert _arm_records_load("off-r4") is False
        assert _arm_records_load("0-device-g4-r4") is True
