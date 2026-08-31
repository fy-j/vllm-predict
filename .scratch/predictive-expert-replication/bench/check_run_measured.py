# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail a run that measured nothing, so exit code 0 means its arms actually ran.

Ticket 01. Three times in this project a run reported success having measured nothing:
an arm whose config was silently ignored because the console script loaded the installed
vLLM; a placed arm that activated replicas nothing routed to; and a DSV4 arm whose base
checkpoint had no chat template, so the chat endpoint raised inside the dataset sampler
before a single request went out and the traces held 54 events. Each printed a warning
to stderr and carried on, and each was read as a result.

The checks live here rather than in each runner because there are three runners, they
were drifting apart, and a check nobody can unit-test is a check that rots.
`analyse_e2e.py`'s `summarize` and `is_connected` are reused rather than reimplemented:
`is_connected` has been wrong twice, both times reading as an all-clear, and one copy of
that judgement is enough.

What counts as measuring nothing, per arm:

  * no TTFT in the benchmark result, which means no request was served;
  * an empty expert-load dump on an arm that records load, which means the feature's
    instrumentation never ran;
  * no placement log line on an arm with a non-zero budget, which means the arm was
  inert; * a placement log line on the `off` or `0` arm, which means an arm that must
  not place did; * for the placed arm, `is_connected` false — replicas activated but
  nothing routed to them.

Usage:
    python check_run_measured.py --results-dir <dir> --arms off 0 43
    python check_run_measured.py --results-dir <dir> --arms 0 --require-trace
    <dir>/trace-b0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analyse_e2e import is_connected, summarize


def _budget_of(arm: str) -> str:
    """The budget an arm name refers to, stripped of every label decoration.

    An arm label is `<budget>[-<transport>][-g<group>][-r<repeat>]`, and every check
    below
    is about the budget alone. This has now cried wolf twice on a label change, and each
    time the run it failed was healthy:

    * the repeat suffix (`off-r1`, `0-r2`) made every repeat look like a placing arm, so
      the guard faulted the stock baseline for having no dump and no activation;
    * the transport and group infixes (`0-device-g1`) made a **zero-budget** arm look
      non-zero, so the guard called the prediction-only arm inert for correctly placing
      nothing.

    So the budget is taken as the leading field rather than by removing known suffixes:
    anything else is a decoration this function must not know about. A guard that
    cries wolf gets deleted.
    """
    return arm.split("-")[0]


def _arm_records_load(arm: str) -> bool:
    """Whether this arm is expected to write an expert-load dump.

    The stock arm has no EPLB recording by design, so an absent dump there is the
    correct outcome rather than the failure it is on every other arm.
    """
    return _budget_of(arm) != "off"


def _arm_should_place(arm: str) -> bool:
    """Whether this arm is expected to activate replicas.

    `off` disables the feature and `0` enables prediction while withholding placement,
    so both must show no activation. String comparison throughout: `off` is not a
    number, and an arithmetic test on it under `set -u` aborts a runner *after* the arm
    has served its whole benchmark, which is how this was learned.
    """
    return _budget_of(arm) not in ("off", "0")


def check(results_dir: Path, arms: list[str], require_trace: Path | None) -> list[str]:
    """Return one message per way this run measured nothing. Empty means it is sound."""
    problems: list[str] = []

    for arm in arms:
        summary = summarize(results_dir, arm)
        label = f"arm {arm}"

        if not summary.get("mean_ttft_ms"):
            problems.append(
                f"{label}: no TTFT in the benchmark result, so no request was served"
            )

        dump = results_dir / f"dump-b{arm}.jsonl"
        empty_dump = not dump.exists() or dump.stat().st_size == 0
        if _arm_records_load(arm) and empty_dump:
            problems.append(
                f"{label}: expert-load dump is empty, so the recording never ran"
            )

        lines = summary.get("placement_log_lines") or 0
        if _arm_should_place(arm):
            if not lines:
                problems.append(
                    f"{label}: no placement log line, so the arm was inert despite a "
                    f"non-zero budget"
                )
            elif not is_connected(summary):
                problems.append(
                    f"{label}: activated {lines} replica(s) but is not connected "
                    f"(completed={summary.get('completed')} "
                    f"failed={summary.get('failed')}) — placed, nothing routed to them"
                )
        elif lines:
            problems.append(
                f"{label}: activated {lines} replica(s), but this arm must not place"
            )

    if require_trace is not None:
        # Imported here so a run without traces does not pay for it, and so this module
        # stays usable when only the benchmark side is being checked.
        from check_trace_nonempty import count_step_annotations

        found = count_step_annotations(require_trace)
        if not found:
            problems.append(
                f"traces under {require_trace} carry no step annotation, so there is "
                f"nothing to attribute"
            )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--require-trace", type=Path, default=None)
    args = parser.parse_args()

    problems = check(args.results_dir, args.arms, args.require_trace)
    if problems:
        print("MEASURED NOTHING — do not read these results:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 5
    print(
        f"[check] all {len(args.arms)} arm(s) served requests and behaved as expected"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
