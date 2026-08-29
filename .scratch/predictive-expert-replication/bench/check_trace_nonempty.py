# SPDX-License-Identifier: Apache-2.0
"""Fail if a trace directory holds no step annotation, i.e. the run measured nothing.

Written because a DSV4 run reported "done", wrote eight rank traces, and had served zero
requests: the base checkpoint has no chat template, the chat endpoint raised before
sending anything, and the traces held 54 events and no `gpu_user_annotation`. Every
harness failure in this project has this shape — a wrong number or a silent no-op, never
an error — so the guard belongs in the runner, not in the reader.

Exit 0 if at least one rank trace carries an `execute_context_*` annotation.
"""

import glob
import gzip
import json
import re
import sys
from pathlib import Path

_STEP = re.compile(r"execute_context_\d+\(\d+\)")


def count_step_annotations(trace_dir: Path | str, verbose: bool = False) -> int:
    """Total `execute_context_*` annotations across a directory's rank traces.

    Returns 0 both when there are no traces and when the traces are empty, because for
    the caller those are the same failure: nothing was captured.
    """
    paths = sorted(glob.glob(str(trace_dir) + "/*.pt.trace.json*"))
    if not paths:
        if verbose:
            print(f"no rank traces under {trace_dir}", file=sys.stderr)
        return 0
    total = 0
    for path in paths:
        opener = gzip.open if path.endswith(".gz") else open
        events = json.load(opener(path, "rt")).get("traceEvents", [])
        hits = sum(
            1
            for e in events
            if e.get("cat") == "gpu_user_annotation"
            and _STEP.search(e.get("name") or "")
        )
        total += hits
        if verbose:
            print(
                f"{path.split('/')[-1]}: {hits} step annotations, {len(events)} events"
            )
    return total


if __name__ == "__main__":
    sys.exit(0 if count_step_annotations(sys.argv[1], verbose=True) else 1)
