# SPDX-License-Identifier: Apache-2.0
"""Fail if a trace directory holds no step annotation, i.e. the run measured nothing.

Written because a DSV4 run reported "done", wrote eight rank traces, and had served
zero requests: the base checkpoint has no chat template, the chat endpoint raised before
sending anything, and the traces contained 54 events and no `gpu_user_annotation` at
all. Every harness failure in this project has this shape — a wrong number or a silent
no-op, never an error — so the guard belongs in the runner, not in the reader.

Exit 0 if at least one rank trace carries an `execute_context_*` annotation.
"""

import glob
import gzip
import json
import re
import sys

pattern = re.compile(r"execute_context_\d+\(\d+\)")
paths = sorted(glob.glob(sys.argv[1] + "/*.pt.trace.json*"))
if not paths:
    print(f"no rank traces under {sys.argv[1]}", file=sys.stderr)
    sys.exit(1)
total = 0
for path in paths:
    opener = gzip.open if path.endswith(".gz") else open
    events = json.load(opener(path, "rt")).get("traceEvents", [])
    hits = sum(
        1
        for e in events
        if e.get("cat") == "gpu_user_annotation" and pattern.search(e.get("name") or "")
    )
    total += hits
    print(f"{path.split('/')[-1]}: {hits} step annotations, {len(events)} events")
sys.exit(0 if total else 1)
