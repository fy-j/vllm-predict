# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read the end-to-end placement run: did the path connect, and did it rebalance?

Compares per-layer rank imbalance between two runs that differ only in the transfer
budget. The load dump records physical-slot load *after* the router applied the
logical-to-physical map, so tokens routed to a replica land on the replica's row and
therefore on its target rank — which is why this dump can show the rebalancing rather
than only the pre-placement distribution.

Forwards are grouped by assignment count rather than filtered to one value. A run with
a short decode is decode-dominated: an earlier version filtered to full prefill steps
and found two of them, which cannot support a comparison, while the 244 decode forwards
beside them can. Load rebalancing in decode is real and measurable even though block
quantization keeps it from converting into time — that distinction belongs in the
reading, not in the filter.

TTFT is reported but is not the headline. This bring-up path neither overlaps the
transfer nor keeps a placement across forwards, so end-to-end latency measures its cost
rather than its benefit, and the expected benefit is below this node's interconnect
noise either way.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from imbalance import (  # noqa: E402
    aggregated_imbalance,
    critical_path_imbalance,
    load_dump,
)

# The line `_run_placement_if_armed` logs once when it activates anything. Matched in
# full: counting the bare word "activated" also matches vLLM's JIT monitor banner,
# which reported eight activations in a run whose budget was zero.
_PLACEMENT_LINE = "Predictive expert replication: activated"

# A worker that died says so in one of these. Informational only: neither is evidence
# of a failed run. `EngineDeadError` is what a clean SIGTERM raises in the API server,
# and `EngineCore encountered a fatal error` is what the surviving ranks raise when a
# killed peer stops answering `execute_dummy_batch`. Failed requests are the signal.
_FATAL_MARKERS = ("Worker failed with error", "EngineCore encountered a fatal error")


# Ranges, not exact counts. A prefill step's token count varies with how much of a
# prompt the scheduler admitted, so exact grouping scattered eight prefill forwards
# across four singleton bands and the minimum-size filter then dropped all of them —
# hiding the only regime where this feature does anything.
_BANDS = (
    (1.0, 1_000.0, "decode"),
    (1_000.0, 50_000.0, "partial prefill"),
    (50_000.0, float("inf"), "full prefill"),
)


def by_assignments(path: Path) -> dict[str, list]:
    """Group a run's forwards into assignment-count ranges."""
    grouped: dict[str, list] = defaultdict(list)
    for forward in load_dump(path):
        live = [layer for layer in forward if layer.live]
        if not live:
            continue
        total = sum(live[0].rank_load)
        for low, high, name in _BANDS:
            if low <= total < high:
                grouped[name].append(forward)
                break
    return grouped


def summarize(results_dir: Path, budget: str) -> dict:
    """Imbalance per assignment band, plus whether the run stayed healthy."""
    out: dict = {"budget": budget}
    dump = results_dir / f"dump-b{budget}.jsonl"
    if not dump.exists():
        out["missing"] = str(dump)
        return out
    grouped = by_assignments(dump)
    out["forwards"] = sum(len(v) for v in grouped.values())
    out["bands"] = {
        str(assignments): {
            "forwards": len(forwards),
            "critical_path": round(
                statistics.median(critical_path_imbalance(f) for f in forwards), 4
            ),
            "aggregated": round(
                statistics.median(aggregated_imbalance(f) for f in forwards), 4
            ),
        }
        for assignments, forwards in grouped.items()
        if len(forwards) >= 3  # two forwards cannot support a comparison
    }
    log = results_dir / f"server-b{budget}.log"
    if log.exists():
        text = log.read_text(errors="replace")
        out["placement_log_lines"] = text.count(_PLACEMENT_LINE)
        out["fatal"] = sum(text.count(marker) for marker in _FATAL_MARKERS)
    bench = results_dir / f"bench-b{budget}.json"
    if bench.exists():
        try:
            payload = json.loads(bench.read_text())
            out["completed"] = payload.get("completed")
            out["failed"] = payload.get("failed")
            out["mean_ttft_ms"] = round(payload.get("mean_ttft_ms") or 0.0, 1)
            out["mean_tpot_ms"] = round(payload.get("mean_tpot_ms") or 0.0, 2)
        except json.JSONDecodeError:
            out["bench"] = "unparsable"
    return out


def is_connected(placed: dict) -> bool:
    """Whether the placed arm actually ran placements and produced valid data.

    Judged on failed requests, never on `fatal`. Killing an arm's server leaves the
    surviving ranks waiting in `execute_dummy_batch` for a peer that is gone, so
    `EngineCore encountered a fatal error` fires at every clean teardown — it marked a
    run with 152 activations and zero failed requests as not connected. A real mid-run
    death shows up in the requests instead: the hung run failed 56 of 120.

    Extracted so it can be tested. This verdict has been wrong twice — once from a
    log pattern that never matched, once from that teardown marker — and both times it
    read as an all-clear, which is the direction that costs a run.
    """
    return (
        bool(placed.get("placement_log_lines"))
        and placed.get("failed") == 0
        and bool(placed.get("completed"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).parent
    parser.add_argument(
        "--results-dir", type=Path, default=here / "results/e2e-placement"
    )
    parser.add_argument("--baseline-budget", default="0")
    parser.add_argument("--placed-budget", default="43")
    args = parser.parse_args()

    baseline = summarize(args.results_dir, args.baseline_budget)
    placed = summarize(args.results_dir, args.placed_budget)
    out = {"baseline": baseline, "placed": placed}

    comparison = {}
    for band, before in baseline.get("bands", {}).items():
        after = placed.get("bands", {}).get(band)
        if after is None:
            continue
        b, a = before["critical_path"], after["critical_path"]
        comparison[band] = {
            "forwards": f"{before['forwards']} / {after['forwards']}",
            "critical_path": f"{b} -> {a}",
            "excess_removed_pct": round(100 * (b - a) / (b - 1), 1) if b > 1 else None,
        }
    out["comparison"] = comparison or "no band is present in both runs"
    # Judged on failed requests, not on `fatal`. Killing an arm's server leaves the
    # remaining ranks waiting in `execute_dummy_batch` for a peer that is gone, so
    # `EngineCore encountered a fatal error` fires at every clean teardown — it marked
    # a run with 152 activations and zero failed requests as not connected. A real
    # mid-run death shows up in the requests instead: the hung run failed 56 of 120.
    # `fatal` stays in the report as information.
    out["connected"] = is_connected(placed)
    print(json.dumps(out, indent=2))
    (args.results_dir / "e2e-summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
