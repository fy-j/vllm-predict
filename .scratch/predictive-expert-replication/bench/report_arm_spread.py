# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report each arm as a distribution, and say whether the effect clears the noise.

Ticket 08's method. A single pass per arm cannot support a conclusion here: between two
runs of the same script on identical workloads, the stock arm alone moved 28% on mean
TTFT and 40% on throughput, because the machine was in a better state. The ceiling being
chased is about 6.7% of a prefill window, so a method that cannot see 40% of drift
cannot see the effect at all.

So each arm is measured several times, arms interleaved within each repeat, and the
report gives the spread beside the median. The rule it applies is deliberately blunt:
**if the `off` arm's own spread is wider than the difference between arms, the run says
nothing**, and that is printed as the verdict rather than left for a reader to notice.

Throughput is reported beside TTFT because it is less tail-sensitive. Mean TTFT at low
concurrency is dominated by a handful of queued requests, and both runs so far moved
coherently in throughput while mean TTFT moved more.

Usage:
    python report_arm_spread.py --results-dir <dir> --arms off 0 43 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

_METRICS = {
    "mean_ttft": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "median_ttft": r"Median TTFT \(ms\):\s+([0-9.]+)",
    "p99_ttft": r"P99 TTFT \(ms\):\s+([0-9.]+)",
    "throughput": r"Request throughput \(req/s\):\s+([0-9.]+)",
}


def read_repeat(results_dir: Path, arm: str, repeat: int, repeats: int) -> dict | None:
    """One arm's numbers from one repeat, or None when that pass is absent."""
    tag = f"b{arm}-r{repeat}" if repeats > 1 else f"b{arm}"
    log = results_dir / f"bench-{tag}.log"
    if not log.exists():
        return None
    text = log.read_text(errors="replace")
    out = {}
    for name, pattern in _METRICS.items():
        found = re.search(pattern, text)
        if found is None:
            return None
        out[name] = float(found.group(1))
    return out


def collect(results_dir: Path, arms: list[str], repeats: int) -> dict[str, dict]:
    """Per arm, the list of values seen for each metric across repeats."""
    series: dict[str, dict] = {}
    for arm in arms:
        rows = [
            row
            for repeat in range(1, repeats + 1)
            if (row := read_repeat(results_dir, arm, repeat, repeats)) is not None
        ]
        if rows:
            series[arm] = {metric: [row[metric] for row in rows] for metric in _METRICS}
            series[arm]["passes"] = len(rows)
    return series


def _spread(values: list[float]) -> float:
    """Peak-to-peak as a share of the median, which is what has to be beaten."""
    if len(values) < 2:
        return float("nan")
    return (max(values) - min(values)) / statistics.median(values)


def report(series: dict[str, dict], baseline: str) -> dict:
    """Median per arm, the baseline's own spread, and whether anything clears it."""
    if baseline not in series:
        raise SystemExit(f"no data for the baseline arm {baseline!r}")

    out: dict = {"baseline": baseline, "arms": {}}
    base_median = statistics.median(series[baseline]["mean_ttft"])
    base_spread = _spread(series[baseline]["mean_ttft"])

    for arm, data in series.items():
        med = {m: statistics.median(data[m]) for m in _METRICS}
        out["arms"][arm] = {
            "passes": data["passes"],
            **{m: round(v, 2) for m, v in med.items()},
            "mean_ttft_spread_pct": round(100 * _spread(data["mean_ttft"]), 1),
            "vs_baseline_pct": round(
                100 * (med["mean_ttft"] - base_median) / base_median, 1
            ),
        }

    # The verdict. An effect smaller than the baseline's own drift is not an effect that
    # this run measured, whichever direction it points.
    effects = [
        abs(v["vs_baseline_pct"]) for arm, v in out["arms"].items() if arm != baseline
    ]
    largest = max(effects) if effects else 0.0
    out["baseline_spread_pct"] = round(100 * base_spread, 1)
    out["resolvable"] = bool(largest > 100 * base_spread)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--baseline", default="off")
    args = parser.parse_args()

    series = collect(args.results_dir, args.arms, args.repeats)
    if not series:
        raise SystemExit(f"no benchmark logs under {args.results_dir}")
    out = report(series, args.baseline)

    print(
        f"{'arm':10}{'n':>3}{'mean TTFT':>11}{'median':>9}{'p99':>10}"
        f"{'req/s':>9}{'spread':>9}{'vs base':>9}"
    )
    for arm, v in out["arms"].items():
        print(
            f"{arm:10}{v['passes']:>3}{v['mean_ttft']:>11.2f}{v['median_ttft']:>9.2f}"
            f"{v['p99_ttft']:>10.2f}{v['throughput']:>9.2f}"
            f"{v['mean_ttft_spread_pct']:>8.1f}%{v['vs_baseline_pct']:>8.1f}%"
        )
    print(
        f"\nbaseline ({args.baseline}) own spread across passes: "
        f"{out['baseline_spread_pct']}%"
    )
    if out["resolvable"]:
        print("VERDICT: the largest effect exceeds the baseline drift; it is readable.")
    else:
        print(
            "VERDICT: no effect exceeds the baseline's own drift. This run does not "
            "measure the feature — add repeats, or reduce the drift, before quoting it."
        )
    (args.results_dir / "arm-spread.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {args.results_dir / 'arm-spread.json'}")


if __name__ == "__main__":
    main()
