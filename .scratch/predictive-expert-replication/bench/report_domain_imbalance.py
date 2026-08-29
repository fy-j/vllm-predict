# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank domains by the expert imbalance a placement could actually recover.

Reports the **per-layer** figures. Summing each rank's load across layers before
comparing ranks lets different layers' peaks land on different ranks and cancel; that
mistake understated a real measurement here about threefold, twice. Every MoE layer is
its own collective waiting on its own slowest rank, so the quantity that corresponds to
time is the sum of per-layer peaks over the sum of per-layer means — and the per-layer
distribution is printed beside it so the within-layer skew is visible rather than
hidden behind one number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from imbalance import (  # noqa: E402
    aggregated_imbalance,
    apply_moves,
    critical_path_imbalance,
    load_dump,
    plan_moves,
)

PREFILL_MIN_ASSIGNMENTS = 4096


def prefill_forwards(path: Path, want: int = 12):
    """The forwards whose recorded assignment count puts them in prefill."""
    forwards = load_dump(path)
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    counts = [sum(sum(x) for x in r["logical_load"]) for r in records]
    order = sorted(range(len(counts)), key=lambda i: -counts[i])
    picked = [i for i in order if counts[i] >= PREFILL_MIN_ASSIGNMENTS][:want]
    return [forwards[i] for i in picked], [counts[i] for i in picked]


def layer_ratios(layers) -> list[float]:
    """Peak-over-mean within each layer, which is what that layer waits for."""
    return [
        max(x.rank_load) / statistics.mean(x.rank_load) for x in layers if x.live
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dumps", nargs="+", type=Path)
    parser.add_argument("--budget", type=int, default=43)
    parser.add_argument("--per-layer-cap", type=int, default=1)
    args = parser.parse_args()

    rows = []
    for path in args.dumps:
        name = path.stem.replace("dump-", "")
        try:
            forwards, counts = prefill_forwards(path)
        except (ValueError, FileNotFoundError) as exc:
            print(f"{name}: unusable — {exc}", file=sys.stderr)
            continue
        if not forwards:
            print(f"{name}: no prefill-regime forward", file=sys.stderr)
            continue
        ratios = [r for f in forwards for r in layer_ratios(f)]
        crit = statistics.mean(critical_path_imbalance(f) for f in forwards)
        agg = statistics.mean(aggregated_imbalance(f) for f in forwards)
        after = statistics.mean(
            critical_path_imbalance(
                apply_moves(
                    f,
                    plan_moves(
                        f, budget=args.budget, per_layer_cap=args.per_layer_cap
                    ),
                )
            )
            for f in forwards
        )
        ratios.sort()
        rows.append(
            {
                "domain": name,
                "forwards": len(forwards),
                "tokens_per_expert": round(
                    statistics.mean(counts) / len(forwards[0]) / 128, 1
                ),
                "per_layer_min": round(ratios[0], 3),
                "per_layer_p50": round(ratios[len(ratios) // 2], 3),
                "per_layer_p90": round(ratios[int(0.9 * (len(ratios) - 1))], 3),
                "per_layer_max": round(ratios[-1], 3),
                "critical_path": round(crit, 4),
                "recoverable_excess_pct": round(100 * (crit - 1), 1),
                "oracle_removes_pct": round(100 * (crit - after) / (crit - 1), 1)
                if crit > 1
                else None,
                "aggregated_misleading": round(agg, 4),
            }
        )

    rows.sort(key=lambda r: -r["recoverable_excess_pct"])
    head = (
        f"{'domain':22s} {'fwd':>4s} {'tok/exp':>8s} "
        f"{'per-layer min/p50/p90/max':>28s} {'crit':>7s} {'excess':>7s} "
        f"{'oracle':>7s} {'agg':>7s}"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        band = (
            f"{r['per_layer_min']:.2f}/{r['per_layer_p50']:.2f}/"
            f"{r['per_layer_p90']:.2f}/{r['per_layer_max']:.2f}"
        )
        print(
            f"{r['domain']:22s} {r['forwards']:>4d} {r['tokens_per_expert']:>8.1f} "
            f"{band:>28s} {r['critical_path']:>7.4f} "
            f"{r['recoverable_excess_pct']:>6.1f}% {str(r['oracle_removes_pct']):>6s}% "
            f"{r['aggregated_misleading']:>7.4f}"
        )
    print(
        "\nexcess = critical_path - 1, the share a perfect placement could remove.\n"
        "agg = rank totals compared after summing layers. Printed only to show how\n"
        "far it understates the per-layer figure; never size a placement with it."
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
