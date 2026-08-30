# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ticket 06 deliverable: assemble the per-run accuracy scores into one report.

Answers the three questions the ticket asks:

  * How accuracy degrades with prediction lookahead, per domain and shape.
  * Whether the leading layers really do predict worse, which is the only
    evidence for `prediction_skip_first_layers`.
  * Whether the accuracy at the chosen lookahead is good enough for a planner to
    act on, or whether the lookahead must shrink and the exposed transfer be
    accepted instead.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

# The planner replicates `max_replicas_per_layer` distinct experts and that default is
# **1** since ticket 07: at a cap of 1 a budget covers every reachable layer and removes
# 48.0% of critical-path excess against a global oracle's 48.3%, where a cap of 2 covers
# 22 layers and removes 23.5%. So recall at 1 is what the planner depends on, and it is
# also `peak_hit_rate` — whether the one expert it would pick is the one that actually
# ran hottest. Recall at 2 stays available through `--k`.
PLANNER_K = 1


def load_reports(results_dir: Path) -> list[dict]:
    """Read every scored run, newest layout only, skipping unreadable files."""
    reports = []
    for path in sorted(results_dir.glob("accuracy-L*.json")):
        try:
            reports.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return reports


def domain_of(label: str) -> str:
    """`L2-code-p1024` -> `code-p1024`."""
    parts = label.split("-", 1)
    return parts[1] if len(parts) > 1 else label


def degradation(reports: Sequence[dict], k: int = PLANNER_K) -> dict:
    """Recall at `k` per lookahead and domain, plus the drop from lookahead 1.

    Raises:
        ValueError: when no lookahead-1 run exists, since the degradation is
            expressed relative to it and would otherwise be silently absent.
    """
    table: dict[str, dict[int, float]] = {}
    errors: dict[str, dict[int, float]] = {}
    for report in reports:
        lookahead = report.get("lookahead")
        if not isinstance(lookahead, int):
            continue  # a mixed-distance dump cannot be placed on this axis
        domain = domain_of(report.get("label", ""))
        table.setdefault(domain, {})[lookahead] = report["overall"][f"recall_at_{k}"]
        errors.setdefault(domain, {})[lookahead] = report["overall"]["count_error"]
    return {
        "recall": table,
        "count_error": errors,
        "k": k,
        # Reported rather than raised. The skip-first-layers evidence lives in a
        # directory holding one lookahead, and `by_layer_curve` does not need a
        # baseline at all, so refusing here took the whole report down with it.
        "has_lookahead_1_baseline": any(1 in per for per in table.values()),
    }


def lookahead_pair(reports: Sequence[dict], k: int = PLANNER_K) -> dict:
    """Compare the two shortest lookaheads, per domain, on the layers both cover.

    Ticket 07's question: shortening the lookahead to 1 should predict better, since 1 was
    always the accurate distance and 2 existed only to buy the host planner a layer of
    latency for its snapshot copy to land in.

    **Per domain, because accuracy is a property of the content's routing.** Keying by
    lookahead alone let the last report win: with code at 0.9 and text at 0.1 the answer
    came out 0.1 for both arms and "not better at 1", printed in the report as the answer
    to the ticket.

    **On the common layers, because the arms do not cover the same ones.** At a lookahead
    of `n` the reachable targets start at `skip_first + n`, so lookahead 1 covers one extra
    *early* layer — and early layers predict worst, which is the entire reason
    `prediction_skip_first_layers` exists. Pooling charges lookahead 1 for a layer its
    opponent never had to predict. Both numbers are returned, because quoting only the
    favourable one is how a comparison stops meaning anything.

    Returns:
        One entry per domain. Each carries the two lookaheads compared, any it ignored,
        the recall at `k` and count error both pooled and restricted to the common layers,
        the layer sets, and whether the shorter lookahead is better — with `common` and
        `better_at_1` `None`, and a `reason`, when there is nothing to compare rather than
        an average of nothing. Never raises: a report that cannot answer is the answer.
    """
    by_domain: dict[str, dict[int, dict]] = {}
    for report in reports:
        lookahead = report.get("lookahead")
        if not isinstance(lookahead, int):
            continue  # a mixed-distance dump has no single distance to place
        domain = domain_of(report.get("label", ""))
        by_domain.setdefault(domain, {})[lookahead] = report

    out: dict[str, dict] = {}
    for domain, by_distance in sorted(by_domain.items()):
        distances = sorted(by_distance)
        # The two shortest, and the rest named rather than dropped: the runner's own
        # default is `LOOKAHEADS="1 2 3"`, and raising here made the whole deliverable
        # table vanish through a caller's `except`.
        compared, ignored = distances[:2], distances[2:]
        entry: dict = {
            "lookaheads": compared,
            "ignored_lookaheads": ignored,
            "overall": {
                f"recall_at_{k}": {
                    d: by_distance[d]["overall"][f"recall_at_{k}"] for d in compared
                },
                "count_error": {
                    d: by_distance[d]["overall"]["count_error"] for d in compared
                },
            },
            "common_layers": [],
            "layers_only_at_1": [],
            "common": None,
            "better_at_1": None,
        }
        if len(compared) < 2:
            entry["reason"] = "only one lookahead measured"
            out[domain] = entry
            continue
        near, far = compared
        layers = {
            d: {int(layer) for layer in by_distance[d].get("by_layer", {})}
            for d in compared
        }
        common = sorted(layers[near] & layers[far])
        entry["common_layers"] = common
        entry["layers_only_at_1"] = sorted(layers[near] - layers[far])
        if not common:
            entry["reason"] = "the two arms share no target layer"
            out[domain] = entry
            continue

        def pooled(distance: int, field: str, common=common, by=by_distance) -> float:
            rows = [by[distance]["by_layer"][str(layer)][field] for layer in common]
            return round(statistics.mean(rows), 4)

        entry["common"] = {
            f"recall_at_{k}": {d: pooled(d, f"recall_at_{k}") for d in compared},
            "count_error": {d: pooled(d, "count_error") for d in compared},
        }
        entry["better_at_1"] = (
            entry["common"][f"recall_at_{k}"][near]
            > entry["common"][f"recall_at_{k}"][far]
        )
        out[domain] = entry
    return out


def by_layer_curve(reports: Sequence[dict], k: int = PLANNER_K) -> dict:
    """Recall at `k` against target layer index, pooled across runs.

    This is the only evidence for skipping leading layers. If early layers do not
    predict worse, the skip costs coverage for nothing.
    """
    lookaheads = {r.get("lookahead") for r in reports}
    if len(lookaheads) > 1:
        raise ValueError(
            f"cannot pool a per-layer curve across lookaheads {sorted(lookaheads)}: "
            "recall falls monotonically with distance, and the earliest target "
            "layers are covered by fewer, shorter-distance runs than later ones, so "
            "pooling inflates exactly the leading layers a skip decision turns on. "
            "Pass reports from one lookahead."
        )
    pooled: dict[int, list[float]] = {}
    for report in reports:
        for layer, row in report.get("by_layer", {}).items():
            pooled.setdefault(int(layer), []).append(row[f"recall_at_{k}"])
    return {
        layer: round(statistics.mean(values), 4)
        for layer, values in sorted(pooled.items())
    }


def skip_recommendation(curve: dict[int, float], tolerance: float = 0.05) -> dict:
    """Find the first target layer from which recall stops improving materially.

    Compares each leading layer against the median of the later half. A layer
    whose recall sits more than `tolerance` below that median is judged unreliable
    and worth skipping.

    Returns:
        The layers judged unreliable, the stable median, and the implied skip.
    """
    if not curve:
        return {
            "unreliable_layers": [],
            "stable_median": None,
            "implied_skip_target_layers": None,
        }
    layers = sorted(curve)
    tail = [curve[layer] for layer in layers[len(layers) // 2 :]]
    stable = statistics.median(tail)
    unreliable = []
    for layer in layers:
        if curve[layer] < stable - tolerance:
            unreliable.append(layer)
        else:
            break  # only a leading run of bad layers justifies a skip
    return {
        "unreliable_layers": unreliable,
        "stable_median": round(stable, 4),
        # The skip is expressed in *source* layers, and a source predicts
        # `lookahead` layers ahead, so the caller applies the offset.
        "implied_skip_target_layers": (max(unreliable) + 1) if unreliable else 0,
    }


def to_markdown(reports: Sequence[dict], k: int = PLANNER_K) -> str:
    deg = degradation(reports, k)
    by_lookahead: dict = {}
    for report in reports:
        by_lookahead.setdefault(report.get("lookahead"), []).append(report)

    lines = [f"### Recall at {k} by lookahead", ""]
    domains = sorted(deg["recall"])
    lookaheads = sorted({la for d in deg["recall"].values() for la in d})
    head = "| domain / shape | " + " | ".join(f"L={la}" for la in lookaheads) + " |"
    rule = "| --- | " + " | ".join("---" for _ in lookaheads) + " |"
    lines += [head, rule]
    for domain in domains:
        cells = []
        for la in lookaheads:
            value = deg["recall"][domain].get(la)
            cells.append("-" if value is None else f"{value:.3f}")
        lines.append(f"| {domain} | " + " | ".join(cells) + " |")

    lines += ["", "### Count error (total-variation distance, lower is better)", ""]
    lines += [head, rule]
    for domain in domains:
        cells = []
        for la in lookaheads:
            value = deg["count_error"][domain].get(la)
            cells.append("-" if value is None else f"{value:.4f}")
        lines.append(f"| {domain} | " + " | ".join(cells) + " |")

    # Ticket 07's question, per domain and on the layers both arms cover: lookahead 1
    # reaches one extra early layer, early layers predict worst, so pooling would charge
    # it for work the other arm never did. Never wrapped in a `try`: the comparison
    # reports what it cannot answer, and swallowing an exception here once made the whole
    # section disappear from the deliverable without a word.
    pairs = lookahead_pair(reports, k)
    for domain, pair in pairs.items():
        if pair["common"] is None:
            lines += [
                "",
                f"### Lookahead comparison for {domain}: not available",
                "",
                f"{pair.get('reason', 'no comparison')} "
                f"(measured: {pair['lookaheads'] or 'none'}).",
            ]
            continue
        near, far = pair["lookaheads"]
        lines += [
            "",
            f"### {domain}: lookahead {near} against {far}, on their common layers",
            "",
            f"| metric | L={near} | L={far} |",
            "| --- | --- | --- |",
        ]
        for source, suffix in (("common", ""), ("overall", ", all layers pooled")):
            for field, fmt in ((f"recall_at_{k}", "{:.3f}"), ("count_error", "{:.4f}")):
                a = pair[source][field][near]
                b = pair[source][field][far]
                lines.append(
                    f"| {field}{suffix} | {fmt.format(a)} | {fmt.format(b)} |"
                )
        ignored = pair["ignored_lookaheads"]
        lines += [
            "",
            f"{len(pair['common_layers'])} common target layers; "
            f"{len(pair['layers_only_at_1'])} covered only at L={near} "
            f"({pair['layers_only_at_1'] or 'none'}). "
            f"Recall at {k} is "
            f"{'better' if pair['better_at_1'] else 'not better'} at L={near}."
            + (f" Lookaheads {ignored} measured but not compared here." if ignored else ""),
        ]

    lines += ["", "### Recall by target layer, per lookahead", ""]
    lines.append(
        "Never pooled across lookaheads: recall falls with distance and the "
        "earliest layers are covered by fewer runs, which inflates them."
    )
    for lookahead, group in sorted(
        by_lookahead.items(), key=lambda kv: (kv[0] is None, kv[0])
    ):
        if not isinstance(lookahead, int):
            continue
        curve = by_layer_curve(group, k)
        skip = skip_recommendation(curve)
        covered = sorted(curve)
        lines.append("")
        lines.append(
            f"- **L={lookahead}** (target layers {covered[0]}-{covered[-1]}): "
            f"stable median {skip['stable_median']}, leading layers below "
            f"tolerance {skip['unreliable_layers'] or 'none'}"
        )
    if not deg["has_lookahead_1_baseline"]:
        lines += [
            "",
            (
                "No lookahead-1 run in this directory, so the degradation columns "
                "have no baseline to be read against. The per-layer curves above "
                "do not need one."
            ),
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).parent
    parser.add_argument("--results-dir", type=Path, default=here / "results/accuracy")
    parser.add_argument("--k", type=int, default=PLANNER_K)
    args = parser.parse_args()

    reports = load_reports(args.results_dir)
    if not reports:
        raise SystemExit(f"no accuracy-L*.json under {args.results_dir}")
    deg = degradation(reports, args.k)
    # The curve is only meaningful within one lookahead, so group first.
    by_lookahead: dict = {}
    for report in reports:
        by_lookahead.setdefault(report.get("lookahead"), []).append(report)
    curves = {
        str(lookahead): by_layer_curve(group, args.k)
        for lookahead, group in sorted(
            by_lookahead.items(), key=lambda kv: (kv[0] is None, kv[0])
        )
        if isinstance(lookahead, int)
    }
    payload = {
        "runs": [r.get("label") for r in reports],
        "degradation": deg,
        "lookahead_pair": lookahead_pair(reports, args.k),
        "by_target_layer_per_lookahead": curves,
        "skip_per_lookahead": {
            lookahead: skip_recommendation(curve) for lookahead, curve in curves.items()
        },
    }
    text = to_markdown(reports, args.k)
    print(text)
    (args.results_dir / "accuracy-summary.md").write_text(text)
    (args.results_dir / "accuracy-summary.json").write_text(
        json.dumps(payload, indent=2)
    )


if __name__ == "__main__":
    main()
