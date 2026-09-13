# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ticket 06: how well the cross-layer gate predicts its target layer's load.

Reads the raw dump written by `VLLM_PREDICTIVE_ACCURACY_DUMP_PATH` - one JSON
object per recorded forward, each holding every source layer's predicted logical
counts beside the target layer's actual recorded load - and reports the two
metrics the ticket asks for, broken down by layer index.

Two deliberate choices:

  * Gate-logit similarity is not reported. A high logit similarity can still
    reorder the selected top-k and therefore mispredict load, so it would flatter
    the prediction without answering whether the planner can act on it.

  * Both sides are normalized to shares before comparison. Under the
    allgather/reduce-scatter backend the recorded load is multiplied by the DP
    size while the prediction is a plain count, so absolute values are not
    comparable; the ranking and the shares are.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path


def normalize(counts: Sequence[float]) -> list[float]:
    """Turn per-expert counts into shares that sum to one.

    An all-zero input is returned unchanged rather than producing NaN: the caller
    decides whether an empty layer is scorable.
    """
    total = float(sum(counts))
    if total <= 0.0:
        return [0.0] * len(counts)
    return [float(c) / total for c in counts]


def top_k_experts(counts: Sequence[float], k: int) -> list[int]:
    """The `k` hottest expert indices, hottest first.

    Ties break towards the lower index so the metric is deterministic; without
    that, a layer whose experts are evenly loaded would score differently run to
    run for reasons that have nothing to do with prediction quality.
    """
    order = sorted(range(len(counts)), key=lambda i: (-float(counts[i]), i))
    return order[: max(0, k)]


def hot_set_recall(
    predicted: Sequence[float], actual: Sequence[float], k: int
) -> float:
    """Fraction of the actual top-`k` experts that the prediction also ranked top-`k`.

    This is the metric the planner cares about: it replicates a small number of
    hot experts, so what matters is whether the right experts are identified, not
    whether their counts are exact.
    """
    actual_top = set(top_k_experts(actual, k))
    if not actual_top:
        return 0.0
    predicted_top = set(top_k_experts(predicted, k))
    return len(actual_top & predicted_top) / len(actual_top)


def count_error(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Total-variation distance between the predicted and actual load shares.

    Zero when the two distributions agree exactly and one when they are disjoint,
    which makes it directly readable as "the fraction of load placed on the wrong
    expert".
    """
    p = normalize(predicted)
    a = normalize(actual)
    return 0.5 * sum(abs(pi - ai) for pi, ai in zip(p, a))


def peak_expert_hit(predicted: Sequence[float], actual: Sequence[float]) -> bool:
    """Whether the single hottest actual expert is also the prediction's hottest.

    Identical to `hot_set_recall(..., k=1)` by construction. Both are reported
    because the hot-set sizes are configurable and 1 need not be among them; when
    it is, `peak_hit_rate` and `recall_at_1` are the same number and must not be
    read as two independent results.
    """
    return top_k_experts(predicted, 1) == top_k_experts(actual, 1)


def _flatten_predicted(predicted, record):
    """Accept a window row, refuse a whole window.

    Ticket 13 gave the snapshot a window dimension and the recorder kept summing only
    over ranks, so records written since carry `[size, num_logical]`. A single row
    flattens and means what it always did. Several rows may **not** be summed: they
    belong to different target layers, and adding them is the distribution that took
    ticket 12's design from 26.8% of excess removed to 3.7%.
    """
    if not predicted or not isinstance(predicted[0], list):
        return predicted
    if len(predicted) == 1:
        return predicted[0]
    raise ValueError(
        f"record for source {record.get('source')} -> target {record.get('target')} "
        f"carries {len(predicted)} window rows. They belong to different target layers "
        f"and cannot be summed; re-dump with one row per pair."
    )


def aggregate(records: Sequence[dict], ks: Sequence[int] = (1, 2, 4, 8)) -> dict:
    """Score every record and reduce to overall and per-target-layer figures.

    Records whose actual load is empty are excluded and counted separately: a
    layer that saw no tokens has no hot set, and scoring it zero would invent a
    misprediction that never happened.

    Raises:
        ValueError: When no record is scorable, which would otherwise average to a
            flawless-looking result over an empty sample.
    """
    scored: list[dict] = []
    skipped = 0
    skipped_empty_predicted = 0
    for record in records:
        actual = record["actual"]
        if float(sum(actual)) <= 0.0:
            skipped += 1
            continue
        predicted = _flatten_predicted(record["predicted"], record)
        if float(sum(predicted)) <= 0.0:
            # Symmetric with the actual-side skip, and for the same reason. An
            # all-zero prediction normalizes to zeros, `top_k_experts` then falls
            # back to index order and returns [0, 1, ...], and the record scores as
            # a near-total miss that never happened.
            skipped_empty_predicted += 1
            continue
        row = {
            "target": int(record["target"]),
            "source": int(record["source"]),
            "count_error": count_error(predicted, actual),
            "peak_hit": 1.0 if peak_expert_hit(predicted, actual) else 0.0,
        }
        for k in ks:
            row[f"recall_at_{k}"] = hot_set_recall(predicted, actual, k)
        scored.append(row)

    if not scored:
        raise ValueError(
            f"no scorable record: of {len(records)}, {skipped} had an empty actual "
            f"load and {skipped_empty_predicted} an empty prediction, so the dump "
            "caught only forwards in which nothing was routed or nothing predicted"
        )

    def summarize(rows: list[dict]) -> dict:
        out = {
            "samples": len(rows),
            "count_error": round(statistics.mean(r["count_error"] for r in rows), 4),
            "peak_hit_rate": round(statistics.mean(r["peak_hit"] for r in rows), 4),
        }
        for k in ks:
            out[f"recall_at_{k}"] = round(
                statistics.mean(r[f"recall_at_{k}"] for r in rows), 4
            )
        return out

    by_layer: dict[int, list[dict]] = collections.defaultdict(list)
    for row in scored:
        by_layer[row["target"]].append(row)

    return {
        "scored": len(scored),
        "skipped_empty_actual": skipped,
        "skipped_empty_predicted": skipped_empty_predicted,
        "overall": summarize(scored),
        "by_layer": {
            str(layer): summarize(rows) for layer, rows in sorted(by_layer.items())
        },
    }


def load_records(path: Path) -> list[dict]:
    """Flatten the per-forward dump into one record per source/target pair."""
    records: list[dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partially flushed final line
            records.extend(payload.get("pairs", []))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Hot-set sizes to report. 2 is the default max replicas per layer.",
    )
    parser.add_argument("--label", default="", help="Domain and shape of this run.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    records = load_records(args.dump)
    if not records:
        print(f"no records in {args.dump}", file=sys.stderr)
        raise SystemExit(1)
    report = aggregate(records, ks=tuple(args.ks))
    report["label"] = args.label
    report["dump"] = str(args.dump)
    # The lookahead is a launch-time setting, so it is recovered from the pairs
    # rather than assumed: source and target differ by exactly the lookahead.
    distances = {r["target"] - r["source"] for r in records}
    report["lookahead"] = sorted(distances) if len(distances) > 1 else distances.pop()
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
