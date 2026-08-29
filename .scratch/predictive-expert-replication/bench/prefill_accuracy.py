# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ticket 10: does the cross-layer gate predict well enough to place on, at prefill?

Ticket 06 answered this for decode, where a forward carries tens of tokens. Prefill
is a different statistical regime — about 1024 tokens per expert against 4 — and it
is the regime the TTFT work targets.

Scoring is not re-implemented: `prediction_accuracy` owns the metrics and is
imported. What is added here is the three things that regime needs.

1. **Prefill forwards are selected by their recorded assignment count**, never by a
   magnitude threshold. `sum(actual)` over a pair *is* the assignment count, so the
   regime is read off the data.

2. **Accuracy is restricted to the peak rank.** The policy only ever considers
   candidates on the rank the layer is waiting on, so accuracy spread over all 128
   experts does not bear on what it can do. Which rank that is has to come from the
   *prediction*, because that is all the planner has — so a wrong peak rank is
   itself a failure mode, and it is reported separately.

3. **The number that decides the build**: placements chosen from predicted load and
   scored against actual load in the same forward, via `imbalance.plan_moves` and
   `imbalance.apply_moves`.

On (3) the ticket's own framing is wrong and worth stating plainly: it says the
realistic figure is the oracle bound "times the accuracy". A product like that
cannot go below zero, and this quantity can — a misled placement adds load to the
rank the layer is actually waiting on, so it overshoots the baseline instead of
merely failing to help. It has to be measured, not scaled.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

import imbalance
from prediction_accuracy import count_error, hot_set_recall

# A prefill forward is one whose post-allgather assignments put many tokens on every
# expert. At 128 experts this is 32 tokens per expert, comfortably above the largest
# bf16 block tier and far above anything a decode step reaches on this node.
PREFILL_MIN_ASSIGNMENTS = 4096


def rank_loads(counts: Sequence[float], ep_size: int) -> list[float]:
    """Per-rank load from a per-logical-expert vector, by canonical ownership.

    Logical experts are laid out rank-major, so rank `r` owns the slice
    `[r * per_rank, (r+1) * per_rank)`. This is the same grouping `load_dump` uses.
    """
    if len(counts) % ep_size:
        raise ValueError(f"{len(counts)} experts do not divide across {ep_size} ranks")
    per_rank = len(counts) // ep_size
    return [
        float(sum(counts[r * per_rank : (r + 1) * per_rank])) for r in range(ep_size)
    ]


def as_layer(counts: Sequence[float], ep_size: int) -> imbalance.Layer:
    """One layer of one forward, built from a per-logical-expert vector."""
    per_rank = len(counts) // ep_size
    groups = [
        [float(x) for x in counts[r * per_rank : (r + 1) * per_rank]]
        for r in range(ep_size)
    ]
    return imbalance.Layer(rank_load=[sum(g) for g in groups], expert_load=groups)


def peak_rank(counts: Sequence[float], ep_size: int) -> int:
    loads = rank_loads(counts, ep_size)
    return max(range(ep_size), key=lambda r: loads[r])


def peak_rank_recall(
    predicted: Sequence[float], actual: Sequence[float], ep_size: int, k: int
) -> float:
    """Hot-set recall among the experts of the rank the *planner* will look at.

    The planner picks its candidates from the predicted peak rank, so that is the
    slice scored here. When the predicted peak rank is not the actual peak rank this
    is still the honest question — the planner is choosing among those experts
    regardless — and `peak_rank_hit` reports how often that happens.
    """
    per_rank = len(predicted) // ep_size
    r = peak_rank(predicted, ep_size)
    lo, hi = r * per_rank, (r + 1) * per_rank
    return hot_set_recall(predicted[lo:hi], actual[lo:hi], k)


def peak_rank_hit(
    predicted: Sequence[float], actual: Sequence[float], ep_size: int
) -> bool:
    """Whether the prediction identifies the rank the layer actually waits on."""
    return peak_rank(predicted, ep_size) == peak_rank(actual, ep_size)


def prefill_pairs(
    records: Sequence[dict], min_assignments: int = PREFILL_MIN_ASSIGNMENTS
) -> list[dict]:
    """Pairs whose recorded assignment count puts them in the prefill regime."""
    out = []
    for record in records:
        for pair in record.get("pairs", []):
            if sum(pair["actual"]) >= min_assignments:
                out.append(pair)
    return out


def group_by_forward(
    records: Sequence[dict], min_assignments: int = PREFILL_MIN_ASSIGNMENTS
) -> list[list[dict]]:
    """Prefill pairs grouped by the forward they came from.

    The deciding number needs a whole forward at once: the transfer budget is spent
    across a forward's layers, so scoring one layer at a time would let every layer
    spend the whole budget — the same error that cost a real run 3x its transfers.
    """
    out = []
    for record in records:
        pairs = [
            p for p in record.get("pairs", []) if sum(p["actual"]) >= min_assignments
        ]
        if pairs:
            out.append(pairs)
    return out


def deciding_benefit(
    forwards: Sequence[Sequence[dict]],
    ep_size: int,
    budget: int,
    min_tokens: float = 0.0,
) -> dict:
    """Excess removed by predicted-chosen placements, scored on actual load.

    Reported beside the oracle — placements chosen from the actual load — because
    the oracle is the bound ticket 00 established and the gap between them is
    exactly what prediction error costs.

    Returns:
        Baseline, predicted-placement and oracle critical-path imbalance, the share
        of excess each removes, and how often the predicted placement ends up worse
        than placing nothing at all.
    """
    rows = []
    harmful = 0
    for pairs in forwards:
        actual = [as_layer(p["actual"], ep_size) for p in pairs]
        predicted = [as_layer(p["predicted"], ep_size) for p in pairs]
        base = imbalance.critical_path_imbalance(actual)
        moves = imbalance.plan_moves(predicted, budget=budget)
        scored = imbalance.critical_path_imbalance(imbalance.apply_moves(actual, moves))
        oracle = imbalance.critical_path_imbalance(
            imbalance.place_globally(actual, budget=budget)
        )
        if scored > base:
            harmful += 1
        rows.append((base, scored, oracle, len(moves)))
    if not rows:
        return {}

    def excess_removed(base: float, after: float) -> float:
        return 0.0 if base <= 1.0 else (base - after) / (base - 1.0) * 100.0

    base_m = statistics.mean(r[0] for r in rows)
    scored_m = statistics.mean(r[1] for r in rows)
    oracle_m = statistics.mean(r[2] for r in rows)
    return {
        "forwards": len(rows),
        "layers_per_forward": round(statistics.mean(len(f) for f in forwards), 1),
        "placements_per_forward": round(statistics.mean(r[3] for r in rows), 1),
        "baseline": round(base_m, 4),
        "predicted": round(scored_m, 4),
        "oracle": round(oracle_m, 4),
        "excess_removed_predicted_pct": round(excess_removed(base_m, scored_m), 1),
        "excess_removed_oracle_pct": round(excess_removed(base_m, oracle_m), 1),
        "forwards_made_worse": harmful,
    }


def accuracy_table(pairs: Sequence[dict], ep_size: int, ks: Sequence[int]) -> dict:
    """Per-lookahead-distance accuracy on the prefill pairs of one dump."""
    by_distance: dict[int, list[dict]] = {}
    for pair in pairs:
        by_distance.setdefault(pair["target"] - pair["source"], []).append(pair)
    out = {}
    for distance, group in sorted(by_distance.items()):
        row = {"pairs": len(group)}
        for k in ks:
            row[f"recall@{k}"] = round(
                statistics.mean(
                    hot_set_recall(p["predicted"], p["actual"], k) for p in group
                ),
                4,
            )
            row[f"peak_rank_recall@{k}"] = round(
                statistics.mean(
                    peak_rank_recall(p["predicted"], p["actual"], ep_size, k)
                    for p in group
                ),
                4,
            )
        row["peak_rank_hit"] = round(
            statistics.mean(
                peak_rank_hit(p["predicted"], p["actual"], ep_size) for p in group
            ),
            4,
        )
        row["count_error"] = round(
            statistics.mean(count_error(p["predicted"], p["actual"]) for p in group), 4
        )
        out[f"lookahead {distance}"] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dumps", nargs="+", type=Path)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--budget", type=int, default=43)
    parser.add_argument("--min-assignments", type=int, default=PREFILL_MIN_ASSIGNMENTS)
    parser.add_argument("--ks", type=int, nargs="+", default=(1, 2, 4))
    args = parser.parse_args()

    report = {}
    for path in args.dumps:
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        pairs = prefill_pairs(records, args.min_assignments)
        if not pairs:
            report[path.name] = {"note": "no prefill-regime pairs"}
            continue
        report[path.name] = {
            "prefill_pairs": len(pairs),
            "accuracy": accuracy_table(pairs, args.ep_size, args.ks),
            "deciding_benefit": deciding_benefit(
                group_by_forward(records, args.min_assignments),
                args.ep_size,
                args.budget,
            ),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
