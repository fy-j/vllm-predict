# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turn a `VLLM_EPLB_DUMP_LOAD_PATH` dump into the figure's data JSON.

`build_rank_imbalance.py` consumed a hand-made JSON, so the figure could not be
rebuilt for another model or another run without redoing that by hand. This is
that step, written down.

**The metric is per layer.** Every MoE layer is its own collective waiting on its own
slowest rank, so the quantity that corresponds to time is each layer's peak over its
own mean, and the model-level figure is the sum of per-layer peaks over the sum of
per-layer means. Summing a rank's load across layers *first* lets different layers'
peaks land on different ranks and cancel: on Korean prompts that aggregate reads 1.24
against a per-layer 1.90. Both are emitted, and `aggregate_imbalance` exists only to
show that gap — never to size a placement.

Bands are set by **tokens per expert**, not by a prefill/decode label, because what
decides whether imbalance costs anything is how many tokens each expert sees relative
to the MoE kernel's `BLOCK_SIZE_M`. Below one block every touched expert costs the same
and the imbalance is free.

Usage:
    python make_imbalance_data.py --dump ../results/h100/dsv4-baseline/dump.jsonl \\
        --out dsv4-imbalance-data.json --block-size-m 128
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _forward_rows(path: Path) -> list[dict]:
    """Read the dump, keeping only records that carry per-rank load."""
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not isinstance(rec, dict) or "rank_load" not in rec:
            raise ValueError(
                f"{path} is not the self-describing record form; re-capture it."
            )
        out.append(rec)
    return out


def _band(rec: dict, block_size_m: int) -> str:
    """Which regime this forward is in, from tokens per expert.

    The threshold that matters is the MoE kernel's `BLOCK_SIZE_M`: at or below one
    block per expert, every touched expert is padded to the same block count, so no
    placement can save time however skewed the routing is.
    """
    ep = len(rec["rank_load"][0])
    logical = len(rec["logical_load"][0])
    per_layer = statistics.mean(sum(r) for r in rec["rank_load"])
    tokens_per_expert = per_layer / logical if logical else 0.0
    del ep
    if tokens_per_expert <= 1.0:
        return "empty"
    return "prefill" if tokens_per_expert > block_size_m else "decode"


def summarize(records: list[dict], block_size_m: int) -> dict:
    """Per-layer share, per-layer peak-over-mean, and the aggregate that hides it."""
    bands: dict[str, list[dict]] = {}
    for rec in records:
        bands.setdefault(_band(rec, block_size_m), []).append(rec)

    out: dict[str, dict] = {}
    for band, recs in bands.items():
        if band == "empty":
            continue
        ep = len(recs[0]["rank_load"][0])
        num_layers = len(recs[0]["rank_load"])
        logical = len(recs[0]["logical_load"][0])

        # Mean over forwards, per layer per rank, expressed as a share of that
        # layer's total so layers of different size are on one scale.
        share = []
        peak_over_mean = []
        for li in range(num_layers):
            per_rank = [[rec["rank_load"][li][r] for rec in recs] for r in range(ep)]
            means = [statistics.mean(x) for x in per_rank]
            total = sum(means) or 1.0
            share.append([100.0 * m / total for m in means])
            peak_over_mean.append(max(means) / (total / ep))

        # The aggregate view: sum each rank across layers, then compare ranks.
        rank_totals = [
            statistics.mean(
                sum(rec["rank_load"][li][r] for li in range(num_layers)) for rec in recs
            )
            for r in range(ep)
        ]
        agg_total = sum(rank_totals) or 1.0

        tokens_per_expert = statistics.mean(
            statistics.mean(sum(r) for r in rec["rank_load"]) / logical for rec in recs
        )
        m_tokens = statistics.mean(
            statistics.mean(sum(r) for r in rec["rank_load"]) for rec in recs
        )

        out[band] = {
            "share_by_layer": share,
            "peak_over_mean": peak_over_mean,
            "aggregate_share": [100.0 * t / agg_total for t in rank_totals],
            "aggregate_imbalance": max(rank_totals) / (agg_total / ep),
            # The figure reads `critical_path`; the same number under the longer name is
            # kept because `imbalance.py` and the RESULTS tables use that spelling.
            "critical_path": sum(max(s) / 100.0 for s in share) / (num_layers / ep),
            "critical_path_imbalance": sum(max(s) / 100.0 for s in share)
            / (num_layers / ep),
            "forwards": len(recs),
            "ep": ep,
            "num_layers": num_layers,
            "num_logical_experts": logical,
            "M": m_tokens,
            "tokens_per_expert": tokens_per_expert,
            # A multinomial floor: even perfectly random routing is not exactly even
            # at finite sample size, so an imbalance at or under this is noise.
            "noise_floor": 1.0 + (ep - 1) ** 0.5 / max(1.0, tokens_per_expert) ** 0.5,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--block-size-m",
        type=int,
        default=128,
        help="MoE kernel block size; the bar below which imbalance is free",
    )
    args = ap.parse_args()

    data = summarize(_forward_rows(args.dump), args.block_size_m)
    if not data:
        raise SystemExit(
            f"{args.dump} held no forward above one token per expert; nothing to plot"
        )
    args.out.write_text(json.dumps(data, indent=1))
    for band, d in sorted(data.items()):
        print(
            f"{band:8} {d['forwards']:>4} forwards  "
            f"ep={d['ep']} layers={d['num_layers']} "
            f"experts={d['num_logical_experts']}  "
            f"tokens/expert {d['tokens_per_expert']:.0f}  "
            f"critical path {d['critical_path_imbalance']:.4f}  "
            f"aggregate {d['aggregate_imbalance']:.4f}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
