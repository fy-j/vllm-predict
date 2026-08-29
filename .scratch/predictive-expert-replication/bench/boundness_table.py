# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build ticket 00's decisive table from the per-concurrency attribution files.

One row per operating point: what a decode step spends on the expert GEMM, what
it spends on collectives, how unevenly the expert GEMM lands across ranks, and
the ceiling on what perfect expert balance could therefore recover.

The recoverable figure is deliberately an upper bound. A decode step waits for
its slowest rank, so the expert GEMM contributes the *peak* rank's time; perfect
balance would reduce that to the mean and no further. A real placement policy
recovers some fraction of that, never more.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

EXPECTED_RANKS = 8


def build_rows(results_dir: Path, num_layers: int = 48) -> list[dict]:
    """Read every `attribution-c*.json` and derive the per-point row."""
    rows = []
    for path in sorted(
        results_dir.glob("attribution-c*.json"),
        key=lambda p: int(p.stem.split("-c")[1]),
    ):
        report = json.loads(path.read_text())
        per_rank = report["per_rank"]
        moe = [r["moe_us_per_layer"] for r in per_rank]
        mean_moe, peak_moe = statistics.mean(moe), max(moe)
        attributed = statistics.mean(r["attributed_ms_per_step"] for r in per_rank)
        # Peak minus mean, over the whole model, is the whole prize.
        recoverable_ms = (peak_moe - mean_moe) * num_layers / 1000.0
        rows.append(
            {
                "concurrency": int(path.stem.split("-c")[1]),
                "ranks": report["ranks_parsed"],
                "ranks_missing": [r["rank"] for r in report.get("ranks_failed", [])],
                # A decode step waits for its slowest rank, so a missing rank can
                # only lower the observed peak - biasing the imbalance and the
                # recoverable figure downwards, which is the direction that
                # flatters a negative verdict. Never silent.
                "partial_rank_set": report["ranks_parsed"] < EXPECTED_RANKS,
                "gen_tokens_per_step": report["mean"]["gen_tokens_per_step"],
                "moe_us_per_layer_mean": round(mean_moe, 1),
                "moe_us_per_layer_peak": round(peak_moe, 1),
                "nccl_us_per_layer": report["mean"]["nccl_us_per_layer"],
                "moe_ms_per_step": round(mean_moe * num_layers / 1000.0, 2),
                "attributed_ms_per_step": round(attributed, 2),
                # Ratio of the means, not the mean of per-rank ratios. The two
                # disagree by 1.7x at c8, where one rank's collectives are an
                # outlier, and a reader dividing the two columns beside this one
                # must get the number this column states.
                "moe_share_of_attributed": round(
                    mean_moe * num_layers / 1000.0 / attributed, 4
                )
                if attributed
                else None,
                "moe_imbalance_peak_over_mean": round(peak_moe / mean_moe, 3)
                if mean_moe
                else None,
                "recoverable_ms_per_step": round(recoverable_ms, 3),
                "recoverable_share_of_step": round(recoverable_ms / attributed, 4)
                if attributed
                else None,
            }
        )
    return rows


def attach_serving(rows: list[dict], results_dir: Path) -> list[dict]:
    """Add measured TPOT where the point's benchmark result survived.

    Attributed GPU time comes to 49% to 67% of TPOT, so a third to a half of the
    step is **unattributed** - idle, host-side, or in kernels outside the four
    classified buckets. A share quoted against attributed time is therefore larger
    than the same share against a whole step, and both are reported so the smaller
    one cannot be omitted.

    The TPOT here is *not* an unprofiled reference: `bench-c*.json` is written by
    the same client run the profile window sits inside. It bounds the step, it does
    not measure a clean one.
    """
    for row in rows:
        path = results_dir / f"bench-c{row['concurrency']}.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        tpot = payload.get("mean_tpot_ms") or 0.0
        if tpot <= 0:
            continue
        row["measured_tpot_ms"] = round(tpot, 2)
        row["attributed_fraction_of_tpot"] = round(
            row["attributed_ms_per_step"] / tpot, 3
        )
        row["unattributed_fraction_of_tpot"] = round(
            1.0 - row["attributed_ms_per_step"] / tpot, 3
        )
        row["moe_share_of_tpot"] = round(row["moe_ms_per_step"] / tpot, 4)
        row["recoverable_share_of_tpot"] = round(
            row["recoverable_ms_per_step"] / tpot, 4
        )
    return rows


def to_markdown(rows: list[dict]) -> str:
    """Render the table, with rank coverage visible and both denominators shown."""
    header = (
        "| 并发 | ranks | gen tok/step | MoE µs/层 (mean/peak) | NCCL µs/层 | "
        "MoE ms/step | attributed ms | TPOT ms | MoE/attributed | MoE/TPOT | "
        "MoE 不均衡 | 可恢复 ms | 可恢复/TPOT |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- |\n"
    )

    def pct(value):
        return "-" if value is None else f"{value * 100:.2f}%"

    def num(value, fmt="{:.2f}"):
        return "-" if value is None else fmt.format(value)

    lines = []
    for r in rows:
        ranks = f"{r['ranks']}/{EXPECTED_RANKS}"
        if r["partial_rank_set"]:
            ranks += " ⚠"
        lines.append(
            f"| {r['concurrency']} | {ranks} | {r['gen_tokens_per_step']} | "
            f"{r['moe_us_per_layer_mean']} / {r['moe_us_per_layer_peak']} | "
            f"{r['nccl_us_per_layer']} | {r['moe_ms_per_step']} | "
            f"{r['attributed_ms_per_step']} | {num(r.get('measured_tpot_ms'))} | "
            f"{pct(r['moe_share_of_attributed'])} | "
            f"{pct(r.get('moe_share_of_tpot'))} | "
            f"{r['moe_imbalance_peak_over_mean']}× | "
            f"{r['recoverable_ms_per_step']} | "
            f"{pct(r.get('recoverable_share_of_tpot'))} |"
        )
    out = header + "\n".join(lines)

    short = [r for r in rows if r["partial_rank_set"]]
    if short:
        out += "\n\n"
        for r in short:
            out += (
                f"⚠ c{r['concurrency']} covers only {r['ranks']} of "
                f"{EXPECTED_RANKS} ranks (missing {', '.join(r['ranks_missing'])}). "
                "A decode step waits for its slowest rank, so a missing rank can "
                "only lower the observed peak: this row's imbalance and "
                "recoverable figures are lower bounds, in the direction that "
                "flatters a negative verdict. Do not quote it as the headline.\n"
            )
    out += (
        "\n`MoE/attributed` divides by classified GPU time; "
        "`MoE/TPOT` divides by the whole step, of which "
        f"{int((1 - max(r.get('attributed_fraction_of_tpot', 1) for r in rows)) * 100)}"
        "% to "
        f"{int((1 - min(r.get('attributed_fraction_of_tpot', 1) for r in rows)) * 100)}"
        "% is unattributed. The TPOT column is measured by the same client run "
        "the profile window sits inside, so it bounds a profiled step rather than "
        "measuring a clean one.\n"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).parent
    parser.add_argument("--results-dir", type=Path, default=here / "results/boundness")
    parser.add_argument("--num-layers", type=int, default=48)
    args = parser.parse_args()

    rows = attach_serving(
        build_rows(args.results_dir, args.num_layers), args.results_dir
    )
    if not rows:
        raise SystemExit(f"no attribution-c*.json under {args.results_dir}")
    print(to_markdown(rows))
    print()
    print(json.dumps(rows, indent=2))
    (args.results_dir / "boundness-table.json").write_text(json.dumps(rows, indent=2))
    (args.results_dir / "boundness-table.md").write_text(to_markdown(rows))


if __name__ == "__main__":
    main()
