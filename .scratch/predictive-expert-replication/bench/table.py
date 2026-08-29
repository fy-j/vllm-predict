# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render the serving results as a table, with the checks that matter inline.

Two columns exist because a benchmark can look complete and still be measuring
the wrong thing. `real batch` is derived from `throughput x TPOT` rather than
taken from the concurrency flag: a smaller prompt count silently caps requests
in flight, which once made three concurrency points report the same batch. And
average prompt and decode lengths are shown against their targets, because
`--custom-output-len` is only an upper bound unless `--ignore-eos` is set.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import regex as re

CASE = re.compile(
    r"bench-recording-(?P<domain>\w+)-p(?P<prompt>\d+)d(?P<decode>\d+)-c(?P<conc>\d+)"
)
BALANCEDNESS = re.compile(r"max_tokens=(\d+), balancedness=([\d.]+)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path(__file__).parent / "results"
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(args.results_dir.glob("bench-recording-*.json")):
        meta = CASE.match(path.name)
        if meta is None:
            continue
        d = json.loads(path.read_text())
        done = max(1, d["completed"])
        rows.append(
            {
                "shape": f"{meta['prompt']}->{meta['decode']}",
                "dom": meta["domain"],
                "conc": int(meta["conc"]),
                "done": f"{d['completed']}/{d['num_prompts']}",
                "failed": d.get("failed") or 0,
                "inp": d["total_input_tokens"] / done,
                "out": d["total_output_tokens"] / done,
                "target_out": int(meta["decode"]),
                "batch": d["output_throughput"] * d["mean_tpot_ms"] / 1000,
                "mtpot": d["mean_tpot_ms"],
                "ptpot": d["p99_tpot_ms"],
                "mttft": d["mean_ttft_ms"],
                "pttft": d["p99_ttft_ms"],
                "e2e": d["p99_e2el_ms"],
                "thr": d["output_throughput"],
                "rps": d["request_throughput"],
            }
        )
    if not rows:
        print("no results yet", file=sys.stderr)
        return

    rows.sort(key=lambda r: (r["shape"], r["conc"]))
    head = (
        f"{'shape':>11} {'dom':>5} {'conc':>5} {'real batch':>11} {'done':>9} "
        f"{'in':>6} {'out':>6} {'mTPOT':>7} {'p99TPOT':>8} {'mTTFT':>7} "
        f"{'p99TTFT':>8} {'p99 e2e':>9} {'tok/s':>7} {'req/s':>7}"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        warn = ""
        if r["failed"]:
            warn = f"  <- {r['failed']} failed"
        elif r["batch"] < r["conc"] * 0.8:
            warn = f"  <- batch only {r['batch'] / r['conc']:.0%} of concurrency"
        print(
            f"{r['shape']:>11} {r['dom']:>5} {r['conc']:>5} "
            f"{r['batch']:>7.0f} ({r['batch'] / 8:>.0f}) {r['done']:>9} "
            f"{r['inp']:>6.0f} {r['out']:>6.0f} {r['mtpot']:>7.1f} {r['ptpot']:>8.1f} "
            f"{r['mttft']:>7.0f} {r['pttft']:>8.0f} {r['e2e']:>9.0f} "
            f"{r['thr']:>7.0f} {r['rps']:>7.2f}{warn}"
        )
    print(
        "\nreal batch = throughput x mean TPOT, total (per rank); "
        "not the concurrency flag"
    )

    log = args.results_dir / "server-recording.log"
    if log.exists():
        samples = [
            float(b)
            for m, b in BALANCEDNESS.findall(log.read_text(errors="replace"))
            if int(m) > 0
        ]
        if samples:
            head = 1 - statistics.median(samples)
            print(
                f"\nEP rank imbalance over the whole run: headroom median "
                f"{head:.1%}, {len(samples)} samples"
            )
            print(
                "  headroom = 1 - balancedness = (peak - mean) / peak, summed "
                "per layer; it is the ceiling on any placement policy's gain"
            )


if __name__ == "__main__":
    main()
