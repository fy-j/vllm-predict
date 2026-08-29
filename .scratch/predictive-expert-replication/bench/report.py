# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ticket 00 report: combine the probe, the serving runs, and the server log.

Produces the deliverable ticket 00 asks for: measured headroom per domain and
request shape, the MoE boundness ratio it must be read against, a cost profile
seeded from measurement, and a proceed/stop recommendation with its evidence.

Run from the repository root:

    python3 .scratch/predictive-expert-replication/bench/report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import regex as re

sys.path.insert(0, str(Path(__file__).parent))
from analyze import (  # noqa: E402
    QWEN3_30B_A3B,
    build_cost_profile,
    recommend,
)

# EplbState logs "avg_tokens=%.2f, max_tokens=%d, balancedness=%.4f" per step.
# balancedness is avg/max, so headroom is 1 - balancedness by construction.
_BALANCEDNESS = re.compile(
    r"avg_tokens=(?P<avg>[\d.]+), max_tokens=(?P<max>\d+), "
    r"balancedness=(?P<bal>[\d.]+)"
)

# Benchmark result filenames are bench-<mode>-<domain>-p<in>d<out>-c<conc>.json
_CASE = re.compile(
    r"bench-(?P<mode>[^-]+)-(?P<domain>[^-]+)-p(?P<prompt>\d+)d(?P<decode>\d+)"
    r"-c(?P<conc>\d+)\.json"
)


def scrape_balancedness(server_log: Path) -> dict:
    """Extract measured EP rank imbalance from the server log.

    Returns:
        Summary statistics, or an empty dict when the log has no samples, which
        means EPLB recording was not enabled for that run.
    """
    if not server_log.exists():
        return {}
    samples = [
        (float(m["avg"]), int(m["max"]), float(m["bal"]))
        for m in _BALANCEDNESS.finditer(server_log.read_text(errors="replace"))
    ]
    # Drop zero-load samples: they are warmup and dummy steps, and a rank with no
    # tokens has no imbalance to report.
    samples = [s for s in samples if s[1] > 0]
    if not samples:
        return {}
    balancedness = sorted(s[2] for s in samples)
    headroom = [1.0 - b for b in balancedness]
    return {
        "samples": len(samples),
        "balancedness_median": round(balancedness[len(balancedness) // 2], 4),
        "balancedness_min": round(balancedness[0], 4),
        "headroom_fraction_median": round(sorted(headroom)[len(headroom) // 2], 4),
        "headroom_fraction_p95": round(sorted(headroom)[int(len(headroom) * 0.95)], 4),
        "headroom_fraction_max": round(max(headroom), 4),
    }


def is_usable_case(payload: dict) -> tuple[bool, str]:
    """Reject a benchmark result that did not actually measure anything.

    `vllm bench serve` is a client. When the server is unreachable it does not
    fail: it prints and saves a report with every latency at 0.00. Such a file
    must never reach the report, because a zero TPOT would look like a win.

    Returns:
        Whether the case is usable, and the reason when it is not.
    """
    completed = payload.get("completed") or 0
    failed = payload.get("failed") or 0
    if completed <= 0:
        return False, "no requests completed"
    if failed:
        # Distinguish request failures from a truncated run: they point at
        # different causes, an overloaded or shared server versus a killed client.
        return False, f"{failed} requests failed, {completed} completed"
    requested = payload.get("num_prompts") or 0
    if requested and completed < requested:
        return False, f"truncated: {completed}/{requested} requests"
    if not (payload.get("mean_tpot_ms") or 0) > 0:
        return False, "zero TPOT, so the server was likely unreachable"
    return True, ""


def collect_serving_results(results_dir: Path) -> tuple[list[dict], list[dict]]:
    """Read the per-case benchmark JSON that `vllm bench serve` wrote.

    Returns:
        The usable cases, and the rejected ones with the reason, so a discarded
        run is visible in the report rather than silently missing.
    """
    cases: list[dict] = []
    rejected: list[dict] = []
    for path in sorted(results_dir.glob("bench-*.json")):
        meta = _CASE.match(path.name)
        if meta is None:
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            rejected.append({"file": path.name, "reason": "unparsable JSON"})
            continue
        usable, reason = is_usable_case(payload)
        if not usable:
            rejected.append({"file": path.name, "reason": reason})
            continue
        cases.append(
            {
                "domain": meta["domain"],
                "prompt_len": int(meta["prompt"]),
                "decode_len": int(meta["decode"]),
                "max_concurrency": int(meta["conc"]),
                "completed": payload.get("completed"),
                "request_throughput": payload.get("request_throughput"),
                "output_throughput": payload.get("output_throughput"),
                "mean_tpot_ms": payload.get("mean_tpot_ms"),
                "p99_tpot_ms": payload.get("p99_tpot_ms"),
                "mean_ttft_ms": payload.get("mean_ttft_ms"),
                "p99_ttft_ms": payload.get("p99_ttft_ms"),
                "p99_e2el_ms": payload.get("p99_e2el_ms"),
            }
        )
    return cases, rejected


def interpolate_ratio(hardware: dict, tokens_per_rank: float) -> float:
    """MoE boundness ratio at an arbitrary concurrency, from the probe's grid."""
    rows = sorted(
        hardware["layer_budget"]["by_tokens_per_rank"],
        key=lambda r: r["tokens_per_rank"],
    )
    below = [r for r in rows if r["tokens_per_rank"] <= tokens_per_rank]
    above = [r for r in rows if r["tokens_per_rank"] >= tokens_per_rank]
    low = below[-1] if below else rows[0]
    high = above[0] if above else rows[-1]
    if high["tokens_per_rank"] == low["tokens_per_rank"]:
        return low["moe_over_read_floor"]
    span = high["tokens_per_rank"] - low["tokens_per_rank"]
    frac = (tokens_per_rank - low["tokens_per_rank"]) / span
    return low["moe_over_read_floor"] + frac * (
        high["moe_over_read_floor"] - low["moe_over_read_floor"]
    )


def overlap_window_us(hardware: dict, lookahead: int, tokens_per_rank: float) -> float:
    """Time available to hide a transfer at a given prediction lookahead.

    A lookahead of one offers a single Attention block. Each additional layer of
    lookahead adds that layer's MoE plus the following Attention.
    """
    rows = sorted(
        hardware["layer_budget"]["by_tokens_per_rank"],
        key=lambda r: abs(r["tokens_per_rank"] - tokens_per_rank),
    )
    nearest = rows[0]
    attention = nearest["attention_us"]
    moe = nearest["moe_us"]
    return attention + (lookahead - 1) * (moe + attention)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).parent
    parser.add_argument("--results-dir", type=Path, default=here / "results")
    parser.add_argument("--lookahead", type=int, default=2)
    parser.add_argument(
        "--tokens-per-rank",
        type=float,
        default=64.0,
        help="Decode concurrency per rank to evaluate at; KV capacity caps this.",
    )
    args = parser.parse_args()

    hardware = json.loads((args.results_dir / "hardware.json").read_text())
    imbalance = scrape_balancedness(args.results_dir / "server-recording.log")
    serving, rejected_cases = collect_serving_results(args.results_dir)

    transfer_us = hardware["expert_transfer"]["idle_transfer_us_max"]
    window_us = overlap_window_us(hardware, args.lookahead, args.tokens_per_rank)
    exposed_us = max(0.0, transfer_us - window_us)
    ratio = interpolate_ratio(hardware, args.tokens_per_rank)
    floor_us = hardware["layer_budget"]["expert_weight_read_floor_us"]
    peak_moe_us = ratio * floor_us

    verdict = None
    if imbalance:
        verdict = recommend(
            headroom_fraction=imbalance["headroom_fraction_median"],
            moe_ratio=ratio,
            exposed_transfer_us=exposed_us,
            peak_moe_us=peak_moe_us,
        )

    device_name = hardware["interconnect"]["device_name"]
    profile = build_cost_profile(
        model=QWEN3_30B_A3B.name,
        dtype="bfloat16",
        ep_size=hardware["ep_size"],
        num_logical_experts=QWEN3_30B_A3B.num_logical_experts,
        device_name=device_name,
        # Per token-expert pair, from the compute-bound end of the probe grid.
        expert_compute_us_per_token=round(
            hardware["layer_budget"]["by_tokens_per_rank"][-1]["moe_us"]
            / (512 * QWEN3_30B_A3B.experts_per_token),
            4,
        ),
        attention_window_us=window_us,
        transfer_latency_us=transfer_us,
        # Idle bandwidth until the under-load measurement replaces it. Marked in
        # the report so it cannot be mistaken for a validated figure.
        usable_transfer_bandwidth_bytes_per_us=round(
            QWEN3_30B_A3B.bytes_per_expert / transfer_us, 1
        ),
    )

    report = {
        "model": QWEN3_30B_A3B.name,
        "interconnect": {
            "has_nvlink": hardware["interconnect"]["has_nvlink"],
            "device": device_name,
            "pcie": hardware["interconnect"]["pcie"],
        },
        "transfer": {
            "one_expert_mib": hardware["expert_transfer"]["mib_per_expert"],
            "idle_transfer_us": transfer_us,
            "lookahead": args.lookahead,
            "overlap_window_us": round(window_us, 1),
            "exposed_us": round(exposed_us, 1),
            "hidden_fraction": round(min(1.0, window_us / transfer_us), 3),
        },
        "boundness": {
            "tokens_per_rank": args.tokens_per_rank,
            "expert_weight_read_floor_us": floor_us,
            "moe_over_read_floor": round(ratio, 2),
        },
        "measured_imbalance": imbalance or "not measured (no EPLB recording log)",
        "serving_cases": serving,
        "rejected_cases": rejected_cases,
        "kv_ceiling": hardware["kv_ceiling"],
        "verdict": None
        if verdict is None
        else {
            "proceed": verdict.proceed,
            "reason": verdict.reason,
            "headroom_fraction": round(verdict.headroom_fraction, 4),
            "moe_over_read_floor": round(verdict.moe_ratio, 2),
            "recoverable_us_per_layer_per_step": round(
                verdict.recoverable_us_per_step, 2
            ),
            "min_residency_steps": verdict.min_residency_steps,
        },
        "caveats": [
            (
                "The MoE boundness ratio comes from a bmm microbenchmark, not "
                "vLLM's fused MoE kernel; treat it as indicative until the "
                "serving run confirms it."
            ),
            (
                "No tuned MoE configuration exists for this device, so the served "
                "MoE kernel is untuned and its absolute timing is pessimistic."
            ),
            (
                "usable_transfer_bandwidth_bytes_per_us is still the idle figure. "
                "It must be replaced by a measurement taken while token dispatch "
                "and combine are running before the planner relies on it."
            ),
        ],
        "cost_profile": profile,
    }

    out = args.results_dir / "ticket-00-report.json"
    out.write_text(json.dumps(report, indent=2))
    (args.results_dir / "cost-profile.json").write_text(json.dumps(profile, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
