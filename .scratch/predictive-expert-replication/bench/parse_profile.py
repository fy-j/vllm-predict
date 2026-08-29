# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute GPU time inside pure-decode steps of a torch profiler trace.

Ticket 00's decisive question is what fraction of TPOT the MoE layer accounts
for. Two attribution errors made the first attempt unusable and this module
exists to prevent both:

1. A profile window taken under real load contains prefill steps. A 2047-token
   prefill costs about 300 ms against a 136 ms decode step, so leaking one
   inflates every per-step figure. Windows come from the runner's own
   `execute_context_C(c)_generation_G(g)` annotation and only `C == 0` is kept.

2. `fused_moe_kernel` fires twice per layer, once for the gate/up projection and
   once for the down projection. Inferring a step count from its call count
   halves every per-step figure. The step count comes from the annotations.

The output is per-decode-step milliseconds by kernel class, which is the form
the boundness question needs: MoE time per layer against the weight-read floor,
both measured with the kernel that actually serves requests.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import gzip
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

KERNEL_CLASSES = ("nccl", "moe_expert", "attention", "other")

_ANNOTATION = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")

# `profiler_config.detailed_trace_annotation` emits a different shape entirely:
# `execute_{tokens}_context_{n}(sq...sk...)`. Recognised only so the failure names
# the annotation format rather than sending the reader to widen the window.
_DETAILED_ANNOTATION = re.compile(r"execute_\d+_context_\d+\(")

# Substring tests, because the mangled kernel names carry template arguments.
_ATTENTION_MARKERS = (
    "sparse_attn_fwd_kernel",
    "QNormRope",
    "_fused_inv_rope",
    "mhc_",
    "flash_fwd",
    "flash::",
    "paged_attention",
    "flashinfer",
)


def parse_annotation(name: str) -> tuple[int, int, int, int] | None:
    """Read a step's batch shape out of the runner's profiler annotation.

    Returns:
        `(ctx_requests, ctx_tokens, gen_requests, gen_tokens)`, or None when the
        name is not a step annotation.
    """
    match = _ANNOTATION.search(name or "")
    if match is None:
        return None
    return tuple(int(g) for g in match.groups())  # type: ignore[return-value]


def classify_kernel(name: str) -> str:
    """Bucket a CUDA kernel into the four classes the attribution reports."""
    if name.startswith("ncclDevKernel") or "nccl" in name:
        return "nccl"
    if "fused_moe_kernel" in name or "grouped_gemm" in name:
        return "moe_expert"
    # DeepSeek V4 runs DeepGEMM, whose MoE kernels are named for their *scheduler*
    # rather than for the word "moe". A grouped scheduler is the tell: the expert GEMM
    # is the only grouped one, and both of its halves carry it —
    # `fp8_gemm_kernel<4096,4096,...GroupedWithOffsetScheduler>` is w13 (gate+up, output
    # 2 x moe_intermediate) and `<4096,2048,...>` is w2 (down, K = moe_intermediate).
    # Without this, DSV4's expert GEMM lands in "other" and the share reads near zero —
    # a wrong number rather than an error. Matching on the substring "gemm" alone would
    # be worse: the attention projections are `sm90_fp8_gemm_1d2d_impl` and are dense.
    if "GroupedWithOffsetScheduler" in name or "GroupedMasked" in name:
        return "moe_expert"
    if any(marker in name for marker in _ATTENTION_MARKERS):
        return "attention"
    return "other"


@dataclass(frozen=True)
class DecodeWindow:
    """One pure-decode step, as a GPU-side time span."""

    start: float
    end: float
    gen_requests: int
    gen_tokens: int


@dataclass
class WindowAttribution:
    """GPU microseconds by kernel class inside one decode step."""

    gen_tokens: int
    gen_requests: int
    by_class: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(KERNEL_CLASSES, 0.0)
    )


def decode_windows(events: list[dict], phase: str = "decode") -> list[DecodeWindow]:
    """Extract one phase's step windows from a trace's events.

    Only `gpu_user_annotation` is used. Each step also emits a CPU-side
    `user_annotation` covering a wider span; counting both would double the step
    count and halve every per-step figure.

    Args:
        events: The trace's events.
        phase: `"decode"` keeps only steps with no prefill token, which is what the
            decode question needs and the default this module was written for.
            `"prefill"` keeps only steps that carry prefill tokens, which is what
            sets the *prefill* ceiling: balancing touches the expert GEMM alone, so
            the ceiling is the expert GEMM's share of a prefill step times what
            perfect balance can recover. Mixing the two phases is what this module
            exists to prevent, so the two are never combined.

    Raises:
        ValueError: On an unknown phase, rather than silently defaulting, since a
            typo would otherwise return the other phase's windows.
    """
    if phase not in ("decode", "prefill"):
        raise ValueError(f"phase must be 'decode' or 'prefill', not {phase!r}")
    windows = []
    detailed = 0
    for event in events:
        if event.get("cat") != "gpu_user_annotation":
            continue
        if _DETAILED_ANNOTATION.search(event.get("name", "") or ""):
            detailed += 1
            continue
        parsed = parse_annotation(event.get("name", ""))
        if parsed is None:
            continue
        ctx_requests, ctx_tokens, gen_requests, gen_tokens = parsed
        has_prefill = bool(ctx_requests or ctx_tokens)
        if has_prefill != (phase == "prefill"):
            continue  # a window of the other phase; never mix the two
        start = float(event["ts"])
        windows.append(
            DecodeWindow(
                start=start,
                end=start + float(event.get("dur") or 0.0),
                gen_requests=gen_requests,
                gen_tokens=gen_tokens,
            )
        )
    if not windows and detailed:
        raise ValueError(
            f"this trace carries {detailed} step annotations in the detailed form "
            "(`detailed_trace_annotation`), which this parser cannot read. Recapture "
            "with it off, or extend `parse_annotation`; the window is not the problem."
        )
    return sorted(windows, key=lambda w: w.start)


def attribute_windows(
    windows: list[DecodeWindow], kernels: list[dict]
) -> list[WindowAttribution]:
    """Sum each kernel's duration into the decode window it starts inside.

    Kernels that start in no decode window - prefill work, and anything between
    steps - are dropped rather than spread, which is what keeps a prefill step
    from contaminating the decode attribution.
    """
    rows = [WindowAttribution(w.gen_tokens, w.gen_requests) for w in windows]
    starts = [w.start for w in windows]
    for event in kernels:
        if event.get("cat") != "kernel":
            continue
        duration = event.get("dur")
        if duration is None:
            continue
        ts = float(event["ts"])
        index = bisect.bisect_right(starts, ts) - 1
        if index < 0 or ts >= windows[index].end:
            continue
        rows[index].by_class[classify_kernel(event.get("name", ""))] += float(duration)
    return rows


def summarize(rows: list[WindowAttribution], num_layers: int) -> dict:
    """Reduce per-step attributions to the figures ticket 00 reports.

    Raises:
        ValueError: when no pure-decode step was captured. Returning zeros here
            would present as a finding - a zero MoE share - rather than a failed
            measurement.
    """
    if not rows:
        raise ValueError(
            "no decode-only steps in this trace; the profile window caught only "
            "prefill, so widen it or profile later into the run"
        )
    steps = len(rows)
    totals = {name: sum(row.by_class[name] for row in rows) for name in KERNEL_CLASSES}
    attributed = sum(totals.values())
    return {
        "decode_steps": steps,
        "gen_tokens_per_step": statistics.mean(row.gen_tokens for row in rows),
        "gen_requests_per_step": statistics.mean(row.gen_requests for row in rows),
        "per_step_ms": {
            name: round(total / steps / 1000.0, 4) for name, total in totals.items()
        },
        "attributed_ms_per_step": round(attributed / steps / 1000.0, 4),
        # Two kernel launches per layer, so the divisor is layers, not calls.
        "moe_us_per_layer": round(totals["moe_expert"] / steps / num_layers, 2),
        "nccl_us_per_layer": round(totals["nccl"] / steps / num_layers, 2),
        "moe_share_of_attributed": round(totals["moe_expert"] / attributed, 4)
        if attributed
        else 0.0,
        "nccl_share_of_attributed": round(totals["nccl"] / attributed, 4)
        if attributed
        else 0.0,
    }


def parse_trace(path: Path, num_layers: int, phase: str = "decode") -> dict:
    """Load one rank's trace and summarize one phase's steps."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as handle:  # type: ignore[operator]
        events = json.load(handle).get("traceEvents", [])
    rows = attribute_windows(decode_windows(events, phase), events)
    result = summarize(rows, num_layers)
    result["rank"] = path.name.split("_")[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument(
        "--phase",
        choices=("decode", "prefill"),
        default="decode",
        help="which steps to attribute; prefill is what sets the prefill ceiling",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    pattern = str(args.trace_dir / "dp*.pt.trace.json*")
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        print(f"no rank traces under {args.trace_dir}", file=sys.stderr)
        raise SystemExit(1)

    per_rank, failed = [], []
    for path in paths:
        try:
            per_rank.append(parse_trace(path, args.num_layers, args.phase))
        except ValueError as exc:
            failed.append({"rank": path.name.split("_")[0], "reason": str(exc)})

    if not per_rank:
        print(json.dumps({"failed": failed}, indent=2))
        raise SystemExit(1)

    moe_per_layer = [r["moe_us_per_layer"] for r in per_rank]
    report = {
        "phase": args.phase,
        "ranks_parsed": len(per_rank),
        "ranks_failed": failed,
        "per_rank": per_rank,
        "mean": {
            "moe_us_per_layer": round(statistics.mean(moe_per_layer), 2),
            "nccl_us_per_layer": round(
                statistics.mean(r["nccl_us_per_layer"] for r in per_rank), 2
            ),
            "moe_share_of_attributed": round(
                statistics.mean(r["moe_share_of_attributed"] for r in per_rank), 4
            ),
            "gen_tokens_per_step": round(
                statistics.mean(r["gen_tokens_per_step"] for r in per_rank), 1
            ),
        },
        # The transduction question: token imbalance only matters if it shows up
        # as a spread in MoE time across ranks.
        "moe_time_imbalance": {
            "min_us_per_layer": min(moe_per_layer),
            "max_us_per_layer": max(moe_per_layer),
            "max_over_mean": round(
                max(moe_per_layer) / statistics.mean(moe_per_layer), 3
            )
            if moe_per_layer
            else None,
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
