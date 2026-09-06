# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Is prediction's launch half dispatch-bound? Ticket 18's stop gate.

Prediction's added cost splits 47.6% barriers / 52.4% launches-and-compute (ticket 11).
That second half is 9.67 ms over 44 source layers, 220 us each, against a gate GEMM
worth about 3 us of arithmetic on this model. If the 220 us is the host dispatching
work, fusing about seven launches into one recovers most of it. If it is kernel time in
one expensive operator, fewer launches recover nothing and ticket 18 does not apply.

**Prediction's operators are found by differencing two arms, not by matching names.**
The first version of this script matched `linear|mm|topk|...` and attributed the whole
model: the stock arm, which runs no prediction at all, reported 3.36 s of "prediction
ops". A name list cannot separate prediction's gate GEMM from the 44 real ones.
Differencing the arms can, because they differ in nothing else.

Three trace facts this rests on, each of which the first version got wrong:

  * `cpu_op` events **nest** — `aten::linear` holds `aten::matmul` holds `aten::mm` — so
    summing them triple-counts. They are counted, never summed.
  * a `cpu_op`'s duration is host time, not kernel time. Device work is the `kernel`
    category, and overlapping kernels are unioned rather than added.
  * host CUDA API time is `cuda_runtime` **and** `cuda_driver`. Triton launches through
    `cuLaunchKernelEx`, which the driver category holds: counting only `cuda_runtime`
    reported 0.94 added launches per source layer where the true figure is 2.72, and
    undercounted host time with it. Both categories are unioned, not summed, because
    host spans nest.
  * `cudaEventSynchronize` is separated from dispatch. Its call count is unchanged
    between the arms while its duration is not, and its sign flips per rank (+1.97 ms on
    dp0, -2.21 ms on dp5). It measures how early a rank arrives, not what it costs, so
    folding it into "host runtime" would price a skew as a dispatch.

Restricted to `execute_context`: in a 12.95 s trace of this workload the real forwards
covered 0.54 s and an aggregate table was dominated by idle `execute_dummy_batch` work.
NCCL kernels are reported apart — a barrier's duration on one rank measures that rank's
earliness, and the rank that arrives last reads zero.

**Every rank is read, not the first.** An earlier version took `sorted(glob(...))[0]`,
which is dp0, and this is the trap ticket 11 fell into with the 9.3 us AllGather: one
rank's view of a collective is its own earliness. The per-rank rows are printed as well
as their mean, because a term whose sign flips across ranks is not a cost.

Usage:
    python attribute_prediction_ops.py --trace-dir results/<run>/trace-b0 \
        --baseline-dir results/<run>/trace-boff --source-layers 44
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

_NCCL = re.compile(r"nccl", re.IGNORECASE)
_EXECUTE_CONTEXT = re.compile(r"execute_context")
# `cuLaunchKernelEx` is the driver-API entry Triton uses; omitting it hid 1.79 of the
# 2.72 launches a source layer adds. `cudaMemcpyAsync` is not a launch and is excluded.
_LAUNCH = (
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
)


def _load(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle).get("traceEvents", [])


def _union_us(spans: list[tuple[float, float]]) -> float:
    """Wall time at least one span covers. Kernels on different streams overlap, so
    adding their durations can exceed the window they sit in."""
    total = 0.0
    end_so_far = float("-inf")
    for start, end in sorted(spans):
        if end <= end_so_far:
            continue
        total += end - max(start, end_so_far)
        end_so_far = max(end_so_far, end)
    return total


def measure(trace: Path) -> dict:
    """One arm's device busy time, host launch time and launches, in real forwards."""
    events = _load(trace)
    windows = [
        (float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0.0)))
        for e in events
        if e.get("ph") == "X" and _EXECUTE_CONTEXT.search(e.get("name", ""))
    ]
    if not windows:
        raise SystemExit(
            f"{trace}: no execute_context annotation, so nothing can be attributed. "
            f"An aggregate table here would be dominated by idle dummy work."
        )
    windows.sort()
    starts = [start for start, _ in windows]

    # Windows are half-open and may nest, so the enclosing one is not always the latest
    # whose start is <= ts: a short window opening inside a longer one would hide it.
    # `ends` running-max makes the scan back terminate.
    import bisect

    ends: list[float] = []
    running = float("-inf")
    for _, end in windows:
        running = max(running, end)
        ends.append(running)

    def _containing(ts: float) -> tuple[float, float] | None:
        index = bisect.bisect_right(starts, ts) - 1
        while index >= 0 and ends[index] >= ts:
            if windows[index][0] <= ts <= windows[index][1]:
                return windows[index]
            index -= 1
        return None

    def _clip(start: float, end: float) -> tuple[float, float] | None:
        """The part of a span that lies inside a real forward, or None.

        Selecting by start timestamp and then adding the *whole* duration let a
        collective that begins just inside a window contribute its entire tail, so
        `busy_us` could exceed `window_us` and the idle term could go negative. With
        `ncclDevKernel` residencies of hundreds of us against windows of about a
        millisecond, that is not a rounding effect on the shares this script reports.
        """
        window = _containing(start)
        if window is None:
            return None
        clipped_end = min(end, window[1])
        return (start, clipped_end) if clipped_end > start else (start, start)

    def inside(ts: float) -> bool:
        return _containing(ts) is not None

    compute: list[tuple[float, float]] = []
    nccl: list[tuple[float, float]] = []
    host: list[tuple[float, float]] = []
    sync: list[tuple[float, float]] = []
    launches = 0
    cpu_ops = 0

    for event in events:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        ts = float(event["ts"])
        if not inside(ts):
            continue
        cat = event.get("cat")
        name = event.get("name", "")
        span = _clip(ts, ts + float(event["dur"]))
        if span is None:
            continue
        if cat == "kernel":
            (nccl if _NCCL.search(name) else compute).append(span)
        elif cat in ("cuda_runtime", "cuda_driver"):
            if name == "cudaEventSynchronize":
                sync.append(span)
            else:
                host.append(span)
            if name in _LAUNCH:
                launches += 1
        elif cat == "cpu_op":
            cpu_ops += 1

    window_us = _union_us(windows)
    compute_us = _union_us(compute)
    # Time parked inside `ncclDevKernel` is not the host stalling the device, and
    # subtracting only compute from the window charges every barrier to dispatch. That
    # error read as "99.2% dispatch-bound" once, which is the answer this script exists
    # to *not* assume. True idle is what neither compute nor a collective covers.
    busy_us = _union_us(compute + nccl)
    return {
        "trace": trace.name,
        "forwards": len(windows),
        "window_us": round(window_us, 1),
        "compute_us": round(compute_us, 1),
        "nccl_us": round(_union_us(nccl), 1),
        "busy_us": round(busy_us, 1),
        "idle_us": round(window_us - busy_us, 1),
        "host_runtime_us": round(_union_us(host), 1),
        "event_sync_us": round(_union_us(sync), 1),
        "launches": launches,
        "cpu_ops": cpu_ops,
    }


def diff(
    prediction: dict,
    stock: dict,
    source_layers: int,
    predicting_forwards: int | None = None,
) -> str:
    """What prediction adds, and whether a fusion can remove it.

    Device *idle* on one rank cannot arbitrate this, which is why it is reported but not
    judged on: added host work on every rank delays every rank's arrival, and on any one
    rank that surfaces as the peers' barrier residency rather than as its own idle. The
    idle figure can therefore fall while the window grows.
    """
    # Not every annotated forward predicts: a decode or dummy forward is gated off, and
    # dividing an *added* quantity by every window understates it. The harness prints
    # the count as "N forwards dumped"; without it this uses every window and says
    # so, because a silently wrong denominator here is a wrong per-layer figure in a
    # document someone will quote.
    forwards = max(predicting_forwards or prediction["forwards"], 1)
    per = source_layers * forwards
    keys = (
        "compute_us",
        "idle_us",
        "nccl_us",
        "launches",
        "cpu_ops",
        "host_runtime_us",
        "event_sync_us",
        "window_us",
    )
    added = {k: prediction[k] - stock[k] for k in keys}

    def rate(key: str) -> float:
        return added[key] / per

    def row(label: str, key: str, unit: str, note: str = "") -> str:
        value = added[key]
        shown = f"{value:9.0f} {unit}" if unit else f"{value:9.0f}"
        return f"{label:22s}{shown}  ({rate(key):7.2f} {unit or ''}/layer){note}"

    lines = [
        f"windows: prediction {prediction['forwards']}, stock {stock['forwards']}; "
        f"per-layer rates divide by {forwards} predicting forward(s)"
        + ("" if predicting_forwards else " (assumed: pass --predicting-forwards)"),
        row("launches added:", "launches", ""),
        row("host aten ops added:", "cpu_ops", ""),
        row("host dispatch added:", "host_runtime_us", "us"),
        row("event-sync blocking added:", "event_sync_us", "us", "  <- skew, see below"),
        row("device compute added:", "compute_us", "us"),
        row("device idle added:", "idle_us", "us", "  <- not judged on, see docstring"),
        row("collective added:", "nccl_us", "us", "  <- ticket 13's half"),
    ]
    compute = rate("compute_us")
    ops = rate("cpu_ops")
    lines.append(
        f"per source layer, prediction costs {compute:.1f} us of device arithmetic and "
        f"{ops:.1f} host aten ops"
    )
    # Shares of the window prediction adds, which is the quantity a fusion or a captured
    # graph can act on. Reported rather than concluded from: the first version of this
    # verdict called the residual "host ops" and pointed at a lever worth 52%, and the
    # measured share is 8.4%.
    window = rate("window_us")
    if window > 0:
        for label, key in (
            ("host dispatch", "host_runtime_us"),
            ("device compute", "compute_us"),
            ("event-sync blocking", "event_sync_us"),
        ):
            lines.append(f"  {label:22s}{rate(key) / window * 100:6.1f}% of added window")
        accounted = sum(
            rate(k) for k in ("host_runtime_us", "compute_us", "event_sync_us")
        )
        lines.append(
            f"  {'gap / waiting':22s}{(window - accounted) / window * 100:6.1f}% of added window"
        )
    lines.append(
        "Read the per-rank table above before the means. `event-sync blocking` flips "
        "sign across ranks (+1.97 ms on dp0, -2.21 ms on dp5): it measures how early a "
        "rank arrives, not what it costs, and folding it into dispatch prices a skew."
    )
    lines.append(
        "Measured outcome, 2026-09-06: fusing this path (ticket 18) returned a median "
        "2.6% of stock TTFT, above the 0.9-1.5% its dispatch share predicts — so some "
        "of the gap does respond to launch count. Do not re-derive a 52% lever from the "
        "ticket text; that label came from eliminating the collective and calling the "
        "residual launches."
    )
    return "\n".join(lines)


def _rank_traces(directory: Path) -> list[Path]:
    """Every rank's trace, because one rank's view of a collective is its earliness.

    Reading `[0]` here priced ticket 18 at a dispatch share taken from dp0 alone, whose
    blocking sync reads +1.97 ms while dp5's reads -2.21 ms. A term whose sign depends on
    which file you opened is not a cost.
    """
    traces = sorted(directory.glob("*.pt.trace.json*"))
    if not traces:
        raise SystemExit(f"{directory}: no trace file")
    return traces


def _mean_over_ranks(directory: Path) -> tuple[dict, list[dict]]:
    per_rank = [measure(trace) for trace in _rank_traces(directory)]
    keys = [k for k, v in per_rank[0].items() if isinstance(v, int | float)]
    mean = {k: sum(arm[k] for arm in per_rank) / len(per_rank) for k in keys}
    mean["ranks"] = len(per_rank)
    return mean, per_rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--source-layers", type=int, default=44)
    parser.add_argument("--predicting-forwards", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    prediction, prediction_ranks = _mean_over_ranks(args.trace_dir)
    stock, stock_ranks = _mean_over_ranks(args.baseline_dir)

    print(f"{'rank':6s}{'window +us':>12s}{'compute +us':>13s}{'evtsync +us':>13s}")
    for left, right in zip(prediction_ranks, stock_ranks):
        print(
            f"{'':6s}{left['window_us'] - right['window_us']:12.0f}"
            f"{left['compute_us'] - right['compute_us']:13.0f}"
            f"{left['event_sync_us'] - right['event_sync_us']:13.0f}"
        )
    print()
    for label, arm in (("prediction", prediction), ("stock", stock)):
        print(
            f"{label:11s} window={arm['window_us']:10.0f} us "
            f"compute={arm['compute_us']:10.0f} idle={arm['idle_us']:10.0f} "
            f"nccl={arm['nccl_us']:10.0f} launches={arm['launches']:7.0f} "
            f"cpu_ops={arm['cpu_ops']:8.0f}"
        )
    print()
    print(diff(prediction, stock, args.source_layers, args.predicting_forwards))
    if args.out:
        args.out.write_text(
            json.dumps({"prediction": prediction, "stock": stock}, indent=2)
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
