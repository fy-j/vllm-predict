# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How much of a real forward window the GPU is actually busy for.

Ticket 06's remaining criterion. The cost this feature was found to have is the host
failing to feed the GPU — occupancy fell from 86.8% to 52.9% once placement was on, with
30.2 ms of gaps over 0.5 ms against a baseline's 3.0 ms — so the measurement that says
whether removing the host synchronisation worked is occupancy inside real forward
windows, per arm.

**Busy time is the union of the kernel intervals, never their sum.** The compute stream,
the token-collective stream and the prediction stream overlap by design, so a sum
exceeds the window it sits in. That is not hypothetical: dividing the expert GEMM by
summed attributed time reported it as 10.76% of a prefill step where wall-clock puts it
at 14.29%, and the same error once read an 8-rank-summed kernel total as a per-rank one
and concluded a window was 10% occupied when seven of eight ranks were at 84%. Every
figure here therefore has a wall-clock denominator.

Windows come from the runner's own `execute_context` annotation, and only the asked
phase's are kept. A trace's aggregate operator table is dominated by idle
`execute_dummy_batch` work — in one 12.95 s capture the real forwards covered 0.54 s —
so an unrestricted total measures the wrong thing entirely.

The window extraction and the kernel classes are `parse_profile`'s, reused rather than
reimplemented: two parsers disagreeing about what a prefill window is would be a defect
no test here could see.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import gzip
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_profile import (  # noqa: E402
    KERNEL_CLASSES,
    classify_kernel,
    decode_windows,
)

# Gaps at least this long are counted as the engine failing to keep the GPU fed. 0.5 ms
# is the threshold the recorded 3.0 ms / 30.2 ms comparison used, kept so the numbers
# are comparable against it.
BUSY_GAP_US = 500.0


def union_busy_us(intervals: list[tuple[float, float]]) -> float:
    """Total time covered by at least one interval.

    Args:
        intervals: `(start, end)` pairs in microseconds, in any order.

    Returns:
        The measure of the union, so concurrent work on several streams counts once.
    """
    if not intervals:
        return 0.0
    busy = 0.0
    current_start, current_end = None, None
    for start, end in sorted(intervals):
        if current_end is None:
            current_start, current_end = start, end
            continue
        if start > current_end:
            busy += current_end - current_start  # type: ignore[operator]
            current_start, current_end = start, end
        elif end > current_end:
            current_end = end
    if current_end is not None:
        busy += current_end - current_start  # type: ignore[operator]
    return busy


def gaps_over(
    intervals: list[tuple[float, float]],
    start: float,
    end: float,
    threshold: float = BUSY_GAP_US,
) -> list[float]:
    """Idle stretches inside `[start, end]` of at least `threshold`.

    A gap before the first kernel counts: the host arriving late is the cost being
    measured, not an artifact to trim.

    Args:
        intervals: Busy `(start, end)` pairs, in any order.
        start: Window start.
        end: Window end.
        threshold: Shortest gap worth reporting.

    Returns:
        Gap lengths in microseconds, in time order.
    """
    gaps = []
    cursor = start
    for busy_start, busy_end in sorted(intervals):
        if busy_start > cursor:
            gaps.append(busy_start - cursor)
        cursor = max(cursor, busy_end)
        if cursor >= end:
            break
    if cursor < end:
        gaps.append(end - cursor)
    return [gap for gap in gaps if gap >= threshold]


@dataclass
class WindowOccupancy:
    """One forward window, as wall-clock against work actually running in it."""

    start: float
    end: float
    busy_us: float
    gap_us_over_threshold: float
    largest_gap_us: float
    busy_ex_nccl_us: float = 0.0
    ctx_tokens: int = 0
    by_class: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(KERNEL_CLASSES, 0.0)
    )

    @property
    def wall_us(self) -> float:
        return self.end - self.start

    @property
    def occupancy(self) -> float:
        """Busy share of the window. Cannot exceed 1, because `busy_us` is a union."""
        return self.busy_us / self.wall_us if self.wall_us > 0 else 0.0

    @property
    def occupancy_ex_nccl(self) -> float:
        """Busy share counting only kernels that are not token collectives.

        Raw occupancy cannot answer "is the host feeding the GPU", because a rank
        blocked inside `ncclDevKernel` waiting for a peer is busy by kernel-time
        accounting and idle in fact. Measured at DP=8, per-rank occupancy tracks NCCL
        residency almost exactly: the ranks that scored 21-29% were the ones with 11-12
        ms of collective time, and the ones at 88% had 58 ms. So the rank that arrives
        first and waits longest inside the collective looks like the busiest one.

        This is the same lesson as "a collective's duration is mostly a measurement of
        what the other rank was doing", applied to occupancy rather than to a barrier.
        """
        return self.busy_ex_nccl_us / self.wall_us if self.wall_us > 0 else 0.0

    @property
    def expert_gemm_share(self) -> float:
        """Expert GEMM against **wall-clock**, which is what sets the ceiling.

        Perfect expert balance can only shorten this term, so the ceiling is this share
        times the fraction of MoE time balance can recover. Against summed attributed
        time the same quantity reads about 3 points lower and understates the ceiling.
        """
        return self.by_class["moe_expert"] / self.wall_us if self.wall_us > 0 else 0.0

    @property
    def expert_gemm_us_per_1k_ctx(self) -> float:
        """Expert GEMM per 1000 prefill tokens, which is comparable across arms.

        Two arms need not split the same requests into the same number of prefill steps
        — measured, the placed arm took 18 to 22 windows per rank where the others took
        11 to 13 — so a per-window millisecond figure compares different amounts of
        work. Per prefill token it is the same work by construction, and expert-GEMM
        time is very nearly linear in tokens above the block-quantization bar.
        """
        if self.ctx_tokens <= 0:
            return 0.0
        return self.by_class["moe_expert"] / (self.ctx_tokens / 1000.0)


def window_occupancies(
    events: list[dict],
    phase: str = "prefill",
    gap_threshold_us: float = BUSY_GAP_US,
) -> list[WindowOccupancy]:
    """Occupancy of every window of one phase in one rank's trace.

    Args:
        events: The trace's events.
        phase: `"prefill"` or `"decode"`. Prefill is what this feature acts on.
        gap_threshold_us: Shortest idle stretch counted as a gap.

    Returns:
        One row per window, in time order.

    Raises:
        ValueError: If the trace holds no window of that phase. A zero occupancy would
            read as a finding rather than as a capture that caught nothing.
    """
    windows = decode_windows(events, phase)
    if not windows:
        raise ValueError(
            f"no {phase} window in this trace: the capture caught none of that phase, "
            f"so there is nothing to attribute. Widen the profile window or check that "
            f"the arm served requests."
        )
    per_window: list[list[tuple[float, float]]] = [[] for _ in windows]
    # Kept separately rather than subtracted afterwards: compute that overlaps a
    # collective would be counted twice by `busy - nccl`, and it overlaps by design.
    ex_nccl: list[list[tuple[float, float]]] = [[] for _ in windows]
    classes: list[dict[str, float]] = [
        dict.fromkeys(KERNEL_CLASSES, 0.0) for _ in windows
    ]
    starts = [w.start for w in windows]
    for event in events:
        if event.get("cat") != "kernel":
            continue
        duration = event.get("dur")
        if duration is None:
            continue
        ts = float(event["ts"])
        # Same rule as `attribute_windows`: a kernel belongs to the window it starts in.
        index = _window_of(starts, ts, windows)
        if index is None:
            continue
        # Clipped at the window end, so a kernel overrunning the window cannot report
        # more busy time than the window has.
        finish = min(ts + float(duration), windows[index].end)
        per_window[index].append((ts, finish))
        kernel_class = classify_kernel(event.get("name", ""))
        classes[index][kernel_class] += finish - ts
        if kernel_class != "nccl":
            ex_nccl[index].append((ts, finish))
    rows = []
    for window, intervals, work, by_class in zip(windows, per_window, ex_nccl, classes):
        gaps = gaps_over(intervals, window.start, window.end, gap_threshold_us)
        rows.append(
            WindowOccupancy(
                start=window.start,
                end=window.end,
                busy_us=union_busy_us(intervals),
                gap_us_over_threshold=sum(gaps),
                largest_gap_us=max(gaps) if gaps else 0.0,
                busy_ex_nccl_us=union_busy_us(work),
                ctx_tokens=window.ctx_tokens,
                by_class=by_class,
            )
        )
    return rows


def _window_of(starts: list[float], ts: float, windows: list) -> int | None:
    """Index of the window a timestamp starts inside, or None."""
    index = bisect.bisect_right(starts, ts) - 1
    if index < 0 or ts >= windows[index].end:
        return None
    return index


def summarize_windows(rows: list[WindowOccupancy]) -> dict:
    """Reduce one rank's windows to medians.

    Medians rather than means: the first window of a capture carries JIT compilation and
    a one-off `torch.stack` kernel load that blocks the host for 50-100 ms, and a mean
    over a handful of windows is dominated by it.

    Raises:
        ValueError: On an empty row list, for the same reason `window_occupancies` does.
    """
    if not rows:
        raise ValueError("no window to summarize; the capture measured nothing")
    return {
        "windows": len(rows),
        "wall_ms": round(statistics.median(r.wall_us for r in rows) / 1000.0, 4),
        "busy_ms": round(statistics.median(r.busy_us for r in rows) / 1000.0, 4),
        "occupancy": round(statistics.median(r.occupancy for r in rows), 4),
        # The one to read for "is the host feeding the GPU"; see its docstring.
        "occupancy_ex_nccl": round(
            statistics.median(r.occupancy_ex_nccl for r in rows), 4
        ),
        "gap_ms_over_threshold": round(
            statistics.median(r.gap_us_over_threshold for r in rows) / 1000.0, 4
        ),
        "largest_gap_ms": round(
            statistics.median(r.largest_gap_us for r in rows) / 1000.0, 4
        ),
        "expert_gemm_ms": round(
            statistics.median(r.by_class["moe_expert"] for r in rows) / 1000.0, 4
        ),
        "expert_gemm_share_of_window": round(
            statistics.median(r.expert_gemm_share for r in rows), 4
        ),
        # The shape-invariant one. Compare arms on this, not on `expert_gemm_ms`.
        "ctx_tokens": round(statistics.median(r.ctx_tokens for r in rows), 1),
        "expert_gemm_us_per_1k_ctx": round(
            statistics.median(r.expert_gemm_us_per_1k_ctx for r in rows), 2
        ),
        "nccl_ms": round(
            statistics.median(r.by_class["nccl"] for r in rows) / 1000.0, 4
        ),
    }


def parse_trace_occupancy(path: Path, phase: str = "prefill") -> dict:
    """Summarize one rank's trace file."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as handle:  # type: ignore[operator]
        events = json.load(handle).get("traceEvents", [])
    result = summarize_windows(window_occupancies(events, phase))
    result["rank"] = path.name.split("_")[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    paths = sorted(
        Path(p) for p in glob.glob(str(args.trace_dir / "dp*.pt.trace.json*"))
    )
    if not paths:
        print(f"no rank traces under {args.trace_dir}", file=sys.stderr)
        raise SystemExit(1)

    per_rank, failed = [], []
    for path in paths:
        try:
            per_rank.append(parse_trace_occupancy(path, args.phase))
        except ValueError as exc:
            failed.append({"rank": path.name.split("_")[0], "reason": str(exc)})
    if not per_rank:
        print(json.dumps({"failed": failed}, indent=2))
        raise SystemExit(1)

    # Medians across ranks, and the range with them: the step is set by the slowest
    # rank,
    # so a mean occupancy hides the one rank that is waiting.
    def across(key: str) -> dict:
        values = [r[key] for r in per_rank]
        return {
            "median": round(statistics.median(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    report = {
        "phase": args.phase,
        "ranks_parsed": len(per_rank),
        "ranks_failed": failed,
        "per_rank": per_rank,
        "across_ranks": {
            key: across(key)
            for key in (
                "wall_ms",
                "busy_ms",
                "occupancy",
                "occupancy_ex_nccl",
                "gap_ms_over_threshold",
                "expert_gemm_ms",
                "expert_gemm_share_of_window",
                "ctx_tokens",
                "expert_gemm_us_per_1k_ctx",
            )
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
