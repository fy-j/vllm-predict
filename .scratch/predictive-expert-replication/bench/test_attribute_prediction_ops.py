# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ticket 18's stop gate has to be trustworthy: it decides whether a kernel is written.
Each of these pins a way this script has already got a profile wrong."""

import json

import pytest
from attribute_prediction_ops import diff, measure


def _trace(tmp_path, events, name="dp0_run.pt.trace.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"traceEvents": events}))
    return path


def _window(ts, dur):
    return {
        "ph": "X",
        "cat": "user_annotation",
        "name": "execute_context",
        "ts": ts,
        "dur": dur,
    }


def _kernel(ts, dur, name="void gemm"):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur}


def _runtime(ts, dur, name="cudaLaunchKernel", cat="cuda_runtime"):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur}


def test_work_outside_a_real_forward_is_not_attributed(tmp_path):
    """The trap this project fell into: a 12.95 s trace whose real forwards covered
    0.54 s, its operator table dominated by idle `execute_dummy_batch` work."""
    arm = measure(
        _trace(
            tmp_path, [_window(1000, 100), _kernel(1010, 10), _kernel(50_000, 9_000)]
        )
    )
    assert arm["compute_us"] == 10.0


def test_a_trace_without_annotations_refuses_rather_than_reporting(tmp_path):
    """Silence here would produce a confident number from the wrong denominator."""
    with pytest.raises(SystemExit, match="execute_context"):
        measure(_trace(tmp_path, [_kernel(10, 5)]))


def test_overlapping_kernels_are_unioned_not_summed(tmp_path):
    """Kernels on different streams overlap, so adding durations can exceed the window
    that holds them and report an occupancy above 100%."""
    arm = measure(_trace(tmp_path, [_window(0, 100), _kernel(0, 50), _kernel(10, 50)]))
    assert arm["compute_us"] == 60.0


def test_time_parked_in_a_collective_is_not_counted_as_idle(tmp_path):
    """Subtracting only compute from the window charges every barrier to dispatch, which
    read as '99.2% dispatch-bound' once, the answer this script exists to not assume."""
    arm = measure(
        _trace(
            tmp_path,
            [
                _window(0, 100),
                _kernel(0, 10),
                _kernel(20, 70, "ncclDevKernel_AllGather"),
            ],
        )
    )
    assert arm["nccl_us"] == 70.0
    assert arm["compute_us"] == 10.0
    assert arm["idle_us"] == 20.0


def test_nested_host_ops_are_counted_and_never_summed(tmp_path):
    """`aten::linear` holds `aten::matmul` holds `aten::mm`; summing triple-counts."""
    ops = [
        {"ph": "X", "cat": "cpu_op", "name": name, "ts": 0, "dur": 30}
        for name in ("aten::linear", "aten::matmul", "aten::mm")
    ]
    arm = measure(_trace(tmp_path, [_window(0, 100), *ops]))
    assert arm["cpu_ops"] == 3


def test_only_launch_api_calls_count_as_launches(tmp_path):
    """`cudaStreamIsCapturing` outnumbers real launches, and a memcpy is not a launch."""
    arm = measure(
        _trace(
            tmp_path,
            [
                _window(0, 100),
                _runtime(0, 5),
                _runtime(10, 5, "cudaStreamIsCapturing"),
                _runtime(20, 5, "cudaMemcpyAsync"),
            ],
        )
    )
    assert arm["launches"] == 1
    assert arm["host_runtime_us"] == 15.0


def test_triton_launches_through_the_driver_api_are_counted(tmp_path):
    """The defect this script shipped with: `cuLaunchKernelEx` is `cuda_driver`.

    Counting only `cuda_runtime` reported 0.94 added launches per source layer where the
    real figure is 2.72, because every Triton kernel launches through the driver API --
    including the counting kernel ticket 03 added. It also undercounted host time, which
    is what the dispatch share is computed from.
    """
    arm = measure(
        _trace(
            tmp_path,
            [
                _window(0, 100),
                _runtime(0, 5),
                _runtime(10, 7, "cuLaunchKernelEx", cat="cuda_driver"),
            ],
        )
    )
    assert arm["launches"] == 2
    assert arm["host_runtime_us"] == 12.0


def test_blocking_event_sync_is_reported_apart_from_dispatch(tmp_path):
    """Its call count is unchanged between the arms while its duration is not, and its
    sign flips per rank. Folding it into dispatch would price a skew as a launch cost."""
    arm = measure(
        _trace(
            tmp_path,
            [
                _window(0, 10_000),
                _runtime(0, 5),
                _runtime(10, 4_000, "cudaEventSynchronize"),
            ],
        )
    )
    assert arm["event_sync_us"] == 4_000.0
    assert arm["host_runtime_us"] == 5.0


def _arm(**overrides):
    base = {
        "forwards": 1,
        "window_us": 0.0,
        "compute_us": 0.0,
        "nccl_us": 0.0,
        "busy_us": 0.0,
        "idle_us": 0.0,
        "host_runtime_us": 0.0,
        "event_sync_us": 0.0,
        "launches": 0,
        "cpu_ops": 0,
    }
    return {**base, **overrides}


def test_the_report_gives_shares_rather_than_a_verdict():
    """The first version concluded "collapsing those ops into one kernel is the lever"
    and pointed at 52% of prediction's cost. The measured dispatch share is 8.4%, and
    ticket 18 returned a median 2.6% of TTFT. A tool that names a lever gets quoted; one
    that prints shares makes the reader do the arithmetic against its own window."""
    report = diff(
        _arm(window_us=10_000.0, compute_us=1_000.0, host_runtime_us=2_000.0),
        _arm(),
        source_layers=44,
    )
    assert "host dispatch" in report and "20.0% of added window" in report
    assert "device compute" in report and "10.0% of added window" in report
    assert "gap / waiting" in report and "70.0% of added window" in report


def test_the_report_warns_that_event_sync_is_a_skew():
    """A reader who folds it into dispatch prices dp0's earliness as a cost."""
    report = diff(_arm(window_us=1_000.0, event_sync_us=100.0), _arm(), source_layers=44)
    assert "flips sign across ranks" in report


def test_a_span_running_past_its_window_is_clipped(tmp_path):
    """Selecting by start and adding the whole duration let a collective that begins
    just inside a window contribute its entire tail, so busy could exceed the window and
    idle could go negative. `ncclDevKernel` residencies are hundreds of us against
    windows of about a millisecond, so this is not a rounding effect."""
    arm = measure(
        _trace(tmp_path, [_window(0, 100), _kernel(90, 1_000, "ncclDevKernel_x")])
    )

    assert arm["nccl_us"] == 10.0
    assert arm["idle_us"] >= 0.0


def test_overlapping_event_syncs_are_unioned_not_summed(tmp_path):
    """The file's own rule: host spans nest, so they are unioned. Subtracting a summed
    sync from a unioned host time under-reports dispatch and can make it negative -- and
    dispatch is the numerator of the share this script exists to report."""
    arm = measure(
        _trace(
            tmp_path,
            [
                _window(0, 1_000),
                _runtime(10, 500, "cudaEventSynchronize"),
                _runtime(20, 400, "cudaEventSynchronize"),
                _runtime(600, 5),
            ],
        )
    )

    assert arm["event_sync_us"] == 500.0
    assert arm["host_runtime_us"] == 5.0
