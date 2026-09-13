# SPDX-License-Identifier: Apache-2.0
"""Hold a small, duty-cycled GEMM load on every GPU so an idle pod is not reclaimed.

This exists for infrastructure, not for measurement, and its one hard requirement is
that it must never be part of a measurement. A benchmark on this branch is read at a
0.3% resolution and vLLM sizes its KV cache from the free memory it observes at
startup, so a process merely *holding a CUDA context* while a server boots changes
what that server measures. So this worker is killed rather than throttled when anything
else appears, and it keeps a bounded lifetime so the supervisor regains control.

Deciding *when* to yield is the supervisor's job, not this worker's. It was this
worker's, and the first version exited immediately every time: in a container
nvidia-smi reports HOST pids, so excluding `os.getpid()` excludes nothing and the
worker read its own eight contexts as an intruder. Anything comparing a pid it owns
against a pid the driver reports has the same defect.

Utilisation as nvidia-smi reports it is the fraction of sample time in which at least
one kernel was resident, so a duty cycle sets it directly: burst, then sleep for
`busy * (1/target - 1)`. The burst is re-timed every iteration rather than calibrated
once, because clocks move.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-util", type=float, default=0.10)
    parser.add_argument(
        "--seconds",
        type=float,
        default=300.0,
        help="Exit after this long so the supervisor can re-check.",
    )
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument(
        "--iters", type=int, default=25, help="GEMMs per burst; sets the burst length."
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA", file=sys.stderr)
        return 1
    devices = list(range(torch.cuda.device_count()))
    mats = {}
    for d in devices:
        with torch.cuda.device(d):
            a = torch.randn(
                args.size, args.size, device=f"cuda:{d}", dtype=torch.float16
            )
            mats[d] = (a, a.clone(), torch.empty_like(a))
    print(
        f"[keepalive] {len(devices)} GPU(s), {args.size}^2 fp16, "
        f"target {args.target_util:.0%}",
        flush=True,
    )

    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        start = time.monotonic()
        # Issued on every device before any sync, so the GPUs are busy concurrently
        # rather than one-eighth of the time each.
        for d in devices:
            with torch.cuda.device(d):
                a, b, out = mats[d]
                for _ in range(args.iters):
                    torch.matmul(a, b, out=out)
        for d in devices:
            torch.cuda.synchronize(d)
        busy = time.monotonic() - start

        target = min(max(args.target_util, 0.01), 0.95)
        time.sleep(busy * (1.0 / target - 1.0))

    for d in devices:
        mats.pop(d, None)
    mats.clear()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
