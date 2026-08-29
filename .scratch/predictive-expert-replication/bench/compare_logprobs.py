# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare two runs' first-token logprobs within a tolerance.

Ticket 03 asks whether source-rank routing preserves the model's output. Diffing
many greedy tokens answers a harsher question than that: greedy decoding is
chaotic, so one bit of difference in a reduction flips a token and every later
token diverges. Two runs of the *same* configuration can fail such a test.

So compare the distribution instead, at the first generated token, where nothing
has had a chance to compound. Routing a logical expert to a second copy of its own
weights must leave the logprobs equal to within BF16 execution tolerance; a real
semantic break moves them far more than that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict[str, float]]:
    """Read one file of per-prompt top-k logprob maps."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Maximum absolute logprob difference. BF16 has about 3 decimal "
        "digits of mantissa, and these are sums over 48 layers, so a few "
        "hundredths is execution noise while a semantic break is far larger.",
    )
    args = parser.parse_args()

    base, cand = load(args.baseline), load(args.candidate)
    if len(base) != len(cand):
        raise SystemExit(f"prompt count differs: {len(base)} vs {len(cand)}")

    worst = 0.0
    worst_where = ""
    top_token_changed = []
    for index, (b, c) in enumerate(zip(base, cand)):
        if not b or not c:
            raise SystemExit(f"prompt {index}: empty logprobs, nothing to compare")
        b_top = max(b, key=lambda k: b[k])
        c_top = max(c, key=lambda k: c[k])
        if b_top != c_top:
            top_token_changed.append((index, b_top, c_top))
        for token in set(b) & set(c):
            delta = abs(b[token] - c[token])
            if delta > worst:
                worst, worst_where = delta, f"prompt {index}, token {token!r}"

    print(f"prompts compared        {len(base)}")
    print(f"largest logprob delta   {worst:.6f}  ({worst_where})")
    print(f"tolerance               {args.tolerance}")
    if top_token_changed:
        print("top-1 token changed:")
        for index, b_top, c_top in top_token_changed:
            print(f"  prompt {index}: {b_top!r} -> {c_top!r}")
    if worst <= args.tolerance and not top_token_changed:
        print("PASS: distributions agree within tolerance and the argmax is unchanged")
    else:
        raise SystemExit(
            "FAIL: routing moved the output distribution beyond execution noise"
        )


if __name__ == "__main__":
    main()
