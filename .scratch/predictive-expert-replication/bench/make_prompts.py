# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build fixed-length prompts from a real dataset for the ticket 00 shapes.

`vllm bench serve` can force the decode length of a HuggingFace dataset but not
its prompt length, so the two agreed request shapes (2k prompt / 1k decode and
1k prompt / 2k decode) cannot be produced from a stock HF run. This filters real
samples to the target prompt length and writes the custom-dataset JSONL that the
benchmark's `custom` path accepts.

Real content matters here: expert routing is content dependent, so padding with
synthetic tokens would flatten the routing skew that ticket 00 exists to measure.
Samples shorter than the target are concatenated with further samples from the
same dataset rather than padded, which keeps the text real.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Fields that together form one request's prompt. InstructCoder splits a code
# task across `instruction` and `input`, and joining them is what makes the
# prompt realistic code rather than a bare sentence.
_PROMPT_FIELDS = ("prompt", "question", "instruction", "problem", "text", "input")

# Answer-side fields, never part of the prompt.
_EXCLUDED_FIELDS = ("output", "solution", "reference", "answer", "response")

_TURN_FIELDS = ("turns", "conversations", "conversation", "messages")


def extract_text(record: dict) -> str:
    """Pull the prompt-side text out of a record without knowing its schema.

    Deliberately never returns an answer field: including the reference output
    would make the prompt unrepresentative of a real request.
    """
    parts = [
        record[key]
        for key in _PROMPT_FIELDS
        if isinstance(record.get(key), str) and record[key].strip()
    ]
    if parts:
        return "\n\n".join(parts)

    for key in _TURN_FIELDS:
        turns = record.get(key)
        if not isinstance(turns, list):
            continue
        texts = []
        for turn in turns:
            if isinstance(turn, dict):
                if turn.get("role") in ("assistant", "system"):
                    continue
                text = turn.get("value") or turn.get("content")
            else:
                text = turn
            if isinstance(text, str) and text.strip():
                texts.append(text)
        if texts:
            return "\n\n".join(texts)

    # Last resort: the longest remaining string field that is not an answer.
    candidates = [
        value
        for key, value in record.items()
        if key not in _EXCLUDED_FIELDS
        and isinstance(value, str)
        and len(value.strip()) > 32
    ]
    return max(candidates, key=len) if candidates else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset path")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--target-prompt-len", type=int, required=True)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--num-prompts", type=int, default=1024)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    low = int(args.target_prompt_len * (1 - args.tolerance))
    high = int(args.target_prompt_len * (1 + args.tolerance))

    try:
        data = load_dataset(args.dataset, split=args.split, streaming=True)
    except Exception:  # noqa: BLE001 - split names vary across datasets
        data = load_dataset(args.dataset, split="test", streaming=True)

    written = 0
    buffer: list[str] = []
    with args.out.open("w") as handle:
        for record in data:
            text = extract_text(record if isinstance(record, dict) else {})
            if not text:
                continue
            buffer.append(text)
            joined = "\n\n".join(buffer)
            length = len(tokenizer(joined, add_special_tokens=False)["input_ids"])
            if length < low:
                continue  # keep accumulating real text rather than padding
            if length > high:
                # Trim to the target on a token boundary, then decode back.
                ids = tokenizer(joined, add_special_tokens=False)["input_ids"]
                joined = tokenizer.decode(ids[: args.target_prompt_len])
            handle.write(json.dumps({"prompt": joined}) + "\n")
            written += 1
            buffer = []
            if written >= args.num_prompts:
                break

    if written == 0:
        print(
            f"no prompts near {args.target_prompt_len} tokens from {args.dataset}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"wrote {written} prompts of ~{args.target_prompt_len} tokens to {args.out}")
    sys.stdout.flush()
    # Leave immediately rather than unwind. The streaming dataset keeps a
    # background download thread that can abort interpreter teardown with a GIL
    # error *after* the prompts are safely written, which makes a completed run
    # look like a failure to any caller checking the exit status.
    os._exit(0)


if __name__ == "__main__":
    main()
