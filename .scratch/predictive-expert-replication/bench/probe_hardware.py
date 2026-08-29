# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ticket 00 hardware probe: everything measurable without serving a model.

Produces the interconnect facts, the expert-transfer cost, the per-layer time
budget, the expert-weight read floor, and the KV-capacity ceiling on decode
concurrency. Writes JSON for `report.py`.

Run from the repository root so the repository's own `vllm` is importable:

    python3 .scratch/predictive-expert-replication/bench/probe_hardware.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from analyze import (  # noqa: E402
    QWEN3_30B_A3B,
    ModelShape,
    reachable_decode_concurrency,
)

MIB = 2**20


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - probe must not crash the run
        return f"<unavailable: {exc}>"


def probe_interconnect() -> dict:
    """Record how ranks are actually connected, which bounds everything else."""
    nvlink = _sh("nvidia-smi nvlink -s")
    has_nvlink = "does not have or support" not in nvlink and bool(nvlink)
    return {
        "has_nvlink": has_nvlink,
        "nvlink_raw": nvlink.splitlines()[:2],
        "topology_matrix": _sh("nvidia-smi topo -m").splitlines()[:10],
        "pcie": _sh(
            "nvidia-smi --query-gpu=name,pcie.link.gen.max,pcie.link.width.max "
            "--format=csv,noheader | head -1"
        ),
        "device_name": _sh(
            "nvidia-smi --query-gpu=name --format=csv,noheader | head -1"
        ),
        "device_count": torch.accelerator.device_count(),
    }


def _time_us(fn, iters: int, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - start) / iters * 1e6


def _time_p2p_us(src: int, dst: int, nbytes: int, iters: int = 50) -> float:
    """Time a cross-device copy, synchronizing BOTH devices.

    `torch.accelerator.synchronize()` only waits on the current device, so timing a
    copy between two other devices measures launch overhead instead of transfer.
    That reads as physically impossible bandwidth, which is how this was caught.
    """
    a = torch.empty(nbytes // 2, dtype=torch.bfloat16, device=f"cuda:{src}")
    b = torch.empty(nbytes // 2, dtype=torch.bfloat16, device=f"cuda:{dst}")
    try:
        for _ in range(10):
            b.copy_(a)
        torch.accelerator.synchronize(src)
        torch.accelerator.synchronize(dst)
        start = time.perf_counter()
        for _ in range(iters):
            b.copy_(a)
        torch.accelerator.synchronize(src)
        torch.accelerator.synchronize(dst)
        return (time.perf_counter() - start) / iters * 1e6
    finally:
        del a, b
        torch.accelerator.empty_cache()


# The guard exists because `torch.accelerator.synchronize()` waits only on the current
# device, so timing a copy between two *other* devices measures launch overhead and
# reads as impossible bandwidth. The ceiling must follow the fabric actually installed,
# not a constant: pinned at PCIe's 70 it rejects every correct NVLink measurement, and
# raised to NVLink's on a PCIe node it stops catching the bug it was written for.
PCIE_GEN5_X16_CEILING_GB_PER_S = 70.0
# NVLink 4 on H100 SXM is 18 links x 25 GB/s = 450 GB/s per direction, and NVSwitch
# lets one pair use all of them. Above 500 the measurement is not waiting.
NVLINK4_CEILING_GB_PER_S = 500.0


def probe_expert_transfer(
    shape: ModelShape, pairs: list[tuple[int, int]], has_nvlink: bool = False
) -> dict:
    """Point-to-point cost of moving one expert, idle.

    This is the *optimistic* figure. The usable bandwidth the cost profile needs
    is lower, because token dispatch and combine share the same fabric; see
    `probe_interconnect_under_load.py`.

    Args:
        shape: The model shape, for one expert's byte size.
        pairs: Rank pairs to time.
        has_nvlink: From `probe_interconnect`. Selects which plausibility ceiling
            applies, since the two fabrics differ by more than 6x.
    """
    nbytes = shape.bytes_per_expert
    ceiling = NVLINK4_CEILING_GB_PER_S if has_nvlink else PCIE_GEN5_X16_CEILING_GB_PER_S
    fabric = "NVLink 4" if has_nvlink else "PCIe Gen5 x16"
    results = {}
    implausible = []
    for src, dst in pairs:
        us = _time_p2p_us(src, dst, nbytes)
        gb_per_s = nbytes / us / 1e3
        results[f"{src}->{dst}"] = {"us": round(us, 1), "gb_per_s": round(gb_per_s, 2)}
        if gb_per_s > ceiling:
            implausible.append(f"{src}->{dst} at {gb_per_s:.0f} GB/s")
    if implausible:
        raise RuntimeError(
            f"Implausible P2P bandwidth, above the {fabric} ceiling of "
            f"{ceiling:.0f} GB/s: {implausible}. The measurement is not waiting for "
            f"the transfer; do not seed a cost profile from it."
        )
    idle = [r["us"] for r in results.values()]
    return {
        "bytes_per_expert": nbytes,
        "mib_per_expert": round(nbytes / MIB, 2),
        "per_pair": results,
        "idle_transfer_us_max": round(max(idle), 1),
        "idle_transfer_us_min": round(min(idle), 1),
    }


def probe_layer_budget(
    shape: ModelShape, ep_size: int, tokens_per_rank: list[int], device: int = 0
) -> dict:
    """Per-layer Attention and MoE time, and the expert-weight read floor.

    The Attention figure is the overlap window a lookahead of one would have.
    The read floor is what MoE costs even with perfectly balanced tokens, so the
    ratio of measured MoE time to it says whether balance can matter at all.
    """
    torch.accelerator.set_device_index(device)
    dtype = torch.bfloat16
    hidden = shape.hidden_size
    q_dim = 32 * shape.head_dim
    qkv_out = q_dim + 2 * shape.num_kv_heads * shape.head_dim
    local_experts = shape.num_logical_experts // ep_size

    w_qkv = torch.randn(qkv_out, hidden, dtype=dtype, device="cuda")
    w_o = torch.randn(hidden, q_dim, dtype=dtype, device="cuda")
    w13 = torch.randn(
        local_experts,
        2 * shape.moe_intermediate_size,
        hidden,
        dtype=dtype,
        device="cuda",
    )
    w2 = torch.randn(
        local_experts, hidden, shape.moe_intermediate_size, dtype=dtype, device="cuda"
    )

    local_bytes = shape.local_expert_bytes(ep_size)
    bandwidth = _measure_hbm_bandwidth_bytes_per_us(device)
    read_floor_us = local_bytes / bandwidth

    rows = []
    for tokens in tokens_per_rank:
        x = torch.randn(tokens, hidden, dtype=dtype, device="cuda")
        xo = torch.randn(tokens, q_dim, dtype=dtype, device="cuda")
        attn_us = _time_us(
            lambda x=x, xo=xo: (F.linear(x, w_qkv), F.linear(xo, w_o)), iters=100
        )
        # With allgather_reducescatter every rank sees all DP ranks' tokens and
        # runs its own local experts over them.
        pairs = max(1, tokens * ep_size * shape.experts_per_token // ep_size)
        per_expert = max(1, pairs // local_experts)
        xe = torch.randn(local_experts, per_expert, hidden, dtype=dtype, device="cuda")

        def moe(xe=xe) -> None:
            h = torch.bmm(xe, w13.transpose(1, 2))
            gate, up = h.chunk(2, dim=-1)
            torch.bmm(F.silu(gate) * up, w2.transpose(1, 2))

        moe_us = _time_us(moe, iters=50)
        rows.append(
            {
                "tokens_per_rank": tokens,
                "attention_us": round(attn_us, 1),
                "moe_us": round(moe_us, 1),
                "moe_over_read_floor": round(moe_us / read_floor_us, 2),
            }
        )
        del x, xo, xe
        torch.accelerator.empty_cache()

    # The empirical floor is MoE time at the smallest token count, where the
    # kernel is streaming expert weights and doing almost no arithmetic. Prefer
    # it over the copy-bandwidth estimate: it is the same kernel being measured.
    empirical_floor_us = min(r["moe_us"] for r in rows)
    for row in rows:
        row["moe_over_read_floor"] = round(row["moe_us"] / empirical_floor_us, 2)

    return {
        "local_expert_mib_per_layer": round(local_bytes / MIB, 1),
        "hbm_copy_bytes_per_us": round(bandwidth, 1),
        "expert_weight_read_floor_us_estimated": round(read_floor_us, 1),
        "expert_weight_read_floor_us": round(empirical_floor_us, 1),
        "read_floor_source": (
            "measured MoE time at minimum tokens per rank, which is dominated by "
            "streaming the local expert weights"
        ),
        "by_tokens_per_rank": rows,
    }


def _measure_hbm_bandwidth_bytes_per_us(device: int) -> float:
    """Measure achievable HBM read bandwidth rather than trusting a datasheet."""
    torch.accelerator.set_device_index(device)
    buf = torch.empty(512 * MIB // 2, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(buf)
    us = _time_us(lambda out=out, buf=buf: out.copy_(buf), iters=30)
    # copy_ reads and writes, so bytes moved is twice the buffer.
    bytes_moved = 2 * buf.numel() * 2
    del buf, out
    torch.accelerator.empty_cache()
    return bytes_moved / us


def probe_kv_ceiling(
    shape: ModelShape,
    ep_size: int,
    gpu_memory_utilization: float,
    weight_gib_per_rank: float,
    workspace_gib: float,
    context_lens: list[int],
) -> dict:
    """Decode concurrency KV capacity allows, which caps MoE boundness.

    More concurrency raises the MoE-time to read-floor ratio, and KV capacity
    caps concurrency, so this is a ceiling on how token-compute bound decode can
    ever become on this card.
    """
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    gib = 2**30
    kv_gib = (
        total_bytes / gib * gpu_memory_utilization - weight_gib_per_rank - workspace_gib
    )
    kv_bytes = int(max(0.0, kv_gib) * gib)
    return {
        "card_gib": round(total_bytes / gib, 2),
        "gpu_memory_utilization": gpu_memory_utilization,
        "assumed_weight_gib_per_rank": weight_gib_per_rank,
        "assumed_workspace_gib": workspace_gib,
        "kv_gib_per_rank": round(kv_gib, 2),
        "kv_bytes_per_token": shape.kv_bytes_per_token,
        "kv_tokens_per_rank": int(kv_bytes // shape.kv_bytes_per_token),
        "by_context_len": [
            {
                "context_len": ctx,
                "sequences_per_rank": reachable_decode_concurrency(
                    kv_bytes, shape.kv_bytes_per_token, ctx
                ),
            }
            for ctx in context_lens
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--weight-gib-per-rank", type=float, default=9.75)
    parser.add_argument("--workspace-gib", type=float, default=2.5)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "results" / "hardware.json",
    )
    args = parser.parse_args()

    shape = QWEN3_30B_A3B
    n = torch.accelerator.device_count()
    pairs = [(0, 1), (0, n // 2), (n - 2, n - 1), (0, n - 1)] if n > 1 else []

    interconnect = probe_interconnect()
    result = {
        "model": shape.name,
        "ep_size": args.ep_size,
        "interconnect": interconnect,
        "expert_transfer": probe_expert_transfer(
            shape, pairs, has_nvlink=interconnect["has_nvlink"]
        ),
        "layer_budget": probe_layer_budget(
            shape, args.ep_size, [1, 8, 32, 64, 128, 256, 512]
        ),
        "kv_ceiling": probe_kv_ceiling(
            shape,
            args.ep_size,
            args.gpu_memory_utilization,
            args.weight_gib_per_rank,
            args.workspace_gib,
            [1024, 2048, 3072, 4096],
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
