# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ticket 00 analysis: turn raw measurements into the feasibility verdict.

Everything here is a pure function over measured numbers, so it is unit-tested
without a GPU. The measurement scripts feed it; `report.py` renders it.

The central quantity is *headroom*: the share of the peak EP rank's MoE time
that perfect load balancing could remove. It is an upper bound, because one
replica of one expert recovers only part of it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelShape:
    """The shape facts that determine memory and bandwidth floors."""

    name: str
    num_moe_layers: int
    num_logical_experts: int
    experts_per_token: int
    hidden_size: int
    moe_intermediate_size: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2

    @property
    def bytes_per_expert(self) -> int:
        """One expert's w13 (gate+up, fused) plus w2 (down)."""
        w13 = 2 * self.moe_intermediate_size * self.hidden_size
        w2 = self.hidden_size * self.moe_intermediate_size
        return (w13 + w2) * self.dtype_bytes

    @property
    def kv_bytes_per_token(self) -> int:
        return (
            self.num_moe_layers
            * self.num_kv_heads
            * self.head_dim
            * 2  # K and V
            * self.dtype_bytes
        )

    def local_expert_bytes(self, ep_size: int) -> int:
        """Expert weights one rank must read per MoE layer."""
        return (self.num_logical_experts // ep_size) * self.bytes_per_expert


QWEN3_30B_A3B = ModelShape(
    name="Qwen/Qwen3-30B-A3B",
    num_moe_layers=48,
    num_logical_experts=128,
    experts_per_token=8,
    hidden_size=2048,
    moe_intermediate_size=768,
    num_kv_heads=4,
    head_dim=128,
)


def expert_counts_to_rank_load(
    counts_per_logical_expert: Sequence[int], ep_size: int
) -> list[int]:
    """Sum per-expert token counts into per-EP-rank load.

    Uses the fixed canonical layout: logical expert `e` lives on rank
    `e // (num_logical // ep_size)`. This is what makes a single hot expert an
    entire rank's problem.

    Args:
        counts_per_logical_expert: Routed token count per logical expert.
        ep_size: EP group size.

    Returns:
        Token count per EP rank.

    Raises:
        ValueError: If the experts do not divide evenly across ranks.
    """
    num_experts = len(counts_per_logical_expert)
    if num_experts % ep_size != 0:
        raise ValueError(
            f"{num_experts} logical experts do not divide EP size {ep_size}."
        )
    per_rank = num_experts // ep_size
    return [
        int(sum(counts_per_logical_expert[r * per_rank : (r + 1) * per_rank]))
        for r in range(ep_size)
    ]


def headroom_fraction(rank_load: Sequence[float]) -> float:
    """Share of the peak rank's load that perfect balancing would remove.

    This is the upper bound on any placement policy's gain at this operating
    point, before transfer cost. Returns 0.0 for an idle or balanced model.
    """
    peak = max(rank_load)
    if peak <= 0:
        return 0.0
    mean = sum(rank_load) / len(rank_load)
    return (peak - mean) / peak


def moe_boundness_ratio(measured_us: float, weight_read_floor_us: float) -> float:
    """Measured MoE time over the time to merely read the local expert weights.

    A ratio near 1 means MoE is expert-weight-bandwidth bound, so load imbalance
    cannot convert into time and no placement policy can help. Reporting it is
    what separates "the policy does nothing" from "this operating point cannot
    reward any policy".
    """
    if weight_read_floor_us <= 0:
        raise ValueError("weight_read_floor_us must be positive.")
    return measured_us / weight_read_floor_us


def min_residency_steps(exposed_us: float, gain_per_step_us: float) -> float:
    """Steps a replica must survive for its exposed transfer to pay for itself.

    Returns:
        The step count, or infinity when the replica gains nothing.
    """
    if gain_per_step_us <= 0:
        return math.inf
    return math.ceil(exposed_us / gain_per_step_us)


def reachable_decode_concurrency(
    kv_bytes_available: int, kv_bytes_per_token: int, context_len: int
) -> int:
    """Concurrent sequences per rank that KV capacity allows.

    This is what caps how token-compute bound decode can become: more
    concurrency raises the MoE boundness ratio, and KV capacity caps concurrency.
    """
    if kv_bytes_per_token <= 0 or context_len <= 0:
        raise ValueError("kv_bytes_per_token and context_len must be positive.")
    return int(kv_bytes_available // (kv_bytes_per_token * context_len))


@dataclass(frozen=True)
class Verdict:
    """The ticket 00 gate: whether the remaining implementation is worth doing."""

    proceed: bool
    reason: str
    headroom_fraction: float
    moe_ratio: float
    recoverable_us_per_step: float
    min_residency_steps: float


# A policy that replicates one expert recovers only part of the imbalance: it
# moves at most one expert's excess off the peak rank, and it loads the target.
# Half the theoretical headroom is a deliberately generous stand-in until the
# planner exists; ticket 04 replaces it with the real per-plan estimate.
SINGLE_REPLICA_CAPTURE = 0.5

# Below this ratio, MoE time is dominated by reading expert weights, so
# rebalancing tokens cannot change the layer's duration.
WEIGHT_BOUND_RATIO = 1.2

# Below this, there is not enough imbalance to be worth any transfer.
MIN_USEFUL_HEADROOM = 0.05


def recommend(
    headroom_fraction: float,
    moe_ratio: float,
    exposed_transfer_us: float,
    peak_moe_us: float,
    single_replica_capture: float = SINGLE_REPLICA_CAPTURE,
) -> Verdict:
    """Decide whether measured headroom justifies continuing.

    Args:
        headroom_fraction: Output of `headroom_fraction`.
        moe_ratio: Output of `moe_boundness_ratio`.
        exposed_transfer_us: Transfer time not hidden by the overlap window.
        peak_moe_us: Measured per-layer MoE time on the peak rank.
        single_replica_capture: Share of headroom one replica can realistically
            take.

    Returns:
        A `Verdict` carrying the numbers the decision rests on.
    """
    recoverable = peak_moe_us * headroom_fraction * single_replica_capture
    residency = min_residency_steps(exposed_transfer_us, recoverable)

    if moe_ratio < WEIGHT_BOUND_RATIO:
        return Verdict(
            proceed=False,
            reason=(
                f"MoE is expert-weight-bandwidth bound (ratio {moe_ratio:.2f} < "
                f"{WEIGHT_BOUND_RATIO}); load balance cannot convert into time "
                f"at this operating point, so a null result here says nothing "
                f"about the policy."
            ),
            headroom_fraction=headroom_fraction,
            moe_ratio=moe_ratio,
            recoverable_us_per_step=recoverable,
            min_residency_steps=residency,
        )
    if headroom_fraction < MIN_USEFUL_HEADROOM:
        return Verdict(
            proceed=False,
            reason=(
                f"Measured headroom {headroom_fraction:.1%} is below "
                f"{MIN_USEFUL_HEADROOM:.0%}; there is not enough rank imbalance "
                f"to recover for any transfer cost to be worth paying."
            ),
            headroom_fraction=headroom_fraction,
            moe_ratio=moe_ratio,
            recoverable_us_per_step=recoverable,
            min_residency_steps=residency,
        )
    return Verdict(
        proceed=True,
        reason=(
            f"Headroom {headroom_fraction:.1%} at MoE ratio {moe_ratio:.2f} "
            f"yields about {recoverable:.1f} us per layer per step, so a "
            f"{exposed_transfer_us:.0f} us exposed transfer amortizes in "
            f"{residency:.0f} steps."
        ),
        headroom_fraction=headroom_fraction,
        moe_ratio=moe_ratio,
        recoverable_us_per_step=recoverable,
        min_residency_steps=residency,
    )


def build_cost_profile(
    *,
    model: str,
    dtype: str,
    ep_size: int,
    num_logical_experts: int,
    device_name: str,
    expert_compute_us_per_token: float,
    attention_window_us: float,
    transfer_latency_us: float,
    usable_transfer_bandwidth_bytes_per_us: float,
) -> dict:
    """Emit a cost profile in the schema the feature's validator enforces.

    `usable_transfer_bandwidth_bytes_per_us` must come from a measurement taken
    while token dispatch and combine are running. An idle figure would
    systematically overstate the policy's benefit.
    """
    return {
        "fingerprint": {
            "model": model,
            "dtype": dtype,
            "ep_size": ep_size,
            "num_logical_experts": num_logical_experts,
            "device_name": device_name,
        },
        "expert_compute_us_per_token": expert_compute_us_per_token,
        "attention_window_us": attention_window_us,
        "transfer_latency_us": transfer_latency_us,
        "usable_transfer_bandwidth_bytes_per_us": (
            usable_transfer_bandwidth_bytes_per_us
        ),
    }
