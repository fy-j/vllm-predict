# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic replica planner for Predictive expert replication.

Turns a per-logical-expert load snapshot into a set of replica placements. The
plan drives point-to-point sends and receives, so **every rank must derive a
bit-identical plan from the same input**: a plan that differs by rank pairs a
sender with no receiver and hangs the engine. Determinism is therefore a
correctness property, not a nicety — every tie breaks on expert then rank id, and
nothing consults rank-local state.

Two measured facts shape the policy (`bench/RESULTS.md`, 2026-08-25):

* The transfer budget is global. Spending it uniformly per layer is behind
  ranking candidates across layers at the same budget, because the per-layer
  excess varies several-fold and a uniform allowance wastes it on layers that were
  not skewed.
* A replica moves **half** an expert's load, because source-rank routing splits
  the source ranks across the two copies. It does not move all of it, and the
  target gains what the source sheds, so reducing the peak is a makespan problem
  rather than a subtraction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.distributed.eplb.eplb_communicator import EplbCommunicator


@dataclass(frozen=True, order=True)
class Placement:
    """One replica to install: everything a transfer and an activation need.

    Attributes:
        layer: Sparse MoE layer index.
        logical_expert: The expert to replicate.
        source_rank: Its canonical owner, which sends.
        target_rank: The rank whose inactive slot receives.
        moved_load: Load expected to leave the source, in the snapshot's units.
            Half the expert's load, by the source-rank split.
    """

    layer: int
    logical_expert: int
    source_rank: int
    target_rank: int
    moved_load: float


# One extra copy halves the source's share, since the source ranks divide evenly
# between the two. The glossary's `K/(K+1)` at K=1.
_SHED_FRACTION = 0.5


def plan_replicas(
    per_expert_load: torch.Tensor,
    ep_size: int,
    budget: int,
    min_tokens: float = 0.0,
) -> list[Placement]:
    """Choose up to `budget` replica placements across all layers.

    Args:
        per_expert_load: `[num_layers, num_logical_experts]` global load, already
            reduced across the EP group so every rank sees the same values.
        ep_size: EP group size. Expert `e` belongs to rank
            `e // (num_logical // ep_size)`.
        budget: Total placements allowed, across all layers. This is
            `max_transfers_per_forward`.
        min_tokens: Refuse a placement that would move less than this. Set it to
            the MoE kernel's `BLOCK_SIZE_M`: below one block a replica saves no
            block, so it saves no time.

    Returns:
        Placements in a deterministic order, at most `budget` of them. Empty when
        nothing helps, which includes a dummy forward whose load is all zero.
    """
    if budget <= 0 or per_expert_load.numel() == 0:
        return []
    num_layers, num_logical = per_expert_load.shape
    if num_logical % ep_size != 0:
        raise ValueError(
            f"{num_logical} logical experts do not divide across {ep_size} EP ranks."
        )
    per_rank_experts = num_logical // ep_size

    # One host transfer, not one per decision: the planner is a per-forward
    # diagnostic-weight path and must not synchronize inside its own loop.
    load = per_expert_load.detach().to("cpu", torch.float64).tolist()

    state = [_LayerState(row, ep_size, per_rank_experts) for row in load]
    chosen: list[Placement] = []
    while len(chosen) < budget:
        best: tuple[float, Placement] | None = None
        for layer_index, layer in enumerate(state):
            candidate = layer.best_move(layer_index, min_tokens)
            if candidate is None:
                continue
            gain, placement = candidate
            # Rank across layers by predicted peak reduction — the only quantity
            # comparable between them. On a tie the *lower* placement wins, so the
            # order is the earlier layer, then the lower expert and rank id, and it
            # cannot depend on iteration order. `-gain` puts the strongest first
            # under the same ascending comparison that orders the tie-break.
            if best is None or (-gain, placement) < (-best[0], best[1]):
                best = (gain, placement)
        if best is None:
            break
        state[best[1].layer].apply(best[1])
        chosen.append(best[1])
    return chosen


class _LayerState:
    """Mutable per-rank load for one layer while the plan is being built."""

    def __init__(self, expert_load: list[float], ep_size: int, per_rank: int):
        self.ep_size = ep_size
        self.per_rank = per_rank
        self.expert_load = list(expert_load)
        self.rank_load = [
            sum(expert_load[r * per_rank : (r + 1) * per_rank]) for r in range(ep_size)
        ]
        self.replicated: set[int] = set()
        self.slots_used: set[int] = set()

    @property
    def live(self) -> bool:
        return sum(self.rank_load) > 0

    def best_move(
        self, layer_index: int, min_tokens: float
    ) -> tuple[float, Placement] | None:
        """The placement on this layer that lowers its peak most, if any does."""
        if not self.live:
            return None
        peak = max(range(self.ep_size), key=lambda r: self.rank_load[r])
        old_peak = self.rank_load[peak]
        best: tuple[float, Placement] | None = None
        for offset in range(self.per_rank):
            expert = peak * self.per_rank + offset
            if expert in self.replicated:
                continue
            value = self.expert_load[expert]
            moved = value * _SHED_FRACTION
            if moved < min_tokens or moved <= 0:
                continue
            for target in range(self.ep_size):
                if target == peak or target in self.slots_used:
                    continue
                trial = list(self.rank_load)
                trial[peak] -= moved
                trial[target] += moved
                gain = old_peak - max(trial)
                if gain <= 0:
                    continue  # relocating the peak is not lowering it
                placement = Placement(
                    layer=layer_index,
                    logical_expert=expert,
                    source_rank=peak,
                    target_rank=target,
                    moved_load=moved,
                )
                # Same ordering as the cross-layer choice: strongest gain, then the
                # lowest expert and target id.
                if best is None or (-gain, placement) < (-best[0], best[1]):
                    best = (gain, placement)
        return best

    def apply(self, placement: Placement) -> None:
        self.rank_load[placement.source_rank] -= placement.moved_load
        self.rank_load[placement.target_rank] += placement.moved_load
        self.replicated.add(placement.logical_expert)
        self.slots_used.add(placement.target_rank)


def local_row_of(logical_expert: int, per_rank_experts: int) -> int:
    """Physical row a canonical expert occupies within its owner rank.

    Physical slots are laid out rank-major with the canonical rows first, so an
    expert's row is its offset inside its owner's block.
    """
    return logical_expert % per_rank_experts


def replica_row_of(per_rank_experts: int, slot: int = 0) -> int:
    """Physical row of a rank's `slot`-th inactive replica row.

    The inactive rows trail the canonical ones, which is what keeps canonical
    ownership fixed while leaving each rank somewhere to receive a replica.
    """
    return per_rank_experts + slot


def canonical_row_of(
    logical_expert: int, per_rank_experts: int, replica_slots_per_rank: int
) -> int:
    """Physical row a logical expert's canonical copy occupies, across all ranks.

    Rank-major with the canonical rows first, so the owner is
    `logical_expert // per_rank_experts` and the offset inside its block is
    `local_row_of`.
    """
    owner = logical_expert // per_rank_experts
    stride = per_rank_experts + replica_slots_per_rank
    return owner * stride + local_row_of(logical_expert, per_rank_experts)


def apply_replica_maps(
    logical_to_physical: torch.Tensor,
    logical_replica_count: torch.Tensor,
    source_local: torch.Tensor,
    placements: Sequence[Placement],
    per_rank_experts: int,
    replica_slots_per_rank: int,
    source_rank: int,
    slot_occupant: dict[int, int],
    slot: int = 0,
) -> set[tuple[int, int]]:
    """Make one layer's active replicas exactly `placements`, in place.

    Replaces inverting the whole physical layout with `compute_logical_maps`, which
    nothing about a single placement needs: one expert gained one copy at a row that
    is known from arithmetic. That inversion runs a 136-iteration Python loop and an
    `.item()` that makes its output shape data-dependent — the reason it asserts CPU
    — and it measured 4.707 ms per layer, or 202 ms for a forward touching 43 layers,
    on the activation path inside the forward.

    `placements` is the layer's **complete** desired set, not an addition to it. A
    replica the current plan no longer wants is reverted, because reversion is a map
    edit with no transfer and keeping one is not neutral: it goes on shedding half of
    an expert that may no longer be hot, onto a rank that may now be the peak. Ticket
    00 measured cross-forward residency at -20.0%, and with nothing reverting,
    replicas accumulated to 47-76 per forward against a transfer budget of 43.

    A full rebuild self-corrects state that this path must undo by hand, so it does:
    a reverted expert is returned to a single copy, and its canonical row is restored
    in both maps. Leaving it at two would route half the source ranks to a row that
    now holds someone else's weights.

    Returns:
        `(target rank, logical expert)` for each replica reverted. The caller must
        clear those rows in the physical layout, which is what `active_replicas`
        reads and therefore the source of truth routing is checked against.

    Every scalar here is a host-side Python int, so there is no device read and no
    synchronization; the writes are ordinary indexed tensor stores.

    Args:
        logical_to_physical: `[num_logical_experts, width]` for this layer, mutated.
        logical_replica_count: `[num_logical_experts]` for this layer, mutated.
        source_local: `[num_logical_experts, 1]` map routing reads, mutated.
        placements: Placements activated on this layer.
        per_rank_experts: Canonical experts per rank.
        replica_slots_per_rank: Replica rows per rank.
        source_rank: The EP rank whose source-local map this is.
        slot_occupant: `target rank -> logical expert` currently holding that rank's
            replica slot. Host-side bookkeeping, mutated to match.
        slot: Which replica row of the target rank to use.

    Raises:
        ValueError: If the map is too narrow to hold a second copy, which would
            otherwise write out of bounds or silently drop the replica.
    """
    if logical_to_physical.shape[-1] < 2:
        raise ValueError(
            f"logical_to_physical is {logical_to_physical.shape[-1]} wide, so a "
            f"second copy cannot be recorded and the replica would never be routed "
            f"to. Widen it to at least 2."
        )
    stride = per_rank_experts + replica_slots_per_rank
    targets = [p.target_rank for p in placements]
    if len(set(targets)) != len(targets):
        # A rank holds one replica slot per layer, so two placements cannot share a
        # target. `plan_replicas` never emits this; accepting it would leave two
        # experts each claiming the same row, both routing tokens to weights that
        # belong to only one of them.
        raise ValueError(
            f"two placements target the same rank in one layer: {targets}. A rank has "
            f"one replica slot per layer, so this plan cannot be expressed."
        )
    wanted = {p.target_rank: p.logical_expert for p in placements}

    # Revert first, so a slot handed from one expert to another is released before it
    # is reclaimed and the two writes cannot fight over the same row.
    reverted: set[tuple[int, int]] = set()
    for target, previous in list(slot_occupant.items()):
        if wanted.get(target) == previous:
            continue
        previous_canonical = canonical_row_of(
            previous, per_rank_experts, replica_slots_per_rank
        )
        logical_to_physical[previous, 0] = previous_canonical
        logical_to_physical[previous, 1] = -1
        logical_replica_count[previous] = 1
        source_local[previous, 0] = previous_canonical
        reverted.add((target, previous))
        del slot_occupant[target]

    for placement in placements:
        expert = placement.logical_expert
        target = placement.target_rank
        replica_row = target * stride + replica_row_of(per_rank_experts, slot)

        # Copies are ordered by ascending physical row, because that is the order
        # `compute_logical_maps` discovers them in as it walks the slots. A replica
        # can land *below* its canonical row — expert 9 owned by rank 2 has canonical
        # row 11, and a replica on rank 0 sits at row 4 — so which index holds which
        # copy is not fixed. Assuming `[canonical, replica]` put the wrong row in the
        # map for exactly those cases.
        canonical = canonical_row_of(expert, per_rank_experts, replica_slots_per_rank)
        rows = sorted((canonical, replica_row))
        logical_to_physical[expert, 0] = rows[0]
        logical_to_physical[expert, 1] = rows[1]
        logical_replica_count[expert] = 2
        # `build_source_local_physical_map` picks copy `source_rank % count`.
        source_local[expert, 0] = rows[source_rank % 2]
        slot_occupant[target] = expert
    return reverted


def transfer_replicas(
    placements: list[Placement],
    expert_weights: Sequence[Sequence[torch.Tensor]],
    expert_buffer: Sequence[torch.Tensor],
    ep_rank: int,
    per_rank_experts: int,
    communicator: EplbCommunicator,
    slot: int = 0,
) -> int:
    """Move each placement's weights into its target's inactive row, layer by layer.

    **One layer per round trip, because the staging buffer is a single workspace
    shared by every layer.** Issuing all layers' receives before one execute would
    have them all write the same buffer rows and clobber each other — the constraint
    spec section 7 states as "transfer i+1's receive must wait for transfer i's
    staging-to-slot copy", and the first version of this function violated it. The
    copy-out therefore happens inside the loop, before the next layer reuses the
    buffer, which is also why this function performs the copy rather than returning
    the work to a caller.

    The plan is identical on every rank, so each derives its own role from it: the
    canonical owner sends, the target receives, everyone else takes part in neither.
    `execute` is called once per layer that has any placement, on every rank, because
    it is collective on the pynccl path — a rank with no role in that layer must still
    take part or the others wait for it.

    Args:
        placements: The plan, identical on every rank.
        expert_weights: Per layer, that layer's weight tensors, each shaped
            `[rows_per_rank, ...]`.
        expert_buffer: Staging tensors, one per weight tensor, **shared across
            layers** — not indexed by layer.
        ep_rank: This rank's id in the EP group.
        per_rank_experts: Canonical rows per rank.
        communicator: An `EplbCommunicator`.
        slot: Which inactive row to receive into.

    Returns:
        Placements this rank received and copied into place.
    """
    by_layer: dict[int, list[Placement]] = {}
    for placement in placements:
        by_layer.setdefault(placement.layer, []).append(placement)

    target_row = replica_row_of(per_rank_experts, slot)
    copied = 0
    for layer in sorted(by_layer):
        received: list[Placement] = []
        for placement in by_layer[layer]:
            source_row = local_row_of(placement.logical_expert, per_rank_experts)
            if ep_rank == placement.source_rank:
                communicator.add_send(
                    [w[source_row] for w in expert_weights[layer]],
                    placement.target_rank,
                    expert_id=placement.logical_expert,
                )
            if ep_rank == placement.target_rank:
                communicator.add_recv(
                    [b[target_row] for b in expert_buffer],
                    placement.source_rank,
                    expert_id=placement.logical_expert,
                )
                received.append(placement)
        communicator.execute()
        # Drain the shared buffer before the next layer reuses it.
        for _ in received:
            for weight, buffer in zip(expert_weights[layer], expert_buffer):
                weight[target_row].copy_(buffer[target_row])
            copied += 1
    return copied


def apply_placements_to_layout(
    layout: torch.Tensor,
    placements: list[Placement],
    ep_size: int,
    canonical_per_rank: int,
    replica_slots_per_rank: int = 1,
) -> torch.Tensor:
    """Name each placement's expert in its target rank's inactive row.

    Per layer, never for all layers: a row inactive in one layer and occupied in
    another would otherwise inherit the other layer's expert, and routing would
    send tokens to a row whose weights were never written for that layer.

    Args:
        layout: `[num_layers, num_physical_experts]` physical-to-logical map. Not
            mutated — the caller keeps the old map to order the weight copy against
            the MoE kernel's last read of the row it overwrites.
        placements: The plan, identical on every rank.
        ep_size: EP group size.
        canonical_per_rank: Canonical rows per rank.
        replica_slots_per_rank: Inactive rows per rank.

    Returns:
        A new layout with the placements activated.

    Raises:
        ValueError: If two placements in one layer target the same rank, which one
            slot per rank cannot express and which would silently overwrite.
    """
    updated = layout.clone()
    per_local = canonical_per_rank + replica_slots_per_rank
    view = updated.view(-1, ep_size, per_local)
    taken: set[tuple[int, int]] = set()
    for placement in placements:
        key = (placement.layer, placement.target_rank)
        if key in taken:
            raise ValueError(
                f"layer {placement.layer} rank {placement.target_rank} already "
                f"holds a replica in this plan; one slot per rank per layer cannot "
                f"hold two."
            )
        taken.add(key)
        view[placement.layer, placement.target_rank, canonical_per_rank] = (
            placement.logical_expert
        )
    return updated


# A placement's identity for reconciliation: the same expert in the same target row of
# the same layer is the same placement, whoever sends it. The source rank is derivable
# from the expert, so including it would only invent spurious differences.
ReplicaKey = tuple[int, int, int]


def active_replicas(
    layout: torch.Tensor,
    ep_size: int,
    canonical_per_rank: int,
    replica_slots_per_rank: int = 1,
) -> set[ReplicaKey]:
    """Read the currently active replicas out of the layout.

    The layout is the single source of truth: routing consults the map, so a separate
    record beside it could disagree with what tokens actually do.

    Returns:
        `(layer, logical expert, target rank)` for every occupied replica row.
    """
    per_local = canonical_per_rank + replica_slots_per_rank
    view = layout.view(-1, ep_size, per_local)
    rows = view[:, :, canonical_per_rank:]
    active: set[ReplicaKey] = set()
    for layer, rank, slot in (rows >= 0).nonzero(as_tuple=False).tolist():
        active.add((layer, int(rows[layer, rank, slot]), rank))
    return active


def reconcile(
    active: set[ReplicaKey], desired: list[Placement]
) -> tuple[set[ReplicaKey], set[ReplicaKey], list[Placement]]:
    """Split a desired plan against what is already active.

    Reversion is what makes this worth doing, and it is **free**: deactivating a
    replica is a map edit, with no transfer and no weight touched, because an inactive
    row attracts no tokens whatever it still holds. So there is never a reason to
    carry a placement the current plan no longer wants — and carrying one is not
    neutral: it keeps moving half of an expert that may no longer be hot, onto a rank
    that may now be the peak, which is the measured -20% on mixed traffic.

    Keeping what is still wanted is equally free, which is what turns "transfer the
    whole set every forward" into "transfer the difference" — and in a steady state
    the difference is empty.

    Returns:
        `(keep, revert, transfer)`: keys to leave alone, keys to deactivate, and the
        placements needing a weight transfer.
    """
    wanted = {(p.layer, p.logical_expert, p.target_rank): p for p in desired}
    keep = active & wanted.keys()
    revert = active - wanted.keys()
    transfer = [wanted[key] for key in sorted(wanted.keys() - active)]
    return keep, revert, transfer


def revert_replicas(
    layout: torch.Tensor,
    revert: set[ReplicaKey],
    ep_size: int,
    canonical_per_rank: int,
    replica_slots_per_rank: int = 1,
) -> torch.Tensor:
    """Mark each reverted replica's row inactive again.

    A map edit only. The row keeps whatever weights it held; nothing routes there
    once it reads -1, which is the invariant ticket 03 verified.
    """
    updated = layout.clone()
    per_local = canonical_per_rank + replica_slots_per_rank
    view = updated.view(-1, ep_size, per_local)
    for layer, _expert, target_rank in revert:
        view[layer, target_rank, canonical_per_rank] = -1
    return updated


def plan_one_layer_on_device(
    expert_load: torch.Tensor,
    ep_size: int,
    min_tokens: float,
) -> torch.Tensor:
    """One layer's placement, decided entirely on the device.

    The device counterpart of `plan_replicas` at `max_replicas_per_layer = 1`, the
    default. Ticket 04. It exists so the plan never has to reach the host: the host
    synchronisation `plan_and_launch` performs once per predicted layer costs 5.28 ms of
    collective waiting per layer, 70% of what prediction adds, and it is there only
    because the planner runs in Python.

    **Bit-identity with the host planner is a correctness property, not a nicety.**
    Every rank derives the plan locally from the same snapshot, so a plan that differs
    by rank pairs a sender with no receiver and hangs the engine. Two things make exact
    agreement achievable rather than approximate:

    * The snapshot is integer, and the only non-integer step is a halving, so every
    value
      here is exactly representable in float64. Comparisons and equalities therefore
      mean
      what they do in Python, and `gain == gain.max()` is safe.
    * Reductions that could depend on block order are avoided. A device argmax over
    floats
      is order-dependent, and two ranks reducing the same values in a different order
      can pick different experts; the winner is selected by an explicit lowest-index
      rule over an equality mask instead.

    Args:
        expert_load: `[num_logical_experts]` predicted load, integer-valued, already
            reduced across the EP group so every rank sees identical values.
        ep_size: EP group size. Expert `e` belongs to rank
            `e // (num_logical // ep_size)`.
        min_tokens: Refuse a placement moving less than this. Below one `BLOCK_SIZE_M` a
            replica saves no block, so it saves no time.

    Returns:
        A `[4]` int64 device tensor `(found, logical_expert, target_rank, moved_x2)`.
        `found` is 0 or 1; the remaining entries are meaningless when it is 0.
        `moved_x2` is twice the moved load, kept integral so the whole result stays
        exact — the caller halves it if it needs the value.
    """
    num_logical = expert_load.numel()
    if num_logical % ep_size != 0:
        raise ValueError(
            f"{num_logical} logical experts do not divide across {ep_size} EP ranks."
        )
    per_rank = num_logical // ep_size
    load = expert_load.to(torch.float64)
    rank_load = load.view(ep_size, per_rank).sum(dim=1)

    empty = torch.zeros(4, dtype=torch.int64, device=expert_load.device)
    if float(rank_load.sum()) <= 0.0:
        # A dummy or padding-only forward. Checked on the host because this is the one
        # value already known there — the caller decided to run at all from it.
        return empty

    # The first maximum, matching Python's `max(range(n), key=...)`, so ties go to the
    # lowest rank id exactly as the host planner's do.
    peak_load = rank_load.max()
    peak = int((rank_load == peak_load).to(torch.int64).argmax())

    peak_experts = load.view(ep_size, per_rank)[peak]
    moved = peak_experts * 0.5

    # Every (expert on the peak rank, target rank) pair at once. 16 x 8 x 8 values for
    # this model, so materialising the trial loads is cheaper than being clever about
    # it.
    trial = rank_load.view(1, 1, ep_size).repeat(per_rank, ep_size, 1)
    idx = torch.arange(ep_size, device=load.device)
    trial[:, :, peak] -= moved.view(per_rank, 1)
    trial[torch.arange(per_rank).view(-1, 1), idx.view(1, -1), idx.view(1, -1)] += (
        moved.view(per_rank, 1)
    )
    gain = peak_load - trial.max(dim=2).values

    admissible = (
        (moved.view(per_rank, 1) >= min_tokens)
        & (moved.view(per_rank, 1) > 0)
        & (idx.view(1, ep_size) != peak)
        & (gain > 0)  # relocating the peak is not lowering it
    )
    if not bool(admissible.any()):
        return empty

    # Strongest gain, then the lowest expert id, then the lowest target id. The
    # flattened order is expert-major, so "first admissible index at the maximum gain"
    # is exactly the host planner's tie-break, and taking it from an equality mask keeps
    # the choice independent of how the reduction was scheduled.
    masked = torch.where(admissible, gain, torch.full_like(gain, float("-inf")))
    flat = masked.reshape(-1)
    best = flat == flat.max()
    winner = int(best.to(torch.int64).argmax())
    offset, target = divmod(winner, ep_size)

    out = empty.clone()
    out[0] = 1
    out[1] = peak * per_rank + offset
    out[2] = target
    # Twice the moved load, which is the original integer expert count.
    out[3] = int(peak_experts[offset])
    return out
