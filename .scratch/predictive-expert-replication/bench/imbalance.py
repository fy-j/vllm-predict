# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-imbalance analysis for the expert-load dump.

Two properties are structural rather than remembered, because both errors were
made for real in this branch:

1. **The imbalance is per layer.** Every MoE layer is its own collective and
   waits for its own slowest rank, so the critical path is the sum of per-layer
   peaks. Summing layers before comparing ranks lets their peaks cancel and
   understated one measurement here about threefold. `aggregated_imbalance` exists
   only to show that gap; nothing else in this module computes it.

2. **A replica does not move an expert's whole load.** Source-rank routing splits
   the source ranks across the copies, so one replica takes half. Crediting it
   with the whole load overstated a placement's reach here by roughly 2x.

There is deliberately no prefill/decode classifier. Imbalance is reported against
each forward's assignment count, so no threshold has to be chosen and no forward
has to be guessed at.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Layer:
    """One MoE layer of one forward.

    Attributes:
        rank_load: Load per EP rank. Taken from the dump rather than derived from
            a logical-to-rank mapping, because physical slots are laid out
            rank-major and reshaping is exact where assuming contiguity is not.
        expert_load: Per-rank list of that rank's own experts' loads.
        moves: Replicas placed by a simulation, for budget accounting.
        _slots: Target ranks already hosting a replica of this layer. One
            replica slot per rank per layer, so a rank hosts at most one.
    """

    rank_load: list[float]
    expert_load: list[list[float]]
    moves: int = 0
    _moved: set = field(default_factory=set)
    _slots: set = field(default_factory=set)

    def copy(self) -> Layer:
        return Layer(
            rank_load=list(self.rank_load),
            expert_load=[list(e) for e in self.expert_load],
            moves=self.moves,
            _moved=set(self._moved),
            _slots=set(self._slots),
        )

    def _slot_taken(self, rank: int) -> bool:
        """A rank holds one replica slot per layer, so it can host one replica."""
        return rank in self._slots

    @property
    def live(self) -> bool:
        return sum(self.rank_load) > 0


def shed_fraction(replicas: int) -> float:
    """Fraction of an expert's load that leaves its canonical owner.

    With `replicas` extra copies there are `replicas + 1` copies in total and the
    source ranks divide evenly between them, so the owner keeps `1/(replicas+1)`.
    This is the glossary's `K/(K+1)` and it is strictly below 1: no amount of
    fan-out moves an expert entirely.
    """
    return replicas / (replicas + 1)


def critical_path_imbalance(layers: Sequence[Layer]) -> float:
    """Sum of per-layer peak rank load over the sum of per-layer means.

    Raises:
        ValueError: When no layer carried load, which would otherwise average to
            a flat 1.0 and read as perfect balance.
    """
    live = [x for x in layers if x.live]
    if not live:
        raise ValueError("no layer in this sample carried load")
    return sum(max(x.rank_load) for x in live) / sum(
        statistics.mean(x.rank_load) for x in live
    )


def aggregated_imbalance(layers: Sequence[Layer]) -> float:
    """Rank totals compared after summing every layer — **the misleading metric**.

    Provided only to quantify how far it understates the critical path. Never use
    it to size or justify a placement.
    """
    live = [x for x in layers if x.live]
    if not live:
        raise ValueError("no layer in this sample carried load")
    n = len(live[0].rank_load)
    totals = [sum(x.rank_load[r] for x in live) for r in range(n)]
    return max(totals) / statistics.mean(totals)


def _best_move(layer: Layer) -> tuple[float, int, int, int] | None:
    """The placement on this layer that lowers its peak most, if any lowers it.

    Candidates come only from the peak rank: replicating an expert elsewhere
    cannot shorten the rank the layer is waiting on. A move that would make the
    target the new, higher peak is not an improvement and is refused.

    Returns:
        `(peak reduction, expert index, source rank, target rank)`, or None.
    """
    load = layer.rank_load
    peak_rank = max(range(len(load)), key=lambda r: load[r])
    if all(r == peak_rank or layer._slot_taken(r) for r in range(len(load))):
        return None  # every other rank's slot for this layer is already used
    old_peak = load[peak_rank]
    best = None
    for index, value in enumerate(layer.expert_load[peak_rank]):
        if value <= 0 or (peak_rank, index) in layer._moved:
            continue
        moved = value * shed_fraction(1)
        # A rank holds one replica slot per layer, so a target can host one
        # replica. Ignoring that let a layer place two replicas on one rank,
        # which is not a placement the hardware can express: measured at 6.6%
        # of placements at one per layer, rising to 27.3% at three.
        for candidate in sorted(range(len(load)), key=lambda r: load[r]):
            if candidate == peak_rank or layer._slot_taken(candidate):
                continue
            trial = list(load)
            trial[peak_rank] -= moved
            trial[candidate] += moved
            gain = old_peak - max(trial)
            if gain > 0 and (best is None or gain > best[0]):
                best = (gain, index, peak_rank, candidate)
            break  # the lightest admissible rank is the best target for this expert
    return best


def _apply(layer: Layer, move: tuple[float, int, int, int]) -> None:
    _, index, source, target = move
    moved = layer.expert_load[source][index] * shed_fraction(1)
    layer.rank_load[source] -= moved
    layer.rank_load[target] += moved
    layer._moved.add((source, index))
    layer._slots.add(target)
    layer.moves += 1


def place_uniformly(layers: Sequence[Layer], per_layer: int) -> list[Layer]:
    """Give every layer the same replica allowance — what a per-layer cap does."""
    out = [x.copy() for x in layers]
    for layer in out:
        if not layer.live:
            continue
        for _ in range(per_layer):
            move = _best_move(layer)
            if move is None:
                break
            _apply(layer, move)
    return out


def plan_moves(
    layers: Sequence[Layer],
    budget: int,
    per_layer_cap: int | None = None,
    eligible: set[int] | None = None,
) -> list[tuple[int, int, int, int]]:
    """The moves `place_globally` would make, as data rather than as an effect.

    Exists so a placement can be *chosen* on one load and *scored* on another. That
    is the measurement that decides whether prediction is good enough to build on:
    the oracle benefit assumes the planner sees the load it is optimizing, and a
    predictor does not.

    Args:
        layers: The forward's layers.
        budget: Total placements allowed.
        per_layer_cap: Most placements one layer may take.
        eligible: Layers that have a replica slot at all. `None` means every layer
            does, which is the current allocation: one row per layer per rank, 432 MiB
            per rank at 48 layers. Restricting it is how that memory would be cut, so
            this is what prices the trade.

    Returns:
        `(layer index, expert index within its rank, source rank, target rank)` in
        the order the planner chose them. Order matters on replay, because each move
        consumes one of the target rank's replica slots.
    """
    out = [x.copy() for x in layers]
    moves: list[tuple[int, int, int, int]] = []
    for _ in range(budget):
        best = None
        for index, layer in enumerate(out):
            if not layer.live:
                continue
            if eligible is not None and index not in eligible:
                continue
            if per_layer_cap is not None and layer.moves >= per_layer_cap:
                continue
            move = _best_move(layer)
            if move is not None and (best is None or move[0] > best[0][0]):
                best = (move, index)
        if best is None:
            break
        move, index = best
        _apply(out[index], move)
        moves.append((index, move[1], move[2], move[3]))
    return moves


def apply_moves(
    layers: Sequence[Layer], moves: Sequence[tuple[int, int, int, int]]
) -> list[Layer]:
    """Replay `plan_moves`' output against a possibly different load.

    A move is applied for what it costs on *this* load, including when this load
    makes it useless: a replica transferred on a wrong prediction still occupies the
    slot and still shifts load, and hiding that would flatter the prediction.
    """
    out = [x.copy() for x in layers]
    for layer_index, expert, source, target in moves:
        layer = out[layer_index]
        moved = layer.expert_load[source][expert] * shed_fraction(1)
        layer.rank_load[source] -= moved
        layer.rank_load[target] += moved
        layer._moved.add((source, expert))
        layer._slots.add(target)
        layer.moves += 1
    return out


def place_globally(
    layers: Sequence[Layer], budget: int, per_layer_cap: int | None = None
) -> list[Layer]:
    """Spend one global transfer budget on the best available move anywhere.

    The transfer budget is global — `max_transfers_per_forward` caps transfers
    across all layers at once — so a per-layer allowance cannot express it. This
    ranks every layer's best candidate together and takes the strongest, which is
    what makes the replica count per layer a *result* rather than a constant.

    Args:
        layers: The forward's layers.
        budget: Total placements allowed.
        per_layer_cap: Optional safety cap so one layer cannot take the whole
            budget. This is what `max_replicas_per_layer` becomes.
    """
    out = [x.copy() for x in layers]
    for _ in range(budget):
        best = None
        for index, layer in enumerate(out):
            if not layer.live:
                continue
            if per_layer_cap is not None and layer.moves >= per_layer_cap:
                continue
            move = _best_move(layer)
            if move is not None and (best is None or move[0] > best[0][0]):
                best = (move, index)
        if best is None:
            break
        move, index = best
        _apply(out[index], move)
    return out


def _realize_to_ratio(
    layer: Layer, ratio: float, budget: int, min_tokens: float
) -> int:
    """Bring one layer's peak under `ratio` x its mean, in place, within `budget`.

    Returns:
        Placements used. The layer is left partly relieved when the budget runs
        out, which is correct: a partial improvement is still an improvement.
    """
    if not layer.live:
        return 0
    target = statistics.mean(layer.rank_load) * ratio
    used = 0
    while used < budget and max(layer.rank_load) > target:
        load = layer.rank_load
        source = max(range(len(load)), key=lambda r: load[r])
        best = None
        for index, value in enumerate(layer.expert_load[source]):
            if value <= 0 or (source, index) in layer._moved:
                continue
            moved = value * shed_fraction(1)
            if moved < min_tokens:
                continue
            for target_rank in range(len(load)):
                if target_rank == source or layer._slot_taken(target_rank):
                    continue
                if load[target_rank] + moved > max(target, load[source] - moved):
                    continue  # would only relocate the peak, not lower it
                # Hottest first: measured equal at one placement per layer and
                # ahead on 18 of 18 forwards above that. See spec section 6.
                if best is None or value > best[0]:
                    best = (value, index, target_rank, moved)
        if best is None:
            break
        _, index, target_rank, moved = best
        layer.rank_load[source] -= moved
        layer.rank_load[target_rank] += moved
        layer._moved.add((source, index))
        layer._slots.add(target_rank)
        layer.moves += 1
        used += 1
    return used


def place_by_threshold_search(
    layers: Sequence[Layer],
    budget: int,
    min_tokens: float = 0.0,
    tolerance: float = 0.005,
) -> list[Layer]:
    """Search the achievable imbalance ratio, and let the placement count follow.

    The transfer budget is global, so the target should be global too: this finds
    the smallest ratio every layer can be brought under while the *total* placements
    stay within `budget`. A layer that is worse therefore draws more replicas on its
    own, which is what makes the per-layer replica count a result of the load rather
    than a configured constant.

    Adapted from UltraEP's quota planner, with one difference that matters: its
    replica absorbs an arbitrary token quota, while source-rank routing splits the
    source ranks, so ours moves exactly half of an expert. The search is the same;
    the move it can make is coarser.

    Args:
        layers: One forward's layers.
        budget: Total placements allowed across all layers.
        min_tokens: Refuse a placement that would move fewer tokens than this. Set
            it to `BLOCK_SIZE_M`: below one block a replica saves no block, so it
            saves no time.
        tolerance: Stop when the bracket on the ratio is narrower than this.
    """
    if budget <= 0:
        return [x.copy() for x in layers]
    live = [x for x in layers if x.live]
    if not live:
        return [x.copy() for x in layers]

    lo, hi = 1.0, max(max(x.rank_load) / statistics.mean(x.rank_load) for x in live)

    def cost(ratio: float) -> tuple[int, list[Layer]]:
        trial = [x.copy() for x in layers]
        spent = 0
        # Worst layers first, so a tight budget is spent where it buys most.
        order = sorted(
            range(len(trial)),
            key=lambda i: (
                -(
                    max(trial[i].rank_load) / statistics.mean(trial[i].rank_load)
                    if trial[i].live
                    else 0.0
                )
            ),
        )
        for i in order:
            spent += _realize_to_ratio(trial[i], ratio, budget - spent, min_tokens)
            if spent >= budget:
                break
        return spent, trial

    best = cost(hi)[1]
    while hi - lo > tolerance:
        mid = (lo + hi) / 2
        spent, trial = cost(mid)
        # Feasible when the budget covered it *and* every layer reached the target.
        reached = all(
            max(x.rank_load) <= statistics.mean(x.rank_load) * mid + 1e-9
            for x in trial
            if x.live
        )
        if spent <= budget and reached:
            hi, best = mid, trial
        else:
            lo = mid
            # When the budget was not the binding constraint the trial is still a
            # valid placement; keep it if it beats what we have.
            if spent <= budget and critical_path_imbalance(
                trial
            ) < critical_path_imbalance(best):
                best = trial
    return best


def bin_by_assignments(
    records: Sequence[dict], edges: Sequence[float]
) -> dict[tuple[float, float], dict]:
    """Group per-forward imbalances into assignment-count bands.

    Replaces a prefill/decode classifier: the regime that matters is set by how
    many tokens each expert sees, which is a continuous quantity, so it is
    reported as one rather than split by a threshold someone has to choose.
    """
    out: dict[tuple[float, float], dict] = {}
    for low, high in zip(edges, edges[1:]):
        vals = [r["imbalance"] for r in records if low <= r["assignments"] < high]
        if vals:
            out[(low, high)] = {
                "n": len(vals),
                "imbalance": round(statistics.mean(vals), 4),
                "median": round(statistics.median(vals), 4),
            }
    return out


def load_dump(path: Path) -> list[list[Layer]]:
    """Read the dump into one list of layers per forward.

    Accepts only the self-describing record form. The older bare-array form is
    rejected rather than guessed at: it carried no per-rank load and no assignment
    count, so reading it required assuming a contiguous expert-to-rank mapping.
    """
    forwards = []
    repaired = 0
    rescaled = False
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not isinstance(rec, dict) or "rank_load" not in rec:
            raise ValueError(
                f"{path} is in the pre-2026-08-24 bare-array form, which lacks "
                "per-rank load and assignment counts. Re-capture it."
            )
        ep = len(rec["rank_load"][0])
        logical = rec["logical_load"]
        per_rank_experts = len(logical[0]) // ep
        layers = []
        for li, ranks in enumerate(rec["rank_load"]):
            groups = [
                logical[li][r * per_rank_experts : (r + 1) * per_rank_experts]
                for r in range(ep)
            ]
            # The two views must be on one scale. A dump written before
            # 2026-08-25 reduced the logical view over the EP group and left the
            # per-rank view raw, so they differ by a factor of the EP size — and
            # the per-rank one is then "how one rank's tokens spread across the
            # ranks", not what each rank does. Prefer the logical grouping, which
            # was always reduced, and say so rather than silently mixing scales.
            grouped = [sum(g) for g in groups]
            recorded = float(sum(ranks))
            if recorded > 0 and abs(recorded - sum(grouped)) > 1e-6 * max(
                1.0, sum(grouped)
            ):
                rank_load = grouped
                rescaled = True
            else:
                rank_load = list(ranks)
                rescaled = False
            layers.append(Layer(rank_load=rank_load, expert_load=groups))
        if rescaled:
            repaired += 1
        forwards.append(layers)
    if repaired:
        print(
            f"note: {repaired} of {len(forwards)} forwards in {path.name} carried a "
            "per-rank view on a different scale from the logical one; the logical "
            "grouping was used. Re-capture with a build from 2026-08-25 or later.",
            file=sys.stderr,
        )
    return forwards
