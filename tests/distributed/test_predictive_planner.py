# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the predictive replica planner.

The planner's output drives point-to-point sends and receives, so every rank must
derive a **bit-identical** plan from the same load. A plan that differs by rank
pairs a sender with no receiver and hangs the engine — the same class of failure a
per-forward diagnostic caused once by branching a collective on `is_dummy`.
"""

from __future__ import annotations

import pytest
import torch

from vllm.distributed.eplb.eplb_state import compute_logical_maps
from vllm.distributed.eplb.predictive import build_source_local_physical_map
from vllm.distributed.eplb.predictive_planner import (
    Placement,
    active_replicas,
    apply_replica_maps,
    canonical_row_of,
    plan_replicas,
)


def _load(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


class TestPlanIsDeterministic:
    """Same input, same plan — on every rank, every time."""

    def test_repeated_calls_give_an_identical_plan(self):
        per_expert = _load([[9.0, 1.0, 1.0, 1.0]])
        first = plan_replicas(per_expert, ep_size=2, budget=1)
        second = plan_replicas(per_expert, ep_size=2, budget=1)
        assert first == second

    def test_ties_break_by_expert_then_rank_not_by_iteration_order(self):
        """Two identical candidates must resolve the same way everywhere."""
        # Experts 0 and 1 both on rank 0, equal load; rank 1 is idle.
        per_expert = _load([[5.0, 5.0, 0.0, 0.0]])
        (placement,) = plan_replicas(per_expert, ep_size=2, budget=1)
        assert placement.logical_expert == 0
        assert placement.target_rank == 1


class TestPlacementsAreLegal:
    def test_the_source_is_the_expert_s_canonical_owner(self):
        per_expert = _load([[1.0, 1.0, 9.0, 1.0]])
        (placement,) = plan_replicas(per_expert, ep_size=2, budget=1)
        # Experts 2 and 3 belong to rank 1, so expert 2's owner is rank 1.
        assert placement.logical_expert == 2
        assert placement.source_rank == 1
        assert placement.target_rank == 0

    def test_a_rank_hosts_at_most_one_replica_per_layer(self):
        per_expert = _load([[8.0, 7.0, 0.0, 0.0, 0.0, 0.0]])
        out = plan_replicas(per_expert, ep_size=3, budget=4)
        per_layer_targets = [p.target_rank for p in out if p.layer == 0]
        assert len(per_layer_targets) == len(set(per_layer_targets))

    def test_an_expert_is_replicated_at_most_once(self):
        per_expert = _load([[9.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        out = plan_replicas(per_expert, ep_size=3, budget=3)
        experts = [p.logical_expert for p in out]
        assert len(experts) == len(set(experts))

    def test_the_target_never_becomes_the_new_peak(self):
        """A placement that only relocates the peak is not an improvement."""
        per_expert = _load([[6.0, 5.0]])
        assert plan_replicas(per_expert, ep_size=2, budget=1) == []


class TestBudgetAndFloor:
    def test_the_budget_is_global_across_layers(self):
        per_expert = _load([[9.0, 1.0, 1.0, 1.0], [9.0, 1.0, 1.0, 1.0]])
        assert len(plan_replicas(per_expert, ep_size=2, budget=1)) == 1

    def test_it_spends_on_the_worse_layer_first(self):
        mild = [5.0, 4.0, 4.0, 4.0]
        harsh = [20.0, 1.0, 1.0, 1.0]
        out = plan_replicas(_load([mild, harsh]), ep_size=2, budget=1)
        assert [p.layer for p in out] == [1]

    def test_a_move_below_the_token_floor_is_refused(self):
        """Under one BLOCK_SIZE_M a replica saves no block, so it saves no time."""
        per_expert = _load([[100.0, 1.0, 1.0, 1.0]])
        assert plan_replicas(per_expert, ep_size=2, budget=1, min_tokens=10) != []
        assert plan_replicas(per_expert, ep_size=2, budget=1, min_tokens=80) == []

    def test_a_balanced_layer_yields_nothing(self):
        per_expert = _load([[4.0, 4.0, 4.0, 4.0]])
        assert plan_replicas(per_expert, ep_size=2, budget=4) == []

    def test_a_zero_load_layer_yields_nothing(self):
        """A dummy or padding-only forward must not produce transfers."""
        per_expert = _load([[0.0, 0.0, 0.0, 0.0]])
        assert plan_replicas(per_expert, ep_size=2, budget=2) == []


class TestPlacementIdentity:
    def test_a_placement_names_everything_a_transfer_needs(self):
        per_expert = _load([[9.0, 1.0, 1.0, 1.0]])
        (placement,) = plan_replicas(per_expert, ep_size=2, budget=1)
        assert isinstance(placement, Placement)
        assert placement.layer == 0
        assert placement.logical_expert == 0
        assert placement.source_rank == 0
        assert placement.target_rank == 1
        assert placement.moved_load > 0


# --------------------------------------------------------------------------
# Ticket 07: the targeted replica transfer
# --------------------------------------------------------------------------


class _FakeCommunicator:
    """Records what a rank would send and receive, without a process group."""

    def __init__(self):
        self.sends: list[tuple[int, int]] = []
        self.recvs: list[tuple[int, int]] = []
        self.executed = 0
        self.stream = "unset"

    def add_send(self, tensors, dst_rank, expert_id):
        self.sends.append((dst_rank, expert_id))

    def add_recv(self, tensors, src_rank, expert_id):
        self.recvs.append((src_rank, expert_id))

    def execute(self):
        self.executed += 1

    def set_stream(self, stream):
        self.stream = stream


def _weights(ep_size, per_rank, slots, hidden=4):
    rows = per_rank + slots
    return [torch.zeros(rows, hidden)]


class TestTransferIssuesMatchedSendsAndRecvs:
    """Every send must have exactly one matching recv on the peer.

    The plan is identical on every rank, so each derives its own role from it. If the
    roles do not pair up the transfer hangs — the same silent-deadlock class as a
    collective branched on per-rank state.

    Every case here spans **more than one layer**. The first version of this suite
    used layer 0 only, which is why it missed that the code indexed the shared
    staging buffer by layer: `buffer[0]` happened to exist.
    """

    def _issue(self, placements, rank, ep_size=4, per_rank=4, slots=1, layers=3):
        from vllm.distributed.eplb.predictive_planner import transfer_replicas

        comm = _FakeCommunicator()
        weights = [_weights(ep_size, per_rank, slots) for _ in range(layers)]
        buffer = _weights(ep_size, per_rank, slots)  # one workspace, shared
        copied = transfer_replicas(
            placements,
            expert_weights=weights,
            expert_buffer=buffer,
            ep_rank=rank,
            per_rank_experts=per_rank,
            communicator=comm,
        )
        return comm, copied

    def test_every_send_has_exactly_one_matching_recv_across_ranks(self):
        placements = [
            Placement(0, 1, 0, 3, 5.0),
            Placement(2, 9, 2, 1, 4.0),
        ]
        sends, recvs = [], []
        for rank in range(4):
            comm, _ = self._issue(placements, rank)
            sends += [(rank, dst, e) for dst, e in comm.sends]
            recvs += [(src, rank, e) for src, e in comm.recvs]
        assert sorted(sends) == sorted(recvs)

    def test_only_the_owner_sends_and_only_the_target_receives(self):
        placements = [Placement(1, 1, 0, 3, 5.0)]
        assert self._issue(placements, 0)[0].sends == [(3, 1)]
        assert self._issue(placements, 0)[0].recvs == []
        assert self._issue(placements, 3)[0].recvs == [(0, 1)]
        assert self._issue(placements, 3)[0].sends == []
        for bystander in (1, 2):
            comm, _ = self._issue(placements, bystander)
            assert comm.sends == [] and comm.recvs == []

    def test_one_execute_per_layer_not_one_for_the_whole_plan(self):
        """The staging buffer is one workspace, so layers cannot share a round trip.

        Batching every layer's receive before a single execute would have them all
        write the same buffer rows and clobber each other.
        """
        placements = [
            Placement(0, 1, 0, 3, 5.0),
            Placement(1, 2, 0, 3, 5.0),
            Placement(2, 3, 0, 3, 5.0),
        ]
        for rank in range(4):
            comm, _ = self._issue(placements, rank)
            assert comm.executed == 3

    def test_two_placements_in_one_layer_share_that_layer_s_round_trip(self):
        placements = [Placement(0, 1, 0, 3, 5.0), Placement(0, 9, 2, 1, 4.0)]
        for rank in range(4):
            comm, _ = self._issue(placements, rank)
            assert comm.executed == 1

    def test_a_layer_index_past_the_first_is_used_verbatim(self):
        """The regression: the shared buffer was being indexed by layer."""
        placements = [Placement(2, 1, 0, 3, 5.0)]
        comm, copied = self._issue(placements, 3, layers=3)
        assert comm.recvs == [(0, 1)]
        assert copied == 1

    def test_an_empty_plan_does_nothing_and_executes_nothing(self):
        for rank in range(4):
            comm, copied = self._issue([], rank)
            assert comm.executed == 0 and copied == 0

    def test_the_source_row_is_the_expert_s_offset_within_its_owner(self):
        from vllm.distributed.eplb.predictive_planner import local_row_of

        assert local_row_of(logical_expert=0, per_rank_experts=4) == 0
        assert local_row_of(logical_expert=6, per_rank_experts=4) == 2
        assert local_row_of(logical_expert=11, per_rank_experts=4) == 3

    def test_the_target_row_is_the_first_slot_past_the_canonical_rows(self):
        from vllm.distributed.eplb.predictive_planner import replica_row_of

        assert replica_row_of(per_rank_experts=16, slot=0) == 16
        assert replica_row_of(per_rank_experts=4, slot=0) == 4


# --------------------------------------------------------------------------
# Ticket 08: activation — the layout the routing map is rebuilt from
# --------------------------------------------------------------------------


class TestApplyPlacementsToLayout:
    """A placement is activated by naming the expert in the target's inactive row.

    Per layer, not for all layers: a row inactive in one layer and occupied in
    another must not inherit the other's expert, or routing would send tokens to a
    row whose weights were never written for that layer.
    """

    def _layout(self, ep_size=2, per_rank=2, slots=1, layers=3):
        from vllm.distributed.eplb.predictive import build_predictive_physical_map

        return build_predictive_physical_map(
            num_layers=layers,
            num_logical_experts=ep_size * per_rank,
            ep_size=ep_size,
            replica_slots_per_rank=slots,
        )

    def _apply(self, layout, placements, ep_size=2, per_rank=2, slots=1):
        from vllm.distributed.eplb.predictive_planner import (
            apply_placements_to_layout,
        )

        return apply_placements_to_layout(
            layout,
            placements,
            ep_size=ep_size,
            canonical_per_rank=per_rank,
            replica_slots_per_rank=slots,
        )

    def test_only_the_named_layer_is_changed(self):
        layout = self._layout()
        out = self._apply(layout, [Placement(1, 0, 0, 1, 5.0)])
        view = out.view(3, 2, 3)
        assert int(view[1, 1, 2]) == 0, "layer 1's target row now holds expert 0"
        assert int(view[0, 1, 2]) == -1, "layer 0 is untouched"
        assert int(view[2, 1, 2]) == -1, "layer 2 is untouched"

    def test_canonical_rows_are_never_disturbed(self):
        layout = self._layout()
        before = layout.clone()
        out = self._apply(layout, [Placement(0, 0, 0, 1, 5.0)])
        view_before = before.view(3, 2, 3)[:, :, :2]
        view_after = out.view(3, 2, 3)[:, :, :2]
        assert torch.equal(view_before, view_after)

    def test_it_does_not_mutate_the_input(self):
        """The caller keeps the old map to order the copy against its last read."""
        layout = self._layout()
        before = layout.clone()
        self._apply(layout, [Placement(0, 0, 0, 1, 5.0)])
        assert torch.equal(layout, before)

    def test_an_empty_plan_leaves_every_slot_inactive(self):
        layout = self._layout()
        out = self._apply(layout, [])
        assert int(out.view(3, 2, 3)[:, :, 2].max()) == -1

    def test_two_placements_in_one_layer_use_different_target_rows(self):
        layout = self._layout(ep_size=3, per_rank=2, slots=1, layers=1)
        out = self._apply(
            layout,
            [Placement(0, 0, 0, 1, 5.0), Placement(0, 2, 1, 2, 4.0)],
            ep_size=3,
            per_rank=2,
        )
        view = out.view(1, 3, 3)
        assert int(view[0, 1, 2]) == 0
        assert int(view[0, 2, 2]) == 2

    def test_a_second_placement_onto_one_rank_is_refused(self):
        """One replica slot per rank per layer; a second would overwrite the first."""
        import pytest

        layout = self._layout()
        with pytest.raises(ValueError, match="already holds a replica"):
            self._apply(
                layout,
                [Placement(0, 0, 0, 1, 5.0), Placement(0, 1, 0, 1, 4.0)],
            )


class TestActivePlacementsAreReadFromTheLayout:
    """The layout is the single source of truth for what is active.

    Keeping a separate set beside it would let the two disagree, and the map is what
    routing actually consults.
    """

    def _layout(self, ep_size=2, per_rank=2, slots=1, layers=2):
        from vllm.distributed.eplb.predictive import build_predictive_physical_map

        return build_predictive_physical_map(
            num_layers=layers,
            num_logical_experts=ep_size * per_rank,
            ep_size=ep_size,
            replica_slots_per_rank=slots,
        )

    def _active(self, layout, ep_size=2, per_rank=2, slots=1):
        from vllm.distributed.eplb.predictive_planner import active_replicas

        return active_replicas(
            layout,
            ep_size=ep_size,
            canonical_per_rank=per_rank,
            replica_slots_per_rank=slots,
        )

    def test_a_fresh_layout_has_none(self):
        assert self._active(self._layout()) == set()

    def test_it_finds_what_was_activated(self):
        from vllm.distributed.eplb.predictive_planner import (
            apply_placements_to_layout,
        )

        layout = apply_placements_to_layout(
            self._layout(),
            [Placement(1, 0, 0, 1, 5.0)],
            ep_size=2,
            canonical_per_rank=2,
        )
        assert self._active(layout) == {(1, 0, 1)}

    def test_the_key_is_layer_expert_and_target(self):
        """Source rank is derivable from the expert, so it is not part of identity."""
        from vllm.distributed.eplb.predictive_planner import (
            apply_placements_to_layout,
        )

        layout = apply_placements_to_layout(
            self._layout(),
            [Placement(0, 3, 1, 0, 5.0)],
            ep_size=2,
            canonical_per_rank=2,
        )
        assert self._active(layout) == {(0, 3, 0)}


class TestReconcile:
    """Keep what is still wanted, revert what is not, transfer only the difference."""

    def _reconcile(self, active, desired):
        from vllm.distributed.eplb.predictive_planner import reconcile

        return reconcile(active, desired)

    def test_an_unchanged_plan_transfers_nothing(self):
        desired = [Placement(0, 1, 0, 1, 5.0)]
        keep, revert, transfer = self._reconcile({(0, 1, 1)}, desired)
        assert transfer == [] and revert == set()
        assert keep == {(0, 1, 1)}

    def test_a_dropped_placement_is_reverted_and_costs_no_transfer(self):
        keep, revert, transfer = self._reconcile({(0, 1, 1)}, [])
        assert revert == {(0, 1, 1)}
        assert transfer == []

    def test_a_new_placement_is_transferred(self):
        desired = [Placement(0, 2, 1, 0, 4.0)]
        keep, revert, transfer = self._reconcile(set(), desired)
        assert [p.logical_expert for p in transfer] == [2]

    def test_a_replaced_placement_reverts_the_old_and_transfers_the_new(self):
        """The same slot, a different expert: both halves must happen."""
        desired = [Placement(0, 3, 1, 1, 6.0)]
        keep, revert, transfer = self._reconcile({(0, 1, 1)}, desired)
        assert revert == {(0, 1, 1)}
        assert [p.logical_expert for p in transfer] == [3]

    def test_the_steady_state_is_free(self):
        """The point of reconciling: an unchanged desired set costs nothing."""
        desired = [Placement(li, li, 0, 1, 5.0) for li in range(43)]
        active = {(p.layer, p.logical_expert, p.target_rank) for p in desired}
        keep, revert, transfer = self._reconcile(active, desired)
        assert len(keep) == 43 and not revert and not transfer


class TestIncrementalMapsMatchFullInversion:
    """Updating the routing maps in place must equal rebuilding them from scratch.

    `compute_logical_maps` inverts the whole physical layout: a 136-iteration Python
    loop plus an `.item()` that makes the output shape data-dependent, which is why it
    asserts CPU. Measured at 4.707 ms per layer, or 202 ms for a forward that touches
    43 layers — on the activation path, inside the forward. Nothing about a single
    placement needs an inversion: one expert gained one copy at a known row.

    The risk of writing in place is that the full rebuild *self-corrects* state the
    incremental path must undo by hand — above all a replica slot reused by a
    different expert, which leaves the previous occupant claiming two copies. These
    tests compare against the inversion rather than against my expectations.
    """

    EP = 4
    PER_RANK = 4
    SLOTS = 1
    NUM_LOGICAL = 16

    def _physical(self):
        """Canonical layout: rank-major, canonical rows first, replica rows -1."""
        width = self.PER_RANK + self.SLOTS
        physical = torch.full((self.EP * width,), -1, dtype=torch.long)
        for rank in range(self.EP):
            for index in range(self.PER_RANK):
                physical[rank * width + index] = rank * self.PER_RANK + index
        return physical

    def _reference(self, physical, source_rank):
        """What the full inversion produces, as the routing maps."""
        logical_map, count = compute_logical_maps(
            physical.unsqueeze(0), self.NUM_LOGICAL
        )
        source_map, source_count = build_source_local_physical_map(
            logical_map[0], count[0], source_rank
        )
        return logical_map[0], count[0], source_map, source_count

    def _incremental(self, placements, source_rank, width):
        """What the in-place update produces from the canonical starting point."""
        physical = self._physical()
        logical_map, count = compute_logical_maps(
            physical.unsqueeze(0), self.NUM_LOGICAL
        )
        logical_map = torch.nn.functional.pad(
            logical_map[0], (0, width - logical_map.shape[-1]), value=-1
        )
        count = count[0].clone()
        source_map, _ = build_source_local_physical_map(
            logical_map, count, source_rank
        )
        occupant: dict[int, int] = {}
        # One call carries a layer's complete desired set, which is how the activation
        # path uses it: anything active and not named is reverted.
        apply_replica_maps(
            logical_to_physical=logical_map,
            logical_replica_count=count,
            source_local=source_map,
            placements=placements,
            per_rank_experts=self.PER_RANK,
            replica_slots_per_rank=self.SLOTS,
            source_rank=source_rank,
            slot_occupant=occupant,
        )
        return logical_map, count, source_map

    def _incremental_sequence(self, batches, source_rank, width):
        """Apply several forwards' plans in turn, each a complete desired set."""
        physical = self._physical()
        logical_map, count = compute_logical_maps(
            physical.unsqueeze(0), self.NUM_LOGICAL
        )
        logical_map = torch.nn.functional.pad(
            logical_map[0], (0, width - logical_map.shape[-1]), value=-1
        )
        count = count[0].clone()
        source_map, _ = build_source_local_physical_map(
            logical_map, count, source_rank
        )
        occupant: dict[int, int] = {}
        for batch in batches:
            apply_replica_maps(
                logical_to_physical=logical_map,
                logical_replica_count=count,
                source_local=source_map,
                placements=batch,
                per_rank_experts=self.PER_RANK,
                replica_slots_per_rank=self.SLOTS,
                source_rank=source_rank,
                slot_occupant=occupant,
            )
        return logical_map, count, source_map

    def _physical_after(self, placements):
        """The layout those placements leave behind, for the reference path."""
        physical = self._physical()
        width = self.PER_RANK + self.SLOTS
        for placement in placements:
            physical[placement.target_rank * width + self.PER_RANK] = (
                placement.logical_expert
            )
        return physical

    @pytest.mark.parametrize("source_rank", [0, 1, 2, 3])
    def test_one_placement_matches(self, source_rank):
        placements = [Placement(0, 1, 0, 2, 10.0)]
        ref_map, ref_count, ref_source, _ = self._reference(
            self._physical_after(placements), source_rank
        )
        got_map, got_count, got_source = self._incremental(
            placements, source_rank, ref_map.shape[-1]
        )

        assert torch.equal(got_count, ref_count), (
            f"replica counts diverged: {got_count.tolist()} vs {ref_count.tolist()}"
        )
        assert torch.equal(got_map, ref_map), "logical_to_physical diverged"
        assert torch.equal(got_source, ref_source), (
            f"the map routing reads diverged for source rank {source_rank}: "
            f"{got_source.reshape(-1).tolist()} vs {ref_source.reshape(-1).tolist()}"
        )

    def test_several_placements_on_distinct_ranks_match(self):
        placements = [
            Placement(0, 1, 0, 2, 10.0),
            Placement(0, 6, 1, 3, 8.0),
            Placement(0, 9, 2, 0, 6.0),
        ]
        ref_map, ref_count, ref_source, _ = self._reference(
            self._physical_after(placements), source_rank=1
        )
        got_map, got_count, got_source = self._incremental(
            placements, source_rank=1, width=ref_map.shape[-1]
        )

        assert torch.equal(got_count, ref_count)
        assert torch.equal(got_map, ref_map)
        assert torch.equal(got_source, ref_source)

    @pytest.mark.parametrize("source_rank", [0, 1])
    def test_reusing_a_replica_slot_releases_the_previous_occupant(self, source_rank):
        """The case a full rebuild handles for free and an in-place update must not.

        Rank 2's one replica slot first holds expert 1, then expert 6. Expert 1 must
        drop back to a single copy; leaving it at two would route half the source
        ranks to a row that now holds someone else's weights.
        """
        first = Placement(0, 1, 0, 2, 10.0)
        second = Placement(0, 6, 1, 2, 9.0)
        ref_map, ref_count, ref_source, _ = self._reference(
            self._physical_after([second]), source_rank
        )
        got_map, got_count, got_source = self._incremental_sequence(
            [[first], [second]], source_rank, ref_map.shape[-1]
        )

        assert got_count[1].item() == 1, (
            "expert 1 lost its replica slot to expert 6, so it must claim one copy"
        )
        assert torch.equal(got_count, ref_count)
        assert torch.equal(got_map, ref_map)
        assert torch.equal(got_source, ref_source)

    def test_a_sweep_of_placement_sequences_matches(self):
        """Randomised equivalence, because the hand-picked cases missed one.

        The ascending-physical-row ordering of copies was wrong in the first version
        and only one of three hand-written cases exposed it. Each expert appears at
        most once per sequence, matching what `plan_replicas` produces; slot reuse
        across the sequence is deliberate.
        """
        generator = torch.Generator().manual_seed(0)

        def randint(high):
            return int(torch.randint(high, (1,), generator=generator).item())

        for trial in range(200):
            length = 1 + randint(4)
            experts: list[int] = []
            placements = []
            for _ in range(length):
                expert = randint(self.NUM_LOGICAL)
                if expert in experts:
                    continue
                experts.append(expert)
                owner = expert // self.PER_RANK
                target = randint(self.EP)
                if target == owner:
                    continue  # a replica on the owner's own rank is not a placement
                if any(p.target_rank == target for p in placements):
                    continue  # one replica slot per rank per layer
                placements.append(Placement(0, expert, owner, target, 1.0))
            if not placements:
                continue
            source_rank = randint(self.EP)

            ref_map, ref_count, ref_source, _ = self._reference(
                self._physical_after(placements), source_rank
            )
            got_map, got_count, got_source = self._incremental(
                placements, source_rank, ref_map.shape[-1]
            )

            assert torch.equal(got_count, ref_count), (
                f"trial {trial}: counts diverged for {placements} at source rank "
                f"{source_rank}"
            )
            assert torch.equal(got_source, ref_source), (
                f"trial {trial}: the map routing reads diverged for {placements} at "
                f"source rank {source_rank}"
            )


class TestReversion:
    """A layer's active replicas must be exactly what the current plan wants.

    Reversion is a map edit with no transfer, so carrying a replica the plan no longer
    wants is never cheaper than dropping it — and it is not neutral either: it keeps
    shedding half of an expert that may no longer be hot, onto a rank that may now be
    the peak. Ticket 00 measured cross-forward residency at -20.0%.

    This was implemented once for the whole-layout path and did not come across to the
    in-forward one, where `reconcile`, `revert_replicas` and `active_replicas` sat
    uncalled while replicas accumulated to 47-76 per forward against a transfer budget
    of 43.
    """

    EP = 4
    PER_RANK = 4
    SLOTS = 1
    NUM_LOGICAL = 16

    def _fresh(self):
        width = self.PER_RANK + self.SLOTS
        physical = torch.full((self.EP * width,), -1, dtype=torch.long)
        for rank in range(self.EP):
            for index in range(self.PER_RANK):
                physical[rank * width + index] = rank * self.PER_RANK + index
        logical_map, count = compute_logical_maps(
            physical.unsqueeze(0), self.NUM_LOGICAL
        )
        logical_map = torch.nn.functional.pad(logical_map[0], (0, 1), value=-1)
        count = count[0].clone()
        source_map, _ = build_source_local_physical_map(logical_map, count, 1)
        return physical, logical_map, count, source_map

    def _apply(self, state, placements, source_rank=1):
        _physical, logical_map, count, source_map = state[:4]
        return apply_replica_maps(
            logical_to_physical=logical_map,
            logical_replica_count=count,
            source_local=source_map,
            placements=placements,
            per_rank_experts=self.PER_RANK,
            replica_slots_per_rank=self.SLOTS,
            source_rank=source_rank,
            slot_occupant=state[4],
        )

    def test_a_replica_the_plan_no_longer_wants_is_reverted(self):
        state = (*self._fresh(), {})
        _physical, logical_map, count, source_map = state[:4]
        self._apply(state, [Placement(0, 1, 0, 2, 10.0)])
        assert count[1].item() == 2

        # Next forward wants a different expert entirely, on a different rank.
        reverted = self._apply(state, [Placement(0, 9, 2, 1, 7.0)])

        assert count[1].item() == 1, "expert 1 is no longer wanted; it must revert"
        assert logical_map[1, 1].item() == -1
        assert source_map[1, 0].item() == canonical_row_of(1, self.PER_RANK, self.SLOTS)
        assert (2, 1) in reverted, (
            "the reverted rank/expert must be reported so the caller can clear the "
            "physical row that `active_replicas` reads"
        )

    def test_a_replica_still_wanted_is_kept_and_not_disturbed(self):
        state = (*self._fresh(), {})
        _physical, logical_map, count, source_map = state[:4]
        placement = Placement(0, 1, 0, 2, 10.0)
        self._apply(state, [placement])
        before = (logical_map.clone(), count.clone(), source_map.clone())

        reverted = self._apply(state, [placement])

        assert reverted == set(), (
            "nothing should be reverted when the plan is unchanged"
        )
        assert torch.equal(logical_map, before[0])
        assert torch.equal(count, before[1])
        assert torch.equal(source_map, before[2])

    def test_an_empty_plan_reverts_everything(self):
        state = (*self._fresh(), {})
        _physical, logical_map, count, source_map = state[:4]
        self._apply(state, [Placement(0, 1, 0, 2, 10.0), Placement(0, 6, 1, 3, 9.0)])

        reverted = self._apply(state, [])

        assert reverted == {(2, 1), (3, 6)}
        assert count.tolist() == [1] * self.NUM_LOGICAL, (
            "a forward that plans nothing must leave no replica behind"
        )

    def test_the_host_mirror_agrees_with_the_layout(self):
        """The guard against the failure `active_replicas` warns about.

        Routing consults the layout; this path's bookkeeping is a host-side dict. If
        they disagree, tokens go somewhere the accounting does not know about.
        """
        physical, logical_map, count, source_map = self._fresh()
        occupant: dict[int, int] = {}
        width = self.PER_RANK + self.SLOTS
        generator = torch.Generator().manual_seed(1)

        for _ in range(50):
            n = int(torch.randint(3, (1,), generator=generator).item())
            placements, seen = [], set()
            for _ in range(n):
                expert = int(
                    torch.randint(self.NUM_LOGICAL, (1,), generator=generator).item()
                )
                target = int(torch.randint(self.EP, (1,), generator=generator).item())
                if target in seen or expert // self.PER_RANK == target:
                    continue
                seen.add(target)
                placements.append(
                    Placement(0, expert, expert // self.PER_RANK, target, 1.0)
                )
            reverted = apply_replica_maps(
                logical_to_physical=logical_map,
                logical_replica_count=count,
                source_local=source_map,
                placements=placements,
                per_rank_experts=self.PER_RANK,
                replica_slots_per_rank=self.SLOTS,
                source_rank=1,
                slot_occupant=occupant,
            )
            # Mirror the caller: clear reverted rows, set placed ones.
            for target, _expert in reverted:
                physical[target * width + self.PER_RANK] = -1
            for placement in placements:
                physical[placement.target_rank * width + self.PER_RANK] = (
                    placement.logical_expert
                )

            from_layout = active_replicas(
                physical.unsqueeze(0), self.EP, self.PER_RANK, self.SLOTS
            )
            from_mirror = {(0, expert, rank) for rank, expert in occupant.items()}
            assert from_layout == from_mirror, (
                f"the host mirror and the layout disagree: {from_mirror} vs "
                f"{from_layout}"
            )
