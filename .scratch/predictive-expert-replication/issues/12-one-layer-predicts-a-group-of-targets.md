# 12: One layer predicts a group of targets, and at K=1 nothing changes

**What to build:** The mechanism that lets several target layers share one snapshot collective,
landing with the group size at **1**, where it is provably the path that runs today.

A source layer evaluates the gates of the next `K` target layers on its **own** pre-dispatch
hidden states, and one AllGather carries all `K` count vectors. The `K` gates take the same
input, so their GEMMs concatenate into one — the runner already does that trick in
`_maybe_fuse_gate_weights` for its own gate. Only every `K`-th layer predicts at all; the
layers between are consumers. Prediction distances inside a group are `1..K`.

**Why K=1 first, and why that is the whole point of this ticket.** This branch has paid for
building on unverified premises several times: a wait that was vacuous, a value published where
nothing read it, a barrier priced from one rank. At `K=1` this mechanism must produce the same
counts, the same plans and the same published maps as the per-layer path it replaces, and that
is assertable by equality rather than by argument. Flipping `K` is then a configuration change
against a mechanism already known to be correct, which is ticket 13.

**Blocked by:** None (can start immediately). Ticket 11 supplies the measurements that justify
it and is answered.

**Status:** ready-for-agent

- [ ] A group size knob, defaulting to **1**, with validation that rejects a value leaving no
      source layer or naming a target past the last layer. It interacts with
      `prediction_lookahead_layers`: inside a group the distances are `1..K`, so the existing
      lookahead knob becomes the group's *first* distance and the two must not be able to
      describe contradictory shapes. Decide which one survives and say so in the rejection
      message.
- [ ] One source layer's prediction produces `K` count vectors from one set of hidden states,
      through one concatenated gate GEMM and one counting kernel launch, not `K` of each. The
      top-k stays per target, because it is vLLM's own kernel and shared with real routing.
- [ ] One AllGather per group carries `[K, num_logical]`, and the snapshot the planner sees is
      `[ep_size, K, num_logical]` with each target layer reading its own row.
- [ ] The registry binds groups rather than adjacent pairs: a source layer binds `K` targets,
      the layers inside its group bind none, and coverage is unchanged — every layer from
      `prediction_skip_first_layers + 1` to the last is still some group's target.
- [ ] The coordinator holds up to `K` inbound pending plans, each keyed by its target layer and
      stamped with the forward that produced it, and every one is still produced and consumed
      within the same forward. Ticket 06's two invariants keep holding: a leftover pending plan
      raises, and a plan from another forward raises.
- [ ] **At K=1 the whole path is bit-identical to the one it replaces**, asserted by equality
      over randomised hidden states and randomised snapshots: the same predicted counts, the
      same plan rows, the same published `logical_to_physical`, `logical_replica_count`,
      source-local map and layout. This is the criterion the ticket exists for; a K=1 default
      that merely "looks right" is worth nothing.
- [ ] The existing suites still pass unchanged, and a real server at K=1 still activates on
      every reachable layer with the same excess removed as today's 26.5-27.0%.
