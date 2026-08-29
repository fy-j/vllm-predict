# 02: Config and accounting say what they mean

**What to build:** Four knobs currently mislead a reader about what they do. Each is small on
its own; together they are the difference between a spec a newcomer can trust and one they
have to re-derive from measurements.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] `prediction_lookahead_layers = 1` is rejected by configuration validation, with the
      reason stated: with the launch and the wait sitting as adjacent statements at the head
      of the MoE forward, a lookahead of 1 gives an overlap window of **zero** and exposes
      the whole transfer. The value is currently accepted, so the setting that looks most
      attractive silently produces the worst result. Ticket 07 lifts this.
- [ ] The MoE kernel's block size is read from the selected kernel configuration and carried
      as a cost-profile field checked at startup, instead of a hardcoded constant. The same
      number gates placement suppression and floors the planner's minimum move, so a wrong
      value either reopens a regime that is settled negative or rejects placements that would
      have paid.
- [ ] `max_transfers_per_forward` counts transfers. It currently increments on planned
      placements *before* reconciliation drops the ones already resident, so it bounds
      coverage rather than bytes. Either the counter moves after reconciliation or the knob
      is renamed to say coverage — but the name and the behaviour must agree.
- [ ] `max_concurrent_transfer_bytes` is enforced. It is declared and never read, while the
      spec makes it the binding constraint on expert-weight bytes in flight and the reason a
      per-forward count was rejected as insufficient.
- [ ] Unit tests for each rejection and each bound, including a test that the byte bound
      actually binds when two transfers would otherwise overlap one layer's dispatch.
