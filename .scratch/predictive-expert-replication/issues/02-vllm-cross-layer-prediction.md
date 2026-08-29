# 02 — vLLM cross-layer prediction and global predicted-load snapshot

**What to build:** Make Qwen3-30B-A3B expose a read-only predictive path that uses a configurable lookahead MoE's gate and router semantics to produce a global predicted-load snapshot, without changing expert placement or request output. Measuring how accurate that prediction is belongs to ticket 06.

**Blocked by:** 01 — vLLM predictive infrastructure and fixed layout.

**Status:** done — implemented, unit-tested, and validated on the 8-GPU node

- [x] Each sparse MoE in the source range binds to the MoE `prediction_lookahead_layers` ahead; the skipped leading layers and the trailing layers bind no target and run no predicted-count collective.
- [x] A topology without a sparse MoE in every decoder layer is rejected, so a lookahead measured in layer indices cannot silently span a different distance.
- [x] Source-local current-MoE hidden states produce target logical-expert predictions before token dispatch, using the target router's selection semantics.
- [x] Predicted counts are indexed by logical expert, produced through a read-only routing path that applies no logical-to-physical mapping and records no actual expert load.
- [x] All EP ranks receive the same `[source rank, logical expert]` Global predicted-load snapshot after dispatch; the communication overlaps current expert compute and completes before combine.
- [x] Dummy/padding-only forwards participate with zero counts and do not mutate replica state.
- [x] Tests validate binding topology under a lookahead greater than one, source-rank count provenance, gathered snapshot equality, valid-token masking, and agreement between predicted-router selection and the target router contract.
