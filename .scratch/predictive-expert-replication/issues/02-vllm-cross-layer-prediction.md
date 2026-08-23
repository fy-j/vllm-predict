# 02 — vLLM cross-layer prediction and global predicted-load snapshot

**What to build:** Make Qwen3-30B-A3B expose a read-only predictive path that uses the immediately following MoE's gate and router semantics to produce a global predicted-load snapshot, without changing expert placement or request output.

**Blocked by:** 01 — vLLM predictive infrastructure and fixed layout.

**Status:** ready-for-agent

- [ ] Every non-final sparse MoE is bound to the adjacent target MoE gate module; the final MoE has no target prediction.
- [ ] Source-local current-MoE hidden states produce target logical-expert predictions before token dispatch, using the target router's selection semantics.
- [ ] All EP ranks receive the same `[source rank, logical expert]` Global predicted-load snapshot after dispatch; the communication overlaps current expert compute and completes before combine.
- [ ] Dummy/padding-only forwards participate with zero counts and do not mutate replica state.
- [ ] Tests validate binding topology, source-rank count provenance, gathered snapshot equality, and agreement between predicted-router selection and the target router contract.
