# 14: The snapshot reduction leaves NCCL, and stops being a barrier

**What to build:** The predicted-load snapshot as a device-side reduction over NVSHMEM instead
of an NCCL collective, so it stops coupling the ranks at all.

The planner needs only the **sum over source ranks** of the `[num_logical]` counts —
`record_prediction` already reduces the snapshot with `sum(dim=0)`, so the per-source-rank
breakdown the AllGather delivers has never been read. That is 512 bytes. NVSHMEM is already
initialised on every worker for the weight transfer, and a put-with-flag reduction needs no
group-wide alignment: a rank reads its peers' contributions as they arrive, and a per-forward
sequence number makes "this forward's values" checkable without a barrier.

**Why this was sequenced after batching rather than instead of it** (kept for the record; the
status above supersedes its conclusion, and "dropped rather than built" is what happened). Batching is pure scheduling
against a mechanism ticket 12 proves inert; this is a new kernel with a new correctness
argument, and its value is only whatever barrier cost ticket 13 leaves behind. If 13 lands the
remaining 11 barriers at a cost that no longer matters, this ticket should be dropped rather
than built, and dropping it is an acceptable outcome.

**Blocked by:** nothing. It was blocked by 13; 13 landed at window 1 as the default, so this
ticket is priced against window 1 directly.

**Status: CLOSED, DO NOT BUILD (2026-09-08). Recomputed at 8k, and there is nothing left to
remove.** The 2026-08-31 verdict below was taken at ~877 tokens per forward and this ticket was
left open on exactly the right doubt: the ceiling is a share of prediction's cost, prediction's
cost is per *forward*, and the window it hides in scales with tokens. Measured at ~4200 tokens
per forward (3 passes, so read the direction not the size): **prediction alone is +0.24% +/- 1.38
mean TTFT against stock, and removing its snapshot AllGather leaves +1.30%.** The cost this
ticket exists to attack has amortised away — at 1k it was +13.5%. A ticket whose ceiling is a
fraction of nothing does not get built. See `RESULTS.md` 2026-09-08.

Ticket 11's isolation already prices what this ticket can return: removing all 44 barriers is
**47.6%** of prediction's added cost (11.04 of 23.21 ms). Applied to a clean interleaved run at
the knee today — stock 171.79, prediction only 190.24, placing at window 1 188.88 — that
ceiling is **8.78 ms**, against a placing gap of **17.09 ms**:

```text
placing now            +17.09 ms  (+9.9%)
placing, perfect 14     +8.31 ms  (+4.8%)   <- a free snapshot exchange, still negative
```

So a *perfect* implementation — the snapshot costing nothing at all — leaves the feature at
**+4.8% mean TTFT**. It halves the gap and does not close it, and the realistic share is less
than the ceiling because a correct reduction still waits for the last arriving rank; the 0.25
ms per barrier is mostly arrival skew, not the 10-20 us of ring latency this kernel would
remove. The remaining **52.4%** is prediction's launches and compute, which is ticket 09's
lever and the larger half.

**The attempt to bound this directly failed, twice, and the second failure is instructive.**
A deterministic rank-identical snapshot (`VLLM_PREDICTIVE_DETERMINISTIC_SNAPSHOT`) removes the
collective while keeping placement armed — unlike ticket 11's probe, which had to withhold
placement. Both runs came in *slower* than the arm they were meant to bound:

* **first, a calibration error of mine.** Rotating the hot expert every forward re-planned
  every layer every forward: 1634 activations over 70 forwards where the real arm does 408.
  It measured a 4x transfer storm, +33.9% TTFT. Fixed by `_PROBE_ROTATE_EVERY = 4` and a phase
  per source layer, which lands 439-483 over 70-80 — calibrated against the real arm.
* **then, an impossibility.** Calibrated, it still cost +38.0%. Both arms carry identical
  canonical load (logical imbalance 1.8912 in each), but the real arm reaches a physical
  1.6598 and the probe 1.8807 — **26.0% of excess removed against 1.2%**. A synthetic snapshot
  places replicas that shed nothing, so it pays placement's whole cost for no benefit. This is
  not fixable: a rank-identical *synthetic* snapshot cannot describe real load, and one that
  could would need the real data, which needs the collective. **No zero-collective probe can
  bound this ticket.** The valid instrument is ticket 11's isolation on the prediction arm.

The probe and its tests are retained: it is a working, calibrated churn generator, and the
4x-churn number it produced by accident prices ticket 15's defect (see below).

**Incidental result, and it belongs to ticket 15.** 4x the transfer rate (5.8 -> 23.3
activations per forward, same traffic) costs **+27% mean TTFT**. Ticket 15's defect — a spent
budget reverting still-valid resident replicas — is exactly what turns steady reuse into that
churn, so its cost is first-order, not second. It also means lookahead's value is in the
*stability* of what it predicts more than in hit rate.

**Status:** ready-for-agent

* [ ] The reduction produces a `[num_logical]` sum that is **bit-identical to the AllGather
      path's**, over randomised per-rank counts including empty and all-zero inputs. Integer
      throughout, for the same reason the planner is: every rank must derive the same plan, and
      a float reduction whose order differs by rank is how two ranks pick different experts.
* [ ] Reads only this forward's values. A per-forward sequence number written with the payload,
      checked on read, and a value from another forward is an invariant violation rather than
      something to average in. Ticket 06's plan-ownership rule is the precedent.
* [ ] **No group-wide barrier and no host read**, asserted by measurement rather than by grep:
      `set_sync_debug_mode("error")` clean, and the host returns in microseconds with 100 ms
      queued on the compute stream, which is the shape ticket 06's test used.
* [ ] Every rank agrees on whether this path is in use, through the existing
      `agree_across_ranks`, and a group that falls back closes what it opened. A rank deciding
      this alone is the deadlock that helper exists to prevent, and the fallback path must not
      land on a configuration the validator forbids — which it did once already.
* [ ] Measured against ticket 13's numbers at the same operating point: collectives per prefill
      window, window wall-clock, and three-arm TTFT. The claim to test is that the remaining
      barrier cost goes to zero, not that the reduction is faster.
