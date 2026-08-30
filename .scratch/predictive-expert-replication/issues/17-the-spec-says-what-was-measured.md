# 17: The spec says what was measured

**What to build:** `spec.md` brought back into agreement with the measurements, so the document
read second by every session stops teaching things that have been withdrawn.

Six disagreements accumulated, and two of them have already cost real work: the batching entry
closed a line of optimisation for a whole session on a single-rank measurement, and the ceiling
figure is quoted in two incompatible units in the same document. This is a documentation ticket
with no code, and it is worth its own slice because a spec that disagrees with the evidence is
how this project has repeatedly rebuilt or rejected the wrong thing.

**Blocked by:** None (can start immediately). Do it *before* ticket 08, which quotes the spec's
figures.

**Status:** ready-for-agent

- [ ] **The ceiling has one unit.** "Further Notes" gives 5.05% of a prefill step, computed from
      a 10.76% expert-GEMM share of **attributed** GPU time, which sums overlapping streams and
      so understates the share. Against wall-clock the same measurement is 11.16% and the ceiling
      is **5.26% of a prefill window**. Pick wall-clock, state the band, and say why the other
      figure existed — the correction is instructive, not embarrassing.
- [ ] **The batching entry is rewritten rather than left struck through.** It currently carries a
      strikethrough and a withdrawal note. Out of Scope is not where a live line of work belongs;
      move it out, and leave behind the reason the original conclusion was wrong (a barrier priced
      from the one rank that never waits) since that reasoning error is the transferable part.
- [ ] **Decision 11's staging workspace is two experts, not one.** "The staging workspace holds
      one expert and is shared by every layer" was true when written and is not now: the put and
      the drain of consecutive layers are not ordered against each other, so it is double-buffered
      and alternates by layer parity. The rejected single-buffer form and why it was unsafe belong
      in the same entry.
- [ ] **Decision 17's window claim is qualified by what the profile found.** The launch moved to
      the predicting layer's MoE tail and the window is the target layer's Attention, which is
      correct. What the spec does not say is that the snapshot collective inside the predicting
      layer overlaps real compute **1.6%** of the time, so `start_snapshot`'s intended hiding
      behind the local expert GEMM does not happen. That is a live cost, not a detail.
- [ ] **The verdict text distinguishes the two halves.** The spec's problem statement and stop
      gate treat the feature as one quantity, and it is now measured as two with opposite signs:
      placement returns about 4% of mean TTFT and captures 70-80% of the ceiling, while prediction
      spends 13.68%. A stop gate on the sum would stop a mechanism that works because of the one
      that pays for it.
- [ ] **Every number cites its measurement.** No new claims are introduced by this ticket; each
      figure it changes points at the `bench/RESULTS.md` section that produced it, and anything
      that cannot be pointed at is deleted rather than reworded.
