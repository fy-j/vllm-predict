# 09: Can a CUDA graph capture the prediction path?

**What to build:** An answer, not a feature. **On the mainline as of 2026-08-30, and now the
largest single lever in the project** — this ticket used to say "exploratory, and deliberately
off the mainline", and the measurement below is why that changed.

**The prize is measured, not argued.** Ticket 11 split prediction's +13.68% mean TTFT in two by
removing only its collectives: **11.04 ms of 23.21 is the 44 per-layer barriers, and 12.17 ms
(52%) is its launches and compute.** In the window profile the same term is 8.3 ms per prefill
window at *unchanged* collective count. Tickets 12, 13 and 14 attack the barrier half; **nothing
in the ticket set attacks the launch half except this one.** Even a perfect grouping leaves
prediction near +4.8% against a 5.26% ceiling, which is break-even; the launch half is what
would make the feature positive.

An earlier version of this ticket priced the prize as "about 2.2 ms of GPU work turns into
+39.9 ms of collective waiting, roughly 700 extra launches". That framing is superseded:
prediction adds **no net compute** — per-rank compute-only occupancy is uniform in both arms and
the absolute compute time is unchanged — so there is no small-GPU-cost-amplified-hugely story.
The launch term is a direct 8.3 ms of window that survives when every prediction collective is
removed.

It stays an *answer* ticket rather than an implementation one for two unchanged reasons. It
contradicts the eager execution the current scope mandates, so adopting it is a scope decision
the operator makes. And whether the predictive stream, its events and a plan that varies per
forward can be captured at all is **unverified** — this branch has already paid for building on
unverified premises more than once.

**Blocked by:** None (can start immediately). It can run in parallel with anything, and it
should start before 08 rather than after: 08's verdict would otherwise be measured against a
cost this ticket may show is removable.

**Status: partly answered from the code, 2026-09-06. Two of its five criteria no longer
need a run, and the prize it was written around is refuted.**

**Its prize is not there.** This ticket priced itself at "12.17 ms of mean TTFT, 8.3 ms of
prefill window", taken from ticket 11 by removing prediction's collectives and calling the
residual *launches*. An 8-rank attribution of the DP=8 profile measures the residual directly
instead of labelling it. Per forward, of the 8.14 ms prediction adds to the window:

```
device compute                     0.15 ms    1.8%
host dispatch                      0.69 ms    8.4%   <- all this ticket can remove
blocking cudaEventSynchronize      0.70 ms    8.5%   (sign flips per rank: skew, not cost)
gap / waiting                      6.61 ms   81.2%
```

So the first-order prize is **0.69 ms per forward, about 0.9-1.5% of mean TTFT**, not 52% of
prediction's cost. The earlier figure came from `sorted(glob(...))[0]` — **one rank**, dp0 — and
from counting only `cuda_runtime`, which excludes `cuLaunchKernelEx` and so misses every Triton
launch. On dp0 the blocking sync reads +1.97 ms; on dp5 it reads **-2.21 ms**. A quantity whose
sign depends on which rank you read is an earliness, not a cost — the same trap ticket 11 fell
into with the 9.3 us AllGather.

**Criterion 1 and 2 are answered by the dispatcher, without a run.** `CUDAGraphDispatcher.dispatch`
returns `CUDAGraphMode.NONE` when `num_tokens > max_cudagraph_capture_size`, and that cap
defaults to **512** on H100 (1024 only on data-center Blackwell). Every prefill forward in this
project's workload carries **~1024 tokens or more**. So at default settings a captured graph is
never replayed for a prefill forward at all, and TTFT is entirely prefill. The question "can a
graph capture the predictive stream and its events" does not arise on the path that matters,
because no graph runs there.

Forcing it to apply means capturing at >= 1024 and padding every prefill up to a captured size.
This branch has already measured what enlarging prefill work costs: raising
`max_num_batched_tokens` made mean TTFT **35% to 46% worse**. Padding to a captured shape is the
same harm on the same metric, and it would be paid on every forward to recover at most 0.69 ms
of dispatch.

**What is left of this ticket is one cheap arm, and it is not graph capture.** Dropping
`--enforce-eager` also enables `VLLM_COMPILE`, whose Inductor fusion reduces prefill kernel count
*without* any graph replay. That attacks the same 0.69 ms and does apply to prefill. It is worth
one interleaved arm and nothing more.

**Relationship to 18, which landed first.** Ticket 18's fused kernel takes prediction from about
2.7 added launches per source layer to 1, so it removes most of the 0.69 ms this ticket was
aiming at. Whatever 18 measures is subtracted from this ticket's remaining value, exactly as 18's
own text predicted.

**What ticket 18 left, measured 2026-09-06.** 18 landed and took prediction from about 2.7
added launches per source layer to 1, worth a median **2.6%** of stock TTFT. That is most of
what this ticket was aiming at, and it beat the 0.9-1.5% first-order dispatch ceiling -- so
some of the 81% gap does respond to launch count. What remains for this ticket is the ~1 launch
per layer 18 could not remove (`out.zero_()`), plus every op outside the prediction path, and
none of it can be replayed on a prefill forward at default settings.

**Recommendation: drop the graph-capture scope, keep one arm.** The prefill path is not
reachable by replay, forcing it there costs more than it returns, and the fusion route already
took the dispatch this ticket was priced on. Run the `--enforce-eager` arm for the Inductor
fusion, record it, and close the ticket either way. It does **not** replace ticket 03 or 18;
both landed and both are what made this ticket small.

**Status:** ready-for-agent (reduced scope: one `--enforce-eager` arm, then close)

- [ ] Whether a graph can capture a region containing the predictive stream and its events at
      all, with the failure mode recorded if not.
- [ ] Whether a plan that changes every forward can live inside a captured graph, or whether it
      forces a replay-with-updated-inputs shape, and what that costs.
- [ ] The launch count and the collective waiting inside real forward windows, captured against
      eager, so the size of the prize is a number. Compare against the term this ticket is aimed
      at: **8.3 ms of prefill window, 12.17 ms of mean TTFT**, measured by removing prediction's
      collectives and finding that much left over. A graph that does not move that term has not
      found the prize, whatever it does to the launch count.
- [ ] A recommendation with its evidence: whether this should replace the fusion work of ticket
      03, complement it, or be dropped. If it would replace it, say so plainly — ticket 03 is
      most of what would become unnecessary.
- [ ] **What it would take to adopt, and what it costs elsewhere**, since the answer feeds a
      scope decision rather than a merge. Eager execution is mandated by `spec.md`'s runtime
      scope and by the operator's own comparison basis; every figure this project has recorded is
      eager. State whether graph capture invalidates them, or only adds an arm.
