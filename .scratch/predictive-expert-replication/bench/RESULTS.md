# Ticket 00 measurement log

Every run is recorded here, including the invalid ones and why they are invalid.
Append a new dated section per session; never overwrite an earlier one.

---

## 2026-08-23 — first session

### Environment

| Item | Value |
| --- | --- |
| Node | 8x NVIDIA GeForce RTX 5090, 31.4 GiB each |
| Interconnect | PCIe Gen5 x16, **no NVLink** (`nvidia-smi nvlink -s`: "does not have or support Nvlink" on all 8) |
| Model | `/models/preset/Qwen/Qwen3-30B-A3B/v1.0`, BF16, 48 layers, 128 experts, top_k 8 |
| Server | **installed vLLM 0.27.1, not a build of this repository** |
| Config | DP=8, EP=8, TP=1, eager, `allgather_reducescatter`, `max-model-len` 4096, `gpu-memory-utilization` 0.90 |
| EPLB | recording-only: `num_redundant_experts=0`, `step_interval=1e9`, `log_balancedness=true` |

### A. Hardware probe — trustworthy

Deterministic, no server involved. Source: `results/hardware.json`.

| Quantity | Measured |
| --- | --- |
| One BF16 expert | **9.00 MiB** |
| Expert P2P transfer, idle | **176-177 us**, ~53.5 GB/s, uniform across all sampled pairs |
| HBM copy bandwidth | 1526 GB/s |
| Local expert weights per layer per rank | 144.0 MiB |
| Expert-weight read floor | **102 us** (empirical: MoE time at 1 token/rank) |

Per-layer time budget, one rank:

| tokens/rank | attention us | MoE us | MoE / read floor |
| --- | --- | --- | --- |
| 1 | 23.2 | 102.1 | 1.00 |
| 8 | 25.3 | 104.3 | 1.02 |
| 32 | 26.7 | 106.3 | 1.04 |
| 64 | 26.8 | 109.1 | 1.07 |
| 128 | 31.2 | 191.0 | **1.87** |
| 256 | 50.5 | 356.9 | 3.50 |
| 512 | 97.8 | 537.8 | 5.27 |

Overlap window versus the 176 us transfer:

| lookahead | window us | hidden |
| --- | --- | --- |
| 1 (attention only) | 26.8 | **15%** |
| 2 | 162.7 | **92%** |
| 3 | 298.6 | 100% |

KV ceiling at 0.90 utilization: 15.97 GiB/rank, 96 KiB/token, 174k tokens/rank
-> 170 / 85 / 56 / 42 sequences per rank at 1k / 2k / 3k / 4k context.

### B. Expert load imbalance — trustworthy, and the key positive result

3532 balancedness samples scraped from the server log, where
`balancedness = sum_layers(mean across ranks) / sum_layers(max across ranks)`,
so headroom is `1 - balancedness` by construction.

| Statistic | headroom |
| --- | --- |
| median | **44.4%** |
| p95 | 52.1% |
| max | 70.4% |

By load level: 44.8% at low load, 39.0% at medium, 44.4% at high. Stable, and far
above what sampling noise would produce, so **Qwen3-30B-A3B has genuine expert
routing skew**. This is unaffected by the concurrency contamination below, because
it is a per-step routing property rather than a latency measurement.

### C. Serving metrics — MEASURED BUT CONTAMINATED, do not cite

Two benchmark loops were alive at once, so all five runs overlap in time and the
throughput and TTFT figures reflect a shared server. Recorded for completeness only.

| start-end | dataset | conc | done/req | in tok/req | mean TPOT | p99 TPOT | mean TTFT | p99 TTFT | out tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14:08:57-14:10:21 | mt-bench | 64 | 128/128 | 87 | 80.32 | 81.73 | 1038 | 1160 | 768 |
| 14:09:06-14:14:27 | NuminaMath-CoT | 64 | 120/120 | 90 | 118.32 | 177.78 | 45367 | 152808 | 379 |
| 14:10:45-14:12:08 | InstructCoder | 64 | 128/128 | 158 | 80.61 | 82.09 | 399 | 1279 | 766 |
| 14:11:41-14:12:21 | mt-bench | 256 | 15/512 | 167 | 80.89 | 81.19 | 876 | 942 | 152 |
| 14:11:41-14:13:35 | InstructCoder | 256 | 77/120 | 143 | 80.11 | 80.48 | 462 | 508 | 535 |

All times in ms. The last two did not merely stop early: 43 of 120 and 497 of
512 requests **failed**, which is what two clients saturating one server looks
like from the client side. Both are rejected by `report.py`.

**The one robust signal in this table**: TPOT sits at 80-81 ms in every run
regardless of concurrency (64 or 256) and domain, and `mean_itl` equals
`mean_tpot` throughout. A compute-bound decode step would grow with batch size.
It does not, so the decode step is dominated by fixed per-step overhead. For
scale, 48 layers of attention plus MoE account for about 6.5 ms of that 80 ms.

### Defects in this session's harness, to fix before re-running

1. **Prompt length was never controlled.** `--hf-output-len` sets the decode
   length, but the stock CLI has no input-length control for HuggingFace datasets.
   Actual prompts were 87-167 tokens, while filenames claimed `p1024` and `p2048`.
   **Neither agreed request shape (2k/1k, 1k/2k) was actually tested.** Fix by
   pre-filtering each dataset to the target band offline and feeding the result
   through the custom-dataset path, or by using the length-controllable sonnet
   dataset for text.
2. **Runs were not serialized.** A previously launched loop was still alive.
3. **Result filenames encoded intent, not measurement.** They must be derived
   from the observed prompt length.
4. `Aeala/ShareGPT_Vicuna_unfiltered` fails against vLLM's own `--dataset-name hf`
   path under `datasets` 5.x (`TypeError: string indices must be integers`), so
   this session's text runs used `philschmid/mt-bench`. It works through
   `make_prompts.py`, which reads the dataset directly, and is the better text
   source because mt-bench has only 80 samples.

### Verdict from this session

**Do not start ticket 03.** Headroom is real at 44%, but at the concurrency this
node reaches, MoE is expert-weight-bandwidth bound (ratio 1.02 at 8 tokens/rank),
so that imbalance does not convert into time. Separately, TPOT is overhead-
dominated rather than MoE-dominated, which caps any MoE-level gain well below the
imbalance figure.

This is a statement about this operating point, not about the policy. Still to
measure: concurrency high enough to leave the weight-bound regime, the two agreed
request shapes with controlled prompt lengths, serialized runs, PCIe utilization
under load to replace the idle bandwidth in the cost profile, and a profile
attributing TPOT so the MoE share is measured rather than inferred.

Caveats: the boundness ratio is a `bmm` microbenchmark, not vLLM's fused MoE
kernel; and no tuned MoE config exists for this device, so the served kernel is
untuned and its absolute timing is pessimistic.

---

## 2026-08-23 — second session: tickets 01 and 02 validated on hardware

First run of **this repository's** predictive code on the 8-GPU node, rather than
the installed build. Weights loaded with `--load-format dummy`, which still
exercises config validation, EPLB provisioning, model construction and binding,
startup normalization, and every forward, without reading 57 GiB.

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /models/preset/Qwen/Qwen3-30B-A3B/v1.0 --load-format dummy --port 8101 \
  --data-parallel-size 8 --enable-expert-parallel \
  --all2all-backend allgather_reducescatter --enforce-eager \
  --max-model-len 2048 --gpu-memory-utilization 0.85 \
  --additional-config '{"predictive_expert_replication":{"enabled":true,
                        "cost_profile_path":"/tmp/predictive-cost-profile.json"}}'
```

| Criterion | Result |
| --- | --- |
| Config validation accepts the approved scope on a real worker | pass |
| Cost-profile fingerprint accepted, model/dtype/EP/device | pass |
| Startup normalization before readiness | pass, **218 ms** (267 ms on the first attempt) |
| Layout is 16 canonical + 1 inactive per rank | pass, as logged by normalization |
| Native EPLB placement suppressed | pass: exactly **one** rearrangement in the whole run, and it is the profile reservation. No `EPLB step:` lines, no non-profile rearrangement. |
| Profile-run buffer reservation still happens | pass, by design, since predictive transfers need those buffers |
| Forward completes with prediction active | pass: 43 source layers x 8 ranks, single request and 24 requests at concurrency 12, **0 failed** |
| No invariant violation or stranded pending plan | pass, no assertion or `RuntimeError` in the log |

`startup_normalization_ms` is reported separately and excluded from serving
latency, as the spec requires.

Two environment issues had to be solved first, neither caused by the feature:

1. The repo tree lacks the vendored `vllm/third_party/{deep_gemm,fmha_sm100,
   tml_fa4,triton_kernels}` and `flashmla/flash_mla_interface.py`, so warmup
   died with `No module named vllm.third_party.flashmla.flash_mla_interface`.
   Symlinked from the installed build and listed in `.git/info/exclude`, so the
   repository stays clean. Note this crashed *after* normalization succeeded.
2. The cost-profile fingerprint compares `model_config.model`, which is the
   filesystem path when serving from disk, not the HuggingFace name. The profile
   must pin the path actually served.

`torch.distributed.all_gather_into_tensor` emits a deprecation warning in favour
of `all_gather_single`. Left as is: four of vLLM's own communicators use the same
call, so changing one file would diverge from the tree.

### What this session does not establish

TPOT here is meaningless: dummy weights and 12-way concurrency. Overlap was still
not measured, and output equivalence cannot be tested with dummy weights. Both
need a real-weight run and belong to ticket 05.

---

## 2026-08-23 — third session: boundness sweep and the feasibility verdict

Real weights, this repository's code, `sweep_boundness.sh`. Short 128/128 shape,
chosen because the spec's 3072-token context caps decode at about 59 sequences per
rank (vLLM reports 181,040 KV tokens per rank), well short of the roughly 128 the
probe grid says is needed to leave the expert-weight-bandwidth-bound regime.

Concurrency here is `--max-concurrency`, the number of requests in flight, not a
rate: `request_rate` was left at `inf`, so RPS is an output. Sequences per rank is
concurrency divided by the 8 DP ranks.

| conc | seq/rank | p99 TPOT ms | out tok/s | req/s | completed |
| --- | --- | --- | --- | --- | --- |
| 128 | 16 | 79.7 | 1233 | 9.63 | 400/400 |
| 256 | 32 | 81.5 | 2395 | 18.71 | 400/400 |
| 512 | 64 | 84.6 | 4440 | 34.69 | 400/400 |
| 1024 | 128 | 83.7 | 4523 | 35.34 | 400/400 |
| 2048 | 256 | 82.9 | 4535 | 35.43 | 400/400 |

**The last two rows are nominal, not real.** Throughput is flat from concurrency
512 onward (4440, 4523, 4535 tok/s), so the server never ran 128 or 256 sequences
per rank; the extra requests queued. The reachable ceiling is about 64 per rank.

**Across the whole reachable range TPOT is flat**: 79.7 to 84.6 ms, +6%, while the
decode batch grew 4x. Measured headroom over the same run was **50.8%** median
across 1187 balancedness samples, so the imbalance is real and large.

### RETRACTED: this sweep never reached the regime it was built to test

The concurrency column above is nominal. Deriving the running batch from
`throughput x TPOT` gives 12, 24, 46, 46, 46 sequences per rank: identical from
concurrency 512 onward, because `--num-prompts 400` caps requests in flight at
400 no matter what `--max-concurrency` says. The sweep explored 12 to 46 per rank
and stopped there for that reason, not because of any hardware limit.

Two further caps were missed. `max_num_seqs` defaults to **128 per rank**, exactly
the threshold the probe grid puts the regime change at, so it must be raised to
explore past it. And KV at a 256-token context allows roughly 700 per rank, so
capacity was never the binding constraint here.

**The verdict below is therefore not supported by this data and is withdrawn.**
It is kept only to record what was claimed and why it was wrong.

### ~~Feasibility verdict: negative for this configuration~~ (withdrawn)

At 64 sequences per rank the probe puts MoE at about 109 us per layer, so 48
layers is about **5.2 ms of a 84.6 ms TPOT — 6.2%**. Eliminating *all* imbalance
in *every* layer would save at most `0.508 x 5.2 ms` ≈ **2.6 ms, or 3.1% of
TPOT**, and one replica of one expert per layer captures only a fraction of that.

So the imbalance does not convert into time here, because the decode step is
dominated by something other than MoE compute. Tickets 07, 08, 04 and 05 should
not proceed on this configuration.

### What this does not establish

This is a conclusion about this node and this configuration, not about the policy.

- **Eager execution is spec-mandated**, and eager launch overhead is the most
  likely occupant of the other 78 ms. Under CUDA graphs TPOT would fall and MoE's
  share would rise — though the predictive stream and event machinery may not be
  graph-capturable, which is itself worth settling before rejecting the policy.
- **No tuned MoE configuration exists for this device**, so the served MoE kernel
  is slower than it should be. That *overstates* MoE's share, meaning the real
  figure is likely below 6.2%.
- One domain (code) and one shape (128/128). Per-domain and per-shape breakdowns
  are still unmeasured.
- The two request shapes the spec fixes were **not** used here, deliberately: they
  cannot reach the regime under test.

### Still unmeasured from ticket 00

- Fixed-RPS TPOT and maximum sustainable RPS under a p99 TPOT SLO. Both need an
  open-loop run with a request rate set; every run so far has been closed-loop
  concurrency.
- Per-domain and per-request-shape imbalance rather than pooled.
- Per-layer interconnect utilization and usable transfer bandwidth under load. The
  cost profile still carries the idle figure.
- Peak memory and KV block count from a no-EPLB run.
- The uniform-routing control.

---

## 2026-08-23 — fourth session: the two agreed request shapes

The first pass at the shapes the spec actually targets. Real weights, this
repository's code, `--ignore-eos` so decode length is set rather than capped, and
`--num-prompts 400` so concurrency is never capped by the prompt count.

Each shape is paired with the dataset whose natural length already sits near its
prompt target, rather than trimming an ill-fitting one into shape: `1024->2048`
with InstructCoder (mean 989 tokens, 5% trimmed) and `2048->1024` with ShareGPT
(mean 2043). Pairing this way cut truncation on the 1k prompts from 88% to 5%.

Regenerate with `python3 bench/table.py`.

| shape | dom | conc | real batch (per rank) | done | in | out | mean TPOT | p99 TPOT | mean TTFT | p99 TTFT | p99 e2e | tok/s | req/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024->2048 | code | 64 | 57 (7) | 400/400 | 1006 | 2048 | 78.8 | 79.1 | 732 | 1739 | 163373 | 723 | 0.35 |
| 1024->2048 | code | 192 | 134 (17) | 400/400 | 1006 | 2048 | 80.7 | 81.3 | 1462 | 3577 | 169188 | 1660 | 0.81 |
| 1024->2048 | code | 384 | 203 (25) | 400/400 | 1006 | 2048 | 82.2 | 83.4 | 3676 | 7054 | 173489 | 2469 | 1.21 |
| 2048->1024 | text | 64 | 57 (7) | 400/400 | 2060 | 1024 | 79.9 | 80.9 | 1065 | 2417 | 84072 | 709 | 0.69 |
| 2048->1024 | text | 192 | 133 (17) | 400/400 | 2060 | 1024 | 83.3 | 84.9 | 2536 | 7095 | 93224 | 1597 | 1.56 |
| 2048->1024 | text | 384 | 200 (25) | 400/400 | 2060 | 1024 | 86.2 | 90.3 | 7035 | 13963 | 97622 | 2327 | 2.27 |

Times in ms. All 400/400 completed with no failures and average lengths on target.

**Real batch is derived from `throughput x mean TPOT`, not read off the
concurrency flag.** At concurrency 384 it is only 200, about 52% of nominal,
because `max_num_batched_tokens` defaults to 2048 while prompts are 1024 or 2048
tokens, so chunked prefill competes with decode for the per-step budget. The trend
is intact — 57, 134, 200 — but concurrency and decode batch are not proportional,
and `--max-num-batched-tokens 8192` would be needed for them to be.

**EP rank imbalance: headroom median 47.4% over 36,936 samples**, measured on
these shapes rather than on the short-context sweep. This is the ceiling on any
placement policy's gain.

### The shape of the result

Raising concurrency 6x barely moves TPOT but multiplies TTFT:

| | conc 64 -> 384 |
| --- | --- |
| p99 TPOT, 1024->2048 | 79.1 -> 83.4 ms, **+5%** |
| p99 TPOT, 2048->1024 | 80.9 -> 90.3 ms, **+12%** |
| p99 TTFT, 1024->2048 | 1739 -> 7054 ms, **+306%** |
| p99 TTFT, 2048->1024 | 2417 -> 13963 ms, **+478%** |

Added load queues rather than making each decode step more expensive, which is
consistent with the decode step being dominated by a fixed per-step cost.

### What this implies, and what is still not established

At 25 sequences per rank the probe puts MoE at about 106 us per layer, so 48
layers is roughly 5.1 ms of an 83.4 ms TPOT: **6.1%**. Balancing every layer
perfectly would save at most `0.474 x 5.1 ms` ≈ **2.4 ms, or 2.9% of TPOT**, and
one replica per layer captures a fraction of that.

No verdict is drawn from this. The arithmetic rests on subtracting a local
microbenchmark from measured TPOT, which leaves about 88% of TPOT unattributed,
and the TTFT explosion suggests the binding constraint may be scheduling rather
than anything inside the decode step. Those are different problems with different
remedies, and only a per-operation profile separates them. That profile is
recorded as a deferred item on ticket 00.

Also still unmeasured: per-domain imbalance (each shape here has one domain),
fixed-RPS and maximum sustainable RPS under an SLO (every run has been closed-loop
concurrency), interconnect utilization under load, peak memory and KV blocks from
a no-EPLB run, and the uniform-routing control.

---

## 2026-08-23 — ticket 03 output equivalence: PASS

Verified on the 8-GPU node with real weights, using
`verify_source_rank_routing.sh` and the startup weight-equality assertion.

| Evidence | Result |
| --- | --- |
| Replica weights vs canonical | **48 of 48 layers byte-identical** (`compared 48 replicated (layer, expert) pairs`) |
| argmax of the first token, 4 prompts | unchanged |
| 96 greedy tokens of text | identical |
| tail logprobs, non-argmax candidates | up to 0.355 apart |

Because every physical copy of a logical expert is provably identical, routing a
source rank to either copy is arithmetically equivalent, so the only thing that
can differ is the order of summation. The distribution shift is therefore
reduction-order noise by construction, not a semantic change, and the criterion
is met.

### Two things this took to establish, both worth keeping

**A control run was necessary.** The first comparison failed on text equality, and
the obvious explanation — cross-process nondeterminism — turned out to be wrong:
running the *canonical* configuration twice produced byte-identical output. So the
difference really was caused by the replica, and only then did the regrouping
explanation become the candidate. Without the control the failure was
uninterpretable in either direction.

**A tolerance was not enough, and the one used was invented.** The 0.05 logprob
tolerance had no calibration behind it; the observed logprobs sit on a visible
BF16 grid with about 0.0125 to 0.03 spacing, and 0.355 is 12 to 28 steps of that.
Nothing in that arithmetic decides whether the shift is noise or a defect. The
weight-equality assertion does, which is why it is now a permanent startup check
rather than a one-off diagnostic. Ticket 07 needs the same guarantee after a real
transfer and can reuse it.

The assertion reports how many replicated pairs it compared. With no placement it
compares nothing and says so: a vacuous pass reads exactly like a real one, and an
earlier version of this check would have passed without a replica present.

### Operational note

`kill -9` on the API server does not take its workers or `VLLM::DPCoordinator`
with it, and `nvidia-smi --query-compute-apps` only lists processes holding device
memory, so it reports zero while helpers still run. Three launches were lost to
this: the first died silently with no traceback, and only the third surfaced the
real cause as `Free memory on device cuda:7 (3.42/31.36 GiB)`. Clean up by pattern
against `VLLM::` and re-check until nothing remains.

---

## 2026-08-23 — fifth session: per-layer load concentration

Measures how the imbalance is *shaped*, which decides how many experts a placement
policy has to touch. Instrumented with `VLLM_EPLB_DUMP_LOAD_PATH`, which appends
per-layer per-logical-expert load for each forward. 133 samples, code domain,
1k prompts, concurrency 64, EPLB recording only with no rearrangement.

| Quantity | Median | p95 | Uniform would be |
| --- | --- | --- | --- |
| Critical-path headroom (Σ per-layer peak vs Σ per-layer mean) | **41.1%** | | 0% |
| Within a layer, share of the peak rank's load to shed to equalize | **39.6%** | 57.3% | 0% |
| Within a layer, hottest expert's share of the peak rank's load | **33.6%** | 56.7% | 6.2% |
| Within a layer, top three experts' share | **68.5%** | | 18.8% |

The skew is strong: one expert carries 33.6% of its rank's load where an even
split would be 6.2%, and three carry 68.5% against 18.8%.

### This invalidates a plan made from a wrongly aggregated version of it

A first pass summed the layer axis before comparing ranks. That gives 1.10x peak
versus mean and suggests only 9.2% needs shedding. Both are wrong for this purpose:
each layer is its own collective and waits for its own slowest rank, so the critical
path is the sum of per-layer peaks, and different layers peak on different ranks, so
summing first lets them cancel. The correctly aggregated figure is 39.6%, not 9.2%,
and the hottest expert's share is 33.6%, not 11.8%.

Both metrics are now named in the glossary, since only one of them can inform a
placement decision.

### Consequence: replicating one expert cannot equalize a layer, at any fan-out

Spreading a single expert of share `s` across `K` target ranks sheds at most
`s x K/(K+1)`, which is strictly below `s`. With `s = 33.6%` against a need of
39.6%:

| Fan-out K | Sheds at most | Enough for 39.6%? |
| --- | --- | --- |
| 1 | 16.8% | no |
| 2 | 22.4% | no |
| 4 | 26.9% | no |
| 7 | 29.4% | no |

Even moving that expert's entire load away sheds 33.6%, still short. So the
limit is not the fan-out and not the chunk granularity: **one expert per layer is
not enough**, and at least two experts must be touched. Chunk granularity itself is
ample — the subset sums of 8 source-rank chunks give 256 achievable split points.

This is the evidence for choosing between fanning one expert out to many ranks and
replicating several different experts. It favours the latter. The decision is open.

---

## 2026-08-24 — sixth session: the decode step, attributed. Ticket 00 answered

The decisive item of ticket 00 was "MoE's share of TPOT, attributed rather than
inferred". It is now measured at four operating points, and the answer is
consistent and negative. The mechanism behind it is identified, which matters
more than the numbers, because it says exactly what would have to change.

Captured with `run_boundness_profile.sh`: one server, one wave of requests per
point (so a prefill burst is followed by a long clean decode phase), a four
second profile window well inside that phase, code prompts at 1024 tokens
decoding 2048 with `--ignore-eos`, `--enforce-eager`, DP=8, EP=8,
`allgather_reducescatter`, no NVLink. Parsed by `parse_profile.py`.

**Corrected 2026-08-24, after review.** The first version of this table divided
by attributed GPU time only, and called the remainder profiling inflation. It is
the opposite: attributed time is 49% to 67% *of* TPOT, so a third to a half of the
step is unattributed, and every share quoted against attributed time is larger
than the same share of a whole step. Both denominators now appear. The share
column was also a mean of per-rank ratios while its neighbours were means of
values, which disagreed by 1.7x at c8; it is now a ratio of the means.

| Concurrency | ranks | gen tok/step | MoE us/layer (mean/peak) | NCCL us/layer | MoE ms/step | attributed ms | TPOT ms | MoE/attributed | **MoE/TPOT** | MoE imbalance | Recoverable ms | **Recoverable/TPOT** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | **5/8** | 1 | 36.0 / 38.2 | 1114.2 | 1.73 | 58.74 | 120.25 | 2.94% | 1.44% | 1.060x | 0.104 | 0.09% |
| 64 | 8/8 | 5.4 | 60.2 / 62.3 | 1372.6 | 2.89 | 74.01 | 122.36 | 3.91% | 2.36% | 1.035x | 0.100 | 0.08% |
| 192 | 8/8 | 16.1 | 86.2 / 90.1 | 1450.4 | 4.14 | 79.61 | 123.07 | 5.20% | 3.36% | 1.046x | 0.188 | 0.15% |
| 384 | 8/8 | 32.2 | 94.2 / 98.1 | 1519.6 | 4.52 | 85.25 | 127.12 | 5.30% | 3.56% | 1.042x | 0.189 | 0.15% |

**c8 covers only 5 of 8 ranks** (dp2, dp3, dp4 produced no decode-only window). A
step waits for its slowest rank, so a missing rank can only lower the observed
peak — biasing that row's imbalance and recoverable figures downwards, in the
direction that flatters this verdict. It is not the headline; c384, with all eight
ranks, is.

The TPOT column is not an unprofiled reference: it is measured by the same client
run the profile window sits inside. It bounds a profiled step.

"Recoverable" is an upper bound by construction: a decode step waits for its
slowest rank, so the expert GEMM contributes the *peak* rank's time and perfect
balance would reduce that to the mean, no further. A real policy gets a fraction
of it.

### Why the imbalance does not convert: block quantization

Token imbalance is 1.66x (39.6% of the peak rank must shed to equalize). MoE
*time* imbalance is 1.04x. The gap is not damping, it is quantization.

`moe_align_block_size` pads each expert's token list up to a multiple of
`BLOCK_SIZE_M`, and the bf16 default tiers are 16 for M<=32, 32 for M<=96, 64 for
M<=512, 128 above. With 128 logical experts and top-8 routing, tokens per expert
is `M/16` where M is the post-allgather token count:

| Concurrency | M | BLOCK_SIZE_M | tokens/expert | blocks/expert |
| --- | --- | --- | --- | --- |
| 8 | ~8 | 16 | 0.5 | 1 |
| 64 | ~43 | 32 | 2.7 | 1 |
| 192 | ~129 | 64 | 8 | 1 |
| 384 | ~256 | 64 | 16 | 1 |

Every touched expert costs exactly one block at every reachable point. An expert
holding 3 tokens costs the same as one holding 12, so imbalance below the block
size is free and balancing it saves nothing.

Imbalance starts costing only when tokens per expert exceeds `BLOCK_SIZE_M`,
which needs M > 2048 concurrent decode tokens. At a 3072-token context this node
serves about 464. **It is short by a factor of 4.4 and cannot reach the regime.**

### Two of this session's own earlier figures were wrong

Both were produced this session and are corrected here rather than left standing.

- A per-step attribution divided the MoE total by `fused_moe_kernel` call count.
  That kernel fires twice per layer, once per projection, so every per-step
  figure was half. `parse_profile.py` now takes the step count from the runner's
  annotation and a test pins it.
- The first profile window mixed prefill into the decode attribution. A
  2047-token prefill costs about 300 ms against a 136 ms decode step. Windows are
  now `execute_context_0(0)_*` only, and a test pins that too.

The extrapolation made from the contaminated data - that concurrency 384 would
show a 24% MoE share and about 7% recoverable - was wrong by more than an order
of magnitude. Measured: 5.69% share, 0.22% recoverable.

### The 102 us expert-weight read floor does not describe the real kernel

Ticket 00 recorded a floor of 102 us for reading 144 MiB of local experts per
layer, from a bmm microbenchmark. The real kernel costs 36 us/layer at one token
per rank, well below that floor, because it only reads the experts that have
tokens: an expert with no block is never loaded. The boundness ratio derived from
the proxy therefore does not transfer, and the measured curve above replaces it.

### What the collectives are, and what they are not

NCCL is 1114 to 1520 us per layer, 20x the expert GEMM. It is *not* redundant
expert traffic - no replication code ran, `num_redundant_experts` was 0. It is
`allgather_reducescatter` token dispatch and combine, forced by the absence of
NVLink.

Its byte volume cannot explain it: about 512 KiB per layer at 54 GB/s is ~10 us
against ~1300 us measured. So it is latency and arrival skew, not bandwidth. At
concurrency 192 the per-rank NCCL spread is 632 to 1746 us/layer while the MoE
spread is 78 to 90, and the two do not track each other. Whatever that time is,
expert placement cannot move it, and the bound on what placement *can* move is
set by the MoE spread alone.

### Recommendation: stop before building the transfer machinery on this node

The ceiling is **0.08% to 0.15% of a decode step** (0.13% to 0.24% if quoted
against attributed GPU time only, which is what the first version of this section
did), against a transfer costing 175 us per expert. This does not justify tickets 07, 04, or 05 here.

The limiting factor is **kernel block granularity**, not the policy, not the
interconnect, and not prediction accuracy. That is a conclusion about this
model shape and this operating range, and it names what would have to change:

- Tokens per expert above `BLOCK_SIZE_M`. Needs M > 2048 concurrent decode
  tokens: more KV per rank, shorter contexts, or a model with fewer, larger
  experts. Qwen3-30B-A3B's 128 experts at top-8 is the worst case for this.
- A tuned config with a smaller `BLOCK_SIZE_M` at large M would move the
  threshold down. Note there is no tuned `E=128,N=768` file for H100, and the
  H200 file that exists uses 128 at M>=1024, so tuning does not obviously help.

### What this does not establish

- 8x H100 was not measured. HBM3 and 80 GB raise the reachable M by roughly 4x,
  to about 1800 at this context, still under the 2048 threshold. NVLink shrinks
  the collectives and so raises MoE's *share*, which helps, but the block
  quantization argument is unchanged by either. The conclusion could still
  invert on a different model shape; it is unlikely to invert on hardware alone.
- Ticket 06 is unmeasured. Prediction accuracy is irrelevant to this verdict:
  even a perfect prediction can only recover the MoE spread.
- A third to a half of the step is **unattributed** — idle, host-side, or in
  kernels outside the four classified buckets. That is the opposite of what this
  section first claimed ("the profile inflates a step"): attributed GPU time is
  *below* TPOT, not above it. Shares against attributed time are therefore upper
  bounds on shares of a real step, and both are now tabulated.
- `ruff` and `pre-commit` are not installed on this node. Only the 88-character
  limit was checked, by hand.

### Harness defect found and fixed: the wrong vLLM was being measured

`vllm serve` is a console script, so `sys.path[0]` is `/usr/local/bin` and the
current directory is never on the path. It therefore loads the *installed* vLLM,
which is stock 0.27.1 and does not even contain `distributed/eplb/predictive.py`.
An entire prediction-accuracy run produced six empty dumps this way, reporting
nothing but an "Unknown vLLM environment variable" warning.

`python3 -m vllm.entrypoints.openai.api_server` does put the current directory on
the path, which is why `verify_source_rank_routing.sh` worked and why tickets 01
to 03 remain validly validated. Both runner scripts now export `PYTHONPATH` and
**abort** unless the loaded vLLM resolves inside this tree.

A second defect in the same area: `vllm serve` starts one API server per DP rank
and each rebuilds the config from serialized engine args. The predictive wiring's
`num_redundant_experts` survives that round trip but its `enable_eplb` does not,
so every rank dies on "num_redundant_experts is set to 8 but EPLB is not
enabled". The accuracy runner uses the single-API-server module form.

---

## 2026-08-24 — ticket 06: the cross-layer gate, measured

First measurement of the feature's core hypothesis: how well the current MoE's
hidden states, run through a later MoE's gate, predict that later layer's load.

Captured by `run_prediction_accuracy.sh`. One server per lookahead, because the
lookahead is a launch-time setting. 32 requests at concurrency 32, decoding 128
tokens with `--ignore-eos`; the two agreed shapes supply the prompt side (code at
1024 tokens, conversational text at 2048), and accuracy is sampled once per
forward, so 135 to 163 forwards gives 5500 to 5900 scorable layer pairs per run.
Scored offline by `prediction_accuracy.py`.

### Recall at 2, by lookahead

Recall at 2 is the operative figure: the planner replicates
`max_replicas_per_layer` distinct experts, which defaults to 2.

| domain / shape | L=1 | L=2 | L=3 |
| --- | --- | --- | --- |
| code p1024 | 0.828 | 0.781 | 0.726 |
| text p2048 | 0.793 | 0.740 | 0.696 |

### Count error (total-variation distance between load shares)

| domain / shape | L=1 | L=2 | L=3 |
| --- | --- | --- | --- |
| code p1024 | 0.0882 | 0.1204 | 0.1450 |
| text p2048 | 0.0829 | 0.1109 | 0.1344 |

Degradation is monotone and close to linear: about five points of recall and
0.03 of count error per layer of distance. Code predicts three to four points
better than conversational text at every distance, which is the content
dependence the shapes were chosen to expose.

Gate-logit similarity is deliberately absent. A high logit similarity can still
reorder the selected top-k and mispredict load, so it would flatter the
prediction without answering the question. `peak_hit_rate` is identical to recall
at 1 by construction and a test pins that, so the two are never read as separate
evidence.

### `prediction_lookahead_layers = 2`: confirmed

Lookahead 1 predicts about five points better but hides only 15% of a 9.00 MiB
transfer behind one Attention block. Lookahead 2 hides 92% and costs those five
points. Lookahead 3 costs five more and buys nothing, because 92% was already
enough. The default stands on measurement now rather than assumption.

### `prediction_skip_first_layers = 3`: confirmed, marginally conservative

The six runs above all used skip 3, so layers 0 to 2 were never sources and the
per-layer curve began at target layer 4 — those runs could not evaluate the
default they were supposed to justify. A seventh run with **skip 0** at lookahead
2 supplies the missing part, covering target layers 2 to 47:

| target layers | mean recall at 2 |
| --- | --- |
| 2 | 0.028 |
| 3 | 0.262 |
| 4 | 0.716 |
| 7-11 | 0.763 |
| 12-46 | 0.735 to 0.814 |
| 47 | 0.449 |

Targets 2 and 3 are unusable and target 4 onward is normal. At lookahead 2 those
two targets come from source layers 0 and 1, so **skip 2 is what the unambiguous
part of the evidence supports**, and the default 3 is one layer conservative.

**Corrected 2026-08-24, after review.** The first version of this section pooled
the per-layer curve across lookaheads and reported "leading layers below
tolerance: none". That was biased: recall falls monotonically with distance, and
the earliest target layers are reachable only by the shortest-distance runs -
target 4 by L=1 alone, target 5 by L=1 and 2 - so pooling lifted exactly the
layers a skip decision turns on. Computed per lookahead instead:

| run | target layers | stable median | leading layers below tolerance |
| --- | --- | --- | --- |
| skip 0, L=2 | 2-47 | 0.750 | **2, 3** |
| skip 3, L=1 | 4-47 | 0.825 | 4, 5, 6 |
| skip 3, L=2 | 5-47 | 0.770 | 5 |
| skip 3, L=3 | 6-47 | 0.701 | none |

Magnitude decides how to read those. Layers 2 and 3, at 0.028 and 0.262 against a
median of 0.750, sit twenty to thirty tolerance-widths low - categorical. The
layers flagged in the skip-3 rows sit about 0.06 low, which is inside this curve's
own noise (layer 5 dips to 0.526 and layer 11 to 0.650 between neighbours near
0.75), so they should not move a default. Target 4 at 0.716 is itself marginal
against a 0.75 to 0.83 median, which makes **keeping the default at 3 the safer
reading** and lowering it to 2 a defensible one - not the other way round, as the
first version implied.

The curve is never pooled across lookaheads any more; `by_layer_curve` raises
rather than allowing it, and the tool reports one curve per lookahead.

**Not established: whether early-layer badness belongs to the source or the
target.** Targets 2 and 3 are simultaneously the earliest targets and the ones
fed by the earliest sources, and this run cannot separate those. Holding the
target fixed and varying the distance does show the expected monotone decline
(target 6: 0.768 / 0.720 / 0.681 at L=1/2/3; target 10: 0.875 / 0.864 / 0.799),
but that is the lookahead effect, not the attribution. The experiment that would
settle it is skip 0 at lookahead 1, which reaches target 2 from source 1. It was
not run because both attributions imply the same action — skip the leading source
layers — and the GPU time went to ticket 03's outstanding observation instead.

A second unexplained observation: target layer 47, the last, scores 0.449. Its
source (45) is a legal source, so if this holds up the trailing layers may
deserve exclusion too. One layer, pooled over two runs, so it is an observation
and not a recommendation.

### This does not change ticket 00's verdict, and was not expected to

Prediction accuracy bounds how much of the *available* MoE time imbalance a
policy can capture. On this node that imbalance is worth 0.08% to 0.15% of a
decode step, so even a perfect gate recovers a fraction of a fraction. Ticket 06
was run because it is cheap, unblocked, and its result is what transfers: to the
Ascend port, and to any model shape whose experts are few enough that balance
converts into time at all.

### Sample-size caveat

Per-layer figures pool two runs of 135 to 163 forwards, so a single layer rests
on a few hundred samples and the curve is visibly noisy — layer 5 dips to 0.526
and layer 11 to 0.650 between neighbours near 0.75. The bucketed rows and the
overall figures are the ones to quote; individual layers are indicative.

---

## 2026-08-24 — ticket 03 complete: an inactive slot attracts no tokens

The last outstanding criterion of ticket 03, verified under real requests by
`verify_inactive_slots.sh`.

```
Predictive expert replication normalized ... to 16 canonical + 1 inactive rows
  per rank in 693.03 ms.
Predictive expert replication: verified 384 inactive physical slots carry no
  routed load.
Successful requests: 32     Mean TPOT (ms): 94.58
```

384 is 48 layers times 8 ranks, one inactive row each, so every slot the layout
reserves was examined.

The check raises on the forward that violates it rather than reporting afterwards:
an inactive slot's weights are never written, so a token routed there reads
uninitialised memory and produces plausible-looking output instead of an error.
It also refuses a layout that marks no slot inactive, so it cannot pass
vacuously — the same failure mode `verify_replica_weight_equality` was hardened
against, and which that check honestly reported in this run ("compared 0
replicated pairs ... this check is vacuous", since no static replica was placed).

### A deadlock this check caused, and the shape that prevents it

The first attempt hung the engine. All eight EngineCores sat in
`No available shared memory broadcast block found in 60 seconds`, with no error
and no tokens returned.

Cause: the check was gated on `not is_dummy`, and it performs an all-reduce.
`is_dummy` differs across DP ranks — an idle rank runs a dummy batch to stay in
lockstep — so some ranks entered the collective and others skipped it.

`_dump_logical_expert_load` never had that guard, which is exactly why it was
safe. All three diagnostics now dispatch from `_run_step_diagnostics()`, which
takes **no arguments**, and a test asserts that signature. With nothing per-rank
available to branch on, the collectives cannot diverge again.

The lesson generalizes beyond this check: **any per-forward diagnostic that
performs a collective must run on every rank unconditionally.** Gating one on
per-rank state deadlocks silently, which is worse than crashing.

---

## 2026-08-25 — the goal moved to prefill and TTFT, and prefill was measured

The operator's goal changed: decode gains need a batch this node cannot reach, so
the target is now prefill and TTFT. That is a different regime and it needed its
own measurement — everything said about prefill before this session rested on 6
forwards from one domain that happened to be on disk.

Captured by `run_imbalance_survey.sh`: native EPLB recording (no replication in
the loop, so this is the baseline a policy would act on), three domains at 120
requests each, 2048-token prompts, two chunked-prefill budgets. 1705 forwards.
Analysed by `bench/imbalance.py`, which has 17 tests and no code path that sums
layers before comparing ranks.

### Prefill imbalance is real and holds across domains

| Domain | per-layer critical path | aggregated | noise floor |
| --- | --- | --- | --- |
| text | 1.5773x | 1.1212x | 1.0112x |
| code | 1.5688x | 1.0831x | 1.0112x |
| math | 1.5045x | 1.1105x | 1.0112x |

Far above the multinomial floor at 1024 tokens per expert, so this is routing
behaviour and not sampling. Note the aggregated column again: the same data reads
1.08 to 1.12x if the layers are summed first.

### The binding constraint is the transfer budget, and its default was sized for decode

Within one forward, with perfect prediction — the ceiling any policy could reach:

| transfers / forward | critical path | excess removed | MoE time saved | of prefill (5090) | of prefill (H100 SXM) |
| --- | --- | --- | --- | --- | --- |
| **4 (default)** | 1.4975 | 9.8% | 3.5% | **1.0%** | **2.4%** |
| 8 | 1.4557 | 17.4% | 6.2% | 1.9% | 4.3% |
| 16 | 1.3918 | 29.0% | 10.3% | 3.1% | 7.2% |
| **48 (one per layer)** | 1.2736 | 50.4% | 17.9% | **5.4%** | **12.5%** |
| 96 | 1.1962 | 64.4% | 22.9% | 6.9% | 16.0% |

At the default of 4 the ceiling is 1% of prefill and not worth building. At 48 it
is 5.4% here and 12.5% on an NVLink machine, and 48 is affordable in prefill:
432 MiB per forward against the 6 GiB of activations the collectives already move
(+7%), and 8.5 ms serialized against a ~200 ms forward (4%). The default of 4 was
right for decode, where one transfer costs 177 us against a 90 us layer; a prefill
layer is 1250 us.

The H100 SXM column is arithmetic from MoE's share of a prefill step, not a
measurement.

### Placements do not survive between forwards, and stale ones do harm

| B=96 | critical path | excess removed |
| --- | --- | --- |
| planned and used in the same forward | 1.1962 | **64.4%** |
| planned on another forward, same domain | 1.4185 | 24.2% |
| planned on another forward, another domain | 1.6623 | **-20.0%** |

Cross-domain gets monotonically *worse* as the budget grows (1.7118 at B=144):
more placements means more wrong placements, and a replica still splits its
expert's source ranks whether or not that expert is still hot. So the lifecycle
cannot accumulate coverage across forwards under mixed traffic — the policy must
re-plan every forward, and `min_residency_steps` / `hot_stable_steps` do not carry
the meaning they have in decode.

Per-layer peak-rank stability measured directly agrees: mode share 1.00 within
code, 0.83 within math, **0.33 within text**, and 0.39 pooled across the three.
An earlier 93% figure came from a single domain that happened to be the stable one.

### Raising the chunked-prefill budget hurts TTFT — the opposite of the hypothesis

| Domain | budget | mean TTFT | p99 TTFT | mean TPOT | output tok/s |
| --- | --- | --- | --- | --- | --- |
| text | 2048 | 1358.3 | 2728.1 | 83.52 | 635.9 |
| text | 8192 | 1895.6 | 3053.3 | 80.47 | 632.0 |
| code | 2048 | 1168.0 | 2432.8 | 78.73 | 685.1 |
| code | 8192 | 1703.4 | 2594.4 | 75.75 | 678.4 |
| math | 2048 | 1140.2 | 2348.5 | 79.04 | 684.3 |
| math | 8192 | 1533.5 | 2369.6 | 74.54 | 697.9 |

Mean TTFT is 35% to 46% **worse** at 8192, p99 worse too, TPOT 3 to 6% better,
throughput flat. The mechanism explains it: at 2048 one prefill per step pipelines
first tokens out at steps 1, 2, 3, 4 — mean 2.5 step-times; at 8192 four prefills
share one step four times as long, so all four wait 4 step-times. A bigger chunk
turns first tokens from a pipeline into a batch.

It also does nothing for the feature: at 8192 tokens per expert rises to 4096 and
blocks per expert to 32, and the recoverable fraction is unchanged (51.5% at
budget 48 against 50.4%). Quantization was already open at 8. **Raising
`max_num_batched_tokens` should be dropped from the plan on both counts.**

### Errors in this session's own analysis, corrected

- **M is the post-allgather token count**, and taking it as one rank's scheduler
  budget put prefill at 1 block per expert instead of 8. The confusion was
  prolonged by reading the first forward of a run, which is a single-rank warmup
  step where the two coincide.
- **A first claim that prefill imbalance was 1.025x** aggregated the layers before
  comparing ranks — the error this branch had already made once and documented.
  Per layer it is 1.55x.
- **A claim that the hottest expert on the peak rank beats the largest predicted
  reduction** was measured without the positive-benefit test, so it was free to
  overshoot. Gated, hottest-first starves: a budget of 96 finds 3 placements and
  removes 17.1% where gated largest-reduction removes 64.4%. Written into the spec
  and withdrawn one commit later.
- A first out-of-sample figure of -5% used that starved planner. The correct
  figures are +24.2% same-domain and -20.0% cross-domain.

### What this does not establish

- **Prediction accuracy in prefill.** Ticket 06 measured it on runs whose forwards
  were overwhelmingly decode. The payoff above is an oracle bound; the realistic
  figure is that times prefill prediction accuracy, which nobody has measured.
- **H100 SXM.** Its column is arithmetic. And SXM is not optional: H100 PCIe has
  no NVSwitch, so the 8-way collective still crosses PCIe and MoE's share of a
  prefill step stays near the 5090's 30%.
- The b8192 imbalance figures rest on **2** prefill forwards — they agree with
  b2048 but cannot stand alone.

### Correction: the leading layers cannot receive a placement

The budget table above allowed placements on all 48 layers. Only **43** can receive
one: a target sits `prediction_lookahead_layers` after its source, sources start
after `prediction_skip_first_layers`, so with the defaults (2 and 3) targets run 5
to 47 and layers 0 to 4 are unreachable. Those five carry 9.2% of the excess —
close to their uniform 10.4% share, which is why the penalty is small.

| transfers / forward | all 48 layers | reachable 43 only | overstated by |
| --- | --- | --- | --- |
| 4 | 9.8% | 9.8% | 0.0 pt |
| 8 | 17.4% | 17.3% | 0.1 pt |
| 16 | 29.0% | 28.3% | 0.7 pt |
| **48** | 50.4% | **47.4%** | 3.0 pt |
| 96 | 64.4% | 60.5% | 3.9 pt |

So the reachable ceiling at one placement per layer is **47.4% of the excess, 16.9%
of MoE time, 5.1% of a prefill step here and 11.8% on an NVLink machine.** At budget
4 nothing changes, because the four strongest candidate layers already lie in range.

The policy comparisons are unaffected: uniform against global, and hottest-first
against largest-reduction, each ran both arms over the same layer set.


### Correction (2026-08-25, later the same day): the per-rank view was on the wrong scale

Every prefill figure above is superseded. The dump field named `rank_load` was
taken from the **un-reduced** `expert_load_pass` while the logical view was
all-reduced over the EP group. Recording happens in the router on each rank's own
tokens, so the un-reduced tensor is *"how one rank's tokens spread across the
ranks"* — a different quantity, smaller by a factor of the EP size. Read together
with the logical view it put the two on scales eight times apart, which is why a
replicated expert appeared to hold 181% of its own rank's load.

The self-consistency check that should have caught this ran on the **first forward
of the run**, a warmup step in which only one rank holds tokens; there the
all-reduce sums seven zeros and the two views coincide, so the check passed
vacuously. That is the same failure mode `verify_replica_weight_equality` and
`verify_inactive_slots_unused` were both hardened against, and the second time a
warmup forward has misled this analysis.

Fixed: one all-reduce now feeds both views, `_logical_from_reduced` is a pure
static method so a caller that already reduced cannot reduce again, and
`load_dump` detects a scale mismatch and falls back to the logical grouping rather
than mixing. Three tests pin it, one of which requires the fixture to have load on
**at least two ranks** so it cannot pass vacuously.

Corrected per domain, 18 prefill forwards, noise floor 1.0112x:

| Domain | per-layer critical path | aggregated | forwards |
| --- | --- | --- | --- |
| text | 1.2801x | 1.0524x | 6 |
| code | 1.5821x | 1.0801x | 6 |
| math | 1.5043x | 1.1128x | 6 |

Combined baseline **1.4559x** (was reported as 1.5518x). Only `text` moved
much — 1.5773x to 1.2801x — because its eight ranks route their local tokens least
alike; `code` and `math` were within 0.002.

| transfers / forward | critical path | excess removed | MoE time saved | of prefill (5090) | of prefill (H100 SXM) |
| --- | --- | --- | --- | --- | --- |
| 4 | 1.4298 | 5.7% | 1.8% | 0.5% | 1.3% |
| 8 | 1.4085 | 10.4% | 3.3% | 1.0% | 2.3% |
| 16 | 1.3736 | 18.0% | 5.7% | 1.7% | 4.0% |
| 43 | 1.2947 | 35.4% | 11.1% | 3.3% | 7.8% |
| 86 | 1.2258 | 50.5% | 15.8% | 4.7% | 11.1% |
| 129 | 1.1909 | 58.1% | 18.2% | 5.5% | 12.7% |

**At one placement per reachable layer (43) the payoff is 3.3% of a prefill step
here and 7.8% on an NVLink machine**, against the 5.1% and 11.8% previously
recorded. At the default budget of 4 it is 0.5% and 1.3%.

Global allocation still beats a uniform per-layer allowance, but by less than
reported: 13% relatively at one per layer, 9% at two,
5% at three (critical path 1.245 against
1.2258 at two). The earlier 15% to 19% came from the mis-scaled
data.


## 2026-08-25 — borrowing from UltraEP: what transferred and what did not

Read against UltraEP (arXiv:2606.04101, `Dots-Infra/UltraEP`), which reports cutting
inter-rank imbalance from 1.30–4.01x to **1.01x** on 256 GPUs at EP64. Three of its
ideas were priced against our data. One is adopted, one is measurably wrong for us,
and one is a real but moderate gain that costs a verified contract.

### Adopted: a token floor per replica, and its value is not a tunable

UltraEP refuses a replica below `ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA` (default
1024). We should refuse below **`BLOCK_SIZE_M`**, and for us that is not a guess:
the kernel pads each expert's token list to a multiple of it, so a replica taking
fewer than one block saves no block and therefore no time. Adopted in
`bench/imbalance.py` as `min_tokens`.

### Rejected by measurement: the binary search on the imbalance ratio

Their planner binary-searches the achievable ratio and lets the placement count
follow — which is exactly the "dynamic replica count per layer" shape we wanted, so
it was implemented (`place_by_threshold_search`) and compared against the existing
greedy at matched budget, both respecting the slot constraint and the token floor:

| transfers / forward | greedy | threshold search | better |
| --- | --- | --- | --- |
| 16 | 1.3736 (18.0%) | 1.3858 (15.4%) | greedy |
| 43 | 1.2947 (35.4%) | 1.3126 (31.4%) | greedy |
| 86 | 1.2261 (50.4%) | 1.2441 (46.5%) | greedy |
| 129 | 1.1921 (57.9%) | 1.2012 (55.9%) | greedy |

**The greedy wins on 18 of 18 forwards**, mean difference 0.0179 of critical path
against a standard deviation of 0.004 to 0.008 — consistent, not noise.

The reason is the same constraint that limits everything else here: **a discrete move
cannot hit a continuous target.** The search sets one ratio and asks every layer to
reach it, which works when a replica can absorb an arbitrary quota. Source-rank
routing moves exactly half of an expert, so the reachable loads are a small discrete
set: a layer either falls short of the ratio or overshoots it and makes the target
the new peak, and the search calls both infeasible and spends the budget for nothing.
The greedy never sets a target, only lowers the peak a step at a time, which fits a
coarse move set better.

**Do not re-try this from the paper.** The solver is not where their advantage lives.

### Priced, not adopted: quota-driven routing

Their advantage lives in the routing granularity. With the real constraints — one
slot per rank, K replicas per layer, quota bounded by the expert's own load, token
floor at `BLOCK_SIZE_M` — quota routing gives:

| replicas / layer | quota routing | source-rank half-split | ahead by |
| --- | --- | --- | --- |
| 1 | 1.2745x (39.8%) | 1.2947x (35.4%) | 4.4 pt |
| 2 | 1.1756x (61.5%) | 1.2261x (50.4%) | 11.1 pt |
| 3 | 1.1118x (75.5%) | 1.1921x (57.9%) | 17.6 pt |

Roughly **12% to 23% more of the excess removed at matched transfer budget**, growing
with the replica count. Note this is far less than a first reading suggested: that
used UltraEP's `excess <= slack` feasibility test, which is a *relaxation* ignoring
whether the replicated experts have that much load to give. Under real constraints
quota routing reaches 1.2745x at one per layer, not 1.0.

It would cost ticket 03's verified "no source rank's chunk is ever split" contract
and a new quota-driven reroute path — vLLM's existing per-token choice is a Knuth
hash, uniform-random rather than quota-driven. That trade is ticket 11.

### Not borrowable: their timing, and why ours differs

UltraEP does **not** hide the transfer. Its README is explicit that
`update_placement` and `weight_sync` are "on the critical path"; only the backward
`grad_reduce` is overlapped. They can afford to pay it: 0.19–0.32 ms of weight sync
at 706–715 GB/s of critical-path bandwidth inside an NVLink domain, with TMA
double-buffering and streaming stores. Our PCIe is 54 GB/s, thirteen times slower,
so paying it outright is not open to us.

What *is* borrowable is the ordering: plan on **exact post-gating load in the current
layer** rather than a cross-layer prediction. In prefill the dispatch collective
takes about 1243 us per layer while one expert transfer takes 175 us — 14% — so the
transfer fits inside the window between gating and the expert GEMM even on PCIe. In
decode the same dispatch is 19 us and the transfer is 900% of it, which is why
lookahead exists at all. If prefill adopts in-layer exact-load planning, prediction
error leaves the payoff estimate entirely, and tickets 02, 06 and 10 stop gating the
prefill path.

### Also fixed here: the greedy was ignoring the slot constraint

`place_globally` allowed two replicas of one layer onto the same target rank, which a
rank holding one replica slot per layer cannot express — 6.6% of placements at one
per layer, 17.4% at two, 27.3% at three. Now enforced in `_best_move` and `_apply`
with three tests. The published payoff figures barely move (50.5% to 50.4% at 86
transfers) because the greedy already preferred the lightest rank, which is rarely the
same rank twice; the constraint is a correctness fix rather than a numbers change.

## 2026-08-25 — first end-to-end placement run: the path works, the lifecycle does not

Two servers differing only in `VLLM_PREDICTIVE_PLACE_PER_FORWARD` (0 and 43), same
prompts, same seed, `--custom-output-len 128`. `run_e2e_placement.sh`, read by
`analyse_e2e.py`.

### The mechanism connects

| | budget 0 | budget 43 |
| --- | --- | --- |
| replicas activated | 0 | **43** |
| worker fatals | 0 | 0 |
| requests completed / failed | 120 / 0 | 120 / 0 |
| load on non-canonical ranks | **0.00%** | **3.12%** |

That last row is the direct evidence: with no replica, each rank's physical load equals
the load of the experts it canonically owns; with replicas active, 3.12% of the load
sits on ranks that do not own the expert. Tokens really are reaching the replicas.

Per-layer critical-path imbalance, over the two well-populated assignment bands:

| band | forwards | critical path | excess removed |
| --- | --- | --- | --- |
| 512 | 122 / 122 | 1.6912 -> 1.6434 | **6.9%** |
| 448 | 122 / 122 | 1.6749 -> 1.6243 | **7.5%** |

### The cost, as predicted

| | budget 0 | budget 43 |
| --- | --- | --- |
| mean TTFT | 907.5 ms | **1708.3 ms** (+88%) |
| mean TPOT | 99.56 ms | 102.70 ms (+3%) |

88% worse TTFT for 7% less load imbalance. That is the expected result for a path with
neither transfer overlap nor a lifecycle, and it is why this run was read for mechanism
rather than for benefit.

### Two behaviours that differ from what the code says it does

**Placements are never reverted.** `run_predictive_placement` returns early on an empty
plan, leaving the layout untouched, so a replica placed once persists through every
later forward. The commit message for this path says the policy "re-plans every
forward"; it does not. It accumulates, and a later placement into an occupied slot
overwrites the earlier one silently, because the one-slot-per-rank check only covers
placements *within* one plan.

This matters beyond tidiness: the 2026-08-25 survey measured cross-forward residency as
**harmful** under mixed traffic (-20.0% across domains). Here it helped, because the run
is one domain. A mixed-traffic run with this code would be expected to do worse than the
baseline, and nothing in the code prevents it.

**Nothing was placed during decode, and that is correct.** The `min_tokens` floor is the
kernel's `BLOCK_SIZE_M` (128), and a decode layer's whole load is 448 to 512 across 128
experts — half of the hottest expert is about 32, far under the floor. Fed a decode
forward offline, the planner returns 0 placements at `min_tokens=128` and 43 at 0. So
the 43 replicas were placed during one of the 8 prefill steps and then persisted.

The consequence for reading the table above: **the 6.9% and 7.5% are prefill-step
placements improving the load distribution that decode forwards then see.** Decode load
rebalancing does not convert into time — block quantization eats it, as measured on
2026-08-24 — so those percentages are evidence the mechanism works, not evidence of a
decode gain.

### What this run could not measure

Only 2 of 263 forwards were full prefill steps: 120 requests with a 128-token decode is
decode-dominated. Measuring the prefill effect needs the request count sized from the
number of prefill forwards wanted — about 800 requests, or a decode length of 1 — which
is the same sizing note ticket 10 carries.

## 2026-08-25 — reversion and overlap: the feature works, and the bottleneck moved

Same two-arm run, with the placement path now reverting what it no longer wants and
transferring on a side stream for the following forward to activate.

### It removes 23.9% of the per-layer excess, in the regime where it should

Grouped by assignment range rather than exact count — a prefill step's token count
varies with how much of a prompt the scheduler admitted, and exact grouping scattered
eight prefill forwards into four singleton bands that the minimum-size filter then
dropped, hiding the only regime this feature acts in:

| band | forwards | critical path | excess removed |
| --- | --- | --- | --- |
| decode | 251 / 251 | 1.6920 -> 1.6950 | **-0.4%** |
| partial prefill | 4 / 4 | 1.5851 -> 1.5013 | 14.3% |
| **full prefill** | 8 / 8 | 1.5950 -> **1.4526** | **23.9%** |

Load on ranks that do not canonically own the expert, which is the direct evidence of
what routing did:

| | prefill steps | decode steps |
| --- | --- | --- |
| budget 0 | 0.00% | 0.00% |
| budget 43 | **3.95%** | **0.00%** |

**Decode's -0.4% is correct, not a regression.** The `min_tokens` floor is the kernel's
`BLOCK_SIZE_M`, and half of a decode layer's hottest expert is about 32 tokens, so a
decode forward plans nothing; reconciliation then reverts the prefill step's replicas
before decode sees them. The previous version's 6.9% in the decode bands came entirely
from *never* reverting, which the survey measured as harmful on mixed traffic.

### The live figure cross-validates the out-of-sample prediction

Offline, planning and scoring on the same forward gave 35.4% of the excess at this
budget; planning on one forward and applying to another within a domain gave 24.2%. The
two-phase design plans on forward N and activates on N+1, so it is the second case, and
it measures **23.9%**. Those agree, which is the first time an offline figure here has
been checked against a live one.

Payoff: 23.9% of the excess is 8.9% of MoE time, so about **2.7% of a prefill step on
this node and 6.2% on an NVLink machine**.

### Still net negative, and the bottleneck is no longer the transfer

| | budget 0 | budget 43 |
| --- | --- | --- |
| mean TTFT | 878.5 ms | 1341.3 ms (+53%) |
| mean TPOT | 97.23 ms | 101.34 ms (+4%) |

Reversion and overlap took TTFT from +88% to +53%, but 2.7% cannot pay for 53%. The
transfer is now on a side stream and activated a forward later, so what remains is
**activation**: `update_mapping` plus `publish_source_local_maps` rebuild the logical
maps and touch every layer, and they land on the prefill step's critical path, which is
exactly what TTFT measures. Ticket 08 asks for activation to change only the rows that
changed; this does a full republish.

So the next bottleneck is named and it is not the one the design worried about.

### What this run still cannot measure

Eight full prefill steps out of 263. The comparison is consistent across all three bands
and cross-validates an independent offline figure, but eight forwards is thin. A run
sized from the prefill-forward count — about 800 requests, or a decode length of 1 — is
what the payoff figure needs before it sizes anything.
