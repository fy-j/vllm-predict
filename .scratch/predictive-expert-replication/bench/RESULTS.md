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

---

# 2026-08-29 — 8x H100 80GB SXM, NVSwitch. First measurements on the new node

Hardware, established rather than assumed (`HANDOFF-2026-08-29-H100.md` step 0):

    Product Name : NVIDIA H100 80GB HBM3   -> SXM, not a PCIe card
    nvidia-smi topo -m : NV18 between all 8 pairs -> full NVSwitch fabric
    nvidia-nvshmem-cu13 3.4.5 present; deep_ep 2.0.0+local importable

**Build.** `uv venv --python 3.12` then `VLLM_USE_PRECOMPILED=1 uv pip install -e .
--torch-backend=auto` succeeded first try on torch 2.13.0+cu130. `.venv/bin/python -c
"import vllm"` resolves to `/mnt/yhong/vllm-predict/vllm/__init__.py`, version
`0.1.dev20306+gfed29c4f8`, and `vllm/distributed/eplb/predictive.py` imports. This is a
genuine build of this tree, unlike the 5090 node's symlinked kernels, so ticket 05's
precondition is met here.

Installing `nvshmem4py-cu13` upgraded `cuda-python` 12.9.7 -> 13.3.1 in the venv.
`import vllm` still works after it; noted because it is the kind of change that breaks a
build silently.

## `probe_hardware.py`

One harness fix was needed first, and it is the kind that would have produced a wrong
number rather than an error. The plausibility guard on P2P bandwidth was a **constant**
pinned to PCIe Gen5's 70 GB/s, so it rejected every correct NVLink measurement as
"implausible". It now selects the ceiling from `probe_interconnect()`'s `has_nvlink`
(NVLink 4: 18 links x 25 GB/s = 450 GB/s per direction, guard at 500). The guard itself
was not removed — it exists because `torch.accelerator.synchronize()` waits only on the
current device, and timing a copy between two *other* devices measures launch overhead.

| | 5090 / PCIe Gen5 x16 | H100 SXM / NV18 |
| --- | --- | --- |
| one 9.00 MiB expert, idle P2P | 176-177 us at 53.5 GB/s | **39.9-41.0 us at 230-237 GB/s** |
| uniform across rank pairs | yes | yes (39.9 to 41.0) |
| one layer's Attention projections | 23-31 us at <=128 tok/rank, 98 us at 512 | **20.5-23.2 us at <=128, 37.5 us at 512** |
| one layer's MoE (bmm proxy) | 102 us at 32 tok/rank | 69.3 us at 32, 107.4 at 512 |
| MoE over expert-weight read floor | 1.02 at 8 tok/rank, 1.87 at 128 | 1.02 at 8, 1.17 at 128, **1.65 at 512** |
| KV per rank | ~150-170k tokens (32 GB card) | **643,873 tokens** (79.1 GiB card, 59.0 GiB KV) |

**Transfer got 4.4x cheaper; the Attention window did not shrink proportionally.** At
512 tokens per rank one layer's Attention projections alone are 37.5 us against a 40.6 us
transfer — 92% of an expert now fits inside a **single** Attention block, where on PCIe a
single block hid 18%. And these projection figures exclude the attention kernel itself,
so the real window is wider. This is the measurement that makes the operator's intended
pipeline shape (predict at layer `i`, transfer during layer `i+1`'s Attention) feasible
at all; see `CURRENT-STATUS.md`'s 2026-08-29 code audit finding 1 for why the current
hook positions still cannot express it.

**The decode verdict is no longer settled by KV capacity.** Ticket 00 closed decode
because clearing the MoE kernel's one-block-per-expert bar needs more than 2048
concurrent decode tokens and the 5090 could serve about 464. Recomputing from the
measured KV ceiling, at `M x topk / num_logical > BLOCK_SIZE_M`:

| context | sequences per rank | total decode tokens | tokens per expert | clears 128? |
| --- | --- | --- | --- | --- |
| 1024 | 628 | 5024 | 314 | **yes, 2.45x** |
| 2048 | 314 | 2512 | 157 | **yes, 1.23x** |
| 3072 | 209 | 1672 | 104 | no |
| 4096 | 157 | 1256 | 79 | no |

So decode clears the bar at contexts up to roughly 2400 tokens and fails above it.
Ticket 00's own estimate for 8x H100 was "about 1800, still under the threshold"; that
was low by 2.8x at 1024 context. **This does not reopen decode by itself.** Three things
gate it: `max_num_seqs` must be raised to reach 628 per rank; the real `BLOCK_SIZE_M`
for `E=128,N=768` on H100 is unknown and the 128 in the table is the hardcoded
`_MOE_BLOCK_SIZE_M` (audit finding 5), so the whole column moves if the tuned config
picks 64; and clearing the bar removes the *reason* imbalance was free, it does not show
that imbalance then converts into time. Treat this as "the constraint that closed decode
is no longer binding at short contexts", not as a positive result.

## `probe_nvshmem.py` — passes, 5 of 5

Ticket 13's gate. It needed the three API corrections the ticket predicted ("expect to
correct them on first contact"), all three now in the script:

- **Python bindings are not on public PyPI.** `nvidia-nvshmem-cu13` ships only the C
  library — `libnvshmem_host.so.3` and the UID bootstrap, no Python module. The bindings
  are `uv pip install --extra-index-url https://pypi.nvidia.com nvshmem4py-cu13`
  (installed 0.3.1, which pulled `nvidia-nvshmem-cu13` 3.7.2).
- **Init is UID-bootstrapped, not implicit.** `nvshmem.core.init(device=Device(),
  uid=..., rank=..., nranks=..., initializer_method="uid")`, with the UID broadcast over
  the existing torch process group. MPI bootstrap is the other option and was not used:
  vLLM is not launched under `mpirun` and `mpi4py` must be built against the same MPI.
- **`Device` moved.** `cuda.core.Device` since cuda-python 13; nvshmem4py's own docs
  still say `cuda.core.experimental`.
- **Allocation is `nvshmem.core.tensor(shape, dtype)`**, not `empty`.
- **There is no `put_on_stream`.** `nvshmem.core.put(dst, src, pe, stream=)` *is* the
  stream-ordered entry point and raises `NotImplementedError` when `stream` is None.

Results:

    [PASS] NCCL up before NVSHMEM
    [PASS] NVSHMEM importable — nvshmem.core
    [PASS] symmetric heap after NCCL — 8 PEs
    [PASS] stream-ordered put — nvshmem.core.put(..., stream=)
    [PASS] wait_event orders the put — expected 8, got 8

    9.00 MiB put: p50 33.0 us (285.9 GB/s), min 31.6, max 35.0

Every one of the three things ticket 13 said would invalidate its design on their own is
answered in its favour. In particular **a plain CUDA event is sufficient to order the
consumer**, so the design needs no NVSHMEM fence and no flag polling — polling being the
per-rank-timing divergence class that deadlocked this branch twice.

**33.0 us, against 289 us on PCIe: 8.8x.** 43 puts per forward is 1.4 ms of fabric time
here against 12.4 ms there. The probe's own threshold for "the planning delay can stay at
one layer and the lookahead need not rise" is 50 us, and 33 clears it.

Note what this does and does not buy. A put at 33 us is *not* faster than the NCCL P2P
copy measured above at 40 us — the two are the same order. The entire value of the put is
that **the host never reads the plan**, which is the 11.02 ms `cudaEventSynchronize` and
the 86.8% -> 52.9% occupancy drop. This is worth stating because the obvious reading of
"8.8x" is a bandwidth win, and it is not one.

Two API facts found while correcting the probe, both of which simplify ticket 13 and are
recorded in it: `nvshmem.core.register_external_tensor` exists, so the model's own
`expert_weights` allocation can be registered with NVSHMEM and put into directly, which
removes the symmetric staging buffer and its extra 9 MiB device-to-device copy from the
design; and `cuda.core.VirtualMemoryResource` exists, which is the allocator ticket 12's
VMM aliasing needs.

## Step 1 — the number that sets the prefill ceiling

`run_placement_profile.sh`, `DOMAIN=ko CONC=8 NUM_PROMPTS=64 OUT_LEN=4 BUDGETS=0`, one
baseline arm with no placement. Attributed with `parse_profile.py --phase prefill`, so
only `execute_context_*` windows carrying prefill tokens are counted — trap 2 of the
handoff. Results in `results/h100/step1-prefill-64.json`.

Two harness gaps had to be closed first, and both would have produced a wrong number
rather than an error:

- **`parse_profile.py` was decode-only.** It kept windows with no prefill token, which
  is what the decode question needs and the opposite of what the ceiling needs. It now
  takes `--phase {decode,prefill}` and refuses to mix them.
- **The runner tied the prompt count to the concurrency.** `--num-prompts "$CONC"` gave
  8 requests and **one prefill window per rank**. `NUM_PROMPTS` is now separate. The
  handoff's own Step 1 command already passed `DOMAIN` and `NUM_PROMPTS`, neither of
  which the script read; both are wired now.

| rank | windows | attributed ms/step | expert GEMM | nccl | attention | other |
| --- | --- | --- | --- | --- | --- | --- |
| dp0 | 8 | 24.3 | 15.56% | 42.35% | 5.28% | 36.81% |
| dp1 | 8 | 113.9 | 10.79% | 72.79% | 1.12% | 15.30% |
| dp2 | 8 | 106.8 | 11.44% | 71.06% | 1.20% | 16.30% |
| dp3 | 8 | 102.4 | 12.12% | 69.61% | 1.25% | 17.02% |
| dp4 | 8 | 124.6 | 9.34% | 75.24% | 1.06% | 14.36% |
| dp5 | 8 | 136.0 | 8.75% | 77.16% | 0.98% | 13.12% |
| dp6 | 8 | 136.0 | 9.34% | 76.72% | 0.95% | 12.99% |
| dp7 | 8 | 136.2 | 8.74% | 77.67% | 0.93% | 12.67% |

**Mean expert GEMM share of attributed prefill GPU time: 10.76%.** The thin 8-request
run gave 10.77% from a single window per rank, so the figure is not sample-limited.
MoE 231.35 us per layer, nccl 1690.56 us per layer. Baseline mean TTFT 303.61 ms,
median 236.68, p99 801.35 over 64 requests at concurrency 8.

| | 5090 / PCIe | H100 SXM / NV18 |
| --- | --- | --- |
| collectives | 66.0% | **77.0%** |
| expert GEMM | 3.67% | **10.76%** |
| attention / norm | 4.9% | ~1.0% |
| ceiling = expert GEMM x 47% perfect balance | 1.37%-1.73% | **~5.1%** |

**The ceiling rose about 3x, to roughly 5% of a prefill step.** Against the handoff's
own decision rule — "low single digits means stop; 15% means the ceiling is ~7% and
there is something to win" — 10.76% lands between the two, at a ceiling near 5%.

**Read the collective share carefully, because the obvious reading is wrong.** NCCL's
share went *up* on the faster fabric, 66% to 77%. That is not a bandwidth statement: at
concurrency 8 over DP=8 each rank holds about one request, so the ranks are unevenly
loaded and most of that time is ranks waiting at a collective for the slowest one. dp0
shows it directly — 24.3 ms of attributed work against dp7's 136.2 ms, 42% nccl against
78%. This is the same conclusion as the 5090's, that the collectives are arrival skew
and latency rather than bytes, and a faster fabric does not shrink skew.

The consequence for the ceiling is that **10.76% is a floor, not a ceiling estimate**.
An evenly loaded operating point would shrink the waiting and raise expert GEMM's share
further. A concurrency sweep is what would pin it; this run does not.

Also measured, and **not** to be read as the recoverable imbalance: MoE *time* per rank
spread `max_over_mean` 1.144. That is the summed-across-layers view, which the glossary
records as understating the per-layer critical path by about 3.7x on this workload. It
is reported because the parser emits it, not because it sizes anything.

## Allocation order priced offline (audit finding 2)

`imbalance.plan_moves_in_layer_order` is new: it reproduces what the online planner
does — one layer at a time, in increasing layer index, each taking
`min(remaining, per_layer_cap)` — so the oracle `plan_moves` has a control differing
*only* in which layer receives the next placement. Both use the same greedy and the
same positive-benefit test.

On a synthetic model shaped like the measured one (43 layers, peak-rank load
concentrated 55/20/15/10 across its experts, hottest layers placed last), at budget 43:

| allocation | layers covered | critical path | excess removed |
| --- | --- | --- | --- |
| baseline | — | 2.3099 | — |
| layer order, cap 2 (**what runs today**) | 22 | 2.0023 | 23.5% |
| layer order, cap 1 | 43 | 1.6768 | 48.0% |
| global ranking, cap 2 (oracle) | 40 | 1.6734 | 48.3% |

Two things follow. **`max_replicas_per_layer=1` captures 99.4% of the oracle's benefit
at the same budget**, as a one-line configuration change, because coverage is what
drives benefit and a layer's second placement chases a much smaller expert than its
first. And the ratio today's allocation achieves against the oracle, 23.5/48.3 = 0.486,
is close to the measured 16.9/34.9 = 0.484 — which is corroboration that allocation
order is the 15%-versus-33% gap, not prediction accuracy.

This is synthetic data, so the ratio agreement is suggestive rather than proof. The
intra-layer concentration is what makes it behave like the real thing: a first attempt
split each peak rank's load evenly between two experts, and then both placements gained
the same amount, global ranking also filled 2 per layer, and the control measured
nothing. That failure is recorded in the test's docstring so it is not repeated.

## Ticket 14 — the third arm, and two harness defects it exposed

`run_e2e_placement.sh` grew a `budget=off` arm: no predictive config at all, so no
`enable_eplb`, no redundant experts, no replica slots. That is the denominator every
TTFT number in this branch has been missing, because both existing arms pass
`enabled = True` and differ only in the transfer budget. Default is now
`BUDGETS="off 0 43"`.

**First stock-server TTFT on this node**, `ko`, 400 requests, CONC=8, OUT_LEN=1:

| arm | mean TTFT | median | p99 | vs `off` |
| --- | --- | --- | --- | --- |
| `off` — feature fully disabled | **184.70 ms** | 132.04 | 739.17 | — |
| `0` — prediction on, placement withheld | **198.81 ms** | 177.39 | 676.16 | **+7.6%** |
| `43` — placing | **242.79 ms** | 174.47 | 2820.04 | **+31.5%** |

`max_replicas_per_layer=1` (lowered from 2 this session) did what the offline pricing
said it would: **344 placement log lines = 43 layers x 8 ranks, one replica each**, so
coverage went from 22 layers to all 43 reachable ones. `connected: true`, 400 requests
completed, 0 failed.

Imbalance recovered, arm `0` against arm `43`, per-layer critical path:

| band | forwards | critical path | excess removed |
| --- | --- | --- | --- |
| full prefill | 15 / 13 | 1.8836 -> 1.6715 | **24.0%** |
| partial prefill | 71 / 74 | 1.8851 -> 1.7061 | **20.2%** |
| decode | 14 / 13 | 1.9688 -> 1.9500 | 1.9% (gated, correctly) |

That is up from the 5090's 15-17%, as the coverage fix predicted, and is now about 69%
of the offline oracle's ~35% rather than half of it.

### The decisive arithmetic

    expert GEMM share of an attributed prefill step        10.76%
    recoverable excess as a share of MoE time              46.9%   (cp 1.8836)
    -> perfect balance is worth                             5.05% of a prefill step
    -> 24.0% of the excess, as measured, is worth           1.21% of a prefill step

    measured cost: prediction alone                        +7.6% mean TTFT
                   prediction + placement                 +31.5% mean TTFT

**Prediction alone costs 1.51x the entire perfect-balance ceiling.** Before a single
replica moves, the mechanism that creates the transfer window has already spent more
than perfect expert balance could ever return. The full feature costs 6.2x the ceiling
and 26x what it actually delivered.

**This is a conclusion about the mechanism, not about the interconnect, and that is what
makes it different from the 5090 result.** Ticket 13 removes the host synchronisation,
which is most of the +22.1% that placement adds on top. It does **not** touch the +7.6%:
that is 43 extra gate matmuls and 43 extra AllGathers per forward, one per source layer,
and a device-side plan leaves every one of them in place. So the ceiling stays below the
floor even with ticket 13 built, on the fastest interconnect NVIDIA currently ships. The
finding transfers to the Ascend port, where the same 43 collectives would be paid.

The one thing that could change it is making prediction itself much cheaper — a single
batched cross-layer snapshot instead of 43 per-layer AllGathers, which
`CURRENT-STATUS.md` lists as not designed. That would have to bring 7.6% under about 2%
to leave room, and it does not address the gate matmuls.

Two caveats, stated because they are the ones that could move the number and neither
rescues it. `OUT_LEN=1` prices TTFT only, so TPOT reads 0.00 in every arm and the decode
cost is not in these figures. And 10.76% is a floor on the expert GEMM share (the
collectives are inflated by arrival skew at concurrency 8), so an evenly loaded operating
point would raise the ceiling — but it would have to raise it about 1.6x just to reach
prediction's own cost, and prediction's cost would rise with it, since more concurrency
does not reduce the number of collectives.

TPOT reads 0.00 in all arms because `OUT_LEN=1` produces no inter-token interval; this
shape prices TTFT only.

Two defects in the runner, both of which wasted a full arm's worth of GPU time and
neither of which failed loudly:

1. **`[[ "$budget" -eq 0 ]]` on a non-numeric arm.** With `budget=off`, the arithmetic
   test treats `off` as a variable name and `set -u` aborts — *after* that arm had
   served its entire 400-request benchmark. Now a string comparison.
2. **`kill -TERM "$PID"` followed by a bare `wait`.** The DP=8 server did not die on
   SIGTERM, so `wait` blocked forever: the `off` arm's server sat idle for 26 minutes
   with its result already on disk, the remaining two arms never started, and nothing in
   any log said why. The teardown now escalates — TERM, a bounded 60 s poll, then KILL —
   because the orphan-reap loop that follows it never ran either. Reaping matters: the
   engine cores reparent to init and hold 73 GiB per GPU until killed.

Worth stating as a pattern, since this is now five harness defects in two sessions
(these two, the PCIe-pinned bandwidth guard, the decode-only trace parser, and the
prompt count tied to concurrency): **this harness fails by producing a wrong number or
by silently doing nothing, not by erroring.** Check that an arm actually ran before
reading its output, and that a run's exit code 0 means all its arms completed.

## The prediction AllGather, measured on H100 — and the batching idea is dead

I proposed batching the 43 per-layer AllGathers into one cross-layer snapshot as the only
remaining escape. **Measured, it is not worth doing**, and the reasoning that motivated
it was inherited from the 5090 and does not hold here.

From the `budget=0` arm's trace (prediction on, placement off), prefill windows only,
NCCL kernels grouped by stream — `dp0`, 8 windows:

| kernel | stream | calls | total | p50 | max |
| --- | --- | --- | --- | --- | --- |
| `ncclDevKernel_AllGatherV_RING_LL` — token dispatch | 23 | 1152 | 53.74 ms | 58.0 us | 85.5 us |
| `ncclDevKernel_Reduce_Sum_bf16_RING_LL` — combine | 23 | 384 | 25.24 ms | 72.9 us | 75.4 us |
| **`ncclDevKernel_AllGather_RING_LL` — prediction snapshot** | **31** | **344** | **3.21 ms** | **9.3 us** | **9.8 us** |

344 calls over 8 windows is exactly **43 per forward**, one per source layer, which
identifies it beyond doubt.

Three things follow:

1. **It is on its own stream (31)**, distinct from the token collectives' stream 23 and
   from the default compute stream. Confirmed empirically rather than from the spec's
   intent. The EPLB group being a separate process group is what buys this.
2. **It costs 0.40 ms per forward**, 9.3 us per layer. Its window is that layer's local
   expert GEMM, measured at 231 us per layer, so the collective occupies **4% of the
   window it is hidden in**. There is nothing left to hide; it is already overlapped
   about as completely as a collective can be.
3. **There is no arrival skew here.** p50 9.3 us against a max of 9.8 us — a 5% spread on
   344 samples. The 5090 recorded 5 us to 11181 us on the same 4 KiB payload, p50 493 us,
   and that two-thousand-fold spread is what made "43 barriers" look expensive. On this
   node the ranks arrive together, because token dispatch has just synchronised them and
   the snapshot is enqueued immediately after it.

So collapsing 43 collectives into 1 would save at most 0.40 ms per forward against a
measured cost of +7.6% TTFT and a ceiling of 5.05%. **That direction is closed.** It cost
one trace read to find out, which is the right order to do it in.

### Where the cost actually is, and one correction to the previous section

Not the collective. A prefill window on this node holds about **1910 kernels**, and
prediction contributes roughly 15 per source layer — a gate GEMM, the router's top-k, and
about a dozen elementwise ops in `predict_local_counts` — so about **645 of those 1910
launches, a third of the window**, exist only to predict. Each is also a Python-level
dispatch on the host, 43 times per forward, in an eager engine.

Occupancy in those windows, same trace: `dp0` 12.9%, `dp4` 106.2%, `dp7` 116.1% (over
100% because kernels on separate streams overlap and the sum double-counts). `dp0` is the
rank that finishes early and waits, which is the same arrival-skew picture the Step 1
attribution showed.

**Correction to the three-arm table above.** Arm `0` differs from arm `off` by more than
prediction: it also enables EPLB *actual-load* recording every forward (the dump and the
balancedness log at interval 1) and it carries the 17-row physical layout instead of 16.
So **+7.6% is an upper bound on prediction's own cost, not a measurement of it.** The
verdict is unchanged, since even the upper bound exceeds the 5.05% ceiling, but the
attribution is not settled and the earlier wording "prediction alone" overstated what was
measured.

What would settle it: one profiled `off` arm, diffed against this trace on kernel count
and GPU busy time per prefill window. That separates prediction's compute from the
recording and the layout, and it is the measurement that decides whether a fused counting
kernel — replacing those dozen elementwise ops with one — is worth writing.

## The measurement that undermines the mechanism: expert load is static across forwards

Asked because the redesign question is "can we get the transfer window without paying 43
gate evaluations", and the answer turns on how fast expert load actually changes.

Method: the `budget=0` dump (105 forwards, no placement, so every forward's load is
unperturbed canonical). Take the prefill-sized forwards (>= 16384 assignments per layer,
46 of them). Plan `budget=43, per_layer_cap=1` on the load of forward `N - lag` and score
it on forward `N`'s load. Lag 0 is the oracle: it plans on the load it is scored against.

| lag (forwards) | pairs | excess removed, mean | median | vs oracle |
| --- | --- | --- | --- | --- |
| 0 — oracle | 46 | 31.7% | 31.6% | 1.00 |
| 1 | 45 | 31.2% | 31.1% | **0.98** |
| 2 | 44 | 31.2% | 31.2% | 0.98 |
| 4 | 42 | 31.2% | 31.2% | 0.99 |
| 8 | 38 | 31.3% | 31.2% | 0.99 |
| 16 | 30 | 31.3% | 31.3% | 0.99 |

**A placement computed from load measured sixteen forwards ago is 99% as good as one
computed from the load it will actually meet.** Per-layer expert load on this workload is
not merely predictable, it is close to static.

This is not circular: the `budget=0` arm places nothing, so both the planning load and the
scoring load are unperturbed, and no placement's effect leaks into either.

### What it implies for the design

Cross-layer prediction exists for exactly one reason — routing maps logical to physical
*before* dispatch, so a placement decided from a layer's own measured load has no window
to transfer in. That reasoning is correct and it is not the only escape. **Actual load
from the previous forward is a second source of the same information, and it is already
recorded and already reduced** by EPLB's `expert_load_pass`, at zero marginal cost.

Planning between forwards rather than inside one collapses both measured cost centres at
once:

| | in-forward prediction (today) | plan from last forward's recorded load |
| --- | --- | --- |
| extra gate GEMMs per forward | 43 | **0** |
| extra AllGathers per forward | 43 | **0** (EPLB's reduction already exists) |
| transfer window | one layer, ~231 us | the inter-forward gap plus all 48 layers |
| host sync inside the forward | yes, 11.02 ms measured | **none** — the plan is known before the forward starts |
| adaptation lag | 1 forward (lookahead 2) | 1 forward |
| accuracy against oracle | 0.98 offline, 24.0% of 31.7% online | **0.98** |

The +7.6% and the +22.1% both trace to work that this arrangement does not do. NVSHMEM
also stops being necessary: with the plan on the host before the forward begins, `ncclSend`
taking a host `c_int` peer is no longer a constraint, which was the entire reason ticket 13
existed.

**The honest consequence is that the design converges on Native EPLB with a short
interval**, plus source-rank routing. The project's premise was that Native EPLB's
historical window reacts too slowly for bursty or phase-changing traffic; on this workload
there is nothing to react to. What would still be genuinely new is re-planning every
forward at negligible cost, which Native EPLB does not offer because its `step_interval`
is thousands of steps.

### The caveat that decides whether this generalises

**This is one domain.** All 400 prompts are `ko`, at concurrency 8, so consecutive forwards
are similar partly by construction of the benchmark. Ticket 00 measured a placement carried
*across domains* at **-20.0%**, so the stability above is a within-mix property and must not
be read as a workload-independent one. A mixed-domain dump is the measurement that decides
whether the redesign holds, and it is cheap: interleave two prompt sets in one run and
repeat this table.

## Where prediction's cost actually goes: it desynchronises the ranks

The `off`-arm profile was captured with parameters identical to the `budget=0` one
(`DOMAIN=ko CONC=8 NUM_PROMPTS=64 OUT_LEN=4`) so the two can be diffed. `run_placement_profile.sh`
gained the same `budget=off` arm as the e2e runner. Per prefill window, mean over 8 ranks,
classified with `parse_profile.classify_kernel`:

| class | off | b0 (prediction on) | delta | off count | b0 count | delta count |
| --- | --- | --- | --- | --- | --- | --- |
| nccl | 40.85 ms | 81.15 ms | **+40.30** | 192 | 235 | **+43** |
| moe_expert | 11.09 ms | 11.10 ms | +0.02 | 96 | 96 | 0 |
| attention | 1.28 ms | 1.29 ms | +0.01 | 53 | 53 | 0 |
| other | 14.65 ms | 16.48 ms | **+1.83** | 830 | 1525 | **+695** |
| total | 67.87 ms | 110.02 ms | +42.15 | | | +738 |

**Prediction's own compute is 1.83 ms** across 695 extra kernels, 2.6 us each — the gate
GEMM, the top-k, and the dozen elementwise ops in `predict_local_counts`. **Its AllGathers
are 0.40 ms** (the +43 nccl kernels are exactly the 43 source layers, at the 9.3 us already
measured). Both are cheap.

**The token collectives grew by ~39.9 ms with no extra kernels and no extra bytes.** Same
192 dispatch and combine calls, same volume, roughly double the duration. That is pure
waiting.

    prediction's own compute      1.83 ms  ┐ cause
    prediction's AllGathers       0.40 ms  ┘
    token dispatch/combine       +39.9 ms  <- effect, ~18x amplification

So the cost mechanism is **desynchronisation**: the 695 extra launches and 43 extra
collectives push each rank's arrival at the next token collective apart, and dispatch and
combine absorb that as duration. It is uneven because each rank predicts over its own token
count, so the busy spread across ranks widens from 3.6x on `off` (22.1-80.6 ms) to 5.6x on
`b0` (24.3-136.2 ms).

### Three consequences, two of which overturn earlier reasoning here

1. **The earlier claim that "the cost is 645 launches" had the right lever and the wrong
   mechanism.** The launches are not expensive in themselves (1.83 ms); what they cause is.
2. **Ticket 13 does not fix this on its own.** A device-side plan and a one-sided put remove
   the host synchronisation and the D2H copy. They leave all 695 launches and all 43
   collectives in place, so the desynchronisation is untouched. Anyone reasoning that
   "device-side planning removes the to-host transfer, therefore the design becomes viable"
   is right about the sync and wrong about this.
3. **Operator fusion moves from routine optimisation to the highest-leverage item.** Of the
   695, about 43 are the gate GEMM and 43-86 the top-k; the remaining ~560-610 are the
   elementwise tail of `predict_local_counts`. Fusing those into one counting kernel per
   layer takes 695 to about 130. If the desynchronisation scales with launch count, the
   +40 ms follows it down.

And a fourth lever that had not been considered: **make prediction's cost uniform across
ranks.** Each rank currently predicts over its own token count, so the expensive ranks get
more expensive and the skew compounds. Work done on a fixed padded shape would cost the same
in total while removing the *variance*, and it is the variance the collectives charge for.

### Caveat on the timings

The two traces come from separate server launches, so the millisecond column carries
run-to-run variation and there is no repeat measurement. The kernel-count column is solid:
all 8 ranks agree to within 732-744. This profile is `OUT_LEN=4` over 64 prompts while the
+7.6% TTFT figure is `OUT_LEN=1` over 400, so **do not convert +42 ms into a percentage**.
What this establishes is the *composition* of the cost — prediction's own work is 4% of what
it adds, the collectives' waiting is 96% — not its magnitude.

## Does balancing shrink the collectives? No — tested, and the ceiling stands

The 5.05% ceiling assumes balancing touches only the expert GEMM. The obvious way it could
be too pessimistic: a rank with a hot expert finishes MoE late, arrives late at the next
collective, and every rank waits — so balancing might pay at the collectives' 77% share
instead. Tested by profiling the placed arm with parameters identical to the other two.

`parse_profile.py --phase prefill`, 8 windows per rank each arm:

| arm | MoE us/layer, mean | max across ranks | MoE share | nccl us/layer |
| --- | --- | --- | --- | --- |
| `b0` predict only | 231.3 | 264.7 | 10.76% | 1690.6 |
| `b43` placed | 124.6 | **247.6 (-6.5%)** | 5.83% | **2155.7 (+27.5%)** |

**The peak fell 6.5% and the collectives rose 27.5%.** The hypothesis has no support: the
machinery's cost dominates any second-order gain in arrival alignment. The ceiling stays at
~5% of a prefill step.

Two cautions on this comparison, both of which say "weak evidence" rather than "wrong
direction". The two runs chunked differently — the per-layer *mean* halved (231.3 -> 124.6),
which cannot be a balancing effect since balancing conserves total expert work and only
moves it between ranks, so the windows are not strictly comparable. And `max_over_mean`
rising 1.144 -> 1.987 is the *aggregated* imbalance view, which this project has twice
recorded as the wrong metric. The peak and the nccl figures are the ones worth reading, and
both point the same way as every other measurement this session.

**Method note for anyone repeating this:** summing MoE time per rank and averaging over
ranks cannot show balance improving, because balancing is conservative in that sum by
construction. The quantity that sets the step is the per-layer max across ranks. That trap
caught me here and it is the same aggregation error the glossary warns about, in a new form.

# 2026-08-29 (later) — DeepSeek-V4-Flash-FP8, and the model-shape lever does not work

Measured because the model shape looked like the biggest available lever: DSV4-Flash does
302 MFLOP of expert work per token against Qwen3-30B-A3B's 75.5, and FP8 halves the time,
so expert GEMM should have been about 2x the share and the ceiling should have roughly
doubled. **It did not. The share is the same and so is the ceiling.**

## Bring-up: four failures, none of them the feature

Recorded because each cost a run and none is documented anywhere.

1. **The `sgl-project` FP8 conversion does not load in this tree.** `_load_w13` dies with
   "size of tensor a (2048) must match tensor b (4096)" — its gate/up fusion layout differs.
   The official `deepseek-ai/DeepSeek-V4-Flash-Base` loads fine. The working invocations on
   this machine (`/mnt/ytji/bench_dsv4_*.sh`) are **SGLang**, not vLLM, so they do not
   transfer.
2. **`--kv-cache-dtype fp8` is required.** DSV4's sparse MLA uses the `fp8_ds_mla` layout
   and asserts on `auto`: every worker dies with "only supports fp8 kv-cache, got auto".
3. **FlashInfer's JIT needs two things this image lacks.** First `nvrtc.h`, which *is* in
   the venv (`nvidia/cuda_nvrtc/include/`) but not on FlashInfer's `-isystem
   /usr/local/cuda/include`; then `-lnvrtc`, whose unversioned dev symlink is missing even
   though `libnvrtc.so.13` is present. Two symlinks fix both. Verify by running `ninja`
   directly in `/root/.cache/flashinfer/*/cached_ops/fused_moe_90` — seconds, instead of
   discovering it after a 13-minute weight load.
4. **A base checkpoint has no chat template**, so `--backend openai-chat` raises inside
   the dataset sampler and **not one request is sent**. The run still reported "8 rank
   traces" and "done": the traces held 54 events and zero annotations. Use
   `--backend openai --endpoint /v1/completions --skip-chat-template`.

Failure 4 is the third time this harness has reported success on a run that measured
nothing, so `run_dsv4_baseline.sh` now fails with `MEASURED NOTHING` (exit 5) unless the
bench log carries a TTFT *and* `check_trace_nonempty.py` finds a step annotation.

## The operating point is degenerate, and that is itself the finding

At `CONC=32 NUM_PROMPTS=256`, per prefill window:

| rank | ms/step | expert GEMM | nccl | attention | other |
| --- | --- | --- | --- | --- | --- |
| dp0 | 57.6 | 10.86% | 51.73% | 12.88% | 24.53% |
| dp4 | 55.3 | 12.99% | 49.25% | 12.21% | 25.55% |
| dp1 | 71.6 | 8.94% | 62.45% | 9.54% | 19.07% |
| dp3 | 78.5 | 9.54% | 61.56% | 9.39% | 19.51% |
| dp2 | 475.3 | 1.54% | **93.02%** | 1.85% | 3.58% |
| dp5 | 466.9 | 1.41% | **93.47%** | 1.80% | 3.32% |
| dp6 | 388.7 | 1.63% | **92.85%** | 1.91% | 3.61% |
| dp7 | 339.9 | 1.82% | **92.76%** | 1.80% | 3.62% |

**Half the ranks spend 93% of the window waiting in collectives**, taking 340-475 ms per
step against the working ranks' 55-79 ms. Mean TTFT 5293 ms, p99 28369 ms over 256
requests. At `CONC=8` it was worse still — 19 ms against 1100 ms, a 58x spread.

So the naive mean of the MoE share, 6.09%, is meaningless: it averages ranks that did no
work. The defensible figure is from the four compute-bound ranks, and it is **10.58%
mean, 10.20% median**.

## The comparison, and the correction

| | Qwen3-30B-A3B | DSV4-Flash FP8 |
| --- | --- | --- |
| experts / top-k | 128 / 8 | 256 / 6 |
| expert FLOP per token | 75.5 MFLOP | 302 MFLOP |
| **expert GEMM share** (compute-bound ranks) | **10.76%** | **10.58%** |
| attention share | ~1% | **9-13%** |
| other share | ~13% | 19-25% |
| per-layer critical path, prefill | 1.8852 | **1.9622** |
| aggregate imbalance | 1.2356 | 1.1090 |
| aggregate understates by | 1.53x | **1.77x** |
| recoverable excess of MoE time | 46.9% | 49.0% |
| **ceiling** | **5.05%** | **≈5.19%** |

**The 2x estimate from FLOPs was wrong**, and the reason is worth keeping: the rest of the
model grew with the experts. DSV4's attention is sparse MLA plus hash compression, 64
heads and lora projections — 9-13% of a step against Qwen's ~1%. Numerator and
denominator moved together, so the ratio did not move. **Changing model shape between
these two does not change the economics.**

What *is* worse on DSV4 is the imbalance itself: per-layer peak/mean reaches **3.077**,
and one rank holds **38.46%** of a layer's tokens against a uniform 12.50%, with a
multinomial noise floor of 1.1491. The aggregate view reads 1.1090 and hides essentially
all of it — a 1.77x understatement, worse than Qwen's 1.53x. More reason never to size a
placement from it.

## Tooling

- `run_dsv4_baseline.sh`, and `check_trace_nonempty.py` as its measured-nothing guard.
- `figures/make_imbalance_data.py` — the dump-to-figure step, which did not exist; the old
  `rank-imbalance-data.json` was hand-made, so the figure could not be rebuilt for another
  model. Cross-validates on the Qwen dump: critical path 1.8852 against the 1.8836
  `analyse_e2e.py` reports, aggregate 1.2356 against the recorded 1.24.
- `figures/build_rank_imbalance.py` now takes `--data/--out/--model` and reads shape from
  the data instead of hardcoding 8x48. `figures/dsv4-rank-imbalance.html` is the DSV4
  report, 43x8.
- `parse_profile.classify_kernel` learned DeepGEMM: a **grouped** scheduler
  (`GroupedWithOffsetScheduler`) is the tell for the expert GEMM — `<4096,4096,...>` is w13
  and `<4096,2048,...>` is w2. Matching "gemm" would have been wrong: the attention
  projections are `sm90_fp8_gemm_1d2d_impl` and dense. Without this the share reads ~0%.

## The load-bearing assumption, tested: cost is linear in source-layer count

Every projection about fusion rested on "the collectives' extra waiting scales with
prediction's launch count", which was an assumption. Tested with a config-only arm:
`prediction_skip_first_layers` raised from 3 to 35 takes the source-layer range
`[skip_first, num_moe_layers - lookahead)` from 43 layers to 11. Same model, same prompts,
same `budget=0`, so only the number of predicting layers changes.
`run_placement_profile.sh` gained a `SKIP_FIRST` passthrough.

Per prefill window, summed over 8 ranks:

| arm | source layers | +kernels | +nccl | +total | per source layer |
| --- | --- | --- | --- | --- | --- |
| `off` | 0 | — | — | — | — |
| `skip=35` | 11 | +1778 | +76.98 ms | +96.03 ms | 161.6 kernels, **7.00 ms** |
| `skip=3` | 43 | +5904 | +322.41 ms | +337.24 ms | 137.3 kernels, **7.50 ms** |

**7.00 against 7.50 ms of extra collective waiting per source layer — within 7%.** The
scaling is linear, so the assumption holds and projections built on it are now grounded.

Three consequences.

**Fusion is quantifiable.** The kernels prediction introduces, per source layer (`dp0`,
diffed against `off`; note this is 17.2 per layer, not the 137 above — that figure divides
an 8-rank sum by the layer count and is wrong by a factor of 8):

| per layer | us/window | kernel | fusable |
| --- | --- | --- | --- |
| 3.0 | 309 | `unrolled_elementwise_kernel` | yes |
| 3.0 | 189 | `vectorized_elementwise_kernel<8>` | yes |
| 2.0 | 107 | `vectorized_elementwise_kernel<4>` | yes |
| 1.0 | 185 | `_scatter_gather_elementwise_kernel` (`scatter_add_`) | yes |
| 1.0 | 142 | `elementwise_kernel<128,4>` | yes |
| 1.0 | 72 | `elementwise_kernel<128,2>` | yes |
| 1.0 | 65 | `vectorized_elementwise_kernel<2>` | yes |
| 1.0 | 401 | `ncclDevKernel_AllGather` — the snapshot | batch only |
| 1.0 | 182 | `topkGating` — the router's top-k | **no** |
| 0.8 | 183 | `nvjet_tst_64x24...` — the gate GEMM | **no** |
| 1.1 | 239 | `_eplb_map_and_record_i32_kernel` | not prediction — EPLB *recording* |

About **11-12 of the 17 are the elementwise tail of `predict_local_counts`**, so fusing
them into one counting kernel removes roughly **473 of the 738 added launches, 64%**. At
linear scaling that takes the added cost down by about the same fraction.

**"Predict fewer layers" is ruled out, now by measurement rather than by guess.** Cost is
7.25 ms per source layer and benefit is about 0.65 points of prefill excess per covered
layer (ticket 12, a straight line with no knee). Both linear, so the ratio does not move.
Only a change to the cost *per layer* improves the economics.

**And `+7.6%` includes work ticket 13 removes.** `plan_and_launch` calls
`copy_event.synchronize()` at line 226, *before* the budget check at 239, so the
`budget=0` arm pays a host D2H synchronisation on **every** predicted layer — 43 per
forward — on top of prediction's compute and its AllGathers. So that arm is not
"prediction alone" in the sense used earlier in this file, and the floor for a
device-planned design is below 7.6%.

### Where this leaves the arithmetic

    measured today, prediction + infra + 43 host syncs        +7.6% mean TTFT
    fusion removes 64% of added launches, cost linear         -> about +2.7%   (extrapolated)
    a device-side plan removes the 43 host syncs              -> lower, share unknown
    measured ceiling                                          about 5.05%

**For the first time in this project the arithmetic can close.** Every earlier verdict
here — including the "break-even at best" written above — assumed the launch-count scaling
that had not been tested, and did not account for the host syncs sitting inside the
prediction-only arm.

Three things keep this a projection rather than a result. The 7.6% -> 2.7% step is a linear
extrapolation and has to be re-measured after fusion. This experiment **cannot** separate
the host sync from the launches, because both scale with source-layer count; only
implementing the device-side plan separates them. And the 5.05% ceiling carries a couple of
points of uncertainty depending on whether it is attributed from the mean rank or the peak
rank.

## Option (a), putting straight into the model's own weights, is ruled out

The operator chose to skip the staging buffer and `register_external_tensor` the expert
weight tensors directly, so registration became a load-bearing premise. Probed before
writing any spec around it (`probe_nvshmem_register.py`, 8 GPUs, no vLLM, no weights).

**It is not available on this stack**, and the reason is a closed loop rather than a
missing setting. Four checks deep:

| # | requirement | fix | result |
| --- | --- | --- | --- |
| 1 | NVSHMEM's per-device memory resource must exist | one `nvshmem.core.tensor((1,), uint8)` | cleared |
| 2 | size must be a multiple of heap granularity (512 MiB default) | `NVSHMEM_CUMEM_GRANULARITY=2097152` + one padding row | cleared |
| 3 | buffer must be CUDA VMM allocated | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | cleared |
| 4 | the VMM handle must be mappable by NVSHMEM | none available | **`CUDA_ERROR_NOT_SUPPORTED`** |

`NVSHMEM_DISABLE_CUDA_VMM=1` removes check 4 and registration then *reports* success for
all 48 layers — 7.59 GiB per rank, and the tensors still work for local compute. But the
first real put fails with **"Buffer registration requires dynamic VMM heap"**: that
registration was not symmetric. Registration needs the VMM heap; the VMM heap needs a
mappable user handle PyTorch does not create. The two settings exclude each other.

Two by-products worth keeping. Check 1's error message ("device that is not initialized
with NVSHMEM") reads like a device-binding bug and is not one — the fix is an allocation,
not a `set_current`. And a **row-granular** put must go through
`nvshmem.bindings.putmem_on_stream(dst_ptr, src_ptr, bytes, pe, stream)`:
`nvshmem.core.put` resolves arguments through nvshmem's tracking table, which holds whole
allocations, so a row slice raises "Tensor not tracked by nvshmem". A device-side
implementation would emit the pointer form anyway.

**So the transfer lands through a symmetric-heap staging buffer**, and its extra local copy
costs 6.3 us for Qwen's 9 MiB, 16.8 us for DSV4's 24 MiB at the measured 3.00 TB/s —
0.01% of a 67.87 ms prefill step. A third route exists: allocate the expert weights *from*
the symmetric heap with `nvshmem.core.tensor()` instead of registering torch memory, which
`probe_nvshmem.py` proves works mechanically. It puts NVSHMEM's allocator underneath the
model's own weights, which weight loading and EPLB's `rearrange` share, so it is a deeper
intrusion than registration would have been and should only be considered if the staging
copy ever shows up in a measurement.

## PyTorch symmetric memory also works, and is what option (a) was reaching for

Probed after NVSHMEM registration was ruled out, because this repository already uses
PyTorch symmetric memory for custom all-reduce
(`vllm/distributed/device_communicators/symm_mem.py`). `probe_symm_mem.py`, 8 GPUs, 6 of 6:

    [PASS] allocate weights in the symmetric pool — 153 MiB per layer, ordinary torch.zeros
    [PASS] rendezvous — world 8, is_symm_mem_tensor True
    [PASS] still usable for local compute — matmul on a canonical row is finite
    [PASS] row-granular write into a peer's replica row — holds 8.0, expected 8.0
    [PASS] all 48 layers — 7.17 GiB per rank
    6.00 MiB row -> peer replica row: p50 25.0 us (252 GB/s)
    buffer_ptrs_dev present: True

It clears the exact obstacle NVSHMEM could not. `get_mem_pool(device)` returns a
`torch.cuda.MemPool`, so the expert weights are allocated by **ordinary `torch.zeros`**
inside `torch.cuda.use_mem_pool(...)` and stay ordinary tensors afterwards — weight loading
and EPLB's `rearrange` keep plain semantics, which the matmul check confirms. `rendezvous`
then makes them addressable, `get_buffer(peer, shape, dtype)` hands back **a torch tensor
view of the peer's memory**, and writing one expert is `peer_view[row].copy_(my_row)` — a
*slice*, which NVSHMEM's tracking table refuses. `buffer_ptrs_dev` is a device-resident
peer-pointer table, so a Triton kernel could compute the destination with no host at all.

| | NVSHMEM register (a) | NVSHMEM staged (b) | PyTorch symm-mem (d) |
| --- | --- | --- | --- |
| weight allocation | **impossible** | untouched | ordinary torch inside a MemPool |
| staging buffer | — | one expert | none |
| extra local copy | — | +6.3 us | none |
| destination may be a slice | — | — | yes |
| one 9 MiB expert | — | 33.0 + 6.3 = **39.3 us** | about **37.5 us** |
| dependencies | nvshmem4py + 2 env vars | nvshmem4py | PyTorch only, already used here |
| device-side pointer table | — | — | `buffer_ptrs_dev` |

The 252 GB/s is below NVSHMEM's 285 because `copy_` launches a generic copy kernel rather
than a tuned path; a Triton kernel over `buffer_ptrs_dev` would likely close that, and the
design wants such a kernel anyway.

### Decision: (b), the staged NVSHMEM route

Chosen 2026-08-29. **(b) does not touch how vLLM allocates expert weights at all**, where
(d) needs a `use_mem_pool` context around weight creation inside `FusedMoE` and the quant
methods. The staged route's price is one expert-sized symmetric buffer and a 6.3 us local
copy — 0.06% of a 67.87 ms prefill step once the 33.0 us put is added — bought against zero
intrusion into the weight-loading path.

**(b) needs no environment variables.** `NVSHMEM_CUMEM_GRANULARITY` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` were only ever attempts to make
registration work for (a); the staged path is what `probe_nvshmem.py` validated 5 of 5 with
a stock configuration.

(d) stays on the record because it is the better shape if the staging copy or the buffer
ever matters, and because it needs no NVSHMEM at all.

# 2026-08-29 (ticket 03) — the fused counting kernel, and what it revealed

Implemented and measured. The elementwise tail of the predicted-count path is now one
Triton kernel, tested by equality against the retained reference over 200 randomised cases
plus padding boundaries, out-of-range ids, both router id widths and an empty batch.

Profiled at the parameters the earlier arms used (`DOMAIN=ko CONC=8 NUM_PROMPTS=64
OUT_LEN=4 BUDGETS=0`), per prefill window summed over 8 ranks:

| arm | nccl | other | total | kernels | kernels/layer/rank |
| --- | --- | --- | --- | --- | --- |
| `off`, no prediction | 326.8 ms | 117.2 | 542.9 | 9366 | — |
| prediction, 12 elementwise ops | 649.2 ms | 131.9 | 880.2 | 15270 | **17.2** |
| prediction, fused kernel | 593.3 ms | 130.3 | 825.9 | 11812 | **7.1** |

**Launches per source layer fell 17.2 to 7.1, a 59% cut — and the extra collective waiting
fell only 17%, 322.4 ms to 266.5 ms.** That is not proportional, and the gap is the finding.

### The per-layer cost decomposes, and the launch-driven part is the smaller one

The earlier linearity test varied the *number of predicting layers*, which moved launches,
host synchronisations and AllGathers together — it could not separate them, and this file
said so at the time. Fusion moves only the launches, so the two components separate:

    per-layer extra collective waiting  =  A (scales with launches)  +  B (fixed per layer)

    before fusion   7.50 ms/layer          launches at 1.00x
    after  fusion   6.20 ms/layer          launches at 0.41x
    -> A = 2.21 ms/layer  (30%)
       B = 5.28 ms/layer  (70%)

**B is essentially the host synchronisation.** The other per-layer fixed cost, the snapshot
AllGather, measures 0.40 ms per forward per rank, which is 0.074 ms per layer summed over
ranks — under 2% of B. So the 5.28 ms is the D2H synchronisation that `plan_and_launch`
performs once per predicted layer, consistent with the 11.02 ms single
`cudaEventSynchronize` measured earlier.

### The projection this corrects

This file previously extrapolated the prediction arm from +7.6% to about +2.7% mean TTFT on
the assumption that cost scaled with launch count. Measured, fusion takes it to about
**+6.3%**, because 70% of the per-layer cost does not scale with launches at all.

**So ticket 06 matters more than ticket 03**, which reverses the ordering intuition this
ticket set was built on. Removing the host synchronisation attacks 70% of prediction's added
cost; fusing the counting attacks 30% and has now delivered its share. Both are still worth
having — 17% is real and the kernel is written — but the arithmetic closes or fails on the
device-side plan, not on the kernel.

Against the 5.05% ceiling: +6.3% is still above it, and only ticket 06 can bring it below.
If removing the host sync recovers most of B, the floor lands near 1% and the feature is net
positive for the first time. That remains an extrapolation, and the lesson from this ticket
is precisely that a per-layer cost can have a large component the obvious lever does not
touch.

## Ticket 03's TTFT re-measurement: the method cannot resolve the effect

Ran the three arms again with the fused kernel, at the configuration the earlier ones used
(`ko`, 400 requests, CONC=8, OUT_LEN=1), to turn the extrapolated `+6.3%` into a measurement.
The guard passed and the placed arm activated on all 43 reachable layers.

| arm | before, mean TTFT | after, mean TTFT | before req/s | after req/s |
| --- | --- | --- | --- | --- |
| feature disabled | 184.70 ms | **132.21 ms** | 43.21 | **60.30** |
| prediction only | 198.81 ms (+7.6%) | 164.20 ms (+24.2%) | 40.14 | 48.56 |
| placing | 242.79 ms (+31.5%) | 258.21 ms (+95.3%) | 32.87 | 30.91 |

**The two runs are not comparable, and the reason is in the first row.** The `off` arm is a
stock server carrying none of this feature's code, and it got 28% faster on mean TTFT and 40%
faster on throughput between the runs — same 400 requests, same 390,152 input tokens. Nothing
in the commit can do that; the machine was simply in a better state, with weights and JIT
caches warm after a day of runs. **The baseline moved eight times further than the ceiling
being chased.**

So no cross-run conclusion is available, and the honest statement is that this configuration
cannot measure a 5% effect.

### What the within-run ratios do and do not say

Each run measures its own `off` arm minutes from the others, so within-run ratios are the
defensible comparison. They say the feature costs *more* after fusion: prediction +7.6% ->
+24.2%, placing +31.5% -> +95.3%.

That is consistent with fusion working rather than failing, which is worth spelling out. The
feature's overhead is largely fixed per-layer host and launch work, and that does not shrink
when the GPU-side and queueing parts of the baseline get faster. A fixed overhead against a
40% faster baseline is a larger fraction of it. The same mechanism was recorded earlier from
the profiling side: launch-bound behaviour gets relatively *worse* on faster hardware, since
launch cost is unchanged while GPU work shrinks.

The lower-variance measurement is the one that already showed fusion working, because both of
its arms come from the same run: launches per source layer 17.2 -> 7.1, and the extra
collective waiting 322.4 -> 266.5 ms, a 17% reduction.

### Consequence for ticket 08

Its method needs fixing before any TTFT verdict, and this is the ticket's own risk rather
than a detail. Options, in increasing cost:

- **Repeat each arm** several times within one run and report a distribution. Cheapest, and
  it at least bounds the variance instead of ignoring it.
- **Interleave the arms** rather than running them in sequence, so a drift in machine state
  cannot land entirely on one of them. Requires restarting servers per block.
- **Make the profile-based attribution primary** and TTFT supporting. GPU time inside
  annotated windows is measured within a single run and has shown a 17% change cleanly,
  where mean TTFT could not see 40%.
- **Report throughput as well as TTFT.** It is less tail-sensitive and moved coherently in
  both runs (43.21/40.14/32.87 and 60.30/48.56/30.91).

Recorded as a warning rather than a fix: no TTFT number from this harness should be quoted
across runs until arms are repeated or interleaved.

## Fixing the ruler: the operating point was wrong, and so were three of my own numbers

Prompted by the observation that concurrency 8 over DP=8 is one request per rank, and that
what the feature can shorten is a prefill *step* — so the point to measure at is the highest
concurrency that does not queue, where TTFT is the step and not the wait behind other
requests.

### The knee is 16, and the theory that said 64 was wrong

`run_concurrency_knee.sh` sweeps one stock server, so no restart drift enters the comparison.
1024 prompts, `ko`, OUT_LEN=1:

| concurrency | mean TTFT | median | p99 | req/s | median vs c=1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 112.4 | 111.6 | 143.6 | 8.9 | 1.00 |
| 4 | 130.3 | 126.5 | 141.6 | 30.6 | 1.13 |
| 8 | 141.2 | 125.4 | **629.2** | 56.5 | 1.12 |
| **16** | 137.4 | **138.7** | 162.3 | 113.5 | **1.24** |
| 32 | 195.8 | 184.2 | 354.4 | 157.7 | 1.65 |
| 48 | 246.6 | 247.0 | 346.2 | 188.6 | 2.21 |
| 64 | 301.8 | 281.9 | 494.9 | 199.6 | 2.53 |
| 96 | 386.5 | 414.2 | 557.7 | 228.7 | 3.71 |
| 128 | 439.7 | 421.9 | 607.5 | 263.9 | 3.78 |

**Concurrency 16 is the highest that does not queue** — median TTFT 1.24x the unloaded value,
against 1.65x at 32 — and it carries twice the throughput of 8. Note also that 8, the point
every earlier figure in this file was taken at, has a p99 of 629 ms against 16's 162: it has a
tail anomaly that 4 and 16 both lack, which is its own reason not to measure there.

**The theoretical estimate of 64 was wrong and the error is instructive.** It assumed a rank
fits eight 958-token prompts in one 8192-token forward, times eight ranks. But under
`allgather_reducescatter` every rank's MoE runs over *all* DP ranks' tokens, so per-rank
compute grows with **total** concurrency rather than with per-rank batching. The no-queue
limit is therefore far below the batching limit.

### Cost and ceiling in one unit at last, and three corrections

The anchors that disagreed 8x disagreed because one of them was **summed across streams**.
Attributed GPU time adds the durations of kernels on the compute stream, the token-collective
stream and the prediction stream, which overlap; a prefill window's **wall-clock** duration is
what compares with TTFT. Measured on the same run, three arms, median across ranks:

| arm | window wall-clock | vs off |
| --- | --- | --- |
| feature disabled | 85.7 ms | — |
| prediction only | 105.5 ms | **+23.1%** |
| placing | 129.4 ms | **+51.0%** |

And in the `off` arm, per rank, within the same run:

    window wall-clock        86.9 ms
    kernel time              72.9 ms   -> 84% occupancy
    expert GEMM              12.42 ms  -> 14.29% of the window
    perfect balance saves     5.83 ms  -> 6.71% of the window
    at the online 24% recovery 1.40 ms -> 1.61% of the window

Three things I had written here are corrected by this:

1. **The ceiling is 6.71% of a prefill window, not 5.05%.** The 5.05% divided by attributed
   time, which double-counts overlapping streams, so it understated the expert GEMM's share:
   14.29% of wall-clock rather than 10.76% of a summed total.
2. **The window is not mostly idle.** I read an 8-rank-summed kernel total as a per-rank one
   and concluded occupancy was ~10%. Seven of eight ranks are at 84%; only `dp0` sits at 35%,
   and it is the rank that receives less work.
3. **The ceiling expressed against TTFT is not 2.6%.** That figure came from the same
   summed-time denominator. With the wall-clock window at 86.9 ms and mean TTFT around 132 ms
   at this concurrency, a prefill step is most of TTFT, not half of it.

So at concurrency 8: ceiling 6.71%, prediction costs 22.8%, placing costs 50.3% — 3.4x and
7.5x the ceiling. Both are properties of the wrong operating point, and both are being
re-measured at the knee.

## At the knee, with a ruler that works: the first readable numbers

Concurrency 16, three passes per arm, arms interleaved within each pass, plus one profiled
three-arm run at the same point. `report_arm_spread.py` and the profile agree, and for the
first time the effect exceeds the measurement's own noise.

### TTFT, three passes each

| arm | mean TTFT | median | p99 | req/s | own spread | vs off |
| --- | --- | --- | --- | --- | --- | --- |
| feature disabled | 172.48 ms | 161.86 | 315.42 | 92.31 | **4.9%** | — |
| prediction only | 198.46 ms | 190.16 | 307.05 | 79.65 | 4.5% | **+15.1%** |
| placing | 344.97 ms | 210.17 | **3406.86** | 45.93 | 14.0% | **+100.0%** |

**The baseline's own spread is 4.9%**, so this method resolves effects above roughly 5% and
both of these clear it. Every earlier TTFT figure in this file was a single pass at
concurrency 8 and had no such floor to be judged against.

The placed arm's mean and median diverge sharply — 345 against 210 — because its p99 is
3407 ms. The tail is the cost, not a uniform slowdown.

### The same run, profiled: the mechanism is visibly working

| arm | window wall-clock | occupancy | expert GEMM | share of window |
| --- | --- | --- | --- | --- |
| feature disabled | 94.6 ms | 72% | 10.01 ms | 10.58% |
| prediction only | 115.7 ms | 99% | 10.93 ms | 9.45% |
| placing | 160.4 ms | 86% | **6.88 ms** | 4.29% |

**Balancing shortens the expert GEMM from 10.01 ms to 6.88 ms**, a 3.13 ms saving against a
perfect-balance ceiling of 4.70 ms — so the placement recovers **67% of the ceiling**. That is
the first direct sight of this feature doing the thing it was built to do, and it is well
above the 24%-of-excess figure carried until now, which was measured on token counts rather
than on time.

The prediction arm reaching 99% occupancy is worth noting on its own: prediction fills the
window with work rather than lengthening the idle parts.

### The accounting, and what ticket 06 decides

    realised benefit, expert GEMM shortened        3.13 ms   (67% of the 4.70 ms ceiling)
    cost today, prediction only                 +21.10 ms   = 7x the benefit
    cost today, placing                         +65.80 ms   = 21x the benefit

    after 06, prediction's residual              +3.10 ms   extrapolated at 0.147, the
                                                            measured share of per-layer cost
                                                            that is not the host sync
    versus the benefit                            3.13 ms

**Prediction's residual after ticket 06 comes out equal to the benefit, to within 1%.** Do not
read that as a positive margin: the extrapolation carries far more uncertainty than 1%, so the
honest statement is that prediction alone would be **indistinguishable from break-even**.

The deciding term is the one that cannot be extrapolated: placing costs **+44.7 ms beyond
prediction**, and after 06 that becomes a one-sided put with no host involvement. Whether its
residual is nearer 1 ms or nearer 20 ms is what ticket 06 measures, and it is the difference
between a feature that pays and one that does not.

So the direction of the answer to "should TTFT improve once the host is out of the loop" is:
the benefit is real, larger than previously credited, and about the same size as the residual
overhead. It is a coin flip that only 06 settles.

### A false alarm in the guard, fixed

The nine-arm run failed with `MEASURED NOTHING` on healthy data. Repeated runs label arms
`off-r1`, `0-r2`, and the guard compared the whole label against `"off"`, so every repeat
looked like a placing arm and the stock baseline was faulted for having no dump and no
activation. Fixed by classifying on the budget with the repeat suffix stripped, with tests for
both directions. A guard that cries wolf is a guard that gets deleted, so this counts as a
defect in the guard rather than a nuisance.

## Ticket 05, 2026-08-30: the one-sided transfer lands, and the recorded ordering claim was wrong

`probe_replica_transfer.py`, 8x H100 SXM, driving the production `OneSidedExpertTransfer`
and `ReplicaTransferEngine` rather than a reimplementation, over all 56 ordered rank pairs.

| check | result |
|---|---|
| transport up after NCCL | 8 PEs, 9.00 MiB staging buffer |
| replica row byte-identical, every rank pair | **112/112** weight tensors |
| same, barrier removed (control) | **61/112** — 51 wrong |
| put + barrier + staging copy, one expert | **p50 53.7 us** (49.3 / 68.5) |
| slowed transfer, exposed wait | 0.456 ms against 0.013 ms unslowed |

**A plain CUDA event does not tell a consumer that a peer's put landed, and this project
had it recorded as verified.** `probe_nvshmem.py` checked it with a `dist.barrier()` inside
the region under test, so the barrier established arrival and the event was never
load-bearing. Ticket 05 inherited the claim, and the transfer would have shipped reading
whatever the row happened to hold — plausible floats, wrong logits, nothing raised. The
control arm above is the direct measurement: **51 of 112 tensors wrong** with the barrier
removed and everything else identical.

`nvshmemx_barrier_all_on_stream` fixes it for 13.9 us of the 53.7 us span, with no host
involvement. It is collective, which is a feature here rather than a cost: the plan is
identical on every rank by construction, so every rank issues the same barriers, and that
avoids the per-rank timing decisions that deadlocked this branch twice. It also replaces a
collective that was already on this path — pynccl's `execute()` was called once per layer
on every rank for the same reason.

**43 layers of transfer is 2.3 ms.** The host synchronisation it removes is 5.28 ms *per
layer*. That is the whole case for the device-planned path, and it is the first time both
sides of it have been measured on the same machine.

### Three defects found by running it, none of which a unit test would have caught

1. **The exposed-time markers were not timing events.** `torch.cuda.Event()` without
   `enable_timing=True` refuses `elapsed_time`, so the accounting ticket 08 needs raised
   `ValueError` on all 8 ranks. The drain event correctly does *not* ask for timing — a
   timed event costs a device write on every record, and that one is on the per-forward
   path.
2. **The slowed-transfer arm measured nothing.** The extra puts were queued *after* the
   transfer's drain event, so the wait saw only its own overhead: 0.012 ms, printed as a
   pass. Moving the delay in front of the transfer took it to 0.456 ms. A threshold guessed
   from the transfer time then produced a false failure at 0.433 vs a made-up 0.588 ms bar,
   so the bar is now the measured unslowed wait.
3. **The probe printed `FAIL` and exited 0.** Fixed by counting failures in `report`.

### Teardown: two things that look like transfer bugs and are not

NVSHMEM keeps its own reference count beside Python's. Dropping the last reference to the
symmetric buffer leaves the allocation tracked, so `finalize` reports every buffer leaked
and then segfaults **every rank after all results have printed** — which reads as a
transfer failure. `free_tensor` before `finalize` is required.

Worth recording because it cost the most time here: I hypothesised that the leak was a race
in the free's collectivity (nvshmem4py documents `free` as collective, and only on `free`,
not on the `free_tensor` wrapper), and built a five-script bisection around that. The
actual cause was that my edit adding `free_tensor` to `close()` had silently failed to
apply, so the code under test still only dropped the reference. The bisection's "clean"
arms were the ones that called `free_tensor`; the "leaking" arms were the ones that went
through `close()`. **Verify that an edit applied before drawing a conclusion from the
behaviour of the file.**

## Ticket 06's precondition, 2026-08-30: a device-issued put works, and is not slower

Ticket 06 asks for no host synchronisation on the per-forward path. Ticket 05's transport
cannot deliver that on its own, and the reason is worth stating precisely because it is the
same reason `ncclSend` could not: **a host-issued put takes its peer, its source pointer and
its byte count as host integers, consumed when the host enqueues.** A plan living on the
device cannot aim one, whatever stream it goes on. Moving the planner to the device (ticket
04) removes the planner's host read; the transfer's remains.

Two escapes exist. Issuing every put the plan might have chosen and masking the rest costs
16 rows to 7 peers, about 63 MiB of egress per rank per layer and roughly 230 us — 10 ms
across 43 layers, against a ceiling of about 5% of a 95 ms window. That spends the whole
benefit to save the sync. The other is to issue the put from inside a kernel, which reads the
plan where it already lives.

`bench/probe_device_put.py`, 8x H100:

| check | result |
|---|---|
| device library ships with the wheel | `libnvshmem_device.a`, sm_90 |
| NVRTC compiles and nvJitLink links it | PASS |
| device-issued put lands, plan never read on the host | **8/8 ranks** |
| 9.00 MiB device-issued put + barrier | **p50 47.6 us** (45.3 / 67.0) |

**47.6 us against 53.7 us host-issued**, so the device-issued route is not a trade at all on
this hardware — it is the same time with the host removed. That was not a given: the first
version measured **206 us**, because one block was moving 9 MiB with its own load/store units
rather than the copy engine. Splitting the payload across 32 blocks is what closes it.

### The toolchain recipe, because each step's error names the wrong cause

Seven things had to be right, and six of them fail with a message pointing somewhere else.

1. `nvidia.nvshmem` is a **namespace package**, so `__file__` is None and `Path(None)` raises
   a `TypeError` about `__fspath__`. Locate it through `__path__`.
2. NVSHMEM's headers include `cuda_runtime.h`, which NVRTC does not supply. Add
   `nvidia-cuda-runtime`'s include directory or the compile dies with a "catastrophic error"
   naming that file, which reads as a broken toolchain.
3. It then wants `cuda/std/cstdint`, which is libcu++ from `nvidia-cuda-cccl`. A third `-I`,
   and it only surfaces once the second is supplied.
4. Link **relocatable PTX against `libnvshmem_device.a`**. The `.bc` files beside it are raw
   LLVM bitcode and nvJitLink rejects them as both `NVJITLINK_INPUT_LIBRARY` and
   `NVJITLINK_INPUT_LTOIR`.
5. `LinkerOptions` has no `relocatable_device_code`; it is a `ProgramOptions` setting.
6. Build the kernel **after** `nvshmem.init`. `cuda.core` loads the module into its own
   current context and NVSHMEM registers in the context init bound; build first and the two
   differ, failing with `CUDA_ERROR_INVALID_HANDLE`.
7. Register with **`library_init`, not `module_init`**: `cuda.core` produces a `CUlibrary`,
   and `module_init` rejects it with that same `CUDA_ERROR_INVALID_HANDLE` — two unrelated
   causes behind one message. `module_finalize` is unusable with anything, because
   `module_init` never sets the `finalize_handle` it then requires.

And the launch itself: `cuda.core.launch` needs a `cuda.core.Stream`, so torch's stream has to
be wrapped through the `__cuda_stream__` protocol rather than replaced — launching onto a
stream torch does not know about would put the transfer outside the ordering everything else
relies on. Kernel arguments must be **numpy scalars**; `ctypes` values and bare Python ints
are both rejected, the latter for having no unambiguous width.

### One of my own checks was the bug

The byte check reported "1/8 ranks received their neighbour's payload" and held at 1/8 across
a kernel rewrite and an added device-side `quiet`. The transfer was correct the whole time:
`torch.tensor([bool])` gave a **bool** tensor, and `all_reduce` with SUM over bool saturates
back to bool, so eight agreeing ranks reduce to 1. Two hypotheses were investigated before the
reduction was. **When a count is stuck at a suspiciously round value, check the counter.**

The device-side `quiet` added while chasing that stays, and the comment blaming it for the 1/8
has been corrected: the check passes without it. Two passing runs of one payload is not
evidence that a stream barrier completes a device-issued nbi put, this project has already
shipped one ordering claim resting on exactly that kind of observation, and the quiet costs
about 1 us.

## Ticket 06's wiring, 2026-08-30: every piece works, and the put kernel crashes a server

The three device-side pieces are built, tested and committed. Each is verified against the
host path it replaces, because the host path is the code that was measured working end to
end and disagreement with it means tokens routed to a row holding another expert's weights.

| piece | verified how | result |
|---|---|---|
| plan, fully tensorised | bit-identity to `plan_replicas` over the randomised sweep, CPU and CUDA | agrees, and `set_sync_debug_mode("error")` passes |
| publish, device scatter | `apply_replica_maps` plus the layout edit as oracle, 200-plan random sequence | agrees on all four map tensors |
| transfer, two kernels | 8 ranks, all 56 ordered rank pairs | **112/112** byte-identical, **p50 36.7-40.0 us** |

Two of those numbers are worth putting beside the thing they replace. The host-issued put of
the same payload measured **53.7 us**, so issuing from a kernel is not a trade — it is the
same time with the host removed. And an empty plan moves nothing, decided on the device on
8/8 ranks, which is what lets a forward that places nothing avoid a host round trip to find
that out.

### In a real server it reaches all 48 layers and then segfaults

Wired into `attach_placement_coordinator` behind `device_issued_transfer`, an 8-rank DP=EP=8
server starts, initialises NVSHMEM after NCCL in every worker, logs `device-issued transfer
active on 48 layers, 8 ranks` on all of them, and launches a device-issued transfer for every
one of the 48 layers. Then, at the startup EPLB rearrange, three workers segfault inside
`progress_channels` / `nvshmemi_proxy_progress` — NVSHMEM's host-side proxy thread.

**Bisected to `put_expert`, with four server runs and one standalone reproduction attempt:**

| arm | outcome |
|---|---|
| NVSHMEM up, `transfer()` skipped entirely | **healthy** — so it is not NVSHMEM coexisting with vLLM |
| barrier only, both kernels skipped | **healthy** — so 48 pipelined stream-ordered barriers per forward are fine |
| barrier + drain, put skipped | **healthy** — so it is not the drain and not the launch machinery |
| full | **segfault**, 3 workers |

And it does **not** reproduce standalone. A script that pipelines 48 transfers per round for
5 rounds with no synchronisation, then runs an all-reduce and a host barrier with the
predictive stream still unwaited — the closest thing to what the server does — survives every
time. So the trigger is something the server supplies that the reproduction does not: the
model's own weight tensors as the put source, real plan values, or the rearrange's own use of
those tensors. That is the next thing to establish, and the cheap way in is to print the plan
and the resolved source address from the kernel for one layer rather than to keep guessing —
two hypotheses were investigated by reasoning tonight and both were wrong.

`device_issued_transfer` therefore **defaults to False**. Enabling it today loses a worker,
and a config default that crashes is worse than one that measures the wrong thing. The host
path is unaffected: the same server on the same commit comes up healthy, logs 8 activation
lines, and serves.

### Two debug switches are kept, and they earned it

`VLLM_PREDICTIVE_SKIP_DEVICE_TRANSFER` and `VLLM_PREDICTIVE_DEVICE_TRANSFER_STAGE`
(`full`, `barrier-only`, `put-only`, `drain-only`) are what produced the table above. The
distinction between "NVSHMEM cannot coexist with this engine" and "one of my two kernels is
wrong" took three server starts to make and would have taken far longer to make by reading
code. `no-barrier` and `put-only` produce wrong bytes on purpose and must never serve.

### The in-kernel quiet was removed, and the note that blamed it was wrong

The put kernel called `nvshmem_quiet()` because the NVSHMEM spec completes a non-blocking put
with one. It was removed while chasing the segfault, and byte equality still holds at 112/112
over every rank pair — so the stream-ordered `barrier_all` is what establishes arrival, and
the quiet was not load-bearing. Removing it did **not** fix the crash, and an earlier comment
in the probe claiming the quiet was needed because "7 of 8 ranks read a buffer the bytes had
not reached" has been corrected: that observation was a bool `all_reduce` saturating, not a
missing quiet.

## Ticket 06, 2026-08-30 afternoon: the segfault was the plan's ownership, and the path now runs

The put kernel that crashed three workers is fine. What was wrong is who owned the plan it
read, and the whole chain is now measured rather than argued — the previous session
reasoned through two hypotheses and both were wrong, so this one starts from an experiment.

`DevicePlacementCoordinator.plan_and_launch` built the `[4]` int64 plan on the **compute**
stream and launched two kernels that read it on the **predictive** stream, then returned,
dropping the last reference. PyTorch's caching allocator returns a freed block to the pool
of the stream it was allocated on and hands it to the next allocation there with no
synchronisation, because cross-stream use is supposed to be declared with `record_stream`.
Nothing in `vllm/distributed/eplb/` called it. So the compute stream overwrote the plan
while the kernels were still queued behind a 40 us transfer.

| step | what was measured |
| --- | --- |
| the mechanism, pure torch | a consumer on a second stream read `[1, 11, 1, 2]` — the poison, not the plan |
| the allocator | the freed `[4]` int64 block came back on the **5th** allocation of that size |
| the production classes | the transfer moved **expert 11** where the plan said expert 3 |
| the link to the crash | a recycled block naming **PE 12345 of 2** killed rank 0 with **exitcode -11**, SIGSEGV |

That last row is the server's crash reproduced on demand. In a real forward the overwrite
is not another plan but whatever the engine allocated, so `pe` is an arbitrary int64 and
NVSHMEM's proxy thread dereferences a peer address computed from it.

**Fix:** each layer gets a plan row in a `[num_layers, 4]` buffer the coordinator owns for
its lifetime. No allocation on the path, nothing to recycle, and no `record_stream`
needed. The row is rewritten only by a later forward, by which point
`activate_and_publish` has already made the compute stream wait on that layer's transfer
event.

`bench/probe_plan_lifetime.py` is the probe, and `tests/distributed/test_device_coordinator.py`
the regression test — which fails on the pre-fix code with "the plan's memory was handed to
another allocation while the transfer was still queued to read it".

### Why every earlier check passed

All of them held the plan in a local variable and synchronised immediately after launching.
`probe_device_transfer.py` does exactly that, 112/112 times. A fake that reads the plan when
the transfer is *issued* cannot catch this; the new probe's fake reads it late, behind a
queued delay, which is what a kernel does.

### End to end on 2 GPUs

This node came back from a restart with **2 H100s instead of 8**, so the runtime scope check
that pinned DP=8 was relaxed to "at least 2" with a startup warning that no measured figure
for this feature survives the change — EP size sets the per-rank expert count and with it the
imbalance there is to recover.

`bench/run_device_transfer_smoke.sh`, DP=EP=2, dummy weights:

    device path armed on 2 workers, 86 layer-launches (43 reachable x 2 ranks)
    replicas actually placed, from the device counter: 8
    host-path fallbacks: 0    crash signatures: 0

The placement counter is the evidence that matters: "transfer launched for layer" is logged
whether or not the plan found anything, so a run can log 86 launches and place nothing. It
is now lowerable through `VLLM_PREDICTIVE_PLACEMENT_REPORT_EVERY`, because at 50 forwards a
functional run never reaches its first report.

### Output equivalence, real weights, DP=2

`verify_source_rank_routing.sh` gained `MODE=device` and `MODE=host`, so the same test can
attribute a difference to the transport:

| arm | greedy text, 4 x 96 tokens | largest first-token logprob delta |
| --- | --- | --- |
| canonical vs canonical (CONTROL) | identical | **0.000000** |
| canonical vs device-issued placement | identical | 0.253 (8 replicas placed) |
| canonical vs host-issued placement | identical | 0.337 (1 replica placed) |

So the device transport introduces nothing the host path does not: placement itself moves
low-probability logprobs, because a replica changes which rank contributes a token's expert
output and the combine sums them in another order. The harness's 0.05 tolerance was invented
rather than calibrated — recorded here before, and now shown to fail on the path that was
already measured working end to end. **Greedy text equality is the criterion this test can
actually decide**, and it holds.

Not decided: nothing in a real server checks that a *dynamically* placed replica holds the
bytes of the expert it claims. The startup checksum check runs where nothing is placed and
says so ("0 pairs, vacuous").

### Two defects found on the way, both in things that were supposed to be checked

**`BLOCK_SIZE_M` was never resolved from the kernel.** Every server logged "could not
resolve ... falling back to 128" and nobody had read the traceback: `resolve_moe_block_size_m`
passed `model.expert_weights[0]`, which EPLB registers as **flattened** `[rows, numel]`
views — measured `(65, 3145728)` and `(65, 1572864)` — into a helper that unpacks
`w2_shape` as `(E, K, N)`. The unit test passed because its fake model supplied
three-dimensional tensors, which the real one does not have. Now read from the layer's
`moe_config`, and the fake matches reality. It resolves to 128 at M=16384 on this node, so
no behaviour changed — but it was a guess and is now a reading.

**`torch.stack` blocks the host on its first call in a process**, 50-101 ms, while its
kernel loads. Not a per-call synchronisation: calls 1-3 measure 0.01-0.04 ms and leave a
side stream running. It is a trap for any measurement, and it silently drained the queued
delay in the first version of the lifetime probe, turning a defect-reproducing arm into a
vacuous pass. Warm the path, then measure — and assert the delay was still pending, which
both the probe and the test now do.

**And a correction of my own first reading:** I reported that `torch.stack` over 0-dim
tensors was a per-layer device synchronisation on the production path. It is not; it is the
one-time kernel load above. `set_sync_debug_mode("error")` does not catch it either, which
is why the sync criterion is now tested by measurement: 100 ms queued on the compute stream,
and the path must return in microseconds. A deliberately inserted `int(plan[0])` makes that
test fail, so it is sensitive.

### Four arms at DP=2, the first TTFT numbers for the device transport

Korean prompts, 120 requests at concurrency 16, `OUT_LEN=1` so the run is prefill-weighted,
every arm measured twice in interleaved blocks. **DP=2, so none of these compares to the
DP=8 figures** — with two ranks the per-layer critical-path imbalance is 1.17 where eight
ranks give 1.885, so there is far less to win and the same 43 transfers to pay for.

| arm | mean TTFT, r1 / r2 | p99 TTFT, r1 / r2 |
| --- | --- | --- |
| `off`, stock server | 406.36 / 260.31 ms | 997.32 / 436.86 ms |
| `0`, prediction only | 259.30 / 263.50 ms | 384.82 / 388.95 ms |
| `43:host`, host-issued transfer | 387.66 / 353.57 ms | 863.80 / 659.33 ms |
| `43:device`, device-issued transfer | 363.20 / 360.34 ms | 503.96 / 483.50 ms |

**The stock arm moved 56% between two repeats of an identical configuration** (406 against
260), so nothing here can be compared against stock. That is the same drift recorded before
at 28%, worse this time, and it is why the arms are measured in interleaved blocks.

**What the device transport buys is the tail.** Against the host-issued path its p99 is 42%
and 27% lower in the two repeats, with no overlap between the arms' ranges, which is what
removing a per-layer host synchronisation should look like: an 11.02 ms `cudaEventSynchronize`
per predicted layer stalls the engine and lands in the tail, not in the mean. The mean
difference is 2.4% and sits inside the host arm's own repeat spread of 9.6%, so **no mean
claim is available from this run** — only the p99 one.

**The mechanism is unchanged by the transport, which is the check that matters.** Both place
from the same planner, and both recover the same share of the full-prefill critical-path
excess:

| arm | full prefill, critical path | excess removed | partial prefill |
| --- | --- | --- | --- |
| `43:device` r1 / r2 | 1.172 -> 1.109 / 1.174 -> 1.113 | **36.6% / 34.8%** | 12.9% / 13.3% |
| `43:host` r1 / r2 | 1.172 -> 1.110 / 1.174 -> 1.111 | 36.1% / 36.2% | 13.4% / 13.0% |

`connected: true` on all four, 120 of 120 requests completed, 0 failed. The device arm
reports **217 replicas placed over 30 forwards** from its own device counter, with no
host-path fallback and no crash signature in either repeat.

So at two ranks the feature recovers a *larger* share of a *much smaller* imbalance — 35% of
an excess of 0.17 rather than 24% of 0.885 — while paying the same transfer count, and
placement costs **+38% mean TTFT against prediction alone**. That is the expected direction
and it says nothing new about the verdict, which is a DP=8 question. What is new is that the
device transport's cost is now measured rather than projected, and it is lower than the host
path's on the tail.

Two harness notes for whoever runs this next. `run_e2e_placement.sh` takes `DP` and arms of
the form `43:device` / `43:host`, so both transports interleave inside one pass instead of
being compared across invocations — which, given the 56% drift above, would have compared
machine states. And it now runs both the server and the bench client from `.venv/bin/python`:
the `vllm` console script runs under system python, where the bench extra's pandas vanished
with the pod restart, and where the working tree is off `sys.path` anyway.

## Ticket 07, 2026-08-30: the window moved, and it exposed the real cost centre

The launch is now at the predicting layer's MoE tail, so the overlap window is the target
layer's Attention and `prediction_lookahead_layers` defaults to 1. Reachable layers went
43 -> 44, because a lookahead of 1 leaves one fewer trailing layer unbound.

### Four repeats finally give a usable baseline, and prediction is nearly free

DP=2, Korean prompts, 120 requests at concurrency 16, `OUT_LEN=1`, three arms x 4 repeats
interleaved. **The stock arm's spread fell to 1.6% at four repeats**, against 56% at two —
so this is the first run on this node whose baseline can be compared against at all.

| arm | mean TTFT per repeat | median | spread | p99 median | vs stock |
| --- | --- | --- | --- | --- | --- |
| `off`, stock | 257.1 / 258.4 / 257.3 / 261.2 ms | 257.8 ms | 1.6% | 435.2 ms | — |
| `0`, prediction only | 248.9 / 261.9 / 268.4 / 267.2 ms | 264.6 ms | 7.4% | 395.1 ms | **+2.6%** |
| `43:device`, placing | 342.4 / 343.9 / 338.2 / 303.7 ms | 340.3 ms | 11.8% | 432.4 ms | **+32.0%** |

**Prediction with its infrastructure now costs +2.6% of mean TTFT**, against +7.6% measured
at DP=8. Placement adds the other +29%, and the goal of parity with stock stands or falls on
that number, not on prediction's.

### The profile says the transfer is free and the *arrival barrier* is not

Two torch profiles at DP=2, prediction-only against placing, restricted to the
`execute_context` prefill windows. Per-kernel, inside those windows, on the placing arm:

| kernel | calls | p50 | p90 | max | total | per prefill window |
| --- | --- | --- | --- | --- | --- | --- |
| `put_expert` | 176 | **1.2 us** | 37.3 us | 47.9 us | 1.42 ms | 0.11 ms |
| `drain_expert` | 176 | **1.1 us** | 1.3 us | 28.1 us | 0.53 ms | 0.04 ms |
| `barrier_on_stream_kernel_threadgroup` | 176 | 5.0 us | **1151.4 us** | **5898.2 us** | **57.23 ms** | **4.40 ms** |

So the transfer this project spent four design rounds on costs **1.95 ms of the whole
window** across both kernels, and `nvshmemx_barrier_all_on_stream` — the arrival mechanism —
costs **29 times that**. Ticket 05 measured the same barrier at **13.9 us** in isolation;
under load its p90 is 1.15 ms and its maximum 5.9 ms. That is not the barrier's own cost. It
is **rank arrival skew, made blocking**: a barrier cannot complete until the peer reaches it,
so every placed layer couples the ranks' timelines once, 44 times per forward.

And it is not hidden by the window it was supposed to sit in. Of the barrier's 57.23 ms,
**1.5% is concurrent with Attention and 11.0% with the expert GEMM** — 87% overlaps no
compute at all, because the compute stream is waiting on the transfer event, which waits on
the barrier, which waits on the other rank.

The attributed prefill step grows accordingly, 48.6 ms to 150.9 ms on dp1, with NCCL per
layer going 166 us to 1415 us while MoE per layer *falls* 368 us to 181 us. The placement is
working and the coupling costs more than it returns.

**The fix this points at is pairwise arrival.** Only the target rank needs to know its
sender's put landed, and the plan is rank-identical, so each rank can compute whether it is
that target on the device. NVSHMEM's put-with-signal plus a signal wait on the receiver
replaces a global barrier per layer with a dependency between the one pair that has data to
exchange. 4.40 ms per forward of exposed barrier against a ceiling near 5% of a step is the
whole reason to do it, and it is the first cost centre this project has found that is both
dominant and clearly removable.

### Two traps, one of them mine and expensive

**Do not edit the source while a measurement run is in flight.** Each arm starts a fresh
server, so a syntax error introduced mid-run kills whichever arm starts next. A 12-arm run
lost its `0` and `43:device` arms of repeat 1 that way and had to be thrown out and redone —
40 minutes, and the log said only `never ready`.

**And do not reflow prose with a script.** The line-length fixes above were attempted with a
rewrapper that split an f-string across lines and then, on a second pass with stale line
numbers, merged a `def` into a docstring. Both produced files that imported fine in the
editor's view and failed at parse. The 88-column limit is worth hand-editing for.

### Correction, same evening: the barrier is exposed but it is not the dominant cost

The section above named the arrival barrier as the cost centre. That was premature — it is
4.40 ms per forward and real, but the +29% is mostly something else, and the CPU side of the
same two traces says what. Per-layer figures, which is the normalisation that matters here
because the two arms' windows are not paired (13 against 9, and this project has drawn a
wrong conclusion from unpaired windows before):

| per MoE layer, rank0 | prediction only | placing | ratio |
| --- | --- | --- | --- |
| `vllm::moe_forward` **host** time | 1.25 ms | **2.20 ms** | 1.76x |
| kernels launched | 30.1 | **83.4** | 2.77x |

**Placement adds about 53 kernel launches per layer and 0.95 ms of host time per layer**,
which over 44 placed layers is roughly **42 ms of extra host work per forward** — the right
order for the +76 ms of mean TTFT that placement costs. And the launches are not the
transfer:

    vectorized_elementwise_kernel   +25.9 per layer
    unrolled_elementwise_kernel      +8.2
    elementwise_kernel               +5.3
    indexSelectSmallIndex            +4.9
    index_elementwise_kernel         +4.1
    reduce_kernel                    +3.7
    put_expert / drain_expert / barrier   0.41 each

So about 48 of the 53 are tiny elementwise and reduce kernels: `plan_one_layer_on_device`,
`publish_plan_on_device`, and the device-side residency and budget bookkeeping. The engine is
eager, so each one is also a host-side dispatch, and 2365 launches per prefill window against
826 is what leaves the GPU idle — the placed arm's window carries **35 ms of kernel time in a
159 ms window**.

**This is ticket 03's defect in a new place.** Prediction's twelve elementwise steps were
fused into one Triton kernel for exactly this reason and it recovered 17% of prediction's
cost. Making the plan device-side removed 5.28 ms per layer of host *synchronisation* and put
about 0.95 ms per layer of host *dispatch* back — which was the right trade at DP=8, where the
sync dominated, and is a bad one here. Fusing plan, publish and bookkeeping into one kernel is
the same bounded piece of work: the plan is an argmax over 128 integers and the publish is a
handful of scatter writes.

Two consequences worth stating plainly. The pairwise-arrival change described above is still
worth doing, but it buys 4.40 ms per forward, not the 42. And **the launch count is now the
feature's dominant cost on both paths**, which is a claim about the mechanism rather than
about this node's rank count — at DP=8 the same 53 launches per layer are still there,
underneath a host synchronisation that was 5.5x larger.

### One caveat on every number above

The four-arm TTFT run and both profiles were measured **before** the stream-ordering fix the
code review found — the compute-to-predictive barrier was recorded on the wrong stream, so the
predictive stream never actually waited for the compute stream. The fix adds a real dependency
that was not there while these numbers were taken, and it can move them either way: the
transfer can no longer start ahead of the plan write, and the drain can no longer overtake the
previous forward's MoE. The per-layer launch-count and host-time findings are unaffected,
because they are counts of work the host does regardless of ordering, but **the TTFT table and
the barrier's duration distribution should be re-measured before either is quoted again.**
A smoke run on the fixed code is healthy: both workers arm, 88 layer-launches, 8 replicas
placed, no crash.

### The staging ordering hazard is closed, 2026-08-30

The review's open finding — a later layer's put landing in the workspace an earlier layer's
drain was still reading — is fixed by alternating two staging buffers by layer parity. Two
layers apart share a buffer again and are ordered by the barrier of the layer between them,
so two is enough. It costs one extra expert of symmetric memory, **9.00 MiB per rank**.

`expert_bytes` deliberately still means *one* expert, because it is also the default cap on
bytes in flight; the transport takes a separate `buffers` count. `staging_stride` has no
usable default — zero would disable the alternation silently, and a silently shared
workspace is precisely what this fixes — so it is rejected.

Re-verified on hardware: `probe_device_transfer.py` 4/4 byte-identical at p50 40.6 us,
`probe_plan_lifetime.py` 4/4, and a 2-rank server armed on both workers with 88
layer-launches and 8 replicas placed.

One thing the fix caught immediately, which is the invariant from ticket 07 doing its job:
`probe_plan_lifetime.py` launches without ever publishing, so the second forward it opened
raised "ended with a pending plan for layer(s) [2]". The probe checks the transport and has no
routing maps, so it now drops the entry explicitly rather than leaving a real check disarmed.
