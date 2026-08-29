# 13 — Decide the placement on the device, and transfer without the host

**What to build:** Move the placement decision and the weight transfer entirely onto
the device, so no host synchronisation is needed anywhere on the per-forward path.

**Blocked by:** ~~NVLink hardware, and `bench/probe_nvshmem.py` passing on it.~~ Nothing.

**Status:** **unblocked 2026-08-29** — 8x H100 SXM with NV18 across all pairs, and
`probe_nvshmem.py` passes 5 of 5 with a 9.00 MiB put at **33.0 us p50** against PCIe's
289 us. All three things this design said would invalidate it on their own are answered
in its favour, including that **a plain CUDA event orders the consumer**, so no fence and
no flag polling is needed. Numbers and the API corrections in `bench/RESULTS.md`,
2026-08-29. Not started.

**Two API facts change the design as written below.** Both were found while correcting
the probe:

1. `nvshmem.core.register_external_tensor` exists. The model's own `expert_weights`
   allocation can be registered with NVSHMEM and put into **directly**, so point 3's
   symmetric staging buffer and its extra 9.00 MiB device-to-device copy are avoidable.
   Registration is a collective, which is fine at startup. Point 3's warning still
   stands — do not *move* `expert_weights` into the symmetric heap, since that
   allocation path is shared with EPLB's `rearrange` — but registering in place is not
   moving.
2. `nvshmem.core.put(dst, src, pe, stream=)` is the stream-ordered entry point; there is
   no `put_on_stream`. Point 4's requirement is met by the ordinary `put`.

**This ticket is the operator's intended pipeline shape** (stated 2026-08-29): predict
layer `i + 1` at layer `i`, complete the transfer and the map update during layer
`i + 1`'s Attention, so that layer's MoE runs with the replica live. That shape needs
the launch at the *predicting* layer's MoE tail, which needs no host read of the plan,
which is only true here. See `CURRENT-STATUS.md`'s 2026-08-29 code audit finding 1 for
what the current hook positions do instead, and why `prediction_lookahead_layers=1`
today has a window of zero rather than one Attention block.

Note the consequence for ticket 12: `N` there is the planning delay in layers, so this
ticket's one-layer pipeline makes the aliasing pool `N + 1 = 2` blocks, 18 MiB, instead
of 5 blocks and 45 MiB. Shortening the pipeline is also the cheapest way to shrink that
memory.

## Why

The per-forward path stalls the CPU. `plan_and_launch` calls
`copy_event.synchronize()` because the planner runs on the host and needs this
forward's predicted load. Measured inside real forward windows: GPU occupancy
**86.8% (baseline) versus 52.9% (placed)**, gaps over 0.5 ms totalling **3.0 ms
versus 30.2 ms**, and one `cudaEventSynchronize` of **11.02 ms**. The CPU normally
runs ahead of the GPU enqueuing work; this throws that run-ahead away once per
predicted layer, and the engine becomes launch-bound — which gets relatively *worse*
on faster hardware, since launch cost is unchanged while GPU work shrinks.

**It is not a stream-placement mistake, and moving only the planner does not fix it.**
`ncclSend`'s peer is a `ctypes.c_int` consumed when the host enqueues
(`vllm/distributed/device_communicators/pynccl_wrapper.py`), and it selects the
transport channel at that moment. There is no device-side peer indirection. So the
host must know which expert goes to which rank; if the plan lives in a device tensor
the host reads it back and the same wait reappears.

The escape is **one-sided put**: the address is computed on the device and there is no
matching receive to pair with. That is NVSHMEM, which needs NVLink. The 8x RTX 5090
node has none, no `pynvshmem`, and no DeepEP.

Enumerating all peer pairs so the schedule is static instead is not viable: 8 ranks is
56 pairs at 9.00 MiB, 504 MiB per layer, 21 GB per forward.

## The design, as settled

1. **The plan stays globally deterministic.** Every rank runs the same deterministic
   kernel over the same allgathered snapshot, so the plans are identical by
   construction. A one-sided put would technically allow per-rank decisions, but then
   the receiver needs a flag and must poll it — and polling is per-rank timing, which
   is the divergence class that deadlocked this branch twice.
2. **One replica per layer**, so the plan is a single argmax: hottest expert on the
   peak rank, lightest target rank. No serial dependency, no matching problem, one
   block of reductions. Measured cost of the cap: 30.9% of prefill excess against
   35.0% uncapped. **Triton is not indicated** — the data is 128 integers and a kernel
   launch (~10 us) already exceeds the work; plain torch reductions suffice.
3. **A symmetric-heap staging buffer, then a device copy into the replica row.** One-
   sided put requires the *destination* to be symmetric-heap allocated, and the replica
   row belongs to the model's own weight tensor. Do not move `expert_weights` into the
   symmetric heap: that allocation path is shared with EPLB's `rearrange`. The extra
   9.00 MiB device-to-device copy is ~10 us and negligible.
4. **Stream-ordered NVSHMEM** (`nvshmemx_*_on_stream`), so an ordinary CUDA event and
   `wait_event` order the consumer. That is a device-side wait. The device-side-only
   put would need the flag and polling of (1).
5. **Integer end to end.** Already done: the snapshot is int64 through the copy and
   the tie-break resolves to the lowest placement. A device argmax over floats is
   order-dependent, and two ranks reducing the same values in a different block order
   can pick different experts.
6. **Memory is out of scope here.** See ticket 12.

## The resulting pipeline, with no host read on it

    1  allgather the predicted snapshot        (exists; identical on every rank)
    2  device argmax                          -> plan tensor
    3  device gather of the chosen expert      -> symmetric staging buffer
    4  nvshmemx_put_on_stream                  -> the peer's staging buffer
    5  device copy                             -> the replica row of that layer
    6  device scatter                          -> source_local_physical_map and counts
    7  target layer: wait_event on the predictive stream

Zero `synchronize()` calls.

## What the probe decides

`bench/probe_nvshmem.py`, 8 GPUs, no vLLM. It answers the three things that would
each invalidate this alone — stream-ordered put available, symmetric heap allocatable
after torch and NCCL are already up, NVSHMEM and NCCL coexisting in one process — and
measures a 9.00 MiB put. **At ~13 us instead of PCIe's 289 us, the planning delay can
stay at one layer and the lookahead need not rise, so prediction accuracy costs
nothing** and several open questions close. Its API names are inferred from semantics
and probed with `getattr`; expect to correct them on first contact.

- [x] `probe_nvshmem.py` passes on the target hardware, with the put latency recorded.
      8x H100 SXM, 5 of 5, 9.00 MiB put 33.0 us p50 (31.6 min, 35.0 max, 285.9 GB/s).
- [ ] `activate()` waits with `current_stream().wait_event(event)` rather than
      `event.synchronize()`. This one needs no NVSHMEM and can be done first: it wants
      only the stream ordering "map writes follow the transfer", not any host
      knowledge, and it is about half of the 126 `cudaEventSynchronize` calls a placed
      run makes.
- [ ] The device argmax produces a plan bit-identical to `plan_replicas` at
      `max_replicas_per_layer = 1`, asserted against the existing host planner.
- [ ] No `synchronize()` on the per-forward path, asserted by a test on the module
      source the way the `.query()` prohibition already is.
- [ ] GPU occupancy inside real forward windows back near the baseline's 86.8%.
