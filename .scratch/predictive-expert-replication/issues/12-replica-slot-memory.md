# 12 — Cut the replica slots' 432 MiB with VMM aliasing

**What to build:** Share one pool of replica-weight blocks across layers instead of
giving every layer its own row, using CUDA virtual-memory aliasing.

**Blocked by:** nothing structurally, but it is orthogonal to removing the host
synchronisation and should not compete with it.

**Status:** deferred — sized and priced, not recommended first

## The cost today

One replica row per layer per rank: 48 x 9.00 MiB = **432 MiB per rank**. That is
0.5% of an 80 GB H200 and 1.4% of a 32 GB 5090, and it comes out of KV cache — which
sets reachable concurrency, which is what decides whether prefill clears the
one-block-per-expert bar at all. So it is not free even at 0.5%.

## Why a single shared buffer does not work, and what does

`fused_moe.py` addresses expert weights as `base_ptr + expert_id * stride_be`. All of
a layer's physical experts must therefore be rows of **one contiguous tensor**
(`expert_weights[layer]`), and there is no per-expert weight pointer table to point
somewhere else. A buffer shared across layers cannot be routed to.

Semantically the sharing is sound: only one layer's MoE runs at a time, so only one
layer's replica needs to be valid at any instant. The obstacle is purely addressing,
and CUDA's virtual memory API removes it: reserve each layer's 17-row span with
`cuMemAddressReserve`, then `cuMemMap` the same physical block at several layers'
replica row. The kernel still sees a contiguous strided tensor.

**One block is not enough.** Layer T's transfer is launched N layers early, so writing
a single shared block at T-N destroys the replica that layers T-N..T-1 still need. It
takes **N+1 blocks**, with layer L's row mapped to block `L mod (N+1)`. UltraEP's TMA
double-buffering is this with N+1 = 2.

| | Memory per rank |
| --- | --- |
| One row per layer (today) | 432 MiB |
| VMM aliasing, N = 4 (5 blocks) | 45 MiB |

## Why not the obvious alternative

Allocating slots on only the worst K layers was measured, and it is a straight line:
about 0.65 points of prefill excess per layer with a slot, no knee anywhere.

| Layers with a slot | Memory | Excess removed |
| --- | --- | --- |
| 48 | 432 MiB | 31.0% |
| 32 | 288 MiB | 25.5% |
| 24 | 216 MiB | 21.6% |
| 16 | 144 MiB | 16.5% |
| 8 | 72 MiB | 9.5% |

Measured on the 2026-08-25 e2e baseline dump, 8 full-prefill forwards, budget 43, one
replica per layer, slots given to the K layers with the worst per-layer imbalance.
`imbalance.plan_moves(..., eligible=...)` is the seam.

So restricting layers trades memory for benefit one for one, while VMM aliasing cuts
memory ~10x at **no** benefit cost. If the memory matters, this ticket is the answer
and the K-layer restriction is not.

- [ ] Replica rows of layers sharing a block index are backed by one physical
      allocation, verified by writing through one layer's row and reading another's.
- [ ] `N + 1` blocks, with the mapping derived from the planning delay rather than a
      constant, and a test that a transfer launched N layers early cannot overwrite a
      replica a nearer layer still needs.
- [ ] The weight-loading path and EPLB's `rearrange` still work. They share
      `expert_weights`, and this ticket changes how it is allocated.
- [ ] Measured memory before and after, and prefill benefit unchanged.
