# Read-only prior art: Ascend predictive expert scheduling

**Purpose:** preserve the useful scheduling idea from the earlier Ascend implementation for CUDA implementation review. This is **not** a porting target and is not part of the vLLM runtime package.

**Source provenance:** MindSpeed-RL commit `0cc99cbb`, from the Ascend fused-MoE implementation and its Qwen3-MoE decoder layer. The source repository is not present on the CUDA server, hence these minimal excerpts.

## CUDA contract supersedes this reference

Do **not** mechanically copy this code. The approved CUDA spec differs:

- CUDA prediction is phase-agnostic; do not preserve the Ascend `not is_prefill` gate.
- CUDA must retain source-rank provenance as a `[source rank, logical expert]` Global predicted-load snapshot, not collapse it with an AllReduce.
- CUDA transfers canonical owner to replica target using EPLB-group NCCL P2P, not expert-weight All-to-All.
- CUDA map activation uses staging, slot-use events, `replica_weight_ready`, and pending-map commit at the target MoE entrance.
- CUDA uses target router semantics rather than the Ascend hard-coded `select_experts` options.

## 1. Cross-layer target-gate prediction

The Ascend fused-MoE layer holds a reference to the next layer's gate and applies it to the current MoE hidden states. The essential pattern is:

```python
predict_routed_logits, _ = self.next_gate_ref[0](hidden_states)
_, topk_ids = select_experts(
    hidden_states=hidden_states,
    router_logits=predict_routed_logits,
    top_k=self.top_k,
    use_grouped_topk=False,
    renormalize=False,
)
num_local_tokens_per_expert = torch.bincount(
    topk_ids.flatten(), minlength=self.num_experts
)
```

The current vLLM implementation replaces `next_gate_ref` with an adjacent `MoERunner` target binding. It must use the target router's actual selection contract, not copy the hard-coded options above.

## 2. Prediction starts before current expert compute

Ascend creates counts on its prediction stream before calling the MoE quantization/expert path:

```python
with torch_npu.npu.stream(next_context.predict_stream):
    next_context.predict_num_local_tokens_per_expert = self.predict(hidden_states)
    next_context.predict_ready_event.record(next_context.predict_stream)
hidden_states.record_stream(next_context.predict_stream)

# Then execute current layer's MoE expert path.
e_hidden_states = self.quant_method.apply(...)
```

CUDA equivalent: derive source-local logical counts before token dispatch; begin the count AllGather only after dispatch; overlap that small collective with current expert GEMM.

## 3. Count communication overlaps grouped expert GEMM

The Ascend implementation waits for the prediction event, starts an asynchronous collective, executes grouped MatMul, then waits before combine:

```python
next_context.predict_ready_event.wait()
handle = dist.all_reduce(
    next_context.predict_num_local_tokens_per_expert,
    group=next_context.ep_group.device_group,
    async_op=True,
)

# grouped expert GEMMs execute here

handle.wait()
next_context.cpu_num_tokens_per_expert_buffer.copy_(
    next_context.predict_num_local_tokens_per_expert,
    non_blocking=True,
)
```

This is timing prior art only. CUDA uses AllGather because its planner needs individual source-rank chunks, not the aggregate produced by this AllReduce.

## 4. Weight scheduling overlaps the following Attention

The Ascend decoder starts expert scheduling on a predictive stream before Attention and completes the weight update before the MLP/MoE:

```python
with torch_npu.npu.stream(self.self_attn.expert_predict_context.predict_stream):
    self.self_attn.expert_predict_ref[0].expert_schedul(
        strategy=self.self_attn.expert_predict_context.strategy
    )

hidden_states = self.self_attn(...)

self.self_attn.expert_predict_ref[0].unpermute_expert()
hidden_states = self.mlp(hidden_states)
```

CUDA equivalent: planner completion precedes next Attention; P2P runs on the predictive stream during Attention; target MoE waits for `replica_weight_ready` and commits the pending source-local physical map at its entrance. CUDA deliberately has no extra global activation barrier.

## What this reference does not answer

It does not establish CUDA NCCL ordering, vLLM router API use, dummy/padding masks, source-local physical-map rewriting, P2P slot safety, or cost-aware lifecycle behavior. Those requirements are authoritative in the local spec and tickets.
