# MoE Expert Placement

本 context 描述 MindSpeed-RL 在 MoE 推理中的专家路由、专家副本与专家权重调度概念。

## Language

### Logical and physical experts

**Logical expert**：由 gate 选择、参与模型语义的专家 ID；它是 token 路由的逻辑对象，不因副本存在而改变。
_Avoid_: physical expert, local expert slot

**Physical expert**：某个 EP rank 上实际保存并执行的逻辑专家权重实例。
_Avoid_: logical expert, expert ID

**Expert replica**：同一个 logical expert 在另一个 EP rank 上动态保存的等价 physical expert，用于分担执行负载；replica 可以创建、复用和回收，canonical owner 不随之迁移。
_Avoid_: redundant expert（除非讨论已有实现的配置字段）

**Replica slot**：每个 EP rank 为一个动态 expert replica 预留的 physical expert 槽位；slot 可以 inactive，但同一时刻最多承载一个 logical expert。
_Avoid_: canonical expert slot, staging buffer

### Routing and placement

**Source rank**：dispatch 开始前保存 token hidden states 的 EP rank。
_Avoid_: owner rank, target rank

**Target rank**：实际保存并执行某个 physical expert 的 EP rank。
_Avoid_: source rank

**Source-rank routing**：根据 source rank 和 logical expert 选择 physical expert；同一 source rank 上选择同一 logical expert 的 token 使用同一 replica。
_Avoid_: token-level routing

**Source-local physical map**：某一 source rank 将 logical expert 映射到当前应使用的 physical expert 的 map；它在 token dispatch 前把逻辑路由 ID 转换为 physical routing ID。
_Avoid_: global placement map, token-level route

**Token-level routing**：为每个 token 与 logical expert pair 独立选择 physical expert replica。
_Avoid_: source-rank routing

**Canonical owner**：logical expert 的稳定 owner，永久保留该专家的权重；动态 replica 只是在其之外增加或回收的执行副本。
_Avoid_: primary token

### Planning

**Predictive expert replication**：使用当前 MoE 层对下一层 logical expert 负载的预测，动态创建并路由到下一层 expert replica；它不使用周期性历史负载重排专家所有权。
_Avoid_: predictive EPLB, full expert migration

**Native EPLB**：vLLM 基于历史实际负载窗口周期性调整 physical expert placement 和 replicas 的负载均衡系统；expert replica 是它使用的一种机制，但两者不是同义词。
_Avoid_: expert replica

**Expert replication infrastructure**：Native EPLB 与 Predictive expert replication 共享的 physical slots、logical/physical maps、weight-transfer buffers 和独立通信组；它本身不包含 placement policy。
_Avoid_: Native EPLB, predictive controller

**Cross-layer gate**：用当前 MoE 的 hidden states 评估目标 MoE 的 routing gate，以预测目标 logical expert 选择的推理机制；它不是目标 MoE 实际执行时的 gate 结果。
_Avoid_: next-layer execution, cached router logits

**Predicted load**：由 cross-layer gate 对当前 forward 中全部有效 routed tokens 预测得到的下一层 logical expert token 数量；它不区分 prefill、decode 或 mixed batch，并排除 dummy/padding tokens。
_Avoid_: actual load, phase-specific load

**Global predicted-load snapshot**：同一 forward 中所有 source rank 的 predicted load 组成的 `[source rank, logical expert]` 计数矩阵；每个 EP rank 通过 AllGather 获得相同副本，供确定性 planner 使用。
_Avoid_: actual load, aggregate-only load

**Actual load**：真实 gate 和 dispatch 完成后观测到的 logical expert token 负载。
_Avoid_: predicted load

**Replica placement plan**：描述每层 logical expert 的 canonical owner、replica 集合及其 target ranks 的计划。
_Avoid_: full expert migration plan

**Expert weight transfer plan**：描述为实现 placement plan 需要发送、接收和保留哪些 physical expert 权重。
_Avoid_: routing plan

**Cost profile**：通过离线 microbenchmark 获得的静态成本数据，用于把预测负载收益、expert weight transfer 时间和 Attention overlap window 统一换算为时间；没有 cost profile 时不创建新 replica。一次性 transfer cost 按 replica 的 guaranteed minimum residency 摊销。
_Avoid_: online calibration, runtime load history

### Evaluation

**Compute-limited benchmark**：固定并发或 RPS，评估 predictive expert replication 对 TPOT、尾延迟、最大可持续 RPS 和 EP rank 负载差异的影响。
_Avoid_: memory-capacity benchmark

**Memory-capacity benchmark**：评估 replica slots 的静态权重开销与动态 MoE buffer 峰值变化，记录 KV block 数、峰值显存、最大并发和 OOM 边界。
_Avoid_: compute-limited benchmark

比较三组配置：无 replica、Native EPLB（相同 slot 数量）、Predictive expert replication（相同 slot 数量）。vLLM PoC 的固定目标环境为单机 8×RTX 5090、BF16、eager、TP=1、DP=8、EP=8。
