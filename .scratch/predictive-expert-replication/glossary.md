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

**Single placement**：一层内 1 个 logical expert 复制到 1 个 target rank，占用 1 个槽。spec 现状。
_Avoid_: "方案 A"

**Expert fan-out**：一层内 1 个 logical expert 复制到 K 个 target rank。卸载上限是该专家占峰值 rank 负载的比例乘以 `K/(K+1)`，**恒小于该比例本身** —— 它把同一块负载分得更碎，不增加可卸载的总量。
_Avoid_: "方案 B", multi-replica

**Multi-expert placement**：一层内 K 个**不同**的 logical expert 各复制到 1 个 target rank。卸载量可跨专家累加，成本与同 K 的 expert fan-out 相同。
_Avoid_: "方案 C"

同一层内一个 rank 只有一个槽，所以 expert fan-out 与 multi-expert placement 在同一层里争用同一批槽位。

**Predictive expert replication**：使用当前 MoE 层对下一层 logical expert 负载的预测，动态创建并路由到下一层 expert replica；它不使用周期性历史负载重排专家所有权。
_Avoid_: predictive EPLB, full expert migration

**Native EPLB**：vLLM 基于历史实际负载窗口周期性调整 physical expert placement 和 replicas 的负载均衡系统；expert replica 是它使用的一种机制，但两者不是同义词。
_Avoid_: expert replica

**Expert replication infrastructure**：Native EPLB 与 Predictive expert replication 共享的 physical slots、logical/physical maps、weight-transfer buffers 和独立通信组；它本身不包含 placement policy。
_Avoid_: Native EPLB, predictive controller

**Cross-layer gate**：用当前 MoE 的 hidden states 评估目标 MoE 的 routing gate，以预测目标 logical expert 选择的推理机制；它不是目标 MoE 实际执行时的 gate 结果。目标 MoE 不一定是相邻层，距离由 prediction lookahead 决定。
_Avoid_: next-layer execution, cached router logits, next-layer gate

**Host-planned placement**：plan 在 host 上计算的安排。snapshot 必须先 D2H 落地、host 再读它,所以"snapshot 就绪"和"plan 可用"之间**必须隔一个 layer 边界**去吸收这次拷贝——这一层就是 lookahead 被迫大于 1 的原因。当前实现是这一种,lookahead=2,窗口是中间那层的 MoE 加目标层的 Attention。

**Device-planned placement**：plan 在 device 上计算(确定性 argmax),传输用单边 put,host 全程不读 plan。不需要那个 layer 边界,于是预测、规划、发起可以都在第 `L` 层内完成,窗口就是第 `L+1` 层的 Attention。它需要 NVLink 与 NVSHMEM(已在本节点验证通过,9 MiB put 33 us)。

**lookahead 不是一个独立旋钮,而是上面这个选择的后果。** 把 `prediction_lookahead_layers` 当作可自由调节的参数是本项目的一处概念错误:在 host-planned 安排下 lookahead=1 的**窗口是 0**(发起与等待是 MoE forward 开头相邻的两行),而配置校验目前接受它。lookahead=1 只有在 device-planned 安排下才有意义。
_Avoid_: 把 lookahead 与 overlap window 当作同一个量(见 Overlap window)

**Prediction lookahead**：预测层与被预测层之间相隔的 sparse MoE 层数。lookahead 越小预测越准，但可用于隐藏 expert 权重传输的 overlap window 越短；它是可配置的，因为这个取舍取决于实际互联带宽。
_Avoid_: adjacent layer, prediction depth

**Overlap window**：从 expert 权重传输发起到目标层 MoE 必须读到该权重之间的可用时间。"传输被 overlap"同时要求两件事：传输与计算在时间上并行，**且**传输没有使这段时间内的互联成为新瓶颈。

它**不等于** `lookahead` 乘以一层的时间——它由发起点和等待点这两个 hook 的位置决定，而 lookahead 只决定预测跨越几层。当前实现两个 hook 都在 MoE module forward 的开头，所以 lookahead=2 的窗口是"中间那层的整个 MoE + 目标层的 Attention"，而 lookahead=1 的窗口是 **0**。把这两个量当作同一个东西是本项目的一处已记录偏差（`CURRENT-STATUS.md` 2026-08-29 code audit 第 1 条）。
_Avoid_: attention window（除非特指单层 Attention 这一项成本）；prediction lookahead（它是层距，不是时间窗口）

**Exposed transfer**：expert 权重传输超出 overlap window、必须由目标层等待的那部分时间。它是 replica 的真实一次性成本，也是推导最小驻留步数的分子。
_Avoid_: transfer latency, hidden transfer

**Staging workspace**：P2P 接收 expert 权重的落地区，随后在确认目标 slot 不再被 MoE kernel 读取后拷入该 slot。由于同一时刻只允许一个 expert 在飞，它只需容纳一个 expert，并由全部 MoE 层共享，**不是 per-layer** 的。
_Avoid_: replica slot, expert buffer

**Usable transfer bandwidth**：在 token dispatch/combine 正在运行时实测到的 rank 间可用带宽。cost model 必须用它而不是空载 P2P 带宽，因为 token 集合通信与 expert 权重传输共享同一条互联，用空载值会系统性高估收益。
_Avoid_: peak bandwidth, idle P2P bandwidth

**Predicted load**：由 cross-layer gate 对当前 forward 中全部有效 routed tokens 预测得到的**目标层** logical expert token 数量；它不区分 prefill、decode 或 mixed batch，并排除 dummy/padding tokens。它按 logical expert 索引，绝不按 physical slot 索引。
_Avoid_: actual load, phase-specific load, physical slot count

**Global predicted-load snapshot**：同一 forward 中所有 source rank 的 predicted load 组成的 `[source rank, logical expert]` 计数矩阵；每个 EP rank 通过 AllGather 获得相同副本，供确定性 planner 使用。
_Avoid_: actual load, aggregate-only load

**Actual load**：真实 gate 和 dispatch 完成后观测到的 logical expert token 负载。
_Avoid_: predicted load

**Replica placement plan**：描述每层 logical expert 的 canonical owner、replica 集合及其 target ranks 的计划。
_Avoid_: full expert migration plan

**Expert weight transfer plan**：描述为实现 placement plan 需要发送、接收和保留哪些 physical expert 权重。
_Avoid_: routing plan

**Hot logical expert**：被 policy 判定为"值得创建 replica"的 logical expert。唯一权威判据是 policy 的正收益检验（预测收益扣除摊销后的传输成本仍为正）；`hot_load_ratio` 只是避免逐层评估全部专家的**候选预筛选**，不构成独立判据。生命周期中的"连续两次 hot 观测"意为 policy 会再次选中同一个 `(logical expert, target rank)` 放置。
_Avoid_: high-load expert（作为独立阈值概念）, hot threshold

**Block-quantization bar**：`M x topk / num_logical_experts > BLOCK_SIZE_M` 这条线。低于它，MoE kernel 把每个被碰到的专家都补齐到同样的 block 数，于是不均衡**本来就不花时间**，任何放置都不可能有收益。它是 decode 被判负的唯一原因，也是 `min_tokens_per_expert` 与 planner 的 `min_tokens` 共用的那个常数（当前硬编码 `_MOE_BLOCK_SIZE_M = 128`）。
_Avoid_: gate（本项目 gate 专指 MoE 路由 gate）, threshold（不加限定时不知道指哪条线）

**Placement suppression**：某个 forward 低于 block-quantization bar 时，跳过规划、传输与发布，把各层的 map 原样留给下一个够大的 forward。它**不是**预测结果，也不是"预测失败"——预测在同一根线上就已经被跳过了。代码里叫 `_gated`，与上面的 _Avoid_ 冲突，且当前实现不可达（`CURRENT-STATUS.md` 2026-08-29 code audit 第 3 条）。
_Avoid_: gate, gating, prediction failure

**Transfer budget**：`max_transfers_per_forward`，**一次 forward** 内允许的放置总数，跨全部层共享。一个单位 = 一层里的一个 logical expert 复制到一个 target rank = 最多一次 9.00 MiB 传输。它按**放置数**计（`_spent += len(plan)`，在 `reconcile` 之前），所以已经常驻、不需要重传的放置也占额度——名字说的是传输，管的是覆盖度。逐层上限是另一个量：`max_replicas_per_layer`。
_Avoid_: max_transfers_per_forward 的字面含义（"传输数"）, per-layer budget, per-request budget

**Cost profile**：通过离线 microbenchmark 获得的静态成本数据，用于把预测负载收益、expert weight transfer 时间和 Attention overlap window 统一换算为时间；没有 cost profile 时不创建新 replica。一次性 transfer cost 按 replica 的 guaranteed minimum residency 摊销。
_Avoid_: online calibration, runtime load history

### Evaluation

**Collective residency**：某个 rank 待在一个集合通信 kernel 里的时长。它主要测的是**这个 rank 比最后到达者早了多少**，所以最后到达的那个 rank 上它接近 0。**绝不能用单个 rank 的这个量给一次集合通信定价**——2026-08-30 实测同一个 prediction snapshot AllGather 在 8 个 rank 上的 p50 是 748/628/595/545/131/588/**9.4**/425 us，那个 9.4 us 的 rank 正是最后到达者。项目曾据此（`dp0` 单 rank）判定"这个 AllGather 只要 9.3 us、没有到达偏斜"，并关掉了批量化这条路，见 `CURRENT-STATUS.md` 2026-08-30 的撤回。
_Avoid_: 拿它当 collective 的成本；拿单 rank 的它当全组的它

**Barrier coupling cost**：多引入一个集合通信所带来的、**不出现在任何 kernel 时长里**的代价——每多一个 barrier，全组就多一次"等 N 个 rank 里最慢的那个"。它只在窗口总时长随 barrier **个数**的增长里可见。2026-08-30 的两点观测：192 个 token 集合通信对应 88.9 ms 窗口，236 个（加上 44 个 prediction AllGather）对应 106.9 ms，**每个 0.463 对 0.453 ms**，与载荷（hidden states 对 512 字节）无关。这条是假设而非定论，隔离探针是 `VLLM_PREDICTIVE_SKIP_SNAPSHOT_ALLGATHER`。
_Avoid_: 把它与 collective residency 混为一谈；bandwidth cost

**Critical-path imbalance**：`Σ_各层(该层最慢 rank 的负载) / Σ_各层(该层均值)`。这是**唯一对应 MoE 总耗时的不均衡指标** —— 每层是独立的集合通信，各自等自己最慢的 rank，所以临界路径是逐层峰值之和。EPLB 自报的 `balancedness` 就是它的倒数。
_Avoid_: load imbalance（不加限定时含义不明）

**Summed-load imbalance**：先把各层负载对每个 rank 求和、再比较 rank 之间的差异。**不要用它做放置决策** —— 不同层的峰值 rank 不同，先求和会让它们互相抵消。实测同一份数据下它给出 1.10x，而 critical-path 指标给出 1.70x（headroom 41%），差约 3 倍。
_Avoid_: 用它替代 critical-path imbalance

**Feasibility checkpoint**：ticket 00 的判定点 —— 先实测 EP rank 间的不均衡幅度、以及它能否转化成时间，再决定搬运机制（P2P、staging、planner、benchmark）值不值得实现。它是**项目决策**，与预测准确度无关（后者是 prediction accuracy）。
_Avoid_: **gate** —— 本项目里 gate 专指 MoE 的路由 gate（见 cross-layer gate），拿它指决策点会造成歧义

**Request shape**：一次 benchmark 固定的 prompt 与 decode 长度组合。本 PoC 固定两种：**2k prompt / 1k decode**（prefill 偏重）与 **1k prompt / 2k decode**（decode 偏重）。前者压 TTFT，后者压 TPOT。
_Avoid_: batch size, sequence length

**Uniform-routing control**：用合成 token 序列（在词表上均匀扫描）构造的对照组，用于确定不均衡度的下限。它**不是**服务结果 —— 均匀 token 会让专家选择趋于均匀，从而系统性低估真实内容产生的不均衡。
_Avoid_: random dataset baseline, synthetic baseline

**Compute-limited benchmark**：固定并发或 RPS，评估 predictive expert replication 对 TPOT、尾延迟、最大可持续 RPS 和 EP rank 负载差异的影响。
_Avoid_: memory-capacity benchmark

**Memory-capacity benchmark**：评估 replica slots 的静态权重开销与动态 MoE buffer 峰值变化，记录 KV block 数、峰值显存、最大并发和 OOM 边界。
_Avoid_: compute-limited benchmark

**Prediction accuracy**：predicted load 与同一层同一 forward 的 actual load 的一致程度，以 **hot-expert 集合重叠率**与**每专家计数误差**表示。gate logit 相似度**不是**这个指标 —— 高 logit 相似度仍可能改变 top-k 的选中集合，从而错判负载。
_Avoid_: logits similarity, cosine similarity

比较三组配置：无 replica、Native EPLB（相同 slot 数量）、Predictive expert replication（相同 slot 数量）。vLLM PoC 的固定目标环境为单机 8×RTX 5090、BF16、eager、TP=1、DP=8、EP=8。

该环境**没有 NVLink**，全部 rank 间流量走 PCIe Gen5 x16，因此它是本策略最不利的环境。任何在此得到的负面结论都应记为**对该节点互联的结论**，而非对策略本身的结论。
