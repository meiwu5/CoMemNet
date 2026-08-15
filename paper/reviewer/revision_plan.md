# CoMemNet 大修执行蓝图

## 1. 返修主线

本次大修不宜继续把贡献表述为“完全 graph-free 的 contrastive learning”。当前代码的预测前向不接收邻接矩阵，但增量节点扩展、两跳子图构造和训练节点集合都使用年度邻接矩阵；当前优化目标也只有 MAE，并不存在数学意义上的对比损失。建议将论文主线收敛为：

> CoMemNet 是一个 adjacency-free predictor + topology-assisted update policy 的持续交通预测框架。预测器在训练和推理前向中不执行图卷积、不读取邻接矩阵；邻接信息仅用于确定新节点的局部更新范围。在线—目标分支提供稳定的漂移估计，并通过受限预算节点更新和紧凑时间记忆缓解遗忘。

为了保留“contrastive”术语，必须补充真正的表示级对比目标。否则应统一改名为 `Dynamic Drift Sampler (DD Sampler)` 和 `dual-branch momentum framework`，删除 self-supervised contrastive learning、positive/negative pairs 等声明。推荐补充对比目标，形成下面的修订架构。

## 2. 推荐返修架构：CoMemNet-R

### 2.1 数据与信息边界

周期 \(\tau\) 到来时，仅允许使用：

- 当前周期训练段 \(D_\tau^{train}\) 的输入、标签和节点元数据；
- 上一周期冻结的模型参数 \(\theta_{\tau-1}\)；
- EMA 目标参数 \(\xi_{\tau-1}\)；
- 每周期一个固定维度的 TMRB 状态，或明确限定容量的节点记忆；
- 当前/上一周期邻接矩阵仅供 topology-assisted update policy 生成局部训练集合，不进入 predictor forward；
- 不允许使用当前周期验证集、测试集或未来年份信息来选节点、调参和模型选择。

必须分别报告两种存储口径：

1. `deployment state`：模型、优化器、EMA、TMRB 和 replay buffer；
2. `metadata-assisted state`：若更新策略保留上一年度邻接/节点元数据，将其字节数单独报告。

当前实现会直接重新读取上一年完整原始流量 `pre_data`。因此不能声称“only compressed memory features are retained”。应二选一：

- 严格协议（推荐）：在周期结束时构建固定预算 exemplar/feature buffer，下一周期禁止读取完整历史 raw `.npz`；
- 访问协议：承认 sampler 可访问上一周期一周原始窗口，并把该窗口计入 replay-buffer size，所有受控 baseline 使用相同窗口与预算。

### 2.2 模型数据流

```text
current train window X_tau ------------------------------+
                                                           |
                         +--> online encoder f_theta -------+--> predictor --> Y_hat
                         |                    |
historical budget B -----+                    +--> online representation q
                                              |
                                              +--> node-wise contrastive loss
                                              |
EMA target encoder f_xi (stop-gradient) ------+--> target representation k

previous/current representations --> drift scorer --> budgeted core nodes
new nodes + core nodes --> optional topology neighborhood policy --> update node set
temporal embeddings + compact previous state --> TMRB --> predictor context
```

需要明确区分三个概念：

- `core-node budget M`：DC sampler 选出的旧节点数量或比例；
- `temporal top-K`：TMRB 内对时间位置/特征位置的选择，不能再写成节点 Top-K；
- `expanded update set`：新节点、core nodes 及可选的 k-hop 邻居并集。

当前 `TMRB.py` 的张量含义显示 `top_k` 实际沿时间/节点维混杂操作，论文将其直接描述为 key nodes 风险很高，需要用 shape annotation 和单元测试重新确认。

### 2.3 真正的对比目标

对共同节点 \(i\in V_{\tau-1}\cap V_\tau\)，使用同一时刻或匹配时间上下文生成在线表示 \(q_i\) 与 stop-gradient 目标表示 \(k_i^+\)。同一 batch 其他节点/时间上下文作为负样本：

\[
\mathcal L_{con}=-\frac{1}{|S|}\sum_{i\in S}
\log\frac{\exp(\mathrm{sim}(q_i,k_i^+)/T)}
{\sum_{j\in S}\exp(\mathrm{sim}(q_i,k_j)/T)}.
\]

总损失：

\[
\mathcal L=\mathcal L_{pred}+\lambda_{con}\mathcal L_{con},
\]

至少做 `MAE only / MAE+contrast` 消融。若实验表明 contrast 无收益，应删除 contrastive 命名，而不是保留空洞公式。

### 2.4 严谨的 Wasserstein 节点分数

不要计算 `wasserstein_distance(pre_prob, cur_prob)`；这会把概率值本身当作样本位置。对共同 bin edges \(b_0,\ldots,b_B\) 和归一化质量 \(p,q\)，一维 Wasserstein-1 应写为：

\[
W_1(p,q)=\sum_{b=1}^{B-1}|F_p(b)-F_q(b)|\Delta b,
\]

或调用 `wasserstein_distance(bin_centers, bin_centers, u_weights=p, v_weights=q)`。分数应在每个预测步/特征维分别计算后求平均，避免把早晚高峰时间顺序完全 flatten。建议增加两个版本：

- `W1-marginal`：原始直方图思想的严格实现；
- `W1-phase-aware`：按 time-of-day slot 分组后求平均，保留峰谷相位。

受控距离包括 L2、cosine、KL、JS、MMD；所有方法使用完全相同的表示、窗口、bin 数和节点预算。

## 3. 审稿意见到实验的映射

| 主题 | 必做修改 | 结果载体 |
|---|---|---|
| 历史信息边界 | 增加 protocol 表，列 raw X/Y、metadata、adjacency、feature buffer、model state 的可用性与字节数 | 方法/实验设置 + response |
| sampler 合理性 | Random、High-error、Feature-L2、Recency、KL、JS、MMD、W1，同 backbone、同 M | 主表或 sampler ablation |
| graph 使用冲突 | Predictor-only no-adj、topology-assisted neighborhood、fixed graph predictor、dynamic/adaptive graph predictor | graph-dependency ablation |
| 公平性与成本 | 参数量、可训练参数、峰值显存、buffer、metadata、每周期时间、累计时间、FLOPs/MACs（可选） | efficiency table |
| target branch | no target、online-only sampler、m=0.90/0.95/0.99/0.995、EMA target | target ablation |
| catastrophic forgetting | task-by-task evaluation matrix \(R_{t,j}\)，ACC/AIP、BWT、FWT、Forgetting | continual-learning table + heatmap |
| 多随机种子 | 至少 3 seeds；主结果报告 mean±std；paired test 或 bootstrap CI | 所有关键表 |
| retraining 边界 | 按季度/月度/事件频率外推累计成本，并至少构造 quarterly split 实验 | scale/frequency figure |
| static SOTA | Retrained-PDFormer、Retrained-STD-MAE（资源不足可选其中一个加 STID） | upper-bound table |
| K 解释 | 固定 K 与比例 K（1/2/5/10%）并列；报告最终 expanded set 比例 | sensitivity figure |
| 鲁棒性 | missing sensors 10/20/40%，noise，delayed additions，abrupt demand shock | robustness table |
| 数据可复现 | sensor IDs、年份范围、缺失率、插值比例、节点/边、split timestamps、构图参数、校验和 | dataset card + scripts |
| baseline 缺失 | PECPM/TFMoE/EAC 跑全数据集；若代码或显存不允许，给复现实证和限制，不留空白破折号 | baseline table |

### 持续学习指标定义

保存矩阵 \(R_{t,j}\)：训练完周期 \(t\) 后，在历史周期 \(j\) 的冻结测试集上评估。误差越低越好时，可报告：

- `Average Incremental Performance`：每个 t 对已见任务平均误差，再对 t 平均；
- `BWT`：将 MAE 转为负效用后采用标准定义，或直接报告 signed error increase 并说明越低越好；
- `Forgetting_j = MAE_{T,j} - min_{t in [j,T]} MAE_{t,j}`；
- `FWT`：训练任务 j 前相对随机初始化/首年模型在 j 上的提升。

不要仅用“每年当前测试集 MAE”证明没有遗忘；该指标只反映适应性。

## 4. 代码修改框架

建议将 `main.py` 的全局流程拆开，保证协议可审计：

```text
src/
  model/
    backbone.py            # adjacency-free predictor encoder
    momentum.py            # online/target EMA and projector/predictor
    tmrb.py                # typed memory state; no disk side effects
    losses.py              # prediction and InfoNCE objectives
  continual/
    protocol.py            # period visibility and no-future-data checks
    buffer.py              # fixed-budget raw/feature exemplar store
    selectors.py           # common Selector interface
    distances.py           # correct W1/L2/KL/JS/MMD
    update_policy.py       # new/core nodes and optional k-hop expansion
  evaluation/
    metrics.py             # MAE/RMSE/MAPE and CL metrics
    matrix.py              # R[t,j] evaluation and persistence
    profiler.py            # time/VRAM/RAM/bytes/params
  data/
    preprocessing.py       # public reproducible preprocessing
    splits.py              # chronological yearly/quarterly splits
scripts/
  run_revision.py          # config matrix, seeds, resume, aggregation
  aggregate_revision.py    # mean/std and LaTeX/CSV tables
config/revision/
  protocol/*.yaml
  sampler/*.yaml
  target/*.yaml
  graph/*.yaml
  robustness/*.yaml
```

关键接口建议：

```python
class NodeSelector:
    def select(self, previous, current, budget, context) -> SelectionResult: ...

@dataclass
class PeriodState:
    model_state: dict
    target_state: dict
    memory_state: Tensor
    replay_buffer: ReplayBuffer
    metadata_state: Optional[MetadataState]

class ContinualEvaluator:
    def evaluate_seen_tasks(self, model, task_id) -> dict: ...
```

每次运行必须写出结构化结果，而不只写日志：

```json
{
  "dataset": "PEMSD3-stream",
  "seed": 0,
  "variant": "w1_target",
  "period": 2014,
  "metrics": {"mae": 0, "rmse": 0, "mape": 0},
  "cl_metrics": {"aip": 0, "bwt": 0, "forgetting": 0},
  "cost": {"train_s": 0, "peak_vram_mb": 0, "buffer_bytes": 0,
           "metadata_bytes": 0, "params": 0},
  "selected_nodes": [],
  "config_hash": "..."
}
```

## 5. 当前代码必须先修的可信度问题

1. `main.py` 最后强制 `args.device = "cuda:0"`，会忽略 `--gpuid`，多卡实验和复现会错卡。
2. `global_train_steps = len(train_loader) // batch_size` 将 batch 数再次除以 batch size，定义错误（虽当前未使用）。
3. 早停代码 patience=50，与论文写的 patience=10 不一致；epoch=50 时基本不会早停。
4. 验证和测试调用 `model.forward` 会更新 `hidden_states_per_year`，导致验证/测试数据污染状态；评估必须无副作用。
5. 加载 checkpoint 后无条件 `reset_target_network()`，会覆盖已保存的 EMA target，破坏跨周期稳定分支。
6. target 参数虽然 `requires_grad=False`，但 EMA 参数仍被注册进总参数；表格应同时报告 predictor、target、total、trainable。
7. 原 `feature` sampler 的 Wasserstein 调用不正确，且 previous/current 分别 min-max normalization，距离不可比；必须共享归一化范围。
8. `feature_mmd` 当前在直方图分支之前先构造 p/q，随后却对原向量算 MMD，逻辑可运行但命名/实现应拆开并测试。
9. high-error selector 的 batch truth offset 在 padded/loader 行为下需验证；应保存实际样本 index。
10. `node_list = list(set(...))` 破坏确定顺序；`k_hop_subgraph` mapping 与训练标签映射必须通过单元测试核验。
11. 当 node_list 为空时，代码仍可能在日志前访问 `args.subgraph`；需要显式空更新分支。
12. 当前 replay 读取完整上一年 raw 文件后只截最后一周。存储声明必须按实际访问协议计算，不能只报告 `hidden_states_per_year`。
13. 当前 node embedding 只有形状 `(1, D)` 并对所有节点 expand，不能称为 node-adaptive embedding；真正的空间差异主要来自历史流量与更新子图策略。
14. TMRB 保存的是每年聚合状态字典，内存随 period 数线性增长，并非严格常数；需限定只保留上一状态或诚实报告增长率。
15. 配置中的 `rho`、正文中的 replay ratio、Top-M、TMRB Top-K 容易混用；建立统一 schema 和符号表。

这些问题修复后再跑大规模实验，否则返修表格可能无法经受复查。

## 6. 实验执行优先级

### P0：决定论文能否成立

1. 冻结 protocol，消除验证/测试状态污染；
2. 正确实现 W1 和相同预算 selector；
3. 增加真正 contrast loss，或全篇删除 contrastive claim；
4. 生成任务评估矩阵与 forgetting/BWT；
5. 澄清 adjacency-free predictor 与 topology-assisted update 的边界；
6. 三数据集、3 seeds 的核心模型和 random/L2/high-error 对照。

### P1：正面覆盖审稿人

1. target/momentum/online sampler 消融；
2. 参数、VRAM、buffer、metadata、逐周期耗时；
3. KL/JS/MMD/W1-phase-aware；
4. fixed graph/dynamic graph/update-without-graph；
5. quarterly 或 synthetic short-cycle experiment；
6. 至少一个强静态 SOTA retrained baseline。

### P2：增强录用概率

1. missing/noise/topology shock robustness；
2. PECPM/TFMoE/EAC 全数据集；
3. 完整 dataset card 与一键预处理；
4. effect size、置信区间和失败案例可视化。

不建议一开始就运行现有 `run_reviewer_experiments.sh`：它缺少 seed matrix、CL evaluation matrix、严格 buffer 协议和结果聚合，而且当前 `feature` W1 仍是错误实现。

## 7. 论文结构修改框架

1. `Introduction`：把 absolute graph-free 改为 predictor forward adjacency-free；给出 full retraining 的规模/频率边界，不再使用未经证明的 quadratic cost。
2. `Problem Definition`：新增 information availability、task boundary、buffer budget 和 deployment protocol。
3. `Methodology`：依次写 predictor、online-target、contrast objective、drift selector、TMRB、topology-assisted update；统一 M、K、rho。
4. `Wasserstein Derivation`：给出 support/bin、CDF、ground cost 与 phase-aware 版本。
5. `Experiments`：拆成 accuracy、continual retention、controlled sampler、target、graph dependency、efficiency、frequency、robustness。
6. `Dataset/Reproducibility`：加入 sensor identifiers、地理范围、缺失统计、插值、split timestamp、脚本与校验和。
7. `Limitations`：承认 metadata-assisted update、年度数据的局限及 abrupt topology 的适用边界。
8. `Response Letter`：每条采用 `Comment / Response / Changes / Evidence` 四段式，精确写页码、表号、代码/数据链接。

标题也建议与最终选择一致：

- 若加入 InfoNCE：`CoMemNet: Momentum-Contrastive Continual Traffic Prediction with Budgeted Drift-Aware Memory`；
- 若不加入 InfoNCE：`CoMemNet: Dual-Branch Drift-Aware Memory for Continual Traffic Prediction`。

## 8. 最小可提交返修包

- 修订论文（首页为逐条 response）；
- response letter 独立源文件；
- 3 seeds 核心结果与 sampler/target/graph 消融；
- task-wise matrix、forgetting/BWT/AIP；
- efficiency + storage protocol 表；
- 一个 static SOTA、一个 short-cycle、一个 missing-sensor 实验；
- 可执行 preprocessing、split manifests、sensor metadata 与 dataset card；
- 固定环境、随机种子、结构化结果和聚合脚本。

这套范围能覆盖 Reviewer 1/2 的全部核心问题以及 Reviewer 3 的 1–6、8。Reviewer 3 建议的跨领域引用可以在 related work 中简短讨论，但不能用引用替代方法与实验修复。
