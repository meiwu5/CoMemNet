# CoMemNet

[English](README.md)

CoMemNet 是面向持续演化传感器网络的交通流预测框架。模型由不直接输入邻接矩阵的预测主干、Online/EMA-Target 双分支、漂移感知节点选择、拓扑辅助局部更新，以及节点自适应时间记忆回放模块 TMRB-N 组成。

仓库同时包含大修补充实验框架，位于 `config/reviewer/` 和 `scripts/`，用于完成节点选择策略、Target 分支、momentum、对比损失、图依赖、灾难性遗忘及计算/存储成本等受控实验。

## 发布资源

- 处理后数据集与数据处理说明：`[DATASET_GOOGLE_DRIVE_URL]`
- 本轮返修的模型权重、日志、配置和汇总结果：`[REVISION_ARTIFACTS_GOOGLE_DRIVE_URL]`

本 Git 仓库不包含数据集、scale 子集、模型权重、日志和实验输出。PEMSD3-stream 使用论文中引用的公开演化路网数据发布版本；PEMSD4(L) 与 PEMSD8(M) 基于 CalTrans PeMS 公开记录处理，处理后数据和说明通过上述数据链接提供。

## 项目结构

```text
CoMemNet/
├── main.py                         # 训练和持续学习评估入口
├── src/model/                      # 预测主干、TMRB 和回放节点选择
├── utils/                          # 数据加载、预处理与评价指标
├── config/                         # 主实验、消融、参数及返修配置
├── data/                           # 年度交通数据、邻接矩阵和缓存划分
├── baseline/                       # baseline 实现
├── scripts/
│   ├── generate_reviewer_configs.py
│   ├── run_reviewer_experiments.sh
│   └── aggregate_reviewer_results.py
├── paper/CoMemNet/                 # 论文源文件与图片
└── paper/reviewer/                 # 审稿意见和返修蓝图
```

## 环境安装

推荐使用 Python 3.10 或更高版本：

```bash
pip install -r requirements.txt
```

也可使用 Conda 环境文件：

```bash
conda env create -f environment.yml
conda activate comemnet
```

主要依赖包括 PyTorch 2.7.1 或更高版本、PyTorch Geometric 2.6.1、NumPy、SciPy、pandas、NetworkX 和 scikit-learn。使用 GPU 时，需要安装与本机 CUDA 驱动匹配的 PyTorch 版本。

RTX 5090 等计算能力为 `sm_120` 的 Blackwell GPU 必须使用 CUDA 12.8 或更高版本的 PyTorch wheel。建议先执行：

```bash
pip install --upgrade torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

PyTorch 2.5.x 的 CUDA 12.4 wheel 不包含 `sm_120` 内核，会报错 `no kernel image is available for execution on the device`。

训练前可检查环境：

```bash
python -c "import torch, scipy, pandas; print(torch.__version__, torch.cuda.is_available())"
```

在 Blackwell GPU 上还应确认输出包含 `sm_120`：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
```

如果出现 `ModuleNotFoundError: pandas` 或 `scipy`，说明当前 Python 环境尚未安装项目依赖。

## 数据目录

三个数据集均包含七个年度周期：

| 数据集 | 年份 | 首年/末年节点数 |
|---|---:|---:|
| PEMSD3-stream | 2011–2017 | 655 / 871 |
| PEMSD4-large | 2009–2015 | 1118 / 2406 |
| PEMSD8-mini | 2012–2018 | 216 / 320 |

代码期望的数据结构为：

```text
data/<dataset>/
├── finaldata/<year>.npz       # 年度原始张量，键名为 x
├── graph/<year>_adj.npz       # 年度邻接矩阵，键名为 x
└── FastData/<year>_30day.npz  # 按时间划分的训练/验证/测试缓存
```

返修配置默认设置 `data_process: 0`，直接使用已有 FastData。只有需要重新生成缓存时才改为 `1`；PEMSD4-large 等缓存单个年度可能占用数 GB。

D4 规模实验的子集不单独下载，可由 `PEMSD4-large` 自动生成：

```bash
python scripts/prepare_scaling_datasets.py \
  --source PEMSD4-large \
  --ratios 0.25 0.50 0.75
```

## 外部 baseline 安装

第三方 baseline 及其数据不直接放入本仓库。运行以下命令可安装实验脚本所需的固定版本 STID、STAEformer、EAC 和 PDFormer：

```bash
bash scripts/setup_external_baselines.sh
```

## 基础训练

在 PEMSD3-stream 上运行 CoMemNet：

```bash
python main.py \
  --conf config/PEMSD3-stream/model.json \
  --dataset PEMSD3-stream \
  --gpuid 0 \
  --seed 0
```

运行其他数据集：

```bash
python main.py --conf config/PEMSD4-large/model.json --dataset PEMSD4-large --gpuid 0 --seed 0
python main.py --conf config/PEMSD8-mini/model.json --dataset PEMSD8-mini --gpuid 0 --seed 0
```

CPU 运行可设置 `--gpuid -1`，但完整实验建议使用 CUDA GPU。

每个数据集还包含 `static.json`、`retrained.json`、`no_TMRB.json`、`no_update.json`、`no_select.json`、`no_replay.json` 和 `no_increase.json` 等基础消融配置。

## 返修补充实验

修改实验模板后，重新生成配置：

```bash
python scripts/generate_reviewer_configs.py
```

先用单随机种子进行完整性检查：

```bash
GPU_ID=0 SEEDS="0" bash scripts/run_reviewer_experiments.sh
```

确认无误后运行三个随机种子：

```bash
GPU_ID=0 SEEDS="0 1 2" bash scripts/run_reviewer_experiments.sh
```

脚本默认启用实验级断点续跑。重新执行相同命令时，会扫描结构化结果并跳过已完成的 dataset/variant/seed 组合。若某个变体在运行中被中断，该变体会从第一个周期重新开始；目前不支持单次实验内部的 epoch 级恢复。如需故意重新运行全部实验，可设置：

```bash
GPU_ID=0 SEEDS="0" RESUME=0 bash scripts/run_reviewer_experiments.sh
```

也可以单独运行某项实验。例如 PEMSD3 的 Wasserstein 节点选择：

```bash
python main.py \
  --conf config/reviewer/PEMSD3-stream/sampler_feature.json \
  --dataset PEMSD3-stream \
  --gpuid 0 \
  --seed 0
```

主要实验组包括：

- 节点选择对照：Wasserstein、Random、Recency、High-error、L2、KL、JS、MMD；
- Target 消融：无 Target、只用 Online 特征采样，以及 momentum 0.90/0.95/0.99/0.995；
- 损失消融：MAE-only，以及 contrastive weight 0.01/0.05/0.1/0.2；
- 图依赖消融：仅训练选中节点，以及使用 1/2/3-hop 邻接扩展。

修订后的 Wasserstein 使用共同 support 和共享归一化范围。启用对比目标时，训练损失由预测 MAE 和节点级 InfoNCE 组成。

## 结果保存路径

普通实验结果保存在：

```text
res/<dataset>/<variant><timestamp>/
```

返修实验结果保存在：

```text
res/reviewer/<dataset>/<variant><timestamp>/
├── <variant>.log
├── <year>/<best-validation-loss>.pkl
└── metrics/summary.json
```

其中：

- `<variant>.log`：逐 epoch 损失、年度指标、时间和显存统计；
- `<year>/*.pkl`：每个周期验证集最优 checkpoint；
- `metrics/summary.json`：任务评估矩阵 `R[t,j]`、AIP-MAE、BWT、Forgetting 和逐周期效率/存储统计。

查找全部结构化结果：

```bash
find res/reviewer -path '*/metrics/summary.json' -print
```

聚合三个随机种子：

```bash
python scripts/aggregate_reviewer_results.py \
  --root res/reviewer \
  --output res/reviewer/aggregate.json
```

最终聚合结果位于：

```text
res/reviewer/aggregate.json
```

其中给出每个数据集和实验变体的 AIP-MAE、Forgetting 和 BWT 的均值与样本标准差。

## 实验协议说明

- 每个年度周期按时间顺序划分训练集、验证集和测试集。
- 验证和测试前向过程不会更新 TMRB 状态。
- checkpoint 中保存的 EMA Target 参数会跨周期保留。
- 预测主干的 forward 不直接读取邻接矩阵；topology-assisted 实验只在扩展局部训练节点集合时使用邻接信息。
- `graph_selected_nodes_only` 实验关闭基于邻接矩阵的邻域扩展。
- 当前 replay 协议会访问历史交通数据完成节点选择，因此实验会分别报告历史数据访问量、邻接元数据量和紧凑记忆状态大小。

审稿意见与实验的完整映射、已知限制及后续优先级参见[返修执行蓝图](paper/reviewer/revision_plan.md)。

## 引用

论文正式发表后将在此补充标准 BibTeX。在此之前，如本仓库对研究有帮助，请引用对应的 CoMemNet 稿件并附上本仓库链接。

## 许可证

当前仓库尚未包含许可证文件。重新分发代码或数据前，请先联系仓库作者。

### 最终主结果多随机种子与 EAC baseline

只补最终 CoMemNet 主方法缺失的 seed（默认检查三个数据集的 seed 0、1、2，仅运行缺失配置）：

```bash
GPU_ID=0 SEEDS="0 1 2" RESUME=1 bash scripts/run_final_main_seeds.sh
```

专用的均值和标准差汇总保存在 `res/reviewer/final_main_multiseed.json`。

在完全相同的冻结时间划分上运行 EAC 官方实现（seed 0）：

```bash
GPU_ID=0 SEED=0 RESUME=1 bash scripts/run_eac_baseline.sh
```

EAC 结果保存在 `res/baseline/EAC/<dataset>/eac-0/metrics/summary.json`。适配层复用 CoMemNet 的相同样本划分，只向 EAC 暴露其所需的交通流通道；EAC 官方模型、MSE 目标、优化器和超参数保持不变。
