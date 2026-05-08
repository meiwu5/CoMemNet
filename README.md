# CoMemNet: Contrastive Sampling with Memory Replay Network for Continual Traffic Prediction

> **Mei Wu, Wenchao Weng, Wenxin Su, Wenjie Tang, Wei Zhou**
>

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/pdf/2605.05738)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

---

## Overview

Traffic networks in the real world continuously expand and evolve — new roads are added, sensor distributions shift, and traffic patterns drift over time. Most existing spatio-temporal models assume a **static graph structure** and fail to adapt to these changes, leading to catastrophic forgetting and poor long-term performance.

**CoMemNet** is a simple yet efficient **dual-branch continual learning framework** for traffic prediction on expanding road networks. It addresses three core challenges:

- 🗺️ **Feature extraction under dynamic topology** — via an embedding-based backbone requiring no explicit graph input
- 🧠 **Catastrophic forgetting under spatio-temporal drift** — via a momentum-updated dual-branch contrastive mechanism
- 💾 **Memory explosion in incremental graphs** — via a lightweight Node-Adaptive Temporal Memory Replay Buffer (TMRB-N)

![CoMemNet Architecture](https://github.com/user-attachments/assets/8bd068bd-9e94-4b40-a267-360f2c0ed69d)

---

## Key Contributions

- **Embedding-based Backbone** — no explicit graph structure input or generation required, enabling seamless adaptation to incremental node/edge additions
- **Dual-Branch Contrastive Framework** — an Online Branch for fast convergence and a momentum-updated Target Branch for stable historical knowledge retention
- **DC Sampler (Dynamic Contrastive Sampler)** — uses Wasserstein Distance to identify nodes with significant distribution shifts, implementing a curriculum-style hard-example-first training strategy
- **TMRB-N (Node-Adaptive Temporal Memory Replay Buffer)** — lightweight gated memory module that stores and updates only key node representations, avoiding memory explosion
- **Two new open-source datasets** — PEMSD4(L) and PEMSD8(M), covering Bay Area and Southern California traffic networks over multiple years

---

## Results

CoMemNet achieves **state-of-the-art performance** on three large-scale real-world datasets across all temporal granularities (15 / 30 / 60 min), while training on only **15–30% of nodes** in later years and achieving the **fastest training speed** among all continual learning baselines.

| Dataset | MAE (60min) | RMSE (60min) | MAPE (60min) | Avg. Time (s/epoch) |
|---------|-------------|--------------|--------------|----------------------|
| PEMSD3(S) | **13.57** | **22.94** | **18.80** | **0.30** |
| PEMSD4(L) | **22.00** | **37.38** | **15.86** | **0.65** |
| PEMSD8(M) | **17.03** | **28.41** | **17.82** | **0.23** |

---

## Repository Structure

```
CoMemNet/
├── main.py              # Main training & evaluation entry point
├── run.sh               # Quick-start shell script
├── requirements.txt     # Python dependencies
├── config/              # Dataset-specific configuration files
├── src/                 # Core model implementation (CoMemNet, DC Sampler, TMRB-N)
├── baseline/            # Baseline models (TrafficStream, STKEC, etc.)
├── distance/            # Wasserstein distance computation utilities
├── embedding/           # Node embedding modules
└── utils/               # Data loading, preprocessing, evaluation helpers
```

---

## Getting Started

### 1. Environment Setup

```bash
conda create -n CoMemNet python==3.12
conda activate CoMemNet
pip install -r requirements.txt
```

### 2. Data Preparation

Download the datasets from Google Drive:

📦 **Dataset & Pre-run Logs**: [Google Drive](https://drive.google.com/file/d/1SjsMsZIIWdKxzKxySROnb4Z84t1mZXXU/view?usp=drive_link)

Place the data in the `data/` directory:
```
CoMemNet/
└── data/
    ├── PEMSD3/
    ├── PEMSD4/
    └── PEMSD8/
```

For baseline experiments, place data in:
- `/baseline/TrafficStream-main/data/`
- `/baseline/STKEC-main/data/`

Then set `data_process: 1` in the config to generate `FastData`.

### 3. Run

```bash
sh run.sh
```

Run logs are saved to the `res/` folder.

---

## Configuration Guide

Key parameters in the `config/` files:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `data_process` | `1` / `0` | `1`: generate FastData; `0`: use existing data |
| `auto_test` | `1` / `0` | `1`: train model; `0`: inference only with saved weights |
| `auto_lr` | `1` / `0` | Enable automatic learning rate scheduling |
| `increase` | `true` / `false` | Include newly added nodes (Vτ \ Vτ₋₁) in training |
| `replay` | `true` / `false` | Enable DC Sampler memory replay |
| `replay_ratio` | float (e.g. `0.05`) | Proportion ρ of replayed nodes relative to total nodes |
| `replay_strategy` | `"feature"` | Use Target branch for node feature extraction |
| `num_hops` | int (e.g. `2`) | Neighborhood hops used by the sampler |
| `is_TMRB` | `true` / `false` | Enable TMRB-N temporal memory replay buffer |
| `is_update` | `true` / `false` | Enable temporal feature update within TMRB-N |
| `select_k` | `true` / `false` | `true`: Top-K feature difference selection; `false`: random selection |

### Recommended Hyperparameters

| Dataset | Batch Size | LR | Epochs | Momentum (β) | LR Decay | ρ |
|---------|------------|----|--------|--------------|----------|---|
| PEMSD3(S) | 128 | 0.01 | 50 | 0.99 | 0.5 | 0.05 |
| PEMSD4(L) | 128 | 0.01 | 50 | 0.99 | 0.5 | 0.03 |
| PEMSD8(M) | 128 | 0.01 | 60 | 0.99 | 0.5 | 0.05 |

---

## Datasets

| Dataset | Years | Max Nodes | Max Edges | Region |
|---------|-------|-----------|-----------|--------|
| PEMSD3(S) | 2011–2017 | 871 | 2,788 | Public |
| PEMSD4(L) | 2009–2015 | 2,406 | 9,773 | Bay Area, CA *(new)* |
| PEMSD8(M) | 2012–2018 | 320 | 1,089 | Southern CA *(new)* |

All datasets are sourced from the [CalTrans PeMS system](https://pems.dot.ca.gov/), sampled every 30 seconds and aggregated into 5-minute intervals. Sensors are filtered by missing rate (<10%), geographic coherence (State Post-Mile < 100), and temporal continuity across years.

---

## Acknowledgements

We thank the authors of [TrafficStream](https://github.com/AprLie/TrafficStream) and [STKEC](https://github.com/UnderReview24/STKEC) for open-sourcing their baseline implementations.

---

## Contact

For questions or issues, feel free to open a GitHub Issue or contact:

- **Mei Wu** — wumei5@sjtu.edu.cn *(Shanghai Jiao Tong University)*
- **Wenchao Weng** — 111124120010@zjut.edu.cn *(Zhejiang University of Technology)*
