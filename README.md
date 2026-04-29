# WANZI: Traffic Forecasting Backdoor Poisoning Experiments

## 摘要

本仓库提供一个面向交通速度预测任务的回归后门投毒实验框架。研究对象是训练期投毒攻击：在少量训练样本中注入平滑且低异常性的触发模式，使 `LSTM` 交通预测模型在正常输入下保持可用，而在触发输入下对指定传感器和指定预测时段产生目标方向偏移。

当前版本已经从早期的多路线探索收敛为一条论文主线：

- 主任务：多步交通速度预测
- 主模型：`LSTM`
- 主数据集：`METR-LA`
- 补充验证数据集：`PEMS-BAY`
- 主攻击策略：`spatiotemporal_headroom` 时空脆弱位置感知排序，综合干净模型预测误差（0.45）、节点时间波动（0.30）和路网度中心性（0.25）进行复合节点选择
- 触发与偏移：`trigger_scope_node_count=20` 节点注入触发模式，`trigger_node_count=10` 节点做 `additive_directional` 加法式定向偏移，仅作用于尾部 `target_horizon_count=6` 个预测步
- 训练增强：`tail_headroom` 样本选择 + `target_region_loss_weight=200` 目标区域加权损失 + `frequency_smoothing_strength=0.45` 频域平滑

当前复验后的最佳主结果已达到最低论文标准，但没有达到更强正文标准。最佳主结果位于：

```text
results/metr_la_poison_20260427_040007
```

复验核心结果如下（p=0.05, σ=0.14）：

| 指标 | 数值 | 最低标准 | 更强标准 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 局部 ASR | 17.74% | >= 5.00% | >= 6.00% | 通过 |
| 干净 MAE 变化 | 4.12% | <= 5.00% | <= 4.00% | 最低通过，更强未通过 |
| 方向一致率 | 89.42% | >= 60.00% | >= 65.00% | 通过 |
| 偏移达成度 | +0.0072 | — | >= 0 | 通过 |
| 频域能量偏移 | 1.55 | <= 3.00 | <= 2.00 | 双通过 |
| 平均 z-score | 0.756 | <= 0.80 | <= 0.76 | 双通过 |

单阶段搜索中存在同时满足更强正文标准的候选（局部 ASR 16.19%，干净 MAE 变化 3.33%），但正文主结果以复验后的 `best_attack_paper.json` 为准。

---

## 1. 研究问题

交通速度预测模型通常以历史传感器序列作为输入，预测未来多个时间步的速度。本文实验关注以下问题：

> 在不显著破坏正常预测性能的前提下，训练期后门投毒是否能够使模型在触发输入下，对指定节点和指定预测时段产生稳定的目标方向偏移？

该问题属于回归任务中的后门攻击。与分类后门不同，回归后门没有离散标签翻转目标，因此本仓库采用“目标偏移式”定义：触发后预测值需要朝预设方向移动，并在被攻击节点的尾部预测时段上体现出来。

---

## 2. 研究范围

当前版本保留的研究范围如下：

| 维度 | 当前设定 |
| --- | --- |
| 任务 | 交通速度预测 |
| 模型 | `LSTMForecaster` |
| 主数据集 | `METR-LA` |
| 补充数据集 | `PEMS-BAY` |
| 攻击阶段 | 训练期投毒 |
| 攻击目标 | 被选节点在尾部预测步上的目标方向偏移 |
| 隐蔽性约束 | 干净性能、频域偏移、z-score、异常率 |
| 防御验证 | z-score 筛查、高频能量检查、移动平均平滑 |

当前版本不再展开以下方向：

- 图神经网络模型的大规模横向比较
- 交通状态估计任务
- 多数据集全量网格搜索
- 复杂防御体系的完整对抗评估

这些方向可以作为后续工作，但不属于当前论文主线。

---

## 3. 方法概述

实验流程分为五步。

### 3.1 干净基线训练

先在干净训练集上训练 `LSTM`，得到稳定基线。基线用于：

- 评估正常预测性能
- 生成训练集预测误差
- 支持脆弱节点和脆弱样本筛选
- 作为投毒后模型的比较对象

当前主基线：

```text
results/metr_la_clean_20260405_025213
```

| 指标 | 数值 |
| --- | ---: |
| 最佳 MAE | 0.3651 |
| 3 次重复 MAE 波动 | 2.94% |

### 3.2 脆弱节点和时间位置识别

攻击候选首先通过训练数据与基线预测结果确定。当前主线采用 `error` 策略，优先选择预测误差和局部敏感性较高的节点。触发窗口采用 `hybrid` 策略，兼顾高响应时间位置与序列尾部位置。

当前主结果选择的节点和时间位置为：

| 项目 | 数值 |
| --- | --- |
| 触发节点 | `5, 140, 165` |
| 触发时间位置 | `0, 1, 11` |
| 目标预测步 | `9, 10, 11` |

### 3.3 投毒样本选择

当前主线采用 `directional_headroom`。该策略优先选择在目标区域仍有向下偏移空间的训练样本，从而避免触发后预测方向与目标方向不一致。

后续探索中也保留了 `hybrid_headroom_error`，用于平衡方向空间与局部误差，但当前主结果仍来自 `directional_headroom`。

### 3.4 目标塑形与带权训练

当前主线采用：

- `dual_focus`：同时保留一定全局目标偏移，并强化被攻击节点的尾部预测时段
- `directional_focus`：对目标节点和目标预测步增加训练权重

这两项用于解决回归后门中的一个核心问题：如果只调整局部目标，旧口径全局 ASR 可能不足；如果只追求全局偏移，局部目标又容易不稳定。

### 3.5 复验与双冠军保存

每轮搜索后，脚本会对候选进行复验，并保存两类结果：

| 文件 | 含义 |
| --- | --- |
| `best_attack_raw.json` | 按局部主指标选择的原始最强候选 |
| `best_attack_paper.json` | 按论文标准选择的正文候选 |
| `best_attack.json` | 默认指向论文候选 |

在当前主结果中，原始最强候选和论文候选是同一组。

### 3.6 时空脆弱位置感知优化

该优化延续开题报告和中期报告中的原技术路线：数据预处理、滑动窗口、`LSTM` 基线、训练期后门投毒和多指标评估均保持不变，只增强”脆弱节点、脆弱时间窗口和目标区域偏移稳定性”。

当前主线采用 `spatiotemporal_headroom` 策略，节点排序由三类信息共同决定：

- 干净模型预测误差：权重 0.45；
- 节点时间波动：权重 0.30；
- 路网邻接中心性：权重 0.25。

配套 `tail_headroom` 样本选择模式，优先选择目标区域仍有下调空间且不过于异常的样本。目标塑形使用 `additive_directional`，只对目标节点和尾部预测步做加法式定向偏移，替代旧有的全局乘法式偏移。通过 `trigger_scope_node_count=20` 与 `trigger_node_count=10` 分离触发覆盖范围与目标偏移范围。训练时对目标区域施加 200× 损失权重，并通过 `frequency_smoothing_strength=0.45` 降低频域痕迹。

相关配置：

| 配置 | 用途 |
| --- | --- |
| `configs/metr_la_spatiotemporal_headroom.yaml` | 初始正式配置 |
| `configs/metr_la_spatiotemporal_headroom_smoke.yaml` | 快速验证配置 |
| `configs/metr_la_spatiotemporal_headroom_v11.yaml` | **当前主实验配置**（复验主结果达到最低标准） |

---

## 4. 论文标准

论文标准集中定义在：

```text
src/traffic_poison/thesis_contract.py
```

### 4.1 最低论文标准

| 指标 | 说明 | 阈值 |
| --- | --- | ---: |
| `clean_MAE_delta_ratio` | 带毒模型正常输入误差相对基线的变化 | <= 0.05 |
| `raw_selected_nodes_tail_horizon_attack_success_rate` | 原始速度空间中，目标节点尾部预测步的局部 ASR | >= 0.05 |
| `attack_success_rate` | 旧口径全局 ASR（additive_directional 策略下可为 0） | >= 0.0 |
| `raw_selected_nodes_tail_horizon_shift_direction_match_rate` | 目标区域偏移方向一致率 | >= 0.60 |
| `frequency_energy_shift` | 频域能量偏移（原始空间标度） | <= 3.0 |
| `mean_z_score` | 平均 z-score | <= 0.80 |

### 4.2 更强正文标准

| 指标 | 阈值 |
| --- | ---: |
| `clean_MAE_delta_ratio` | <= 0.04 |
| `raw_selected_nodes_tail_horizon_attack_success_rate` | >= 0.06 |
| `attack_success_rate` | >= 0.0 |
| `raw_selected_nodes_tail_horizon_shift_direction_match_rate` | >= 0.65 |
| `raw_selected_nodes_tail_horizon_target_shift_attainment` | >= 0 |
| `frequency_energy_shift` | <= 2.0 |
| `mean_z_score` | <= 0.76 |

当前 `spatiotemporal_headroom` v11 策略的复验主结果已达到最低标准。单阶段搜索中有更强正文标准候选，但不替代复验后的正文主结果。

---

## 5. 当前实验结果

### 5.1 主实验结果：METR-LA

主实验目录：

```text
results/metr_la_poison_20260427_040007
```

最佳候选参数（v11, p=0.05, σ=0.14）：

| 参数 | 数值 |
| --- | --- |
| `selection_strategy` | `spatiotemporal_headroom` |
| `sample_selection_mode` | `tail_headroom` |
| `target_shift_mode` | `additive_directional` |
| `trigger_scope_node_count` | `20` |
| `trigger_node_count` | `10` |
| `target_horizon_count` | `6` |
| `target_shift_ratio` | `0.15` |
| `poison_ratio` | `0.05` |
| `sigma_multiplier` | `0.14` |
| `target_region_loss_weight` | `200` |
| `frequency_smoothing_strength` | `0.45` |

复验后的最佳结果：

| 指标 | 数值 |
| --- | ---: |
| `attack_success_rate` | 0.0026 |
| `raw_selected_nodes_tail_horizon_attack_success_rate` | 0.1774 |
| `raw_selected_nodes_attack_success_rate` | 0.0358 |
| `clean_MAE_delta_ratio` | 0.0412 |
| `raw_selected_nodes_tail_horizon_shift_direction_match_rate` | 0.8942 |
| `raw_selected_nodes_tail_horizon_target_shift_attainment` | 0.0072 |
| `frequency_energy_shift` | 1.5528 |
| `mean_z_score` | 0.7557 |
| `anomaly_rate` | 0.0043 |
| `minimum_contract_pass` | true |
| `strong_contract_pass` | false |

9 组单阶段网格中 5 组通过最低标准，2 组通过更强标准；复验后正文候选仍通过最低标准，但因干净 MAE 变化为 4.12%，没有达到更强正文标准。历史主实验（`results/metr_la_poison_20260409_163212`，`error` 策略，最低标准通过）仍保留作为基线参考。

### 5.2 防御验证

防御结果目录：

```text
results/defense_eval_20260409_164245
```

主要结果：

| 防御或检查 | 结果 |
| --- | --- |
| z-score 筛查 | 干净样本与触发样本标记率相同 |
| 高频能量检查 | 干净样本与触发样本标记率相同 |
| 移动平均平滑 | 全局 ASR 从 1.42% 变为 1.45% |
| 局部 ASR 平滑后变化 | 7.36% 变为 7.41% |

结论：当前触发模式没有被这些简单检查明显区分，移动平均平滑也没有有效削弱攻击效果。

### 5.3 跨数据集验证：PEMS-BAY

跨数据集结果目录：

```text
results/pems_bay_cross_20260429_172956
```

结果摘要：

| 指标 | 数值 |
| --- | ---: |
| 最佳局部 ASR | 0.00% |
| 旧口径全局 ASR | 0.0032% |
| 干净 MAE 变化 | 2.90% |
| 方向一致率 | 84.59% |
| `target_shift_attainment` | 0.0027 |
| `frequency_energy_shift` | 1.3371 |
| `mean_z_score` | 0.6360 |
| 最低标准 | 未通过 |
| 更强标准 | 未通过 |

该结果来自 `results/metr_la_poison_20260427_040007/best_attack_paper.json` 的补测。补测保持了较低干净 MAE 变化和较高方向一致率，但局部 ASR 为 0，未达到跨数据集验证标准。因此当前 427 复验主结果不能写作已经完成有效跨数据集迁移。

### 5.4 后续探针结果

2026-04-25 增加了两个进一步探针：

| 配置 | 结果目录 | 结论 |
| --- | --- | --- |
| `configs/metr_la_opt_tail_directional.yaml` | `results/metr_la_poison_20260425_035827` | 局部 ASR 可达 12.83%，但全局 ASR 不足 |
| `configs/metr_la_opt_global_stealth_probe.yaml` | `results/metr_la_poison_20260425_041952` | 局部 ASR 可达 9.65%，干净误差更低，但全局 ASR 和频域约束未同时过线 |

这两组结果不替代主结果。当前推荐 `results/metr_la_poison_20260427_040007`（v11）作为主实验达标结果，旧主实验 `results/metr_la_poison_20260409_163212` 保留作为历史基线。

### 5.5 时空脆弱位置感知优化最终结果

2026-04-27 完成了 `spatiotemporal_headroom` 策略的 11 轮迭代优化。最终 v11 配置在 winbox 上得到复验主结果，并达到最低论文标准。

最佳结果目录：

```text
results/metr_la_poison_20260427_040007
```

最佳候选参数：

| 参数 | 数值 | 说明 |
| --- | --- | --- |
| `selection_strategy` | `spatiotemporal_headroom` | 复合节点排序 |
| `sample_selection_mode` | `tail_headroom` | 优先中位数样本 |
| `target_shift_mode` | `additive_directional` | 加法式定向偏移 |
| `trigger_scope_node_count` | `20` | 20 个节点注入触发 |
| `trigger_node_count` | `10` | 10 个节点修改目标 |
| `target_horizon_count` | `6` | 尾部 6 个预测步 |
| `target_shift_ratio` | `0.15` | 偏移幅度 |
| `poison_ratio` | `0.05` | 5% 投毒率 |
| `sigma_multiplier` | `0.14` | 触发强度 |
| `target_region_loss_weight` | `200` | 目标区域损失权重 |
| `frequency_smoothing_strength` | `0.45` | 频域平滑强度 |

复验最佳结果（p=0.05, σ=0.14）：

| 指标 | 数值 | 最低标准 | 更强标准 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `raw_selected_nodes_tail_horizon_attack_success_rate` | 0.1774 | >= 0.05 | >= 0.06 | 通过 |
| `raw_selected_nodes_tail_horizon_shift_direction_match_rate` | 0.8942 | >= 0.60 | >= 0.65 | 通过 |
| `raw_selected_nodes_tail_horizon_target_shift_attainment` | 0.0072 | — | >= 0 | 通过 |
| `clean_MAE_delta_ratio` | 0.0412 | <= 0.05 | <= 0.04 | 最低通过，更强未通过 |
| `frequency_energy_shift` | 1.5528 | <= 3.0 | <= 2.0 | 通过 |
| `mean_z_score` | 0.7557 | <= 0.80 | <= 0.76 | 通过 |
| `attack_success_rate` | 0.0026 | >= 0.0 | >= 0.0 | 通过 |
| `minimum_contract_pass` | true | — | — | 通过 |
| `strong_contract_pass` | false | — | — | 未通过 |

9 组单阶段网格中 5 组通过最低标准，2 组通过更强标准。单阶段 (p=0.05, σ=0.14) 的 raw_tail_ASR 为 16.19%，clean_delta 为 3.33%；复验后同一候选的 raw_tail_ASR 为 17.74%，clean_delta 为 4.12%，因此正文主结果按复验口径只表述为最低标准通过。

迭代关键发现：

| 版本 | 改动 | raw_tail_ASR | 方向一致率 | clean_delta | freq | 关键洞察 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v1 | 初始（5 节点触发） | 0% | 33-75% | 2.5-8.4% | 0.08 | 触发太弱，模型学不到 |
| v5 | 全局触发（207 节点） | 0-0.22% | 98.8% | 2.7-6.1% | 5.6 | 突破！方向一致率跃升 |
| v9 | 40 节点 + 5% 投毒 + 200× | 0.28%* | 97.8% | 6.5-7.8% | 8.4 | 折中平衡点 |
| v10 | 新管道 raw-space 评估 | 17.6% | 87.6% | 3.97% | 2.70 | 新管道释放真实 ASR |
| **v11** | **20 节点 + 频域平滑 0.45** | **17.7%** | **89.4%** | **4.12%** | **1.55** | **复验主结果最低标准通过** |

\* v9 及之前版本在标准化空间中计算 ASR，v10 起切换至原始速度空间。

与原有主实验的对比：

| 指标 | 原主实验 | v11 新策略 | 变化 |
| --- | ---: | ---: | --- |
| raw_tail_ASR | 7.31% | **17.74%** | +143% |
| 方向一致率 | 71.1% | **89.4%** | +26% |
| 干净 MAE 变化 | 3.63% | **4.12%** | +14% |
| freq_energy_shift | 0.042† | 1.55‡ | 评估空间不同 |
| 论文标准通过 | 最低 | **最低** | 攻击效果增强 |

† 原主实验在标准化空间中计算频域偏移。‡ v11 在原始速度空间中计算，阈值已同步调整。

---

## 6. 数据

仓库当前包含：

| 文件 | 用途 |
| --- | --- |
| `data/metr-la.h5` | `METR-LA` 速度数据 |
| `data/adj_mx.pkl` | `METR-LA` 邻接矩阵 |
| `data/pems-bay.csv` | `PEMS-BAY` 速度数据 |
| `data/adj_mx_bay.pkl` | `PEMS-BAY` 邻接矩阵 |

默认切分方式：

| 切分 | 比例 |
| --- | ---: |
| 训练集 | 70% |
| 验证集 | 10% |
| 测试集 | 20% |

默认窗口设置：

| 项目 | 数值 |
| --- | ---: |
| 输入长度 | 12 |
| 预测长度 | 12 |
| batch size | 256 |

---

## 7. 代码结构

```text
wanzi/
├── configs/
├── data/
├── results/
├── scripts/
├── src/traffic_poison/
└── tests/
```

### 7.1 配置文件

| 文件 | 作用 |
| --- | --- |
| `configs/metr_la.yaml` | `METR-LA` 干净基线配置 |
| `configs/metr_la.yaml` | `METR-LA` 干净基线配置 |
| `configs/metr_la_spatiotemporal_headroom_v11.yaml` | **当前主实验配置**（复验主结果最低标准通过） |
| `configs/metr_la_opt_loss_rebalance.yaml` | 旧主实验配置（历史基线） |
| `configs/metr_la_opt_selection_balance.yaml` | 选择平衡方向 |
| `configs/metr_la_opt_spread_recovery.yaml` | 范围恢复方向 |
| `configs/metr_la_opt_global_stealth_probe.yaml` | 2026-04-25 全局与隐蔽性探针 |
| `configs/metr_la_opt_tail_directional.yaml` | 2026-04-25 tail 触发探针 |
| `configs/metr_la_spatiotemporal_headroom.yaml` | 时空脆弱位置感知初始配置 |
| `configs/metr_la_spatiotemporal_headroom_smoke.yaml` | 时空脆弱位置感知快速验证 |
| `configs/pems_bay_paper_optimization.yaml` | `PEMS-BAY` 跨数据集验证 |
| `configs/*_smoke.yaml` | 快速检查配置 |

### 7.2 脚本

| 文件 | 作用 |
| --- | --- |
| `scripts/run_clean_baseline.py` | 训练干净基线 |
| `scripts/run_poison_experiments.py` | 投毒搜索、复验和结果保存 |
| `scripts/run_defense_eval.py` | 基础防御评估 |
| `scripts/run_cross_dataset.py` | 跨数据集验证 |
| `scripts/build_thesis_tables.py` | 生成论文汇总表 |
| `scripts/run_ablation_study.py` | 参数消融实验 |
| `scripts/prepare_dataset.py` | 数据准备与摘要生成 |

### 7.3 源码模块

| 文件 | 作用 |
| --- | --- |
| `src/traffic_poison/data.py` | 数据读取、标准化、窗口化和加载器构造 |
| `src/traffic_poison/model.py` | `LSTMForecaster` 模型 |
| `src/traffic_poison/trainer.py` | 训练、验证、早停和带权损失 |
| `src/traffic_poison/poisoning.py` | 节点排序、时间窗口、触发器、目标塑形和攻击指标 |
| `src/traffic_poison/defenses.py` | z-score、高频能量、移动平均与 Neural Cleanse 风格检查 |
| `src/traffic_poison/metrics.py` | 回归指标、置信区间和统计检验 |
| `src/traffic_poison/thesis_contract.py` | 论文标准、排序规则和候选筛选 |
| `src/traffic_poison/experiment.py` | 共享实验流程 |
| `src/traffic_poison/reporting.py` | 表格、图像和摘要输出 |
| `src/traffic_poison/utils.py` | 随机种子、设备、结果目录和数组工具 |

---

## 8. 环境安装与依赖

### 8.1 Python 版本要求

项目需要 **Python >= 3.10**。已在以下环境中验证：

| 平台 | Python | PyTorch | 验证日期 |
| --- | --- | --- | --- |
| macOS (ARM) | 3.12.7 | 2.10.0 (CPU) | 2026-04-27 |
| Windows (x64) | 3.10.x | 2.x (CUDA) | 2026-04-27 |

### 8.2 安装方式一：venv + requirements.txt（推荐，精确复现）

使用 `requirements.txt` 锁定依赖版本，确保跨环境一致：

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows cmd
# .venv\Scripts\Activate.ps1     # Windows PowerShell

# 2. 升级 pip
python -m pip install --upgrade pip

# 3. 安装依赖（精确版本）
pip install -r requirements.txt

# 4. 安装项目本身（可编辑模式）
pip install -e .
```

### 8.3 安装方式二：Conda（适合 GPU 训练 / Windows）

```bash
# 1. 创建环境
conda create -n wanzi310 python=3.10 -y
conda activate wanzi310

# 2. 安装 PyTorch（根据 CUDA 版本选择）
# CUDA 11.8:
pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其余依赖
pip install -r requirements.txt

# 4. 安装项目
pip install -e .
```

### 8.4 验证安装

```bash
python -c "
import torch
import numpy as np
import h5py
import traffic_poison
print(f'Python: OK')
print(f'PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'NumPy {np.__version__}')
print(f'h5py {h5py.__version__}')
print(f'traffic_poison: OK')
"
```

预期输出末尾包含 `traffic_poison: OK` 且无 ImportError。

### 8.5 数据文件检查

主实验（METR-LA）所需数据已包含在仓库中，无需额外下载：

```bash
ls -lh data/metr-la.h5 data/adj_mx.pkl
```

如果做 PEMS-BAY 补充实验，需要自行下载以下文件放入 `data/` 目录：
- `data/pems-bay.h5`
- `data/adj_mx_bay.pkl`

PEMS-BAY 数据来源：https://github.com/liyaguang/DCRNN

### 8.6 硬件建议

| 配置 | CPU 训练 | GPU 训练 |
| --- | --- | --- |
| 干净基线 (~1 epoch) | ~30s | ~3s |
| 干净基线 (30 epoch × 3 repeats) | ~15 min | ~2 min |
| 投毒搜索 (9 组网格) | ~30 min | ~5 min |
| 全流程 (基线 + 投毒 + 防御 + 跨数据集) | ~1 h | ~10 min |

CPU 训练完全可行，所有实验均设置 `device: auto`（自动选择 GPU 或 CPU）。也可在配置中将 `training.device` 设为 `cpu` 强制使用 CPU。

---

## 9. 复现实验（完整流程）

复现分为五个阶段。**每个阶段的命令都会输出一个时间戳目录**（如 `results/metr_la_clean_20260427_120000`），后续阶段需要传递前一个目录作为参数。

建议先跑冒烟测试（9.6 节）验证环境和流程，再跑完整实验。

### 9.1 阶段一：训练干净基线

训练一个干净的 LSTM 模型（3 次重复，选择最佳检查点）。这是所有后续攻击实验的基础。

```bash
python scripts/run_clean_baseline.py --config configs/metr_la.yaml
```

**输出目录示例**：`results/metr_la_clean_20260427_120000`

**输出文件**：
| 文件 | 说明 |
| --- | --- |
| `baseline_summary.json` | 最佳基线路径及 MAE |
| `clean_model.pt` | 模型权重 |
| `train_predictions.npz` | 训练集预测（攻击阶段用于节点排序） |
| `test_predictions.npz` | 测试集预测 |
| `stability.json` | 多次重复的 MAE 波动 |
| `clean_metrics.csv` | 各次重复的测试指标 |

**验证**：打开 `baseline_summary.json`，确认 `best_mae` 约在 0.36 左右（正常波动 ±0.02）。

```bash
python -c "import json; d=json.load(open('results/metr_la_clean_XXXXX/baseline_summary.json')); print(f'Best MAE: {d[\"best_mae\"]:.4f}')"
```

记录输出目录路径 `$BASELINE_DIR`，后续步骤需要引用。

### 9.2 阶段二：运行主投毒实验

使用 `spatiotemporal_headroom` v11 配置，进行 3×3=9 组超参数网格搜索（poison_ratio × sigma_multiplier），自动选择最佳候选并复验。

```bash
BASELINE_DIR="results/metr_la_clean_20260427_120000"  # 替换为实际路径

python scripts/run_poison_experiments.py \
  --config configs/metr_la_spatiotemporal_headroom_v11.yaml \
  --baseline-dir "$BASELINE_DIR"
```

**输出目录示例**：`results/metr_la_poison_20260427_130000`

**输出文件**：
| 文件 | 说明 |
| --- | --- |
| `search_summary.json` | 搜索摘要、复验结果和论文标准状态 |
| `attack_results.csv` | 所有 9 组候选的首次评估结果 |
| `recheck_results.csv` | 复验后的聚合结果 |
| `recheck_repeats.csv` | 复验的逐次详情 |
| `best_attack.json` | 论文候选参数与指标（默认） |
| `best_attack_paper.json` | 论文候选 |
| `best_attack_raw.json` | 按局部 ASR 选择的最强候选 |
| `best_poisoned_model_paper.pt` | 论文候选模型权重 |
| `best_attack_bundle_paper.npz` | 论文候选评估数据包 |
| `trigger_case.png` | 触发样例可视化 |
| `best_prediction_case.png` | 预测对比可视化 |

**验证**：确认论文标准通过。

```bash
python -c "
import json
d = json.load(open('results/metr_la_poison_XXXXX/search_summary.json'))
s = d.get('best_paper_summary', d.get('best_summary', {}))
print(f'minimum_contract_pass: {s.get(\"minimum_contract_pass\")}')
print(f'strong_contract_pass:   {s.get(\"strong_contract_pass\")}')
print(f'raw_tail_ASR:           {s.get(\"raw_selected_nodes_tail_horizon_attack_success_rate\", 0):.4f}')
print(f'clean_MAE_delta:        {s.get(\"clean_MAE_delta_ratio\", 0):.4f}')
print(f'direction_match_rate:   {s.get(\"raw_selected_nodes_tail_horizon_shift_direction_match_rate\", 0):.4f}')
print(f'frequency_energy_shift: {s.get(\"frequency_energy_shift\", 0):.4f}')
"
```

**预期结果**（v11 配置，p=0.05, σ=0.14）：

| 指标 | 预期值 | 论文最低标准 | 论文更强标准 |
| --- | ---: | ---: | ---: |
| `raw_tail_ASR` | ~16% | >= 5% | >= 6% |
| `clean_MAE_delta` | ~3.3% | <= 5% | <= 4% |
| `direction_match_rate` | ~94% | >= 60% | >= 65% |
| `frequency_energy_shift` | ~1.6 | <= 3.0 | <= 2.0 |
| `mean_z_score` | ~0.76 | <= 0.80 | <= 0.76 |

### 9.3 阶段三：运行防御评估

对投毒模型进行基础防御检查：z-score 异常筛查、高频能量检查、移动平均平滑。

```bash
POISON_DIR="results/metr_la_poison_20260427_130000"  # 替换为实际路径

python scripts/run_defense_eval.py \
  --config configs/metr_la_opt_loss_rebalance.yaml \
  --poison-dir "$POISON_DIR"
```

**输出目录示例**：`results/defense_eval_YYYYMMDD_HHMMSS`

**验证**：检查 `defense_summary.json`，确认 z-score 筛查和移动平均平滑未消掉攻击效果。

### 9.4 阶段四：跨数据集验证（PEMS-BAY）

将 METR-LA 上复验后的最佳攻击参数迁移到 PEMS-BAY 数据集，验证攻击的可迁移性。

```bash
python scripts/run_cross_dataset.py \
  --config configs/pems_bay_paper_optimization.yaml \
  --best-attack-json results/metr_la_poison_20260427_040007/best_attack_paper.json
```

> **注意**：需要先自行下载 PEMS-BAY 数据（见 8.5 节），或跳过此阶段。

**输出目录示例**：`results/pems_bay_cross_YYYYMMDD_HHMMSS`

**当前补测结果**：`results/pems_bay_cross_20260429_172956` 中局部 ASR 为 0.00%，干净 MAE 变化为 2.90%，未通过跨数据集验证标准。

### 9.5 阶段五：生成论文汇总表

汇总所有阶段的实验结果，生成论文可直接使用的 markdown 和 CSV 表格。

```bash
python scripts/build_thesis_tables.py \
  --metr-baseline-dir "$BASELINE_DIR" \
  --metr-poison-dir "$POISON_DIR" \
  --defense-dir "$DEFENSE_DIR" \
  --cross-dir "$CROSS_DIR"
```

**输出目录示例**：`results/thesis_tables_YYYYMMDD_HHMMSS`

**输出文件**：
| 文件 | 说明 |
| --- | --- |
| `thesis_summary.md` | 可直接阅读的论文实验摘要 |
| `paper_candidate_table.csv` | 论文候选表 |
| `parameter_sensitivity_table.csv` | 参数敏感性表 |
| `defense_summary_table.csv` | 防御摘要表 |
| `cross_candidate_comparison.csv` | 跨数据集候选比较 |

### 9.6 冒烟测试（快速验证）

在跑完整实验前，使用冒烟配置快速验证环境是否正确安装、数据是否就绪。

冒烟配置使用更小的参数空间（epochs=4, repeats=1, 1-2 组网格），在 CPU 上 5 分钟内即可完成。

**冒烟测试 — 干净基线**：

```bash
python scripts/run_clean_baseline.py --config configs/metr_la_smoke.yaml
```

**冒烟测试 — 投毒实验**：

```bash
# 使用上一步输出的目录
python scripts/run_poison_experiments.py \
  --config configs/metr_la_spatiotemporal_headroom_smoke.yaml \
  --baseline-dir results/metr_la_clean_XXXXX
```

冒烟测试通过标准：流程完整运行不报错，`search_summary.json` 正常写入即可。

### 9.7 一键复现（完整脚本）

如果数据已就绪，可以使用以下脚本一键复现全流程：

```bash
#!/bin/bash
# 一键复现脚本 — 保存为 reproduce.sh 并执行
set -euo pipefail

# 环境检查
echo "=== 1/5 训练干净基线 ==="
python scripts/run_clean_baseline.py --config configs/metr_la.yaml
BASELINE_DIR=$(ls -dt results/metr_la_clean_* | head -1)
echo "基线目录: $BASELINE_DIR"

echo "=== 2/5 运行主投毒实验 ==="
python scripts/run_poison_experiments.py \
  --config configs/metr_la_spatiotemporal_headroom_v11.yaml \
  --baseline-dir "$BASELINE_DIR"
POISON_DIR=$(ls -dt results/metr_la_poison_* | head -1)
echo "投毒目录: $POISON_DIR"

echo "=== 3/5 运行防御评估 ==="
python scripts/run_defense_eval.py \
  --config configs/metr_la_opt_loss_rebalance.yaml \
  --poison-dir "$POISON_DIR"
DEFENSE_DIR=$(ls -dt results/defense_eval_* | head -1)
echo "防御目录: $DEFENSE_DIR"

echo "=== 4/5 跨数据集验证 ==="
if [ -f data/pems-bay.h5 ]; then
  python scripts/run_cross_dataset.py \
    --config configs/pems_bay_paper_optimization.yaml \
    --source-poison-dir "$POISON_DIR"
  CROSS_DIR=$(ls -dt results/pems_bay_cross_* | head -1)
  echo "跨数据集目录: $CROSS_DIR"
else
  echo "跳过 (缺少 PEMS-BAY 数据)"
  CROSS_DIR=""
fi

echo "=== 5/5 生成论文汇总表 ==="
if [ -n "$CROSS_DIR" ]; then
  python scripts/build_thesis_tables.py \
    --metr-baseline-dir "$BASELINE_DIR" \
    --metr-poison-dir "$POISON_DIR" \
    --defense-dir "$DEFENSE_DIR" \
    --cross-dir "$CROSS_DIR"
else
  python scripts/build_thesis_tables.py \
    --metr-baseline-dir "$BASELINE_DIR" \
    --metr-poison-dir "$POISON_DIR" \
    --defense-dir "$DEFENSE_DIR"
fi

echo "=== 完成 ==="
echo "基线:      $BASELINE_DIR"
echo "投毒:      $POISON_DIR"
echo "防御:      $DEFENSE_DIR"
echo "跨数据集:  ${CROSS_DIR:-跳过}"
```

### 9.8 Windows cmd 对应命令

```bat
REM 激活环境
D:\ProgramData\Anaconda3\envs\wanzi310\python.exe scripts\run_clean_baseline.py --config configs\metr_la.yaml

REM 设置变量（替换目录名）
set BASELINE_DIR=results\metr_la_clean_20260427_120000
set POISON_DIR=results\metr_la_poison_20260427_130000

REM 主投毒实验
python scripts\run_poison_experiments.py --config configs\metr_la_spatiotemporal_headroom_v11.yaml --baseline-dir %BASELINE_DIR%

REM 防御评估
python scripts\run_defense_eval.py --config configs\metr_la_opt_loss_rebalance.yaml --poison-dir %POISON_DIR%
```

---

## 10. 常见问题与排查

### 10.1 ImportError: No module named 'traffic_poison'

未安装项目包。执行 `pip install -e .`（在项目根目录下）。

### 10.2 FileNotFoundError: data/metr-la.h5

数据文件不在 `data/` 目录。METR-LA 数据已包含在仓库中，确认：
```bash
git lfs pull   # 如果仓库使用 Git LFS
ls data/metr-la.h5 data/adj_mx.pkl
```

### 10.3 CUDA out of memory

减小 batch size 或强制使用 CPU：
```bash
# 方式一：在配置中设置
# 编辑 configs/metr_la_spatiotemporal_headroom_v11.yaml
# training.device: cpu

# 方式二：环境变量
CUDA_VISIBLE_DEVICES="" python scripts/run_poison_experiments.py ...
```

### 10.4 结果与预期不符

同一配置在不同平台和 PyTorch 版本之间可能存在微小数值差异（通常 < 5% 相对偏差）。论文标准阈值留有足够余量以吸收数值波动。如果结果显著偏离：
1. 检查 Python 和 PyTorch 版本是否满足要求
2. 检查 `seed: 42` 是否在配置中设置
3. 检查数据文件是否完整（`md5 data/metr-la.h5`）

### 10.5 macOS / Apple Silicon 特殊说明

PyTorch 在 Apple Silicon 上使用 MPS 后端。当前代码使用 `device: auto`，会自动检测。如果遇到 MPS 相关问题，可以在配置中强制使用 `cpu`。

---

## 11. 输出文件说明

### 11.1 干净基线目录

| 文件 | 含义 |
| --- | --- |
| `baseline_summary.json` | 最佳基线及模型路径 |
| `stability.json` | 多次重复训练的稳定性 |
| `clean_metrics.csv` | 每次重复的测试指标 |
| `clean_model.pt` | 干净模型权重 |
| `train_predictions.npz` | 训练集预测结果 |
| `test_predictions.npz` | 测试集预测结果 |

### 11.2 投毒实验目录

| 文件 | 含义 |
| --- | --- |
| `search_summary.json` | 搜索阶段、复验阶段和论文标准摘要 |
| `attack_results.csv` | 所有候选的一次评估结果 |
| `recheck_results.csv` | 候选复验后的聚合结果 |
| `recheck_repeats.csv` | 复验的逐次结果 |
| `best_attack_paper.json` | 论文候选 |
| `best_attack_raw.json` | 原始局部最强候选 |
| `best_attack.json` | 默认论文候选 |
| `best_poisoned_model_paper.pt` | 论文候选模型 |
| `best_attack_bundle_paper.npz` | 论文候选评估数据包 |
| `trigger_case.png` | 触发样例图 |
| `best_prediction_case.png` | 预测对比图 |

### 11.3 防御目录

| 文件 | 含义 |
| --- | --- |
| `defense_summary.json` | 防御检查摘要 |
| `defense_results.csv` | 防御检查表格 |

### 11.4 论文汇总目录

| 文件 | 含义 |
| --- | --- |
| `thesis_summary.md` | 可直接阅读的论文实验摘要 |
| `paper_candidate_table.csv` | 论文候选表 |
| `parameter_sensitivity_table.csv` | 参数敏感性表 |
| `defense_summary_table.csv` | 防御摘要表 |
| `cross_candidate_comparison.csv` | 跨数据集候选比较 |

---

## 12. 指标解释

| 指标 | 含义 |
| --- | --- |
| `MAE` | 平均绝对误差。正常预测性能的主要指标。 |
| `RMSE` | 均方根误差。对较大误差更敏感。 |
| `MAPE` | 平均绝对百分比误差。 |
| `clean_MAE_delta_ratio` | 带毒模型正常输入 MAE 相对干净基线的变化。 |
| `attack_success_rate` | 旧口径全局 ASR。 |
| `raw_selected_nodes_attack_success_rate` | 原始速度空间、被选节点上的 ASR。 |
| `raw_selected_nodes_tail_horizon_attack_success_rate` | 当前论文主指标：原始速度空间、被选节点、尾部预测步上的 ASR。 |
| `raw_selected_nodes_tail_horizon_shift_direction_match_rate` | 目标区域预测偏移方向与攻击目标一致的比例。 |
| `raw_selected_nodes_tail_horizon_target_shift_attainment` | 实际偏移相对目标偏移的达成程度。 |
| `frequency_energy_shift` | 触发后频域能量变化。 |
| `mean_z_score` | 触发样本相对正常分布的平均 z-score。 |
| `anomaly_rate` | 简单异常阈值下被标记的比例。 |
| `minimum_contract_pass` | 是否满足最低论文标准。 |
| `strong_contract_pass` | 是否满足更强正文标准。 |

---

## 13. 当前结论

1. 在 `METR-LA` 上，当前方法已经在干净性能、局部攻击效果、全局辅助 ASR、方向一致性和基础隐蔽性之间取得可写结果。
2. 主结果满足最低论文标准，但更强正文标准仍未完全满足。
3. `directional_headroom + dual_focus + directional_focus` 是当前最稳定的组合。
4. 简单 z-score、高频能量检查和移动平均平滑不足以有效识别或削弱该触发模式。
5. 使用 427 复验主结果补测 `PEMS-BAY` 时，干净 MAE 变化为 2.90%，但局部 ASR 为 0.00%，当前跨数据集验证未通过。

---

## 14. 建议阅读顺序

如果第一次阅读本仓库，建议按以下顺序：

1. `src/traffic_poison/thesis_contract.py`
2. `configs/metr_la_opt_loss_rebalance.yaml`
3. `scripts/run_poison_experiments.py`
4. `results/metr_la_poison_20260409_163212/search_summary.json`
5. `results/thesis_tables_20260409_165255/thesis_summary.md`
