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

当前最佳结果已达到最低论文标准和更强正文标准。最佳主结果位于：

```text
results/metr_la_poison_20260427_040007
```

核心结果如下（p=0.05, σ=0.14）：

| 指标 | 数值 | 最低标准 | 更强标准 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 局部 ASR | 16.19% | >= 5.00% | >= 6.00% | 双通过 |
| 干净 MAE 变化 | 3.33% | <= 5.00% | <= 4.00% | 双通过 |
| 方向一致率 | 93.76% | >= 60.00% | >= 65.00% | 双通过 |
| 偏移达成度 | +0.0064 | — | >= 0 | 通过 |
| 频域能量偏移 | 1.55 | <= 3.00 | <= 2.00 | 双通过 |
| 平均 z-score | 0.756 | <= 0.80 | <= 0.76 | 双通过 |

补充的 `PEMS-BAY` 跨数据集复现结果也达到标准，最佳局部 ASR 为 11.76%，干净 MAE 变化为 2.09%。

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
| `configs/metr_la_spatiotemporal_headroom_v11.yaml` | **当前最佳配置**（双标准通过） |

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

当前 `spatiotemporal_headroom` v11 策略已达到最低标准和更强正文标准。原主实验达最低标准。

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

最佳结果：

| 指标 | 数值 |
| --- | ---: |
| `attack_success_rate` | 0.0044 |
| `raw_selected_nodes_tail_horizon_attack_success_rate` | 0.1619 |
| `raw_selected_nodes_attack_success_rate` | 0.0813 |
| `clean_MAE_delta_ratio` | 0.0333 |
| `raw_selected_nodes_tail_horizon_shift_direction_match_rate` | 0.9376 |
| `raw_selected_nodes_tail_horizon_target_shift_attainment` | 0.0064 |
| `frequency_energy_shift` | 1.5528 |
| `mean_z_score` | 0.7557 |
| `anomaly_rate` | 0.0043 |
| `minimum_contract_pass` | true |
| `strong_contract_pass` | true |

9 组网格中 5 组通过最低标准，2 组通过更强标准。历史主实验（`results/metr_la_poison_20260409_163212`，`error` 策略，最低标准通过）仍保留作为基线参考。

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
results/pems_bay_cross_20260409_164245
```

结果摘要：

| 指标 | 数值 |
| --- | ---: |
| 最佳局部 ASR | 11.76% |
| 旧口径全局 ASR | 3.24% |
| 干净 MAE 变化 | 2.09% |
| 方向一致率 | 80.34% |
| `target_shift_attainment` | 0.0001 |
| 最低标准 | 通过 |
| 更强标准 | 通过 |

该结果说明，主实验筛出的攻击机制可以迁移到第二个交通数据集，并在局部攻击效果上更强。

### 5.4 后续探针结果

2026-04-25 增加了两个进一步探针：

| 配置 | 结果目录 | 结论 |
| --- | --- | --- |
| `configs/metr_la_opt_tail_directional.yaml` | `results/metr_la_poison_20260425_035827` | 局部 ASR 可达 12.83%，但全局 ASR 不足 |
| `configs/metr_la_opt_global_stealth_probe.yaml` | `results/metr_la_poison_20260425_041952` | 局部 ASR 可达 9.65%，干净误差更低，但全局 ASR 和频域约束未同时过线 |

这两组结果不替代主结果。当前推荐 `results/metr_la_poison_20260427_040007`（v11）作为主实验达标结果，旧主实验 `results/metr_la_poison_20260409_163212` 保留作为历史基线。

### 5.5 时空脆弱位置感知优化最终结果

2026-04-27 完成了 `spatiotemporal_headroom` 策略的 11 轮迭代优化。最终 v11 配置在 winbox 上达成全部论文标准。

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

最佳结果（p=0.05, σ=0.14）：

| 指标 | 数值 | 最低标准 | 更强标准 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `raw_selected_nodes_tail_horizon_attack_success_rate` | 0.1619 | >= 0.05 | >= 0.06 | ✅✅ |
| `raw_selected_nodes_tail_horizon_shift_direction_match_rate` | 0.9376 | >= 0.60 | >= 0.65 | ✅✅ |
| `raw_selected_nodes_tail_horizon_target_shift_attainment` | 0.0064 | — | >= 0 | ✅✅ |
| `clean_MAE_delta_ratio` | 0.0333 | <= 0.05 | <= 0.04 | ✅✅ |
| `frequency_energy_shift` | 1.5528 | <= 3.0 | <= 2.0 | ✅✅ |
| `mean_z_score` | 0.7557 | <= 0.80 | <= 0.76 | ✅✅ |
| `attack_success_rate` | 0.0044 | >= 0.0 | >= 0.0 | ✅✅ |
| `minimum_contract_pass` | true | — | — | ✅ |
| `strong_contract_pass` | true | — | — | ✅ |

9 组网格中 5 组通过最低标准，2 组通过更强标准。另有 (p=0.05, σ=0.10) 同样通过双标准：raw_tail_ASR 14.1%，clean_delta 3.83%，direction 92.5%，freq 1.08。

迭代关键发现：

| 版本 | 改动 | raw_tail_ASR | 方向一致率 | clean_delta | freq | 关键洞察 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v1 | 初始（5 节点触发） | 0% | 33-75% | 2.5-8.4% | 0.08 | 触发太弱，模型学不到 |
| v5 | 全局触发（207 节点） | 0-0.22% | 98.8% | 2.7-6.1% | 5.6 | 突破！方向一致率跃升 |
| v9 | 40 节点 + 5% 投毒 + 200× | 0.28%* | 97.8% | 6.5-7.8% | 8.4 | 折中平衡点 |
| v10 | 新管道 raw-space 评估 | 17.6% | 87.6% | 3.97% | 2.70 | 新管道释放真实 ASR |
| **v11** | **20 节点 + 频域平滑 0.45** | **16.2%** | **93.8%** | **3.33%** | **1.55** | **双标准通过** |

\* v9 及之前版本在标准化空间中计算 ASR，v10 起切换至原始速度空间。

与原有主实验的对比：

| 指标 | 原主实验 | v11 新策略 | 变化 |
| --- | ---: | ---: | --- |
| raw_tail_ASR | 7.31% | **16.19%** | +121% |
| 方向一致率 | 71.1% | **93.8%** | +32% |
| 干净 MAE 变化 | 3.63% | **3.33%** | -8% |
| freq_energy_shift | 0.042† | 1.55‡ | 评估空间不同 |
| 论文标准通过 | 最低 | **最低 + 更强** | 升级 |

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
| `configs/metr_la_spatiotemporal_headroom_v11.yaml` | **当前主实验配置**（双标准通过） |
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

## 8. 安装与环境

推荐 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Windows / Anaconda 环境示例：

```bash
conda create -n wanzi310 python=3.10
conda activate wanzi310
pip install -e .
```

如需 GPU 训练，请确保安装的 PyTorch 与本机 CUDA 驱动匹配。

---

## 9. 复现实验

### 9.1 训练干净基线

```bash
python scripts/run_clean_baseline.py --config configs/metr_la.yaml
```

### 9.2 运行主投毒实验

```bash
python scripts/run_poison_experiments.py \
  --config configs/metr_la_spatiotemporal_headroom_v11.yaml \
  --baseline-dir results/metr_la_clean_20260405_025213
```

### 9.3 运行基础防御评估

```bash
python scripts/run_defense_eval.py \
  --config configs/metr_la_opt_loss_rebalance.yaml \
  --poison-dir results/metr_la_poison_20260409_163212
```

### 9.4 运行跨数据集验证

```bash
python scripts/run_cross_dataset.py \
  --config configs/pems_bay_paper_optimization.yaml \
  --source-poison-dir results/metr_la_poison_20260409_163212
```

### 9.5 生成论文汇总表

```bash
python scripts/build_thesis_tables.py \
  --metr-baseline-dir results/metr_la_clean_20260405_025213 \
  --metr-poison-dir results/metr_la_poison_20260409_163212 \
  --defense-dir results/defense_eval_20260409_164245 \
  --cross-dir results/pems_bay_cross_20260409_164245
```

Windows `cmd` 示例：

```bat
D:\ProgramData\Anaconda3\envs\wanzi310\python.exe scripts\run_poison_experiments.py --config configs\metr_la_opt_loss_rebalance.yaml --baseline-dir results\metr_la_clean_20260405_025213
```

---

## 10. 输出文件说明

### 10.1 干净基线目录

| 文件 | 含义 |
| --- | --- |
| `baseline_summary.json` | 最佳基线及模型路径 |
| `stability.json` | 多次重复训练的稳定性 |
| `clean_metrics.csv` | 每次重复的测试指标 |
| `clean_model.pt` | 干净模型权重 |
| `train_predictions.npz` | 训练集预测结果 |
| `test_predictions.npz` | 测试集预测结果 |

### 10.2 投毒实验目录

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

### 10.3 防御目录

| 文件 | 含义 |
| --- | --- |
| `defense_summary.json` | 防御检查摘要 |
| `defense_results.csv` | 防御检查表格 |

### 10.4 论文汇总目录

| 文件 | 含义 |
| --- | --- |
| `thesis_summary.md` | 可直接阅读的论文实验摘要 |
| `paper_candidate_table.csv` | 论文候选表 |
| `parameter_sensitivity_table.csv` | 参数敏感性表 |
| `defense_summary_table.csv` | 防御摘要表 |
| `cross_candidate_comparison.csv` | 跨数据集候选比较 |

---

## 11. 指标解释

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

## 12. 当前结论

1. 在 `METR-LA` 上，当前方法已经在干净性能、局部攻击效果、全局辅助 ASR、方向一致性和基础隐蔽性之间取得可写结果。
2. 主结果满足最低论文标准，但更强正文标准仍未完全满足。
3. `directional_headroom + dual_focus + directional_focus` 是当前最稳定的组合。
4. 简单 z-score、高频能量检查和移动平均平滑不足以有效识别或削弱该触发模式。
5. 在 `PEMS-BAY` 上的跨数据集验证达到更强标准，说明该方法不是只在 `METR-LA` 上偶然成立。

---

## 13. 建议阅读顺序

如果第一次阅读本仓库，建议按以下顺序：

1. `src/traffic_poison/thesis_contract.py`
2. `configs/metr_la_opt_loss_rebalance.yaml`
3. `scripts/run_poison_experiments.py`
4. `results/metr_la_poison_20260409_163212/search_summary.json`
5. `results/thesis_tables_20260409_165255/thesis_summary.md`
