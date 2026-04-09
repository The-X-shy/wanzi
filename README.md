# WANZI：交通预测回归任务中的后门投毒实验

本仓库研究的是交通速度预测场景中的训练期投毒问题。当前版本已经从早期的多路线并行探索，收紧为一条适合论文正文的主线：

- 主任务：交通状态预测
- 主模型：`LSTM`
- 主数据集：`METR-LA`
- 补充验证数据集：`PEMS-BAY`
- 主攻击家族：`error + hybrid + 3 nodes + trigger_steps=3`
- 当前推荐优化方向：`loss rebalance`

这条主线的目标不是单纯追求最大局部峰值，而是在尽量不破坏正常预测性能的前提下，让被攻击节点在最后几个预测步上稳定地产生目标方向偏移，并满足论文写作时需要的最低结果线。

---

## 1. 当前论文主线

当前仓库默认讨论的是下面这个问题：

> 对交通速度预测模型做训练期投毒后，模型在正常输入上保持基本可用；当输入中出现触发模式时，被攻击节点在最后几个预测步上向目标方向偏移。

当前版本已经不再把重点放在：

- 更换为更大的图模型
- 扩大防御方法集合
- 对更多数据集做全量搜索

当前版本真正保留的重点是：

- 建立稳定的 `LSTM` 预测基线
- 在 `METR-LA` 上完成主实验
- 在 `PEMS-BAY` 上做补充复现
- 统一论文结果标准
- 保存“原始最强结果”和“最适合正文的结果”

---

## 2. 与开题报告和中期报告的关系

### 2.1 一致的部分

当前实验与开题报告、中期报告在这些方面是一致的：

- 使用公开交通数据集
- 先建立干净的预测基线
- 再做训练期后门投毒
- 同时考察正常预测性能、攻击效果、隐蔽性和基础防御
- 识别脆弱节点和脆弱时间位置
- 采用“主数据集 + 补充验证数据集”的结构

### 2.2 收紧后的部分

当前版本相对开题报告做了两处明确收紧：

- 只保留了交通状态预测，不再同时展开交通状态估计
- 攻击实现已经从最早设想中的“标签完全不变”演化为“目标偏移式回归后门”

这两处收紧与中期报告是一致的。中期报告强调的是围绕交通预测主任务，把脆弱节点、脆弱时间位置、回归任务的攻击评价和基础防御验证做完整，而不是把研究范围继续铺大。

### 2.3 中期报告结构与代码的对应关系

| 中期报告中的工作块 | 当前代码中的对应位置 | 说明 |
| --- | --- | --- |
| 数据集选择与适配 | `configs/`、`scripts/prepare_dataset.py`、`src/traffic_poison/data.py` | 数据路径、切分、标准化、窗口化 |
| 干净基线训练 | `scripts/run_clean_baseline.py`、`src/traffic_poison/model.py`、`src/traffic_poison/trainer.py` | `LSTM` 基线训练与保存 |
| 脆弱节点与时间定位 | `src/traffic_poison/poisoning.py` | 节点选择、时间位置选择、窗口筛选 |
| 投毒样本与触发器构造 | `src/traffic_poison/poisoning.py`、`scripts/run_poison_experiments.py` | 投毒样本筛选、触发器生成、目标偏移 |
| 回归攻击效果评价 | `src/traffic_poison/poisoning.py`、`src/traffic_poison/metrics.py` | 局部攻击效果与全局辅助指标 |
| 隐蔽性分析 | `src/traffic_poison/poisoning.py`、`src/traffic_poison/defenses.py` | 频域、异常率、`z-score` 等 |
| 基础防御验证 | `scripts/run_defense_eval.py`、`src/traffic_poison/defenses.py` | 简单筛查与平滑实验 |
| 跨数据集补充验证 | `scripts/run_cross_dataset.py` | 将主实验筛出的候选复现到第二数据集 |
| 论文结果收口 | `src/traffic_poison/thesis_contract.py`、`scripts/build_thesis_tables.py` | 统一论文标准、排序和汇总 |

---

## 3. 论文结果标准

当前仓库不再把早期 README 中的旧值当作唯一门槛，而是采用统一的论文结果契约，由 `src/traffic_poison/thesis_contract.py` 管理。

### 3.1 最低可写线

在 `METR-LA` 上，当前默认的最低论文线是：

| 指标 | 最低线 |
| --- | --- |
| `clean_MAE_delta_ratio` | `<= 5%` |
| `raw_selected_nodes_tail_horizon_attack_success_rate` | `>= 5.0%` |
| `attack_success_rate` | `>= 1.50%` |
| `raw_selected_nodes_tail_horizon_shift_direction_match_rate` | `>= 0.60` |

### 3.2 更强的正文线

当前用于判断“是否足够强、足够稳”的更高标准是：

| 指标 | 更强正文线 |
| --- | --- |
| `clean_MAE_delta_ratio` | `<= 4%` |
| `raw_selected_nodes_tail_horizon_attack_success_rate` | `>= 6.0%` |
| `raw_selected_nodes_tail_horizon_target_shift_attainment` | `>= 0` |

### 3.3 为什么同时保存两个冠军

当前版本会同时保存两类冠军：

- `best_attack_raw.json`
  - 只看局部主指标谁最高
- `best_attack_paper.json`
  - 先看是否满足论文门槛，再按论文排序规则选出最适合正文的结果

兼容旧流程时，`best_attack.json` 默认指向 `best_attack_paper.json`。

---

## 4. 数据集

### 4.1 `METR-LA`

- 用途：主实验
- 内容：洛杉矶交通速度时间序列
- 主要文件：
  - `data/metr-la.h5`
  - `data/adj_mx.pkl`

### 4.2 `PEMS-BAY`

- 用途：补充验证
- 内容：湾区交通速度时间序列
- 主要文件：
  - `data/pems-bay.csv`
  - `data/adj_mx_bay.pkl`

### 4.3 为什么保留这两个数据集

这与中期报告的收口方式一致：

- `METR-LA` 用来支撑主结论
- `PEMS-BAY` 用来证明当前方法不是只在一个数据集上偶然成立

---

## 5. 当前推荐的实验路线

当前真正推荐使用的配置只有下面三份：

- 干净基线：`configs/metr_la.yaml`
- 主实验：`configs/metr_la_opt_loss_rebalance.yaml`
- 跨数据集补充验证：`configs/pems_bay_paper_optimization.yaml`

其余 `main / local_error / directional / staged / decoupled / followup` 等配置保留为历史探索记录，不再是当前默认主线。

### 5.1 当前主实验的核心设定

当前推荐的主实验来自 `configs/metr_la_opt_loss_rebalance.yaml`，核心设定如下：

- 数据集：`METR-LA`
- 模型：`LSTM`
- 输入长度：`12`
- 预测步长：`12`
- 投毒比例：`0.018 / 0.02`
- 触发节点数：`3`
- 触发时间长度：`3`
- 触发窗口家族：`hybrid`
- 样本筛选：`directional_headroom`
- 标签塑形：`dual_focus`
- 训练重点：`directional_focus`

### 5.2 当前保留的三个精调方向

本轮审查后保留了三条仍然不偏离论文主线的优化方向：

| 方向 | 目的 | 最终表现 |
| --- | --- | --- |
| `spread_recovery` | 稍微放宽影响范围，补强旧口径 `ASR` | 过最低线，但不是最优 |
| `selection_balance` | 让带毒样本不过于极端，提高方向干净度 | 过最低线，方向达成更好 |
| `loss_rebalance` | 只调整训练重点和目标塑形比例，寻找更强的正文结果 | 当前最佳 |

当前默认采用的是 `loss_rebalance`。

---

## 6. 代码结构

### 6.1 顶层目录

```text
wanzi/
├── configs/
├── data/
├── results/
├── scripts/
├── src/traffic_poison/
└── tests/
```

### 6.2 `configs/`

配置文件目录，用来定义实验搜索空间和默认参数。

| 文件 | 作用 |
| --- | --- |
| `configs/metr_la.yaml` | 干净基线配置 |
| `configs/metr_la_opt_spread_recovery.yaml` | 方向 A：spread recovery |
| `configs/metr_la_opt_selection_balance.yaml` | 方向 B：selection balance |
| `configs/metr_la_opt_loss_rebalance.yaml` | 方向 C：loss rebalance，当前推荐主线 |
| `configs/*_smoke.yaml` | 每个方向的快速检查配置 |
| `configs/pems_bay_paper_optimization.yaml` | 将主实验最佳候选复现到 `PEMS-BAY` |

### 6.3 `scripts/`

脚本层对应的是“实验流程”。

| 文件 | 作用 |
| --- | --- |
| `scripts/run_clean_baseline.py` | 训练干净 `LSTM` 基线 |
| `scripts/run_poison_experiments.py` | 主实验入口，负责搜索、复验、保存双冠军 |
| `scripts/run_defense_eval.py` | 对论文冠军做基础防御评估 |
| `scripts/run_cross_dataset.py` | 将论文冠军复现到 `PEMS-BAY` |
| `scripts/build_thesis_tables.py` | 汇总主实验、防御和跨数据集结果，生成论文摘要 |
| `scripts/prepare_dataset.py` | 数据准备辅助脚本 |

### 6.4 `src/traffic_poison/`

源码层对应的是“方法实现”。

| 文件 | 作用 |
| --- | --- |
| `config.py` | 读取 YAML，补默认值 |
| `data.py` | 数据读取、标准化、窗口切分、数据加载 |
| `model.py` | `LSTMForecaster` 模型定义 |
| `trainer.py` | 训练循环、早停、带权损失 |
| `metrics.py` | 常规回归指标，如 `MAE / RMSE / MAPE` |
| `poisoning.py` | 投毒核心实现：节点/时间选择、触发器、样本筛选、目标塑形、攻击与隐蔽性指标 |
| `defenses.py` | 基础防御方法，如 `z-score` 检查与移动平均平滑 |
| `experiment.py` | 实验流程的统一封装 |
| `reporting.py` | 保存图表、表格和预测示例 |
| `thesis_contract.py` | 统一论文门槛、排序规则、双冠军选择逻辑 |
| `utils.py` | 结果目录、随机种子、设备等通用工具 |

### 6.5 `tests/`

测试目录主要覆盖：

- 论文结果契约是否被正确执行
- 双冠军选择是否稳定、可重复
- 没有候选过论文线时，是否会保存“最接近过线”的结果
- 不同候选是否会因为关键参数被错误合并

---

## 7. 指标说明

当前仓库里的指标分成五类：正常预测性能、攻击效果、方向一致性、隐蔽性、论文标准。

### 7.1 正常预测性能

| 指标 | 含义 |
| --- | --- |
| `MAE` | 平均绝对误差。越小越好，是当前最重要的干净性能指标。 |
| `RMSE` | 均方根误差。越小越好，对大误差更敏感。 |
| `MAPE` | 平均相对误差。越小越好。 |
| `spread` | 多次重复训练后结果的波动程度。越小越稳。 |

### 7.2 正常性能保持

| 指标 | 含义 |
| --- | --- |
| `clean_MAE_delta_ratio` | 带毒模型在正常输入上的 `MAE` 相比干净基线恶化了多少。越小越好。 |

这个指标反映的是：攻击成功是不是建立在“把正常模型弄坏很多”的代价上。

### 7.3 攻击效果

| 指标 | 含义 |
| --- | --- |
| `attack_success_rate` | 旧口径全局攻击成功率。它看的是更宽范围上的平均偏移效果。 |
| `raw_selected_nodes_tail_horizon_attack_success_rate` | 当前论文主指标。只看被攻击节点、原始速度空间、最后几个预测步。 |

这两个指标的区别是：

- `attack_success_rate` 更宽、更旧，适合做辅助对照
- `raw_selected_nodes_tail_horizon_attack_success_rate` 更贴近现在真正想证明的攻击目标，所以它是当前论文主指标

### 7.4 方向一致性

| 指标 | 含义 |
| --- | --- |
| `raw_selected_nodes_tail_horizon_shift_direction_match_rate` | 在目标区域里，有多少比例的预测偏移方向是对的。越高越好。 |
| `raw_selected_nodes_tail_horizon_target_shift_attainment` | 实际偏移距离与目标偏移距离的接近程度。`>= 0` 更理想，长期明显为负说明方向虽然有时命中，但整体还不够贴近目标。 |

这两项指标解决的是一个关键问题：不是只看“偏了没有”，还要看“是不是按希望的方向偏”。

### 7.5 隐蔽性

| 指标 | 含义 |
| --- | --- |
| `frequency_energy_shift` | 触发后高频能量变化有多大。越小越不显眼。 |
| `mean_z_score` | 在简单异常标准下偏离常态的程度。越小越不显眼。 |
| `anomaly_rate` | 被简单异常检测标记的比例。越小越好。 |

### 7.6 论文标准相关

| 字段 | 含义 |
| --- | --- |
| `minimum_bar_met` | 是否达到最低论文线 |
| `strong_bar_met` | 是否达到更强的正文线 |
| `paper_and_raw_same_candidate` | 原始最强结果和论文最优结果是不是同一组 |

---

## 8. 当前结果

### 8.1 干净基线

冻结的主基线目录：

- `results/metr_la_clean_20260405_025213`

当前主线使用的基线结果：

| 指标 | 数值 |
| --- | --- |
| 最佳 `MAE` | `0.3651` |
| 波动 `spread` | `2.94%` |

### 8.2 三个优化方向的比较

| 方向 | 论文冠军 local ASR | legacy ASR | clean MAE drift | direction match | target attainment | 是否过最低线 |
| --- | --- | --- | --- | --- | --- | --- |
| `spread_recovery` | `6.85%` | `1.65%` | `3.97%` | `73.36%` | `0.0017` | 是 |
| `selection_balance` | `6.99%` | `1.71%` | `3.85%` | `68.59%` | `0.0195` | 是 |
| `loss_rebalance` | `7.31%` | `1.74%` | `3.63%` | `71.11%` | `-0.0006` | 是 |

当前推荐保留 `loss_rebalance` 作为主结果方向。

### 8.3 当前主实验最佳结果

主实验目录：

- `results/metr_la_poison_20260409_163212`

当前结果中：

- `best_attack_raw.json` 和 `best_attack_paper.json` 是同一组候选
- 当前论文冠军来自 `directional_headroom + dual_focus + directional_focus`

关键结果如下：

| 类型 | 数值 |
| --- | --- |
| local raw-space ASR | `7.31%` |
| legacy ASR | `1.74%` |
| clean MAE drift | `3.63%` |
| direction match | `71.11%` |
| target attainment | `-0.0006` |
| minimum paper bar | `met` |
| strong paper bar | `not met` |

解释：

- 最低论文线已经通过
- 更强正文线还没有完全通过，主要因为 `target_shift_attainment` 还没有稳定转正

### 8.4 防御验证

防御结果目录：

- `results/defense_eval_20260409_164245`

当前结论是：

- `z-score` 和简单频域检查几乎无法把带毒样本与正常样本明显区分开
- 移动平均平滑没有明显压制攻击效果
- 旧口径 `ASR` 在简单平滑后从 `1.42%` 变为 `1.45%`

这说明当前攻击提升不是靠非常突兀的异常波形换来的。

### 8.5 跨数据集补充验证

跨数据集结果目录：

- `results/pems_bay_cross_20260409_164245`

最佳补充验证结果：

| 指标 | 数值 |
| --- | --- |
| local raw-space ASR | `11.76%` |
| legacy ASR | `3.24%` |
| clean MAE drift | `2.09%` |
| direction match | `80.34%` |
| target attainment | `0.0001` |
| minimum paper bar | `met` |
| strong paper bar | `met` |

这说明当前方法不仅在 `METR-LA` 上成立，在 `PEMS-BAY` 上也能复现出更强的局部效果。

### 8.6 论文汇总目录

当前主线的论文汇总目录：

- `results/thesis_tables_20260409_165255`

其中最重要的文件是：

- `thesis_summary.md`
- `thesis_summary.json`

---

## 9. 结果目录怎么读

### 9.1 干净基线目录

示例：`results/metr_la_clean_20260405_025213`

主要看：

- `baseline_summary.json`
- `stability.json`
- `metrics.csv`

### 9.2 主实验目录

示例：`results/metr_la_poison_20260409_163212`

主要看：

- `search_summary.json`
  - 整体搜索摘要
- `best_attack_raw.json`
  - 原始最强结果
- `best_attack_paper.json`
  - 最适合论文正文的结果
- `best_attack.json`
  - 默认等同于 `best_attack_paper.json`
- `recheck_results.csv`
  - 多次复验结果
- `best_poisoned_model_paper.pt`
  - 论文冠军模型
- `best_poisoned_model_raw.pt`
  - 原始冠军模型

### 9.3 防御目录

示例：`results/defense_eval_20260409_164245`

主要看：

- `defense_summary.json`

### 9.4 跨数据集目录

示例：`results/pems_bay_cross_20260409_164245`

主要看：

- `cross_dataset_summary.json`

### 9.5 审查与方向比较目录

示例：`results/code_review_20260409_165341`

主要看：

- `review_summary.md`
- `direction_comparison.csv`

---

## 10. 如何复现实验

建议使用当前的 `wanzi310` 环境。

### 10.1 干净基线

```bash
conda run -n wanzi310 python scripts/run_clean_baseline.py --config configs/metr_la.yaml
```

### 10.2 主实验

```bash
conda run -n wanzi310 python scripts/run_poison_experiments.py --config configs/metr_la_opt_loss_rebalance.yaml
```

### 10.3 基础防御评估

```bash
conda run -n wanzi310 python scripts/run_defense_eval.py --config configs/metr_la_opt_loss_rebalance.yaml --poison-dir results/metr_la_poison_20260409_163212
```

### 10.4 跨数据集补充验证

```bash
conda run -n wanzi310 python scripts/run_cross_dataset.py --config configs/pems_bay_paper_optimization.yaml --source-poison-dir results/metr_la_poison_20260409_163212
```

### 10.5 生成论文汇总

```bash
conda run -n wanzi310 python scripts/build_thesis_tables.py ^
  --metr-baseline-dir results/metr_la_clean_20260405_025213 ^
  --metr-poison-dir results/metr_la_poison_20260409_163212 ^
  --defense-dir results/defense_eval_20260409_164245 ^
  --cross-dir results/pems_bay_cross_20260409_164245
```

---

## 11. 当前仓库状态

截至当前版本，仓库的结论可以概括为：

- 论文主线已经固定
- 双冠军保存逻辑已经建立
- 三个可行优化方向已经完整跑完
- `loss_rebalance` 是当前最优方向
- `METR-LA` 已经过最低论文线
- `PEMS-BAY` 补充验证成立
- 更强正文线仍未完全达成，因此还有继续打磨空间，但不需要再大幅改动研究主线

如果你是第一次读这个仓库，最推荐的入口顺序是：

1. `configs/metr_la_opt_loss_rebalance.yaml`
2. `src/traffic_poison/thesis_contract.py`
3. `scripts/run_poison_experiments.py`
4. `results/metr_la_poison_20260409_163212/search_summary.json`
5. `results/thesis_tables_20260409_165255/thesis_summary.md`
