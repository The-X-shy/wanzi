# WANZI: 交通预测回归任务中的后门投毒实验

本仓库研究的是交通速度预测场景中的训练期投毒问题。当前版本已经从“多条探索线并行”收紧为一条适合论文正文的主线：

- 主任务：交通状态预测，不再同时展开交通状态估计
- 主模型：`LSTM`
- 主数据集：`METR-LA`
- 补充验证数据集：`PEMS-BAY`
- 主攻击家族：`error + hybrid`
- 当前重点方法：
  - `directional_headroom`：优先挑选本来就更容易被向下拉的样本
  - `dual_focus`：整体预测轻微下移，同时让最后 `3` 个预测步更明显地下移
  - `directional_focus`：训练时额外强调被攻击节点和最后 `3` 个预测步

当前仓库的目标不是继续追更大的模型，也不是把防御部分做成独立课题，而是围绕下面这件事给出完整证据：

> 在尽量不破坏正常预测性能的前提下，让被攻击节点在最后 `3` 个预测步上稳定地产生目标方向偏移。

---

## 1. 与开题报告和中期报告的关系

### 1.1 当前主线与报告的一致部分

当前实验与本地保存的“开题报告”和“中期报告”在下面几件事上是一致的：

- 使用公开交通数据集开展实验
- 先建立 `LSTM` 干净基线
- 再实施训练期后门投毒
- 评估正常预测性能、攻击效果、隐蔽性和基础防御
- 采用一个主数据集和一个补充数据集的结构
- 重点分析脆弱节点、脆弱时间位置和触发器隐蔽性

### 1.2 当前主线相对开题报告的收缩

开题报告里的表述范围更大，当前实验已经按中期报告进行了收缩：

- 开题报告同时写了“交通状态估计与预测”，当前实际只做“交通状态预测”
- 开题报告最初写过“带触发器样本注入训练集，保持标签不变”，当前实现已经演化为“目标偏移式回归后门”

这里要特别说明：

- 当前实现不是严格意义上的“标签完全不变式投毒”
- 当前实现是在触发样本上对训练目标施加受控下移，使模型在触发出现时向攻击者希望的方向偏移

这更符合中期报告里“让触发输入的输出向目标轨迹偏移”的描述，也更符合当前回归任务的实验设计。

### 1.3 中期报告中的结构与代码对应关系

| 中期报告中的工作块 | 当前仓库中的对应位置 | 说明 |
| --- | --- | --- |
| 数据集筛选与适配 | `configs/`、`scripts/prepare_dataset.py`、`src/traffic_poison/data.py` | 负责数据路径、读入方式、切分比例、标准化和滑动窗口 |
| 数据预处理与样本生成 | `src/traffic_poison/data.py` | 负责缺失值处理、标准化、窗口切片、`DataLoader` 构造 |
| LSTM 基准模型 | `src/traffic_poison/model.py`、`src/traffic_poison/trainer.py`、`scripts/run_clean_baseline.py` | 负责模型定义、训练、验证、基线结果输出 |
| 脆弱节点/时间识别 | `src/traffic_poison/poisoning.py` | 负责节点排序、时间位置排序、候选区域筛选 |
| 触发器设计与投毒实现 | `src/traffic_poison/poisoning.py`、`scripts/run_poison_experiments.py` | 负责触发器生成、训练集投毒、目标偏移、搜索与复验 |
| 攻击效果评估 | `src/traffic_poison/poisoning.py`、`src/traffic_poison/metrics.py` | 负责主指标、辅助指标、局部视角和原始空间视角 |
| 隐蔽性分析 | `src/traffic_poison/poisoning.py`、`src/traffic_poison/defenses.py` | 负责频域、异常率、`z-score` 等检查 |
| 基础防御验证 | `scripts/run_defense_eval.py`、`src/traffic_poison/defenses.py` | 负责简单平滑、异常筛查、频率筛查 |
| 跨数据集补充验证 | `scripts/run_cross_dataset.py` | 负责将主数据集筛出的候选复现到第二数据集 |
| 论文结果收口 | `src/traffic_poison/thesis_contract.py`、`scripts/build_thesis_tables.py` | 负责统一结果标准、排序规则、论文汇总表 |

---

## 2. 当前推荐实验主线

当前仓库保留了多份历史配置，但真正推荐用于论文主线复现的只有下面三份：

- 干净基线：`configs/metr_la.yaml`
- 主实验：`configs/metr_la_paper_loss_focus.yaml`
- 补充验证：`configs/pems_bay_paper_loss_focus.yaml`

其余 `main / local_error / directional / decoupled / staged` 配置保留为历史探索记录，不再是默认主线。

### 2.1 当前主实验的基本设定

当前主线在 [configs/metr_la_paper_loss_focus.yaml](configs/metr_la_paper_loss_focus.yaml) 中固定为：

- 数据集：`METR-LA`
- 模型：`LSTM`
- 输入长度：`12`
- 预测步长：`12`
- 投毒比例：`0.018 / 0.02`
- 触发节点数：`3`
- 触发时间长度：`3`
- 目标偏移比例：`0.08`
- 触发窗口族：`hybrid`
- 样本选择方式：`local_error_ratio` 与 `directional_headroom`
- 标签塑形方式：`flat` 与 `dual_focus`
- 训练重点方式：`uniform` 与 `directional_focus`

### 2.2 当前主实验的逻辑顺序

1. 跑出稳定的干净 `LSTM` 基线
2. 在训练集上选择要投毒的样本、节点和时间位置
3. 构造平滑触发器，并对触发样本的训练目标施加目标偏移
4. 重新训练带毒模型
5. 在测试集上分别看：
   - 正常输入下模型是否基本保持正常
   - 触发输入下模型是否在目标区域明显偏移
   - 触发器是否过于显眼
6. 对候选结果做多次复验
7. 将主数据集上表现足够好的候选拿到 `PEMS-BAY` 做补充验证

---

## 3. 数据集

### 3.1 `METR-LA`

- 用途：主实验
- 内容：洛杉矶路网交通速度时间序列
- 文件：
  - `data/metr-la.h5`
  - `data/adj_mx.pkl`

### 3.2 `PEMS-BAY`

- 用途：补充验证
- 内容：湾区路网交通速度时间序列
- 文件：
  - `data/pems-bay.csv`
  - `data/adj_mx_bay.pkl`

### 3.3 为什么是这两个数据集

这和中期报告的收口逻辑一致：

- `METR-LA` 负责主实验，保证主结论集中
- `PEMS-BAY` 负责补充验证，证明当前方法不是只在一个数据集上偶然成立

---

## 4. 代码结构

### 4.1 顶层目录

```text
wanzi/
├─ configs/
├─ data/
├─ results/
├─ scripts/
├─ src/traffic_poison/
└─ tests/
```

### 4.2 `configs/`

配置文件目录，决定实验跑什么。

- `metr_la.yaml`
  - 干净基线配置
- `metr_la_paper_loss_focus.yaml`
  - 当前推荐主实验配置
- `metr_la_paper_loss_focus_smoke.yaml`
  - 当前主实验的快速检查版本
- `pems_bay_paper_loss_focus.yaml`
  - 当前补充验证配置
- 其余 `metr_la_paper_main.yaml`、`metr_la_paper_local_error.yaml`、`metr_la_paper_directional.yaml` 等
  - 历史探索配置，保留作对照

### 4.3 `scripts/`

脚本层对应的是“实验流程”。

- `run_clean_baseline.py`
  - 训练干净 `LSTM` 基线
  - 产出基线性能、稳定性和最佳模型
- `run_poison_experiments.py`
  - 主实验入口
  - 负责候选组合搜索、带毒模型训练、结果排序和复验
- `run_defense_eval.py`
  - 对最佳带毒模型做基础防御检查
- `run_cross_dataset.py`
  - 把主数据集筛出的候选复现到第二数据集
- `build_thesis_tables.py`
  - 汇总主结果、防御结果和跨数据集结果，生成论文用表
- `prepare_dataset.py`
  - 数据准备辅助脚本
- `generate_synthetic_data.py`
  - 生成合成数据的辅助脚本，不属于论文主线

### 4.4 `src/traffic_poison/`

源码层对应的是“方法实现”。

- `config.py`
  - 读取 YAML 配置，补齐默认值
- `data.py`
  - 读数据、补缺失、标准化、滑动窗口切片、构造 `Dataset / DataLoader`
- `model.py`
  - `LSTMForecaster` 模型定义
- `trainer.py`
  - 训练循环、验证、早停、带权损失
- `metrics.py`
  - 常规回归指标：`MAE / MAPE / RMSE`
- `poisoning.py`
  - 攻击核心文件
  - 负责脆弱节点/时间排序、触发器生成、训练集投毒、攻击指标和隐蔽性指标
- `defenses.py`
  - 基础防御方法，如 `z-score` 异常筛查和移动平均平滑
- `experiment.py`
  - 把数据、模型、训练和评估组织成统一实验接口
- `reporting.py`
  - 保存表格、画训练曲线、画预测例图
- `thesis_contract.py`
  - 定义论文结果标准、最低线、排序规则和跨数据集复验门槛
- `utils.py`
  - 随机种子、设备、结果目录、JSON 保存等通用工具

### 4.5 `tests/`

测试目录目前覆盖的是论文主线相关部分。

- `test_poisoning_directional.py`
  - 检查方向性样本筛选、标签塑形和训练重点逻辑
- `test_thesis_contract.py`
  - 检查论文结果标准和候选排序是否一致
- `test_weighted_dataset.py`
  - 检查带权训练数据是否正确传入训练器

---

## 5. 结果文件怎么读

### 5.1 干净基线阶段

干净基线目录通常形如：

- `results/metr_la_clean_YYYYMMDD_HHMMSS`

关键文件包括：

- `clean_metrics.csv`
  - 每次重复训练的基线结果
- `stability.json`
  - 三次或多次重复的波动情况
- `baseline_summary.json`
  - 适合直接引用的基线摘要
- `clean_model.pt`
  - 最优干净模型

### 5.2 主实验阶段

主实验目录通常形如：

- `results/metr_la_poison_YYYYMMDD_HHMMSS`

关键文件包括：

- `attack_results.csv`
  - 每个候选组合的完整结果
- `recheck_results.csv`
  - Top 候选的多次复验结果
- `best_attack.json`
  - 当前选出来的最佳结果
- `search_summary.json`
  - 当前主实验的整体摘要
- `best_poisoned_model.pt`
  - 最优带毒模型

### 5.3 防御评估阶段

- `results/defense_eval_YYYYMMDD_HHMMSS`

关键文件：

- `defense_results.csv`
- `defense_summary.json`

### 5.4 跨数据集补充验证阶段

- `results/pems_bay_cross_YYYYMMDD_HHMMSS`

关键文件：

- `cross_candidate_summary.csv`
- `cross_family_summary.csv`
- `cross_dataset_summary.json`

### 5.5 论文汇总阶段

- `results/thesis_tables_YYYYMMDD_HHMMSS`

关键文件：

- `paper_candidate_table.csv`
- `defense_summary_table.csv`
- `selection_strategy_comparison.csv`
- `window_mode_comparison.csv`
- `thesis_summary.md`
- `thesis_summary.json`

---

## 6. 指标说明

当前仓库采用“常规性能指标 + 攻击指标 + 隐蔽性指标 + 论文标准”的结构。

### 6.1 常规性能指标

这些指标回答“模型正常预测得怎么样”。

#### `MAE`

平均绝对误差。

值越小，说明预测值和真实值平均偏差越小。

#### `MAPE`

平均绝对百分比误差。

值越小，说明相对误差越小。

#### `RMSE`

均方根误差。

值越小，说明较大误差点更少。

### 6.2 正常性能保持指标

这些指标回答“带毒模型在正常输入下被破坏了多少”。

#### `clean_MAE`

带毒模型在干净测试集上的 `MAE`。

#### `clean_MAE_delta`

带毒模型在干净测试集上的 `MAE` 相比干净基线多了多少。

#### `clean_MAE_delta_ratio`

`(clean_MAE - baseline_MAE) / baseline_MAE`

这是论文里最重要的约束指标之一。

值越小越好，说明攻击没有明显破坏正常预测能力。

### 6.3 攻击成功指标

这些指标回答“触发器到底有没有让预测朝目标方向偏移”。

#### `attack_success_rate`

这是仓库最早就有的旧口径指标。

做法是：

1. 先把某个区域内的真实值求平均
2. 再把目标值定义为 `真实均值 × (1 - target_shift_ratio)`
3. 如果触发后的预测均值落在目标区间附近，就记为一次成功

这个指标还保留着，但现在不再单独作为正文主结论。

#### `raw_global_attack_success_rate`

在原始速度空间、全节点、全预测步上的攻击成功率。

#### `raw_selected_nodes_attack_success_rate`

只看被选中的攻击节点，不限制预测步。

#### `raw_tail_horizon_attack_success_rate`

只看最后若干预测步，不限制节点。

#### `raw_selected_nodes_tail_horizon_attack_success_rate`

当前论文主指标。

只看：

- 被攻击节点
- 最后 `3` 个预测步
- 原始速度空间

它直接对应现在论文真正关心的区域，所以是当前仓库的主指标。

### 6.4 偏移强度与方向指标

这些指标回答“模型是不是朝着预期方向偏了，而且偏得有多充分”。

#### `mean_prediction_shift_ratio`

`(clean_prediction_mean - triggered_prediction_mean) / |clean_prediction_mean|`

如果是正值，表示触发器让预测往下移。

值越大，表示平均下移更明显。

#### `target_shift_attainment`

`realized_shift_ratio / target_shift_ratio`

直观理解：

- `1`：达到了设定的目标偏移强度
- `0`：几乎没有形成目标偏移
- 负值：方向错了

#### `shift_direction_match_rate`

触发后确实朝目标方向移动的样本比例。

值越大，说明方向更稳定。

### 6.5 隐蔽性指标

这些指标回答“触发器是不是太显眼了”。

#### `time_domain_amplitude`

带毒输入和干净输入在时域上的平均扰动幅度。

越小越隐蔽。

#### `frequency_energy_shift`

带毒输入和干净输入在频域能量上的平均差异。

越小越说明触发器没有引入明显的频谱异常。

#### `anomaly_rate`

在简单异常检测阈值下，被判成异常的比例。

越小越好。

#### `mean_z_score`

相对于干净样本统计量的平均 `z-score`。

越小越说明带毒样本整体上不突出。

### 6.6 论文结果标准

这些标准在 [src/traffic_poison/thesis_contract.py](src/traffic_poison/thesis_contract.py) 中统一定义。

当前最低论文线大致是：

- `baseline_best_mae <= 0.37`
- `baseline_spread <= 0.05`
- `clean_MAE_delta_ratio <= 0.05`
- `main_local_asr >= 0.05`
- `legacy_asr >= 0.015`
- `shift_direction_match_rate >= 0.60`
- `frequency_energy_shift <= 0.05`
- `mean_z_score <= 0.80`

强正文线比这更严格。

---

## 7. 当前结果

当前推荐引用的结果来自：

- 主实验目录：`results/metr_la_poison_20260408_163958`
- 防御目录：`results/defense_eval_20260408_170304`
- 跨数据集目录：`results/pems_bay_cross_20260408_165342`
- 汇总目录：`results/thesis_tables_20260408_170837`

### 7.1 `METR-LA` 干净基线

- 最稳基线目录：`results/metr_la_clean_20260405_025213`
- 最优 `MAE`：`0.3651`
- 三次波动：`2.94%`

### 7.2 `METR-LA` 主实验

当前有两组最值得引用的结果：

| 类型 | 说明 | 局部主指标 | 干净性能变化 | 旧口径 ASR | 局部方向达成 |
| --- | --- | --- | --- | --- | --- |
| 最高结果 | 局部效果最强，但不适合作为正文主结果 | `7.10%` | `3.70%` | `1.48%` | `0.0035` |
| 正文推荐结果 | 更均衡，适合论文正文引用 | `6.14%` | `3.96%` | `1.70%` | `0.0042` |

说明：

- “最高结果”已经超过之前的 `5.61%` 旧主线
- “正文推荐结果”更适合写论文，因为它同时保住了正常性能、旧口径对照值和方向一致性

### 7.3 `PEMS-BAY` 补充验证

- 最佳局部主指标：`9.50%`
- 干净性能变化：`2.06%`

这说明当前方法不是只在 `METR-LA` 上偶然成立。

### 7.4 基础防御验证

当前结论：

- `z-score` 和简单频率检查区分能力有限
- 简单平滑对局部主指标影响很小

这意味着当前提升不是靠特别突兀的异常波形换来的。

---

## 8. 推荐运行顺序

### 8.1 跑干净基线

```bash
python scripts/run_clean_baseline.py --config configs/metr_la.yaml
```

### 8.2 跑主实验

```bash
python scripts/run_poison_experiments.py --config configs/metr_la_paper_loss_focus.yaml --baseline-dir <clean_output>
```

### 8.3 跑基础防御验证

```bash
python scripts/run_defense_eval.py --config configs/metr_la_paper_loss_focus.yaml --poison-dir <poison_output>
```

### 8.4 跑补充数据集验证

```bash
python scripts/run_cross_dataset.py --config configs/pems_bay_paper_loss_focus.yaml --source-poison-dir <poison_output>
```

### 8.5 生成论文汇总表

```bash
python scripts/build_thesis_tables.py \
  --metr-baseline-dir <clean_output> \
  --metr-poison-dir <poison_output> \
  --defense-dir <defense_output> \
  --cross-dir <cross_output>
```

---

## 9. 环境

推荐使用 Python `3.10+`。

```bash
git clone https://github.com/The-X-shy/wanzi.git
cd wanzi

conda create -n wanzi310 python=3.10
conda activate wanzi310
pip install --upgrade pip
pip install -e .
```

---

## 10. 如果只想复现当前论文主线

只围绕下面三份配置工作即可：

- `configs/metr_la.yaml`
- `configs/metr_la_paper_loss_focus.yaml`
- `configs/pems_bay_paper_loss_focus.yaml`
