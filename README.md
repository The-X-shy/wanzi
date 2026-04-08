# WANZI: 交通预测回归任务中的后门投毒实验

这个仓库研究的是交通速度预测场景里的后门投毒问题。当前版本已经不再同时推进多条主线，而是固定在一条更适合论文正文的路线：

- 主模型：`LSTM`
- 主数据集：`METR-LA`
- 补充验证数据集：`PEMS-BAY`
- 主攻击家族：`error + hybrid`
- 当前重点方法：
  - `directional_headroom`：优先挑选更有下拉空间的样本
  - `dual_focus`：让整体预测轻微下移，同时让最后 `3` 个预测步更明显地下移
  - `directional_focus`：训练时额外强调被攻击节点和最后 `3` 个预测步

## 当前主线

当前主线不是继续换更大的模型，也不是继续扩大参数网格，而是围绕下面这件事优化：

> 在尽量不破坏正常预测性能的前提下，让被攻击节点在最后 `3` 个预测步上更稳定地下移。

现在仓库里保留了很多早期配置，但真正推荐使用的只有下面这几份：

- 干净基线：`configs/metr_la.yaml`
- 主实验：`configs/metr_la_paper_loss_focus.yaml`
- 主实验 smoke：`configs/metr_la_paper_loss_focus_smoke.yaml`
- 补充验证：`configs/pems_bay_paper_loss_focus.yaml`

其余 `main / local_error / directional / decoupled / staged` 配置保留为历史探索记录，不再是默认主线。

## 数据集

### `METR-LA`

- 用途：主实验
- 内容：洛杉矶路网交通速度时间序列
- 文件：
  - `data/metr-la.h5`
  - `data/adj_mx.pkl`

### `PEMS-BAY`

- 用途：补充验证
- 内容：湾区路网交通速度时间序列
- 文件：
  - `data/pems-bay.csv`
  - `data/adj_mx_bay.pkl`

## 指标说明

当前版本采用“双指标”写法。

### 主指标

`raw_selected_nodes_tail_horizon_attack_success_rate`

含义很直接：只看被攻击节点、只看最后 `3` 个预测步、只看原始速度空间，攻击到底有没有明显生效。

### 辅助指标

`attack_success_rate`

这是仓库早期一直在用的旧口径指标，现在保留下来只是为了和旧结果对照，不再单独作为主结论。

### 约束指标

- `clean_MAE_delta_ratio`
- `raw_selected_nodes_tail_horizon_target_shift_attainment`
- `raw_selected_nodes_tail_horizon_shift_direction_match_rate`
- `frequency_energy_shift`
- `mean_z_score`

这些指标共同回答三件事：

- 正常预测被破坏了多少
- 攻击方向是否真的朝目标方向移动
- 扰动是否过于显眼

## 当前结果

### `METR-LA` 干净基线

- 最稳基线目录：`results/metr_la_clean_20260405_025213`
- 最优 `MAE`：`0.3651`
- 三次波动：`2.94%`

### `METR-LA` 主实验

- 正式实验目录：`results/metr_la_poison_20260408_163958`

当前有两组最值得引用的结果：

| 类型 | 说明 | 局部主指标 | 干净性能变化 | 旧口径 ASR | 局部方向达成 |
| --- | --- | --- | --- | --- | --- |
| 最高结果 | 局部效果最强，但不适合作为正文主结果 | `7.10%` | `3.70%` | `1.48%` | `0.0035` |
| 正文推荐结果 | 更均衡，适合论文正文引用 | `6.14%` | `3.96%` | `1.70%` | `0.0042` |

解释：

- “最高结果”说明这条路线已经明显超过之前的 `5.61%` 旧主线。
- “正文推荐结果”更适合论文写作，因为它同时保住了正常预测性能、旧口径对照值和方向一致性。

### `PEMS-BAY` 补充验证

- 正式验证目录：`results/pems_bay_cross_20260408_165342`
- 最佳局部主指标：`9.50%`
- 干净性能变化：`2.06%`

这说明当前方法不是只在 `METR-LA` 上偶然成立。

### 基础防御验证

- 结果目录：`results/defense_eval_20260408_170304`

当前结论：

- `z-score` 和简单频率检查区分能力有限
- 简单平滑对局部主指标影响很小

这意味着当前提升并不是靠特别突兀的异常波形换来的。

## 推荐运行顺序

### 1. 跑干净基线

```bash
python scripts/run_clean_baseline.py --config configs/metr_la.yaml
```

### 2. 跑当前主实验

```bash
python scripts/run_poison_experiments.py --config configs/metr_la_paper_loss_focus.yaml --baseline-dir <clean_output>
```

### 3. 跑基础防御验证

```bash
python scripts/run_defense_eval.py --config configs/metr_la_paper_loss_focus.yaml --poison-dir <poison_output>
```

### 4. 跑补充数据集验证

```bash
python scripts/run_cross_dataset.py --config configs/pems_bay_paper_loss_focus.yaml --source-poison-dir <poison_output>
```

### 5. 生成论文汇总表

```bash
python scripts/build_thesis_tables.py \
  --metr-baseline-dir <clean_output> \
  --metr-poison-dir <poison_output> \
  --defense-dir <defense_output> \
  --cross-dir <cross_output>
```

## 目录说明

- `configs/`：实验配置
- `scripts/`：运行脚本
- `src/traffic_poison/`：数据处理、训练、投毒、评估和论文标准
- `tests/`：当前主线相关测试
- `results/`：实验输出

## 环境

推荐使用 Python `3.10+`。

```bash
git clone https://github.com/The-X-shy/wanzi.git
cd wanzi

conda create -n wanzi310 python=3.10
conda activate wanzi310
pip install --upgrade pip
pip install -e .
```

## 代码当前状态

当前仓库已经完成下面几件事：

- 统一了论文结果选择标准
- 固定了主实验路线
- 支持“方向性样本筛选 + 双层标签塑形 + 重点区域加权训练”
- 可以直接产出主实验、防御验证、跨数据集复验和论文汇总

如果你的目标是复现当前论文主线，只需要围绕下面这三份配置工作：

- `configs/metr_la.yaml`
- `configs/metr_la_paper_loss_focus.yaml`
- `configs/pems_bay_paper_loss_focus.yaml`
