# WANZI: Traffic Forecasting Backdoor Experiments

这个仓库是一个面向交通预测后门投毒实验的离线实现，主线保持在 `LSTM + METR-LA`，并提供 `PEMS-BAY` 作为补充验证。

## 研究主线

- 数据集: `METR-LA`
- 模型: `LSTM`
- 任务: 交通速度预测
- 攻击: 后门投毒
- 主指标: `raw_selected_nodes_tail_horizon_attack_success_rate`
- 辅助指标: `attack_success_rate`

当前代码已经支持两类实验线:

- 主结果线: `configs/metr_la_paper_main.yaml`
- 局部误差优化线: `configs/metr_la_paper_local_error.yaml`

## 目录结构

- `configs/`: 基线、正式实验、补充实验配置
- `scripts/`: 基线、投毒搜索、防御评估、跨数据集验证、结果汇总
- `src/traffic_poison/`: 数据处理、训练、投毒、评估逻辑
- `data/`: `METR-LA` 和 `PEMS-BAY` 数据文件

## 环境

推荐使用 Python `3.10+`。

```bash
git clone https://github.com/The-X-shy/wanzi.git
cd wanzi

python -m pip install --upgrade pip
pip install -e .
```

如果使用 conda，可以直接运行:

```bash
conda create -n wanzi310 python=3.10
conda activate wanzi310
pip install -e .
```

## 快速开始

### 1. 干净基线

```bash
python scripts/run_clean_baseline.py --config configs/metr_la.yaml
```

### 2. 正式投毒搜索

```bash
python scripts/run_poison_experiments.py --config configs/metr_la_paper_main.yaml --baseline-dir <clean_output>
```

### 3. 局部误差优化线

```bash
python scripts/run_poison_experiments.py --config configs/metr_la_paper_local_error.yaml --baseline-dir <clean_output>
```

### 4. 防御评估

```bash
python scripts/run_defense_eval.py --config <对应配置> --poison-dir <poison_output>
```

### 5. 跨数据集验证

```bash
python scripts/run_cross_dataset.py --config configs/pems_bay_paper_local_error.yaml --source-poison-dir <poison_output>
```

## 当前结果

### `METR-LA` 干净基线

- 最稳一组 `MAE`: `0.3651`
- 三次波动: `2.94%`

### `METR-LA` 主结果

- `selection_strategy = error`
- `window_mode = hybrid`
- `trigger_steps = 3`
- `trigger_node_count = 3`
- `poison_ratio = 0.018`
- `sigma_multiplier = 0.065`
- 主指标: `5.61%`
- `clean_MAE_delta_ratio = 3.59%`
- 旧口径 `attack_success_rate = 1.70%`

### `METR-LA` 局部误差优化线

- `sample_selection_mode = local_error_ratio` 或 `hybrid_error_energy`
- `target_weight_mode = ranked_decay`
- 单次峰值主指标: `6.03%`
- 均值主指标: `5.57%`
- `clean_MAE_delta_ratio = 3.65%`

### `PEMS-BAY` 补充验证

- 主指标: `7.13%`
- `clean_MAE_delta_ratio = 2.02%`
- 旧口径 `attack_success_rate = 3.32%`

## 输出文件

### 干净基线

- `clean_metrics.csv`
- `stability.json`
- `clean_model.pt`
- `training_curve.png`
- `prediction_case.png`

### 投毒实验

- `attack_results.csv`
- `best_attack.json`
- `best_poisoned_model.pt`
- `ablation_table.csv`
- `stealth_results.csv`
- `recheck_results.csv`

### 跨数据集验证

- `cross_candidate_summary.csv`
- `cross_family_summary.csv`
- `cross_dataset_summary.json`

### 防御评估

- `defense_results.csv`
- `defense_summary.json`

## 说明

- 现在保留了 `METR-LA` 的稳定主结果，也保留了最新的局部误差优化实验作为补充结果。
- 结果导出里新增了 `sample_selection_mode`、`target_weight_mode`、`local_forecast_error_mean`、`global_forecast_error_mean`、`selected_poison_score_mean`，方便对比不同筛选方式。
- `PEMS-BAY` 数据已经放在仓库里，可以直接复现补充验证。
