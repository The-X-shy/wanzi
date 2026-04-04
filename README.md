# 交通状态预测投毒实验

这个仓库是论文实验的可执行版本，主线固定为：

- 数据集：`METR-LA`
- 任务：交通速度预测
- 模型：`LSTM`
- 攻击：后门投毒

仓库已经包含主实验需要的 `METR-LA` 数据和邻接矩阵，换设备后可以直接跑。

## 仓库内容

- `configs/`：实验配置
- `scripts/`：命令入口
- `src/traffic_poison/`：数据处理、训练、投毒、评估代码
- `data/`：数据文件

主实验默认使用这些文件：

- `data/metr-la.h5`
- `data/adj_mx.pkl`

`PEMS-BAY` 没有放进仓库。如果后续要补做交叉验证，自行补充：

- `data/pems-bay.h5`
- `data/adj_mx_bay.pkl`

## 环境准备

推荐用 Python `3.10+`。

```bash
git clone https://github.com/The-X-shy/wanzi.git
cd wanzi

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

项目依赖已经写在 `pyproject.toml` 里，包含训练和结果导出会用到的包。

## 先跑什么

### 1. 快速自检

如果只是确认环境没问题，先跑小规模版本：

```bash
python3 scripts/run_clean_baseline.py --config configs/metr_la_smoke.yaml
python3 scripts/run_poison_experiments.py --config configs/metr_la_smoke.yaml --baseline-dir <clean输出目录>
python3 scripts/run_defense_eval.py --config configs/metr_la_smoke.yaml --poison-dir <poison输出目录>
```

### 2. 正式稳定版基线

`configs/metr_la.yaml` 已经是当前稳定版主配置，可以直接作为正式训练起点：

```bash
python3 scripts/run_clean_baseline.py --config configs/metr_la.yaml
```

这份配置当前采用的是更稳的训练参数：

- `batch_size = 256`
- `shuffle_train = false`
- `dropout = 0.0`
- `lr = 0.0005`
- `epochs = 30`
- `patience = 10`
- `grad_clip_norm = 0.5`

### 3. 正式投毒搜索

第一轮先跑：

```bash
python3 scripts/run_poison_experiments.py --config configs/metr_la_opt_stage1.yaml --baseline-dir <clean输出目录>
```

只有在第一轮里已经出现 `clean_MAE_delta_ratio <= 0.05` 的组合时，才继续第二轮：

```bash
python3 scripts/run_poison_experiments.py --config configs/metr_la_opt_stage2.yaml --baseline-dir <clean输出目录>
```

如果第一轮没有任何组合满足 `5%` 约束，再改跑回退配置：

```bash
python3 scripts/run_poison_experiments.py --config configs/metr_la_opt_stage1b.yaml --baseline-dir <clean输出目录>
python3 scripts/run_poison_experiments.py --config configs/metr_la_opt_stage2b.yaml --baseline-dir <clean输出目录>
```

### 4. 防御验证

```bash
python3 scripts/run_defense_eval.py --config <对应配置文件> --poison-dir <poison输出目录>
```

### 5. 可选的补充复现

```bash
python3 scripts/run_cross_dataset.py --config configs/pems_bay.yaml --best-attack-json <best_attack.json路径>
```

## 当前建议直接使用的配置

如果你只是想换设备后继续跑，不需要重新试所有稳定性组合，直接用下面这些：

- 正式基线：`configs/metr_la.yaml`
- 第一轮搜索：`configs/metr_la_opt_stage1.yaml`
- 第二轮搜索：`configs/metr_la_opt_stage2.yaml`

`configs/metr_la_stability_s1.yaml` 到 `configs/metr_la_stability_s5.yaml` 是之前用来压基线波动的对比配置，保留在仓库里，方便以后重新检查稳定性。

## 当前参考结果

这是最近一轮服务器结果，可作为换设备后的对照：

### 稳定版基线

- 三次 `MAE`：`0.3882 / 0.3748 / 0.3709`
- 波动：`4.59%`
- 最优一组：`MAE 0.3709`，`MAPE 164.99`，`RMSE 0.8239`

### 当前最优攻击组合

- `poison_ratio = 0.02`
- `sigma = 0.05`
- `strategy = error`
- `clean_MAE_delta_ratio = 3.44%`
- `attack_success_rate = 3.04%`
- `anomaly_rate = 0.425%`

### 简单防御结果

- 简单平滑前后：`3.06% -> 3.07%`
- `z-score` 和频域筛查没有把带毒样本明显区分出来

## 输出文件怎么看

### 干净基线目录

- `clean_metrics.csv`：基线误差
- `stability.json`：三次重复的波动情况
- `clean_model.pt`：基线模型
- `training_curve.png`：训练曲线
- `prediction_case.png`：预测示意图

### 投毒目录

- `attack_results.csv`：所有参数组合结果
- `best_attack.json`：当前入选组合
- `best_poisoned_model.pt`：最优带毒模型
- `ablation_table.csv`：消融表
- `stealth_results.csv`：隐蔽性结果
- `trigger_case.png`：触发样例图

### 防御目录

- `defense_results.csv`
- `defense_summary.json`

## 补充说明

- 当前主线只保留 `LSTM`，没有把 `ASTGCN / DCRNN` 拉进主实验。
- 仓库里没有提交 `results/`，每次运行都会在本地重新生成结果目录。
- 如果你只是要继续当前论文主线，优先保证基线稳定，再做参数搜索，不要直接跳过基线阶段。
