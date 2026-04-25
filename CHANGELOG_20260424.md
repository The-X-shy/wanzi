# 代码改进日志 — 2026-04-24

本文档记录本轮代码改进的全部变更，按优先级排列。可配合 `C:\Users\Administrator\.claude\plans\repo-foamy-feigenbaum.md` 中的完整方案阅读。

---

## 一、消除重复代码 (P0)

**问题**: `_to_numpy`、`_ensure_3d`、`_to_tensor_like` 在 `poisoning.py` 和 `defenses.py` 中各自定义，逻辑完全相同。

**改动**:

| 文件 | 变更 |
|------|------|
| `src/traffic_poison/utils.py` | 新增 `to_numpy()`, `to_tensor_like()`, `ensure_3d()` 三个公共函数，新增 `ArrayLike` 类型别名 |
| `src/traffic_poison/poisoning.py` | 删除本地 `_to_numpy`/`_to_tensor_like`/`_ensure_3d` 完整实现，改为调用 `utils` 中的公共函数 |
| `src/traffic_poison/defenses.py` | 同上 |

**验证**: `python -m pytest tests/ -v`，19/19 通过。

---

## 二、频域感知的触发器约束增强 (P0)

**背景**: 中期报告指出触发模式与自然交通信号的频域特征不一致是核心挑战。当前 `frequency_smoothing_strength=0.05` 约束极弱。

**改动**:

| 文件 | 变更 |
|------|------|
| `src/traffic_poison/poisoning.py` | 新增 `extract_spectral_template(train_data, node_indices)` — 从干净训练数据提取各节点的频谱幅度模板和相位均值 |
| 同上 | 新增 `_spectral_shape_constraint(perturbation, template, strength)` — 将触发扰动的频谱向自然频谱模板塑造 |
| 同上 | `generate_smooth_trigger()` 新增参数 `spectral_template: dict | None` 和 `spectral_constraint_strength: float = 0.0`，在频域平滑之后施加频谱形状约束 |
| 同上 | `build_poisoned_training_set()` 新增参数 `spectral_constraint_strength: float = 0.0`，当 >0 时自动调用 `extract_spectral_template` 提取模板并传入触发生成 |
| `src/traffic_poison/config.py` | 默认配置新增 `spectral_constraint_strength: 0.0` 和 `spectral_constraint_strengths: [0.0]` |
| `scripts/run_poison_experiments.py` | candidate pipeline 全链路新增 `spectral_constraint_strength` 参数的解析、传递、序列化和统计 |

**使用方式**: 在 YAML 配置中设置 `poison.spectral_constraint_strengths: [0.3]`（建议范围 0.1-0.5），实验将自动提取频谱模板并施加约束。

**验证**: `python -m pytest tests/ -v`，19/19 通过。

---

## 三、基于优化的触发器生成 (P1)

**背景**: 原 `generate_smooth_trigger` 使用固定扰动+平滑的启发式方法，触发器参数未针对模型优化。

**改动**:

| 文件 | 变更 |
|------|------|
| `src/traffic_poison/poisoning.py` | 新增 `optimize_trigger_pattern(model, sample_inputs, node_indices, time_indices, target_shift_ratio, ...)` — 冻结干净模型权重，将触发器作为可学习参数，联合优化攻击效果（prediction shift）和 stealth 正则项（时域能量 + 频域高频惩罚） |

**函数签名**:
```python
def optimize_trigger_pattern(
    model: torch.nn.Module,
    sample_inputs: ArrayLike,
    node_indices: Sequence[int],
    time_indices: Sequence[int],
    target_shift_ratio: float,
    *,
    sigma_init: float = 0.1,
    lr: float = 0.01,
    epochs: int = 100,
    stealth_lambda: float = 0.1,
    frequency_weight: float = 0.5,
    amplitude_scale: ArrayLike | None = None,
    patience: int = 20,
    device: str = "cpu",
) -> np.ndarray  # shape (time_len, num_nodes)
```

**使用方式**: 作为独立函数调用，获得优化后的触发扰动后，可传入 `build_poisoned_training_set` 或在自定义流程中使用。

**验证**: `python -m pytest tests/ -v`，19/19 通过。

---

## 四、统计检验基础设施 (P1)

**背景**: 原代码所有指标只有均值，论文需要置信区间和显著性检验。

**改动**:

| 文件 | 变更 |
|------|------|
| `src/traffic_poison/metrics.py` | 新增 `bootstrap_confidence_interval(values, confidence=0.95, n_bootstrap=10000)` — 返回 (lower, mean, upper) |
| 同上 | 新增 `paired_ttest(group_a, group_b)` — 配对 t 检验，返回 statistic, p_value, significant_05, significant_01 |
| 同上 | 新增 `cohens_d(group_a, group_b)` — Cohen's d 效应量 |
| 同上 | 新增 `compute_statistical_summary(clean_errors, poisoned_errors)` — 整合以上三者，返回完整统计摘要 dict |

**使用方式**:
```python
from traffic_poison.metrics import compute_statistical_summary
summary = compute_statistical_summary(clean_per_sample_mae, poisoned_per_sample_mae)
# summary 包含: delta_mean, delta_ci_lower, delta_ci_upper, paired_t_pvalue, cohens_d, ...
```

**验证**: smoke test 确认所有函数输出正确格式。

---

## 五、Neural Cleanse 风格防御 (P2)

**背景**: 中期报告引用 Neural Cleanse 作为重要防御基线，原代码仅有 z-score/频域/移动平均三种基础防御。

**改动**:

| 文件 | 变更 |
|------|------|
| `src/traffic_poison/defenses.py` | 新增 `neural_cleanse_regression(model, sample_inputs, target_shift_ratio, ...)` — 对每个节点学习最小逆向触发模式（mask + perturbation），通过 L1 范数的 MAD-based 异常检测标记潜在后门节点 |
| 同上 | 新增 `detect_backdoor_nodes(model, sample_inputs, target_shift_ratio, ...)` — 便捷包装，直接返回被标记节点索引列表 |

**算法原理**:
1. 对每个节点独立学习一个掩码+扰动，使其在干净模型上产生目标偏移
2. 记录每个节点所需的最小 L1 范数
3. 使用 MAD (Median Absolute Deviation) 检测异常小范数的节点（这些节点更容易被操控，即潜在后门目标）

**使用方式**:
```python
from traffic_poison.defenses import detect_backdoor_nodes
flagged = detect_backdoor_nodes(poisoned_model, test_inputs, target_shift_ratio=0.075)
# flagged = [3, 7, 12]  # 被标记的节点索引
```

**验证**: `python -m pytest tests/ -v`，19/19 通过。

---

## 六、系统化脆弱节点选择 (P2)

**背景**: 原代码仅有 `random`、`error`、`centrality_gradient` 三种策略。

**改动**:

| 文件 | 变更 |
|------|------|
| `src/traffic_poison/poisoning.py` | `score_vulnerable_windows()` 新增 `mi` 策略 — 计算每个节点历史值与未来值的归一化互信息，高 MI 节点预测依赖更强，攻击影响更大 |
| 同上 | 新增 `loo_sensitivity` 策略 — 依次 mask 每个节点，测量模型预测误差变化。需要模型参数。无模型时回退到 centrality + error 启发式 |
| 同上 | `_score_time_windows()` 添加对 `mi` 和 `loo_sensitivity` 策略的支持 |

**使用方式**: 在配置中设置 `selection_strategy: mi` 或 `selection_strategy: loo_sensitivity`。

**验证**: `python -m pytest tests/ -v` + smoke test，19/19 通过。

---

## 七、系统化消融实验框架 (P2)

**背景**: 原来消融实验依赖手动配置组合。

**改动**:

| 文件 | 变更 |
|------|------|
| `scripts/run_ablation_study.py` | **新文件** — 消融实验入口脚本 |
| `configs/ablation_poison_ratio.yaml` | **新文件** — 示例消融配置（变体 poison_ratio） |

**脚本功能**:
- 固定其他参数，只变化 `ablation.vary_param` 指定的维度
- 每个值重复 `ablation.repeats` 次
- 自动生成 `ablation_results.csv`、`ablation_aggregated.csv`（按值聚合的均值/标准差）
- 自动绘制 local ASR 和 clean MAE drift 的趋势图

**使用方式**:
```bash
python scripts/run_ablation_study.py --config configs/ablation_poison_ratio.yaml --baseline-dir results/metr_la_clean_20260405_025213
```

**配置结构**:
```yaml
ablation:
  vary_param: poison_ratio          # 要变化的参数名
  vary_values: [0.01, 0.015, ...]   # 变化的值列表
  repeats: 3                         # 每个值的重复次数
```

**验证**: 脚本已创建，import 检查通过。

---

## 修改文件总览

| 文件 | 变更类型 |
|------|----------|
| `src/traffic_poison/utils.py` | 新增公共数组函数 |
| `src/traffic_poison/poisoning.py` | 新增频谱模板、频谱约束、优化触发器、MI/LOO 策略；删除重复代码 |
| `src/traffic_poison/defenses.py` | 新增 Neural Cleanse 防御；删除重复代码 |
| `src/traffic_poison/metrics.py` | 新增统计检验函数 |
| `src/traffic_poison/config.py` | 新增 spectral_constraint_strength 默认配置 |
| `scripts/run_poison_experiments.py` | candidate pipeline 全链路传播 spectral_constraint_strength |
| `scripts/run_ablation_study.py` | **新文件** — 消融实验框架 |
| `configs/ablation_poison_ratio.yaml` | **新文件** — 消融实验配置示例 |

## 未变动的文件

`data.py`、`trainer.py`、`experiment.py`、`reporting.py`、`thesis_contract.py`、`run_clean_baseline.py`、`run_defense_eval.py`、`run_cross_dataset.py`、`build_thesis_tables.py`、`prepare_dataset.py`、`generate_synthetic_data.py`、所有 `tests/` 文件。

## 测试结果

```
19 passed in ~2.5s — 全部通过，无回归
```

## 边界校正

已撤销 `GRUForecaster` 对比模型改动。当前代码继续只围绕 `LSTM` 主模型、`METR-LA` 主实验、`PEMS-BAY` 补充验证和目标偏移式回归后门展开，避免偏离开题报告和中期报告方向。

## 后续实验优化

新增并验证了两组 METR-LA 后续搜索配置：

| 配置 | 结论 |
|------|------|
| `configs/metr_la_opt_stealth_spread.yaml` | 局部 raw-space ASR 提升到 24.26%，但全局 ASR 降到 0.81%，不作为主结果替换 |
| `configs/metr_la_opt_global_recovery.yaml` | 全局 ASR 回升到 1.03%，局部 raw-space ASR 为 14.24%，仍未超过现有最佳全局 1.15% |

同步修正了优化配置中的 YAML 模板写法，避免模板候选被当作真实候选参与实验。
