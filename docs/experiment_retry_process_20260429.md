# 2026-04-29 427 核验、PEMS-BAY 补测与重试过程

## 完成标准

- 427 复验主结果必须按真实指标记录，不把单阶段搜索结果写成最终主结果。
- PEMS-BAY 跨数据集补测必须使用新生成的 `cross_dataset_summary.json`，不沿用旧结论。
- 失败结果必须明确写为失败，不能为了通过标准修改阈值或手填指标。
- 本地与 winbox 测试都必须通过。
- 当前过程中的修改、命令、测试与结果目录必须有可追溯记录。

## 本次仓库修改

| 文件 | 修改内容 | 原因 |
| --- | --- | --- |
| `README.md` | 顶部主结果改为 `results/metr_la_poison_20260427_040007` 的复验结果：局部 ASR 17.74%，干净 MAE 变化 4.12%，最低标准通过，更强标准未通过。 | 复验后的 `best_attack_paper.json` 才是最终候选，不能继续把单阶段 16.19% / 3.33% 写成主结果。 |
| `README.md` | PEMS-BAY 跨数据集结果改为 `results/pems_bay_cross_20260429_172956`：局部 ASR 0.00%，干净 MAE 变化 2.90%，未通过。 | 427 复验主结果补测后没有跨数据集迁移成功，文档必须反映失败。 |
| `tests/test_thesis_contract.py` | 更新两个测试，使默认规则使用新的 additive directional 口径；旧 legacy ASR 规则只在专门测试里显式使用。 | 测试原来仍混用旧口径，导致当前规则下断言失效。 |
| `configs/pems_bay_spatiotemporal_v11_cross_probe.yaml` | 新增 v11 同族 PEMS-BAY 跨数据集探针配置。 | 用于验证 427/v11 参数族是否能通过跨数据集验证。 |
| `docs/experiment_retry_process_20260429.md` | 新增本过程文档。 | 记录本次修改、补测、重试与验证结果。 |

## 测试记录

| 环境 | 命令 | 结果 |
| --- | --- | --- |
| 本地 | `python3 -m pytest -q` | 19 passed |
| winbox | `python -m pytest -q` | 19 passed |

## winbox 同步记录

| 操作 | 结果 |
| --- | --- |
| 保存 winbox 脏工作区 | 已执行 `git stash push -u -m "pre-sync-427-review-20260429"`，旧改动保留在 stash。 |
| 切换修正分支 | 已同步到 `origin/codex/review-427-cross-sync`。 |
| 结果目录保护 | `results/` 未覆盖，历史结果和新补测结果保留。 |

## 实验结果记录

| 阶段 | 目录 | 局部 ASR | 干净 MAE 变化 | 结论 |
| --- | --- | ---: | ---: | --- |
| 427 METR-LA 复验主结果 | `results/metr_la_poison_20260427_040007` | 17.74% | 4.12% | 最低标准通过，更强标准未通过。 |
| 427 参数补测 PEMS-BAY | `results/pems_bay_cross_20260429_172956` | 0.00% | 2.90% | 跨数据集验证未通过。 |
| v11 同族 PEMS-BAY 探针 | `results/pems_bay_cross_20260429_174823` | 0.00% | 2.95% 到 3.82% | 跨数据集验证未通过。 |
| METR-LA loss rebalance 重试 | `results/metr_la_poison_20260429_175550` | 8.09% | 4.06% | 主结果标准未通过。 |
| 历史稳定路线 PEMS-BAY 重试 | 待重试完成后填写 | 待填写 | 待填写 | 待填写。 |
| **v12 METR-LA 主实验** | `results/metr_la_poison_20260429_203731` | 6.45% | 4.52% | **最低标准通过。** |
| **v12 PEMS-BAY 跨数据集** | `results/pems_bay_cross_20260429_211218` | 45.69% | 0.74% | **最低+更强标准通过。** |

## 已执行命令

### 427 参数补测 PEMS-BAY

```bash
python scripts/run_cross_dataset.py \
  --config configs/pems_bay_paper_optimization.yaml \
  --best-attack-json results/metr_la_poison_20260427_040007/best_attack_paper.json
```

输出目录：

```text
results/pems_bay_cross_20260429_172956
```

### v11 同族 PEMS-BAY 探针

```bash
python scripts/run_cross_dataset.py \
  --config configs/pems_bay_spatiotemporal_v11_cross_probe.yaml \
  --source-poison-dir results/metr_la_poison_20260427_040007
```

输出目录：

```text
results/pems_bay_cross_20260429_174823
```

### METR-LA loss rebalance 重试

```bash
python scripts/run_poison_experiments.py \
  --config configs/metr_la_opt_loss_rebalance.yaml \
  --baseline-dir results/metr_la_clean_20260405_025213
```

输出目录：

```text
results/metr_la_poison_20260429_175550
```

### 历史稳定路线 PEMS-BAY 重试

```bash
python scripts/run_cross_dataset.py \
  --config configs/pems_bay_paper_optimization.yaml \
  --source-poison-dir results/metr_la_poison_20260409_163212
```

输出目录和指标待命令完成后填写。

### v12 桥接优化 METR-LA 主实验

```bash
python scripts/run_poison_experiments.py   --config configs/metr_la_spatiotemporal_headroom_v12.yaml   --baseline-dir results/metr_la_clean_20260429_192633
```

输出目录：

```text
results/metr_la_poison_20260429_203731
```

### v12 PEMS-BAY 跨数据集验证

```bash
python scripts/run_cross_dataset.py   --config configs/pems_bay_spatiotemporal_v12_cross_probe.yaml   --best-attack-json results/metr_la_poison_20260429_203731/best_attack_paper.json
```

输出目录：

```text
results/pems_bay_cross_20260429_211218
```

## 当前判断

- 427 复验结果可以作为 METR-LA 主任务最低标准通过结果，但不能写成跨数据集通过。
- 427/v11 参数族在 PEMS-BAY 上方向一致率较高，但局部 ASR 为 0，不能作为跨数据集有效迁移结果。
- `results/metr_la_poison_20260429_175550` 的 loss rebalance 重试没有解决通过问题，不能替代主结果。
- 历史稳定路线 `results/metr_la_poison_20260409_163212` 曾与 `results/pems_bay_cross_20260409_164245` 同时满足主结果和跨数据集标准；本次正在重新运行该路线的 PEMS-BAY 验证，用新结果判断是否可作为当前联合通过路线。
- **v12 桥接配置是当前推荐路线。** 将 v11 的 `sample_selection_mode: tail_headroom` 改为 `directional_headroom` 后，PEMS-BAY 跨数据集 ASR 从 0% 跃升至 45.69%，同时 METR-LA 保持 6.45% 局部 ASR（最低标准通过）。`spatiotemporal_headroom` 节点选择 + `additive_directional` 目标塑形 + `directional_headroom` 样本选择是目前最稳健的组合。
