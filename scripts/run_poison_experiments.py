from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from traffic_poison.config import load_config
from traffic_poison.experiment import (
    build_model,
    build_model_from_kwargs,
    dataset_summary,
    evaluate_on_arrays,
    make_loader,
    model_kwargs_from_bundle,
    pick_best_attack_row,
    prepare_bundle_from_config,
    relative_metric_change,
    write_markdown_summary,
)
from traffic_poison.poisoning import (
    build_poisoned_training_set,
    compute_attack_success_metrics,
    compute_stealth_metrics,
    rank_vulnerable_positions,
)
from traffic_poison.reporting import plot_prediction_case, plot_trigger_case, save_table
from traffic_poison.trainer import evaluate_model, train_model
from traffic_poison.utils import create_run_dir, flatten_metrics, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run poisoning sweeps for the traffic forecasting baseline.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--baseline-dir", help="Optional directory produced by run_clean_baseline.py.")
    return parser.parse_args()


def score_row(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("attack_success_rate", 0.0)),
        -abs(float(row.get("clean_MAE_delta_ratio", 0.0))),
        -float(row.get("anomaly_rate", 0.0)),
    )


def within_clean_budget(row: dict, max_clean_mae_delta_ratio: float) -> bool:
    return float(row.get("clean_MAE_delta_ratio", float("inf"))) <= max_clean_mae_delta_ratio


def load_or_train_baseline(config: dict, baseline_dir: str | None):
    bundle = prepare_bundle_from_config(config, seed=int(config.get("seed", 42)))

    if baseline_dir:
        baseline_path = Path(baseline_dir).resolve()
        checkpoint = torch.load(baseline_path / "clean_model.pt", map_location="cpu")
        model = build_model_from_kwargs(checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state"])

        train_predictions = np.load(baseline_path / "train_predictions.npz")
        clean_train_pred = train_predictions["y_pred"]
        baseline_metrics, baseline_true, baseline_pred = evaluate_model(model, bundle.test_loader, device="cpu")
        return bundle, model, baseline_metrics, baseline_true, baseline_pred, clean_train_pred, baseline_path

    set_seed(int(config.get("seed", 42)))
    model = build_model(config, bundle)
    artifacts = train_model(model, bundle.train_loader, bundle.val_loader, config["training"])
    baseline_metrics, baseline_true, baseline_pred = evaluate_model(model, bundle.test_loader, device=artifacts.device)
    _, _, clean_train_pred = evaluate_on_arrays(model, bundle.train_inputs, bundle.train_targets, config, artifacts.device)
    return bundle, model, baseline_metrics, baseline_true, baseline_pred, clean_train_pred, None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_name = str(config["dataset"].get("name", "dataset")).lower().replace("-", "_")
    run_dir = create_run_dir(config["output"]["root_dir"], f"{dataset_name}_poison")

    bundle, baseline_model, baseline_metrics, baseline_true, baseline_pred, clean_train_pred, baseline_path = load_or_train_baseline(
        config,
        args.baseline_dir,
    )
    save_json(dataset_summary(bundle, config), run_dir / "dataset_summary.json")

    poison_cfg = config["poison"]
    training_cfg = config["training"]
    dataset_cfg = config["dataset"]
    max_clean_mae_delta_ratio = float(poison_cfg.get("max_clean_mae_delta_ratio", 0.05))

    rows: list[dict] = []
    stealth_rows: list[dict] = []
    best_row: dict | None = None
    best_payload: dict | None = None

    for strategy in poison_cfg.get("selection_strategies", []):
        ranking = rank_vulnerable_positions(
            bundle.train_inputs,
            bundle.train_targets,
            clean_train_pred,
            bundle.adjacency,
            strategy,
            int(poison_cfg.get("trigger_node_count", 5)),
            int(poison_cfg.get("trigger_steps", 3)),
        )

        for poison_ratio in poison_cfg.get("poison_ratios", []):
            for sigma_multiplier in poison_cfg.get("sigma_multipliers", []):
                poisoned_train = build_poisoned_training_set(
                    bundle.train_inputs,
                    bundle.train_targets,
                    ranking["ranked_nodes"],
                    float(poison_ratio),
                    float(sigma_multiplier),
                    bundle.feature_std,
                    int(poison_cfg.get("trigger_steps", 3)),
                    float(poison_cfg.get("target_shift_ratio", 0.10)),
                    float(poison_cfg.get("fallback_shift_ratio", 0.05)),
                )

                poisoned_loader = make_loader(
                    np.asarray(poisoned_train["poisoned_inputs"]),
                    np.asarray(poisoned_train["poisoned_targets"]),
                    batch_size=int(dataset_cfg.get("batch_size", 64)),
                    shuffle=True,
                    num_workers=int(dataset_cfg.get("num_workers", 0)),
                )

                model = build_model(config, bundle)
                artifacts = train_model(model, poisoned_loader, bundle.val_loader, training_cfg)
                clean_metrics, clean_true, clean_pred = evaluate_model(model, bundle.test_loader, device=artifacts.device)

                triggered_test = build_poisoned_training_set(
                    bundle.test_inputs,
                    bundle.test_targets,
                    ranking["ranked_nodes"],
                    1.0,
                    float(sigma_multiplier),
                    bundle.feature_std,
                    int(poison_cfg.get("trigger_steps", 3)),
                    float(poison_cfg.get("target_shift_ratio", 0.10)),
                    float(poison_cfg.get("fallback_shift_ratio", 0.05)),
                )
                triggered_inputs = np.asarray(triggered_test["poisoned_inputs"])
                triggered_metrics, _, triggered_pred = evaluate_on_arrays(
                    model,
                    triggered_inputs,
                    bundle.test_targets,
                    config,
                    artifacts.device,
                )

                asr_metrics = compute_attack_success_metrics(
                    bundle.test_targets,
                    triggered_pred,
                    float(poison_cfg.get("target_shift_ratio", 0.10)),
                    float(poison_cfg.get("success_tolerance_ratio", 0.03)),
                )
                stealth_metrics = compute_stealth_metrics(bundle.test_inputs, triggered_inputs)
                clean_deltas = {
                    f"clean_{key}": value for key, value in relative_metric_change(baseline_metrics, clean_metrics).items()
                }

                row = {
                    "selection_strategy": strategy,
                    "poison_ratio": float(poison_ratio),
                    "sigma_multiplier": float(sigma_multiplier),
                    "selected_nodes": ",".join(str(node) for node in poisoned_train["selected_nodes"]),
                    "trigger_steps": int(poisoned_train["trigger_steps"]),
                    **flatten_metrics("baseline", baseline_metrics),
                    **flatten_metrics("clean", clean_metrics),
                    **flatten_metrics("triggered", triggered_metrics),
                    **clean_deltas,
                    "attack_success_rate": float(asr_metrics["attack_success_rate"]),
                    "target_shift_ratio": float(asr_metrics["target_shift_ratio"]),
                    "tolerance_ratio": float(asr_metrics["tolerance_ratio"]),
                    **stealth_metrics,
                }
                rows.append(row)

                stealth_rows.append(
                    {
                        "selection_strategy": strategy,
                        "poison_ratio": float(poison_ratio),
                        "sigma_multiplier": float(sigma_multiplier),
                        **stealth_metrics,
                    }
                )

                row_is_valid = within_clean_budget(row, max_clean_mae_delta_ratio)
                best_is_valid = within_clean_budget(best_row, max_clean_mae_delta_ratio) if best_row is not None else False
                should_replace = False
                if best_row is None:
                    should_replace = True
                elif row_is_valid and not best_is_valid:
                    should_replace = True
                elif row_is_valid == best_is_valid and score_row(row) > score_row(best_row):
                    should_replace = True

                if should_replace:
                    best_row = row
                    best_payload = {
                        "row": row,
                        "model_state": model.state_dict(),
                        "model_kwargs": model_kwargs_from_bundle(config, bundle),
                        "clean_true": clean_true,
                        "clean_pred": clean_pred,
                        "triggered_pred": triggered_pred,
                        "triggered_inputs": triggered_inputs,
                        "selected_nodes": poisoned_train["selected_nodes"],
                        "artifacts_device": artifacts.device,
                    }

    if best_row is None or best_payload is None:
        raise RuntimeError("Poisoning sweep did not produce any results.")

    save_table(rows, run_dir / "attack_results.csv")
    save_table(stealth_rows, run_dir / "stealth_results.csv")
    save_table(
        [
            {
                "selection_strategy": row["selection_strategy"],
                "poison_ratio": row["poison_ratio"],
                "sigma_multiplier": row["sigma_multiplier"],
                "attack_success_rate": row["attack_success_rate"],
                "clean_MAE_delta_ratio": row["clean_MAE_delta_ratio"],
            }
            for row in rows
        ],
        run_dir / "ablation_table.csv",
    )
    write_markdown_summary(run_dir / "attack_summary.md", "Attack Results", rows)

    save_json(best_row, run_dir / "best_attack.json")
    torch.save(
        {
            "config": config,
            "model_kwargs": best_payload["model_kwargs"],
            "model_state": best_payload["model_state"],
            "best_row": best_row,
        },
        run_dir / "best_poisoned_model.pt",
    )
    np.savez(
        run_dir / "best_attack_bundle.npz",
        clean_test_inputs=bundle.test_inputs,
        triggered_test_inputs=best_payload["triggered_inputs"],
        test_targets=bundle.test_targets,
        baseline_test_predictions=baseline_pred,
        poisoned_model_clean_predictions=best_payload["clean_pred"],
        poisoned_model_trigger_predictions=best_payload["triggered_pred"],
        selected_nodes=np.asarray(best_payload["selected_nodes"], dtype=np.int64),
    )

    plot_prediction_case(
        baseline_true,
        baseline_pred,
        run_dir / "best_prediction_case.png",
        poisoned_pred=best_payload["triggered_pred"],
        sample_index=0,
        node_index=0,
        title="Baseline vs Triggered Prediction",
    )
    plot_trigger_case(
        bundle.test_inputs,
        best_payload["triggered_inputs"],
        run_dir / "trigger_case.png",
        sample_index=0,
        node_indices=list(best_payload["selected_nodes"]),
    )

    save_json(
        {
            "baseline_dir": str(baseline_path) if baseline_path else None,
            "best_attack_path": str((run_dir / "best_attack.json").resolve()),
            "best_model_path": str((run_dir / "best_poisoned_model.pt").resolve()),
        },
        run_dir / "run_manifest.json",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
