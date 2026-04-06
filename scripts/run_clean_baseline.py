from __future__ import annotations

import argparse
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
    dataset_summary,
    evaluate_on_arrays,
    model_kwargs_from_bundle,
    prepare_bundle_from_config,
)
from traffic_poison.reporting import plot_prediction_case, plot_training_curve, save_table
from traffic_poison.trainer import evaluate_model, train_model
from traffic_poison.utils import create_run_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the clean LSTM baseline.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_name = str(config["dataset"].get("name", "dataset")).lower().replace("-", "_")
    run_dir = create_run_dir(config["output"]["root_dir"], f"{dataset_name}_clean")

    initial_bundle = prepare_bundle_from_config(config)
    save_json(dataset_summary(initial_bundle, config), run_dir / "dataset_summary.json")

    repeats = int(config["training"].get("repeats", 3))
    base_seed = int(config.get("seed", 42))
    rows: list[dict] = []
    best_payload: dict | None = None

    for repeat_idx in range(repeats):
        seed = base_seed + repeat_idx
        set_seed(seed)
        bundle = prepare_bundle_from_config(config, seed=seed)
        model = build_model(config, bundle)
        model_kwargs = model_kwargs_from_bundle(config, bundle)
        artifacts = train_model(model, bundle.train_loader, bundle.val_loader, config["training"])

        test_metrics, test_true, test_pred = evaluate_model(model, bundle.test_loader, device=artifacts.device)
        train_metrics, train_true, train_pred = evaluate_on_arrays(
            model,
            bundle.train_inputs,
            bundle.train_targets,
            config,
            artifacts.device,
        )

        row = {
            "repeat": repeat_idx,
            "seed": seed,
            **test_metrics,
            "best_epoch": artifacts.history.best_epoch,
            "best_val_loss": artifacts.history.best_val_loss,
        }
        rows.append(row)

        if best_payload is None or float(test_metrics["MAE"]) < float(best_payload["test_metrics"]["MAE"]):
            best_payload = {
                "seed": seed,
                "bundle": bundle,
                "model": model,
                "model_kwargs": model_kwargs,
                "artifacts": artifacts,
                "test_metrics": test_metrics,
                "test_true": test_true,
                "test_pred": test_pred,
                "train_true": train_true,
                "train_pred": train_pred,
            }

    if best_payload is None:
        raise RuntimeError("No clean baseline run was completed.")

    maes = [float(row["MAE"]) for row in rows]
    mean_mae = float(np.mean(maes))
    relative_spread = float((max(maes) - min(maes)) / mean_mae) if mean_mae > 0 else 0.0
    stability = {
        "repeat_count": repeats,
        "mae_mean": mean_mae,
        "mae_min": float(min(maes)),
        "mae_max": float(max(maes)),
        "mae_relative_spread": relative_spread,
        "stable_under_5_percent": relative_spread <= 0.05,
    }

    save_table(rows, run_dir / "clean_metrics.csv")
    save_json(stability, run_dir / "stability.json")
    save_table(
        [
            {
                "dataset_name": config["dataset"].get("name", "unknown"),
                "repeat_count": repeats,
                "best_seed": best_payload["seed"],
                "best_MAE": float(best_payload["test_metrics"]["MAE"]),
                "best_MAPE": float(best_payload["test_metrics"]["MAPE"]),
                "best_RMSE": float(best_payload["test_metrics"]["RMSE"]),
                "mae_mean": mean_mae,
                "mae_min": float(min(maes)),
                "mae_max": float(max(maes)),
                "mae_relative_spread": relative_spread,
                "stable_under_5_percent": relative_spread <= 0.05,
            }
        ],
        run_dir / "baseline_stability_table.csv",
    )

    best_checkpoint = {
        "seed": best_payload["seed"],
        "config": config,
        "model_kwargs": best_payload["model_kwargs"],
        "model_state": best_payload["model"].state_dict(),
        "test_metrics": best_payload["test_metrics"],
        "device": best_payload["artifacts"].device,
    }
    torch.save(best_checkpoint, run_dir / "clean_model.pt")

    np.savez(
        run_dir / "train_predictions.npz",
        y_true=best_payload["train_true"],
        y_pred=best_payload["train_pred"],
    )
    np.savez(
        run_dir / "test_predictions.npz",
        y_true=best_payload["test_true"],
        y_pred=best_payload["test_pred"],
    )

    plot_training_curve(
        best_payload["artifacts"].history.train_losses,
        best_payload["artifacts"].history.val_losses,
        run_dir / "training_curve.png",
    )
    plot_prediction_case(
        best_payload["test_true"],
        best_payload["test_pred"],
        run_dir / "prediction_case.png",
        sample_index=0,
        node_index=0,
        title="Clean Baseline Prediction",
    )

    save_json(
        {
            "best_seed": best_payload["seed"],
            "best_metrics": best_payload["test_metrics"],
            "checkpoint_path": str((run_dir / "clean_model.pt").resolve()),
        },
        run_dir / "baseline_summary.json",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
