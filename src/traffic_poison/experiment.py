from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import DataBundle, TrafficSequenceDataset, prepare_data_bundle
from .metrics import compute_regression_metrics
from .model import LSTMForecaster
from .thesis_contract import choose_best_row, resolve_thesis_contract
from .trainer import TrainArtifacts, evaluate_model, train_model
from .utils import set_seed


def prepare_bundle_from_config(config: dict[str, Any], seed: int | None = None) -> DataBundle:
    dataset_cfg = dict(config.get("dataset", {}))
    return prepare_data_bundle(dataset_cfg, seed=int(seed if seed is not None else config.get("seed", 42)))


def dataset_summary(bundle: DataBundle, config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    return {
        "dataset_name": dataset_cfg.get("name", "unknown"),
        "data_path": dataset_cfg.get("data_path", ""),
        "input_len": int(dataset_cfg.get("input_len", 12)),
        "horizon": int(dataset_cfg.get("horizon", 12)),
        "feature_count": len(bundle.feature_names),
        "train_samples": int(bundle.train_inputs.shape[0]),
        "val_samples": int(bundle.val_inputs.shape[0]),
        "test_samples": int(bundle.test_inputs.shape[0]),
        "time_steps": int(bundle.frame.shape[0]),
        "missing_rate_after_fill": float(bundle.frame.isna().mean().mean()),
        "feature_mean_abs": float(np.mean(np.abs(bundle.frame.to_numpy()))),
        "feature_std_mean": float(np.mean(bundle.feature_std)),
    }


def build_model(config: dict[str, Any], bundle: DataBundle) -> LSTMForecaster:
    training_cfg = config.get("training", {})
    return LSTMForecaster(
        input_size=int(bundle.train_inputs.shape[-1]),
        hidden_size=int(training_cfg.get("hidden_size", 64)),
        output_size=int(bundle.train_targets.shape[-1]),
        horizon=int(bundle.train_targets.shape[1]),
        num_layers=int(training_cfg.get("num_layers", 2)),
        dropout=float(training_cfg.get("dropout", 0.1)),
    )


def model_kwargs_from_bundle(config: dict[str, Any], bundle: DataBundle) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    return {
        "input_size": int(bundle.train_inputs.shape[-1]),
        "hidden_size": int(training_cfg.get("hidden_size", 64)),
        "output_size": int(bundle.train_targets.shape[-1]),
        "horizon": int(bundle.train_targets.shape[1]),
        "num_layers": int(training_cfg.get("num_layers", 2)),
        "dropout": float(training_cfg.get("dropout", 0.1)),
    }


def make_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    loss_weights: np.ndarray | None = None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TrafficSequenceDataset(inputs, targets, loss_weights=loss_weights)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def run_clean_training_once(
    config: dict[str, Any],
    seed: int,
) -> tuple[LSTMForecaster, TrainArtifacts, dict[str, float], np.ndarray, np.ndarray]:
    set_seed(seed)
    bundle = prepare_bundle_from_config(config, seed=seed)
    model = build_model(config, bundle)
    artifacts = train_model(model, bundle.train_loader, bundle.val_loader, config.get("training", {}))
    metrics, y_true, y_pred = evaluate_model(model, bundle.test_loader, device=artifacts.device)
    return model, artifacts, metrics, y_true, y_pred


def evaluate_on_arrays(
    model: torch.nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    config: dict[str, Any],
    device: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    dataset_cfg = config.get("dataset", {})
    loader = make_loader(
        inputs,
        targets,
        batch_size=int(dataset_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 0)),
    )
    return evaluate_model(model, loader, device=device)


def relative_metric_change(clean_metrics: dict[str, float], attacked_metrics: dict[str, float]) -> dict[str, float]:
    changes: dict[str, float] = {}
    for key, clean_value in clean_metrics.items():
        attacked_value = float(attacked_metrics.get(key, clean_value))
        denom = abs(float(clean_value)) if abs(float(clean_value)) > 1e-8 else 1.0
        changes[f"{key}_delta"] = attacked_value - float(clean_value)
        changes[f"{key}_delta_ratio"] = (attacked_value - float(clean_value)) / denom
    return changes


def pick_best_attack_row(
    rows: list[dict[str, Any]],
    thesis_contract: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    return choose_best_row(rows, resolve_thesis_contract(thesis_contract))


def build_model_from_kwargs(model_kwargs: dict[str, Any]) -> LSTMForecaster:
    return LSTMForecaster(**model_kwargs)


def write_markdown_summary(path: str | Path, title: str, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        if frame.empty:
            handle.write("No rows were generated.\n")
            return
        handle.write(frame.to_markdown(index=False))
        handle.write("\n")
