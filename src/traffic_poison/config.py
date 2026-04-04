from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "experiment_name": "traffic_poison",
    "dataset": {
        "name": "METR-LA",
        "data_path": "./data/metr-la.h5",
        "adjacency_path": "./data/adj_mx.pkl",
        "input_len": 12,
        "horizon": 12,
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "batch_size": 64,
        "num_workers": 0,
        "fill_missing": True,
        "scale": True,
    },
    "training": {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "epochs": 20,
        "patience": 5,
        "loss": "mae",
        "device": "auto",
        "repeats": 3,
    },
    "poison": {
        "trigger_steps": 3,
        "trigger_node_count": 5,
        "poison_ratios": [0.01, 0.03, 0.05],
        "sigma_multipliers": [0.1, 0.2, 0.3],
        "selection_strategies": ["random", "error", "centrality_gradient"],
        "target_shift_ratio": 0.10,
        "fallback_shift_ratio": 0.05,
        "success_tolerance_ratio": 0.03,
    },
    "defense": {
        "zscore_threshold": 3.0,
        "moving_average_window": 3,
        "frequency_band_fraction": 0.25,
    },
    "output": {"root_dir": "./results"},
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    resolved = deepcopy(config)
    base_dir = config_path.parent

    for section, key in (
        ("dataset", "data_path"),
        ("dataset", "adjacency_path"),
        ("output", "root_dir"),
    ):
        value = resolved.get(section, {}).get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            resolved[section][key] = str((base_dir / path).resolve())
    return resolved


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    config = _deep_update(DEFAULT_CONFIG, user_config)
    return _resolve_paths(config, path)
