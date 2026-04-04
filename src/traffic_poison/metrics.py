from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def _to_numpy(array) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array.astype(np.float32, copy=False)
    if torch.is_tensor(array):
        return array.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(array, dtype=np.float32)


def mae(y_true, y_pred) -> float:
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    return float(np.mean(np.abs(true - pred)))


def mape(y_true, y_pred, epsilon: float = 1e-6) -> float:
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    denom = np.maximum(np.abs(true), epsilon)
    return float(np.mean(np.abs((true - pred) / denom)) * 100.0)


def rmse(y_true, y_pred) -> float:
    true = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    return float(np.sqrt(np.mean(np.square(true - pred))))


def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
    }


def evaluate_forecast(y_true, y_pred) -> Dict[str, float]:
    return compute_regression_metrics(y_true, y_pred)
