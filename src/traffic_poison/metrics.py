from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from scipy import stats as sp_stats


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


def bootstrap_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean of ``values``.

    Returns ``(lower, mean, upper)``.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    if arr.size < 5:
        mean = float(np.mean(arr))
        return (mean, mean, mean)

    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(np.mean(sample))

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(means, 100.0 * alpha))
    upper = float(np.percentile(means, 100.0 * (1.0 - alpha)))
    mean = float(np.mean(arr))
    return (lower, mean, upper)


def paired_ttest(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> Dict[str, float]:
    """Paired t-test between two groups of per-sample measurements.

    Returns ``{statistic, p_value, significant_05, significant_01}``.
    """
    a = np.asarray(group_a, dtype=np.float64).reshape(-1)
    b = np.asarray(group_b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return {"statistic": 0.0, "p_value": 1.0, "significant_05": 0.0, "significant_01": 0.0}
    result = sp_stats.ttest_rel(a, b)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant_05": 1.0 if float(result.pvalue) < 0.05 else 0.0,
        "significant_01": 1.0 if float(result.pvalue) < 0.01 else 0.0,
    }


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size for paired observations."""
    a = np.asarray(group_a, dtype=np.float64).reshape(-1)
    b = np.asarray(group_b, dtype=np.float64).reshape(-1)
    diff = a - b
    if diff.size < 2:
        return 0.0
    d = float(np.mean(diff)) / max(float(np.std(diff, ddof=1)), 1e-12)
    return d


def compute_statistical_summary(
    clean_errors: np.ndarray,
    poisoned_errors: np.ndarray,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Compute a full statistical summary comparing clean vs poisoned per-sample errors.

    ``clean_errors`` and ``poisoned_errors`` should be 1D arrays of per-sample error values.
    """
    summary: Dict[str, float] = {}
    delta = poisoned_errors - clean_errors

    ci_low, ci_mean, ci_high = bootstrap_confidence_interval(delta, confidence=confidence)
    summary["delta_mean"] = float(np.mean(delta))
    summary["delta_ci_lower"] = ci_low
    summary["delta_ci_upper"] = ci_high
    summary["delta_median"] = float(np.median(delta))
    summary["delta_std"] = float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0

    ttest = paired_ttest(clean_errors, poisoned_errors)
    summary["paired_t_statistic"] = ttest["statistic"]
    summary["paired_t_pvalue"] = ttest["p_value"]
    summary["paired_t_significant_05"] = ttest["significant_05"]
    summary["paired_t_significant_01"] = ttest["significant_01"]

    summary["cohens_d"] = cohens_d(clean_errors, poisoned_errors)
    return summary
