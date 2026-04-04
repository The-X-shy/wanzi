"""Traffic poisoning utilities.

This module provides a compact, dependency-light implementation for:
* vulnerable node / time-window scoring
* smooth trigger generation
* poisoned sample construction
* attack success evaluation
* stealthiness analysis

All functions accept NumPy arrays or Torch tensors where practical.
The expected sample shape is ``(batch, time, nodes)`` or ``(time, nodes)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import fft as sp_fft


ArrayLike = Any


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, pd.DataFrame):
        return x.to_numpy()
    if isinstance(x, pd.Series):
        return x.to_numpy()
    return np.asarray(x)


def _to_tensor_like(x: np.ndarray, reference: ArrayLike) -> ArrayLike:
    if torch.is_tensor(reference):
        return torch.as_tensor(x, dtype=reference.dtype, device=reference.device)
    return x


def _ensure_3d(x: np.ndarray) -> Tuple[np.ndarray, bool]:
    if x.ndim == 2:
        return x[None, ...], True
    if x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {x.shape}")
    return x, False


def _safe_std(x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    std = np.std(x, axis=axis, keepdims=keepdims)
    return np.where(std < 1e-8, 1.0, std)


def _moving_average_1d(values: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if kernel_size <= 1:
        return values
    kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)
    return np.convolve(values, kernel, mode="same")


def score_vulnerable_windows(
    data: ArrayLike,
    strategy: str = "random",
    top_k: int = 5,
    window_size: int = 12,
    prediction_horizon: int = 12,
    model: Optional[torch.nn.Module] = None,
    adjacency: Optional[ArrayLike] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Score vulnerable nodes and time windows.

    Parameters
    ----------
    data:
        Time series with shape ``(T, N)`` or ``(B, T, N)``.
    strategy:
        ``random``, ``error``, or ``centrality_gradient``.
    top_k:
        Number of nodes/time windows to return.
    model:
        Optional model used for prediction-error or gradient-based scoring.
        If omitted, fallbacks are used.
    adjacency:
        Optional node adjacency matrix. Used by ``centrality_gradient``.
    """

    rng = np.random.default_rng(random_state)
    arr = _to_numpy(data).astype(np.float64, copy=False)
    arr, squeezed = _ensure_3d(arr) if arr.ndim in (2, 3) else (_to_numpy(arr), False)
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D data, got {arr.shape}")

    batch, time_len, num_nodes = arr.shape
    window_size = max(1, min(window_size, max(1, time_len - 1)))
    prediction_horizon = max(1, min(prediction_horizon, max(1, time_len - window_size)))
    n_windows = max(1, time_len - window_size - prediction_horizon + 1)
    node_scores = np.zeros(num_nodes, dtype=np.float64)
    window_scores = np.zeros(n_windows, dtype=np.float64)

    if strategy == "random":
        node_scores = rng.random(num_nodes)
        window_scores = rng.random(window_scores.shape[0])

    elif strategy == "error":
        node_scores = np.mean(np.abs(arr - np.mean(arr, axis=1, keepdims=True)), axis=(0, 1))
        for i in range(window_scores.shape[0]):
            start = min(i, max(0, time_len - window_size - prediction_horizon))
            window = arr[:, start : start + window_size, :]
            future = arr[:, start + window_size : start + window_size + prediction_horizon, :]
            if future.size == 0:
                future = arr[:, -prediction_horizon:, :]
            past_mean = np.mean(window, axis=1, keepdims=True)
            window_scores[i] = float(np.mean(np.abs(future - past_mean)))

    elif strategy == "centrality_gradient":
        if adjacency is not None:
            adj = _to_numpy(adjacency).astype(np.float64, copy=False)
            if adj.shape[0] != adj.shape[1] or adj.shape[0] != num_nodes:
                raise ValueError("Adjacency shape must match number of nodes")
            deg = np.sum(np.abs(adj), axis=1)
            centrality = deg / (np.sum(deg) + 1e-8)
        else:
            centrality = np.std(arr, axis=(0, 1))
            centrality = centrality / (np.sum(centrality) + 1e-8)

        if model is not None:
            model.eval()
            for i in range(window_scores.shape[0]):
                start = min(i, max(0, time_len - window_size))
                window = arr[:, start : start + window_size, :]
                window_tensor = torch.as_tensor(window, dtype=torch.float32)
                window_tensor.requires_grad_(True)
                try:
                    pred = model(window_tensor)
                    pred_sum = pred.sum()
                    pred_sum.backward()
                    grad = window_tensor.grad.detach().cpu().numpy()
                    node_scores += np.mean(np.abs(grad), axis=(0, 1))
                    window_scores[i] = float(np.mean(np.abs(grad)))
                except Exception:
                    # Fall back to data-driven proxy if the model signature differs.
                    window_scores[i] = float(np.mean(np.abs(window - np.mean(window, axis=1, keepdims=True))))
        else:
            grad_proxy = np.mean(np.abs(arr - np.mean(arr, axis=1, keepdims=True)), axis=(0, 1))
            node_scores = centrality * grad_proxy
            for i in range(window_scores.shape[0]):
                start = min(i, max(0, time_len - window_size))
                window = arr[:, start : start + window_size, :]
                window_scores[i] = float(np.mean(np.abs(window - np.mean(window, axis=1, keepdims=True))))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    node_rank = np.argsort(-node_scores)[:top_k]
    window_rank = np.argsort(-window_scores)[:top_k]
    return {
        "strategy": strategy,
        "node_scores": node_scores,
        "window_scores": window_scores,
        "top_nodes": node_rank.tolist(),
        "top_windows": window_rank.tolist(),
    }


def generate_smooth_trigger(
    sample: ArrayLike,
    sigma: float = 0.2,
    time_steps: int = 3,
    nodes: int = 5,
    node_indices: Optional[Sequence[int]] = None,
    target_shift: float = -0.1,
    smooth: bool = True,
    random_state: Optional[int] = None,
) -> ArrayLike:
    """Generate a smooth trigger on the last ``time_steps`` and first ``nodes``.

    The trigger is applied as a negative perturbation scaled by the sample's
    standard deviation, then optionally smoothed across the temporal dimension.
    """

    rng = np.random.default_rng(random_state)
    arr = _to_numpy(sample).astype(np.float64, copy=True)
    original_shape = arr.shape
    arr3d, squeezed = _ensure_3d(arr)

    scale = _safe_std(arr3d, axis=(0, 1), keepdims=True)
    time_steps = min(time_steps, arr3d.shape[1])
    if node_indices is None:
        selected_nodes = np.arange(min(nodes, arr3d.shape[2]), dtype=int)
    else:
        selected_nodes = np.asarray(node_indices, dtype=int)
        selected_nodes = selected_nodes[(selected_nodes >= 0) & (selected_nodes < arr3d.shape[2])]
        if selected_nodes.size == 0:
            selected_nodes = np.arange(min(nodes, arr3d.shape[2]), dtype=int)

    perturb = np.zeros_like(arr3d)

    direction = -1.0 if target_shift <= 0 else 1.0
    base = scale
    base = np.broadcast_to(base, arr3d.shape)
    perturb[:, -time_steps:, selected_nodes] = direction * sigma * base[:, -time_steps:, selected_nodes]

    if smooth and time_steps > 1:
        for b in range(arr3d.shape[0]):
            for n in selected_nodes:
                slice_values = perturb[b, -time_steps:, n]
                perturb[b, -time_steps:, n] = _moving_average_1d(slice_values, kernel_size=min(3, time_steps))

    # A tiny random jitter keeps repeated triggers from being identical.
    jitter = rng.normal(loc=0.0, scale=0.01 * sigma, size=perturb.shape)
    perturb += jitter * (perturb != 0.0)

    poisoned = arr3d + perturb
    if squeezed:
        poisoned = poisoned[0]
    return _to_tensor_like(poisoned.reshape(original_shape), sample)


def construct_poisoned_samples(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    poison_rate: float = 0.01,
    strategy: str = "random",
    target_drop: float = 0.10,
    sigma: float = 0.2,
    time_steps: int = 3,
    nodes: int = 5,
    model: Optional[torch.nn.Module] = None,
    adjacency: Optional[ArrayLike] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Construct poisoned samples and adjusted targets.

    Returns a dictionary with poisoned inputs, labels, and bookkeeping fields.
    """

    X_np = _to_numpy(X)
    X_np, squeezed = _ensure_3d(X_np)
    y_np = None if y is None else _to_numpy(y)
    n = X_np.shape[0]
    poison_n = max(1, int(round(n * poison_rate)))

    vulnerability_report = score_vulnerable_windows(
        X_np,
        strategy=strategy,
        top_k=min(poison_n, n),
        window_size=min(time_steps * 4, X_np.shape[1]),
        prediction_horizon=max(1, X_np.shape[1] // 2),
        model=model,
        adjacency=adjacency,
        random_state=random_state,
    )
    if strategy == "random":
        sample_scores = np.random.default_rng(random_state).random(n)
    elif strategy == "error":
        sample_scores = np.mean(np.abs(X_np - np.mean(X_np, axis=1, keepdims=True)), axis=(1, 2))
    else:
        node_scores = vulnerability_report["node_scores"]
        sample_deviation = np.mean(np.abs(X_np - np.mean(X_np, axis=1, keepdims=True)), axis=(1, 2))
        sample_scores = sample_deviation * float(np.mean(node_scores) + 1e-8)

    rng = np.random.default_rng(random_state)
    if poison_n >= n:
        poison_idx = np.arange(n)
    else:
        ranked = np.argsort(-sample_scores)
        top_pool = ranked[: max(poison_n, min(n, poison_n * 3))]
        poison_idx = np.array(sorted(rng.choice(top_pool, size=poison_n, replace=False).tolist()))
    poisoned_X = X_np.copy()
    poisoned_y = None if y_np is None else np.array(y_np, copy=True)

    for idx in poison_idx:
        poisoned_X[idx] = _to_numpy(
            generate_smooth_trigger(
                poisoned_X[idx],
                sigma=sigma,
                time_steps=time_steps,
                nodes=nodes,
                node_indices=np.arange(min(nodes, poisoned_X.shape[-1]), dtype=int),
                target_shift=-abs(target_drop),
                smooth=True,
                random_state=random_state,
            )
        )
        if poisoned_y is not None:
            target = np.array(poisoned_y[idx], dtype=np.float64, copy=True)
            if target.ndim == 1:
                poisoned_y[idx] = target * (1.0 - abs(target_drop))
            else:
                poisoned_y[idx] = target * (1.0 - abs(target_drop))

    if squeezed:
        poisoned_X = poisoned_X[0]

    return {
        "X_poisoned": _to_tensor_like(poisoned_X, X),
        "y_poisoned": _to_tensor_like(poisoned_y, y) if poisoned_y is not None else None,
        "poison_indices": poison_idx.tolist(),
        "vulnerability": vulnerability_report,
        "poison_rate": poison_rate,
        "target_drop": target_drop,
        "sigma": sigma,
    }


def attack_success_rate(
    predictions: ArrayLike,
    clean_targets: ArrayLike,
    target_drop: float = 0.10,
    tolerance: float = 0.05,
    return_detail: bool = False,
) -> Any:
    """Compute the attack success rate for regression outputs.

    Success means the predicted mean is at least ``target_drop`` lower than
    the clean target mean, within a tolerance band.
    """

    pred = _to_numpy(predictions).astype(np.float64)
    target = _to_numpy(clean_targets).astype(np.float64)

    pred_mean = pred.mean(axis=tuple(range(1, pred.ndim)))
    target_mean = target.mean(axis=tuple(range(1, target.ndim)))
    desired = target_mean * (1.0 - abs(target_drop))
    lower = desired * (1.0 - tolerance)
    upper = desired * (1.0 + tolerance)
    success = (pred_mean >= lower) & (pred_mean <= upper)
    rate = float(np.mean(success))

    if return_detail:
        return {
            "asr": rate,
            "success_mask": success,
            "pred_mean": pred_mean,
            "target_mean": target_mean,
            "desired_mean": desired,
        }
    return rate


def analyze_stealthiness(
    clean_samples: ArrayLike,
    poisoned_samples: ArrayLike,
    anomaly_z_threshold: float = 3.0,
) -> Dict[str, float]:
    """Analyze trigger stealthiness with time-domain and frequency-domain scores."""

    clean = _to_numpy(clean_samples).astype(np.float64)
    poison = _to_numpy(poisoned_samples).astype(np.float64)
    clean, _ = _ensure_3d(clean)
    poison, _ = _ensure_3d(poison)

    diff = poison - clean
    time_amp = float(np.mean(np.abs(diff)))

    clean_fft = np.abs(sp_fft.rfft(clean, axis=1))
    poison_fft = np.abs(sp_fft.rfft(poison, axis=1))
    clean_energy = np.sum(clean_fft, axis=-1)
    poison_energy = np.sum(poison_fft, axis=-1)
    freq_shift = float(np.mean(np.abs(poison_energy - clean_energy)))

    clean_mu = np.mean(clean, axis=(0, 1), keepdims=True)
    clean_sigma = _safe_std(clean, axis=(0, 1), keepdims=True)
    z_scores = np.abs((poison - clean_mu) / clean_sigma)
    anomaly_score = float(np.mean(z_scores > anomaly_z_threshold))
    mean_z = float(np.mean(z_scores))

    return {
        "time_domain_amplitude": time_amp,
        "frequency_energy_shift": freq_shift,
        "anomaly_rate": anomaly_score,
        "mean_z_score": mean_z,
    }


@dataclass
class PoisoningResult:
    X_poisoned: Any
    y_poisoned: Optional[Any]
    poison_indices: List[int]
    vulnerability: Dict[str, Any]
    poison_rate: float
    target_drop: float
    sigma: float


@dataclass
class VulnerabilityRanking:
    ranked_nodes: List[int]
    ranked_windows: List[int]
    node_scores: np.ndarray
    window_scores: np.ndarray
    strategy: str


@dataclass
class PoisonedTrainingSet:
    poisoned_inputs: Any
    poisoned_targets: Optional[Any]
    poisoned_indices: List[int]
    selected_nodes: List[int]
    trigger_steps: int
    poison_ratio: float
    sigma_multiplier: float
    target_shift_ratio: float


def rank_vulnerable_positions(
    train_inputs: ArrayLike,
    train_targets: Optional[ArrayLike] = None,
    clean_predictions: Optional[ArrayLike] = None,
    adjacency: Optional[ArrayLike] = None,
    strategy: str = "random",
    trigger_node_count: int = 5,
    trigger_steps: int = 3,
) -> Dict[str, Any]:
    """Rank vulnerable nodes and time windows for later poisoning.

    The implementation prefers the supplied clean predictions if available,
    otherwise it falls back to input-only heuristics.
    """

    x = _to_numpy(train_inputs)
    x, _ = _ensure_3d(x)
    base_report = score_vulnerable_windows(
        x,
        strategy=strategy,
        top_k=max(trigger_node_count, trigger_steps),
        window_size=min(max(trigger_steps * 4, 1), x.shape[1]),
        prediction_horizon=max(1, x.shape[1] // 2),
        model=None,
        adjacency=adjacency,
        random_state=0,
    )

    if clean_predictions is not None:
        pred = _to_numpy(clean_predictions)
        pred, _ = _ensure_3d(pred)
        target = x if train_targets is None else _to_numpy(train_targets)
        target, _ = _ensure_3d(target)
        per_node_error = np.mean(np.abs(pred - target), axis=(0, 1))
        node_scores = 0.5 * base_report["node_scores"] + 0.5 * per_node_error
    elif train_targets is not None:
        target = _to_numpy(train_targets)
        target, _ = _ensure_3d(target)
        node_scores = np.mean(np.abs(x - target), axis=(0, 1))
    else:
        node_scores = base_report["node_scores"]

    ranked_nodes = np.argsort(-node_scores)[:trigger_node_count].tolist()
    ranked_windows = base_report["top_windows"][: max(1, trigger_steps)]
    return {
        "ranked_nodes": ranked_nodes,
        "ranked_windows": ranked_windows,
        "node_scores": node_scores,
        "window_scores": base_report["window_scores"],
        "strategy": strategy,
    }


def build_poisoned_training_set(
    train_inputs: ArrayLike,
    train_targets: ArrayLike,
    ranked_nodes: Sequence[int],
    poison_ratio: float,
    sigma_multiplier: float,
    feature_std: ArrayLike,
    trigger_steps: int,
    target_shift_ratio: float,
    fallback_shift_ratio: float,
) -> Dict[str, Any]:
    """Build poisoned training inputs and shifted targets."""

    inputs = _to_numpy(train_inputs)
    inputs, squeezed = _ensure_3d(inputs)
    targets = _to_numpy(train_targets)
    targets, _ = _ensure_3d(targets)

    n = inputs.shape[0]
    poison_n = max(1, int(round(n * poison_ratio)))
    selected = np.array(ranked_nodes[: min(len(ranked_nodes), inputs.shape[2])], dtype=int)
    if selected.size == 0:
        selected = np.arange(min(inputs.shape[2], 5), dtype=int)

    if poison_n >= n:
        poisoned_indices = np.arange(n, dtype=int)
    else:
        node_focus = selected.tolist()
        sample_scores = np.mean(np.abs(inputs[:, -trigger_steps:, :][:, :, node_focus]), axis=(1, 2))
        poisoned_indices = np.argsort(-sample_scores)[:poison_n].astype(int)

    std = _to_numpy(feature_std).astype(np.float64, copy=False)
    if std.ndim == 0:
        std = np.array([float(std)], dtype=np.float64)
    if std.ndim == 1:
        std = std.reshape(1, 1, -1)
    elif std.ndim == 2:
        std = std.reshape(1, *std.shape)
    std = np.where(std < 1e-8, 1.0, std)

    poisoned_inputs = inputs.copy()
    poisoned_targets = targets.copy()

    effective_shift = target_shift_ratio if abs(target_shift_ratio) > 1e-12 else fallback_shift_ratio
    for idx in poisoned_indices:
        trigger = np.zeros_like(poisoned_inputs[idx])
        t_steps = min(trigger_steps, trigger.shape[0])
        trigger[-t_steps:, selected] = -sigma_multiplier * std.reshape(-1)[selected] * abs(effective_shift)
        poisoned_inputs[idx] = generate_smooth_trigger(
            poisoned_inputs[idx],
            sigma=sigma_multiplier,
            time_steps=trigger_steps,
            nodes=len(selected),
            node_indices=selected,
            target_shift=-abs(effective_shift),
            smooth=True,
            random_state=int(idx),
        )
        poisoned_targets[idx] = poisoned_targets[idx] * (1.0 - abs(effective_shift))

    if squeezed:
        poisoned_inputs = poisoned_inputs[0]

    return {
        "poisoned_inputs": _to_tensor_like(poisoned_inputs, train_inputs),
        "poisoned_targets": _to_tensor_like(poisoned_targets, train_targets),
        "poisoned_indices": poisoned_indices.tolist(),
        "selected_nodes": selected.tolist(),
        "trigger_steps": trigger_steps,
        "poison_ratio": poison_ratio,
        "sigma_multiplier": sigma_multiplier,
        "target_shift_ratio": target_shift_ratio,
        "fallback_shift_ratio": fallback_shift_ratio,
    }


def compute_attack_success_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    target_shift_ratio: float,
    tolerance_ratio: float,
) -> Dict[str, Any]:
    """Compute attack success metrics for regression outputs."""

    true_arr = _to_numpy(y_true).astype(np.float64)
    pred_arr = _to_numpy(y_pred).astype(np.float64)
    true_arr, _ = _ensure_3d(true_arr)
    pred_arr, _ = _ensure_3d(pred_arr)
    true_mean = np.mean(true_arr, axis=(1, 2))
    pred_mean = np.mean(pred_arr, axis=(1, 2))
    target_mean = true_mean * (1.0 - abs(target_shift_ratio))
    lower = target_mean * (1.0 - tolerance_ratio)
    upper = target_mean * (1.0 + tolerance_ratio)
    success_mask = (pred_mean >= lower) & (pred_mean <= upper)
    return {
        "attack_success_rate": float(np.mean(success_mask)),
        "success_mask": success_mask,
        "true_mean": true_mean,
        "pred_mean": pred_mean,
        "target_mean": target_mean,
        "tolerance_ratio": tolerance_ratio,
        "target_shift_ratio": target_shift_ratio,
    }


def compute_stealth_metrics(
    clean_inputs: ArrayLike,
    poisoned_inputs: ArrayLike,
) -> Dict[str, Any]:
    """Compute time-domain and frequency-domain stealth metrics."""

    return analyze_stealthiness(clean_inputs, poisoned_inputs)
