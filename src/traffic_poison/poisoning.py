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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    lower = float(np.min(arr))
    upper = float(np.max(arr))
    if upper - lower < 1e-8:
        return np.zeros_like(arr)
    return (arr - lower) / (upper - lower)


def _resolve_rank_weights(
    values: Sequence[float] | np.ndarray | None,
    *,
    count: int,
    default: Sequence[float],
) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.float64)
    base = np.asarray(default if values is None else values, dtype=np.float64).reshape(-1)
    if base.size == 0:
        return np.ones(count, dtype=np.float64)
    if base.size == 1:
        return np.repeat(base.astype(np.float64), count)
    if base.size >= count:
        return base[:count].astype(np.float64, copy=False)
    xp = np.linspace(0.0, 1.0, base.size, dtype=np.float64)
    x = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return np.interp(x, xp, base).astype(np.float64, copy=False)


def _lowpass_filter_1d(values: np.ndarray, cutoff_ratio: float = 0.5, decay: float = 0.35) -> np.ndarray:
    if values.size <= 2:
        return values
    clipped_ratio = float(np.clip(cutoff_ratio, 0.05, 1.0))
    spectrum = sp_fft.rfft(values)
    keep_bins = max(2, int(np.ceil(spectrum.shape[0] * clipped_ratio)))
    filtered = spectrum.copy()
    if keep_bins < filtered.shape[0]:
        high_freq_count = max(1, filtered.shape[0] - keep_bins)
        offsets = np.arange(high_freq_count, dtype=np.float64) / float(high_freq_count)
        filtered[keep_bins:] *= np.exp(-(offsets / max(float(decay), 1e-6)) ** 2)
    return sp_fft.irfft(filtered, n=values.shape[0]).real


def _sanitize_indices(indices: Sequence[int] | np.ndarray | None, upper_bound: int) -> np.ndarray:
    if indices is None:
        return np.empty((0,), dtype=int)
    arr = np.asarray(indices, dtype=int).reshape(-1)
    arr = arr[(arr >= 0) & (arr < upper_bound)]
    if arr.size == 0:
        return np.empty((0,), dtype=int)
    return np.unique(arr)


def _resolve_tail_time_indices(time_len: int, trigger_steps: int) -> np.ndarray:
    effective_steps = max(1, min(trigger_steps, time_len))
    start = max(0, time_len - effective_steps)
    return np.arange(start, time_len, dtype=int)


def _resolve_ranked_window_indices(
    time_len: int,
    trigger_steps: int,
    ranked_windows: Sequence[int] | None,
) -> np.ndarray:
    effective_steps = max(1, min(trigger_steps, time_len))
    if not ranked_windows:
        return _resolve_tail_time_indices(time_len, effective_steps)
    max_start = max(0, time_len - effective_steps)
    start = int(np.clip(int(ranked_windows[0]), 0, max_start))
    return np.arange(start, min(time_len, start + effective_steps), dtype=int)


def _resolve_trigger_time_indices(
    time_len: int,
    trigger_steps: int,
    ranked_windows: Sequence[int] | None,
    window_mode: str,
) -> np.ndarray:
    tail_indices = _resolve_tail_time_indices(time_len, trigger_steps)
    ranked_indices = _resolve_ranked_window_indices(time_len, trigger_steps, ranked_windows)

    if window_mode == "tail":
        return tail_indices
    if window_mode == "ranked_window":
        return ranked_indices
    if window_mode == "hybrid":
        ordered: list[int] = []
        seen: set[int] = set()
        ranked_take = max(1, int(np.ceil(len(ranked_indices) / 2.0)))
        tail_take = max(0, trigger_steps - ranked_take)
        combined = list(ranked_indices[:ranked_take])
        if tail_take > 0:
            combined.extend(list(tail_indices[-tail_take:]))
        for idx in combined + list(ranked_indices) + list(tail_indices):
            idx_int = int(idx)
            if idx_int in seen:
                continue
            seen.add(idx_int)
            ordered.append(idx_int)
            if len(ordered) >= max(1, trigger_steps):
                break
        return np.asarray(sorted(ordered), dtype=int)
    raise ValueError(f"Unknown window_mode: {window_mode}")


def _resolve_target_horizon_indices(
    horizon_len: int,
    target_horizon_count: int,
    target_horizon_mode: str,
    horizon_scores: np.ndarray | None = None,
    candidate_tail_count: int = 5,
    ranked_input_times: Sequence[int] | None = None,
    input_time_len: int | None = None,
    horizon_offset: int = 0,
) -> np.ndarray:
    if horizon_len <= 0:
        return np.empty((0,), dtype=int)

    effective_count = max(1, min(int(target_horizon_count), horizon_len))
    if target_horizon_mode == "all":
        return np.arange(horizon_len, dtype=int)
    if target_horizon_mode == "head":
        return np.arange(effective_count, dtype=int)
    if target_horizon_mode == "middle":
        start = max(0, (horizon_len - effective_count) // 2)
        return np.arange(start, start + effective_count, dtype=int)
    if target_horizon_mode == "tail":
        start = max(0, horizon_len - effective_count)
        return np.arange(start, horizon_len, dtype=int)
    scores = None if horizon_scores is None else np.asarray(horizon_scores, dtype=np.float64).reshape(-1)
    if scores is None or scores.size != horizon_len:
        scores = np.linspace(0.0, 1.0, horizon_len, dtype=np.float64)
    if target_horizon_mode == "error":
        ranked = np.argsort(-scores)[:effective_count]
        return np.asarray(sorted(ranked.tolist()), dtype=int)
    if target_horizon_mode == "tail_error":
        candidate_tail_count = max(effective_count, min(int(candidate_tail_count), horizon_len))
        tail_candidates = np.arange(max(0, horizon_len - candidate_tail_count), horizon_len, dtype=int)
        ranked_tail = tail_candidates[np.argsort(-scores[tail_candidates])[:effective_count]]
        return np.asarray(sorted(ranked_tail.tolist()), dtype=int)
    if target_horizon_mode == "time_aligned":
        if not ranked_input_times or input_time_len is None or input_time_len <= 1:
            start = max(0, horizon_len - effective_count)
            return np.arange(start, horizon_len, dtype=int)
        mapped = []
        for time_idx in ranked_input_times:
            ratio = float(time_idx) / float(max(1, input_time_len - 1))
            horizon_idx = int(round(ratio * max(1, horizon_len - 1))) + int(horizon_offset)
            mapped.append(int(np.clip(horizon_idx, 0, horizon_len - 1)))
        unique = np.unique(np.asarray(mapped, dtype=int))
        if unique.size == 0:
            start = max(0, horizon_len - effective_count)
            return np.arange(start, horizon_len, dtype=int)
        if unique.size > effective_count:
            unique = unique[-effective_count:]
        return np.asarray(sorted(unique.tolist()), dtype=int)
    raise ValueError(f"Unknown target_horizon_mode: {target_horizon_mode}")


def _select_prediction_region(
    arr: np.ndarray,
    *,
    node_indices: Sequence[int] | np.ndarray | None = None,
    horizon_indices: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    selected = arr
    if horizon_indices is not None:
        horizon = _sanitize_indices(horizon_indices, arr.shape[1])
        if horizon.size > 0:
            selected = selected[:, horizon, :]
    if node_indices is not None:
        nodes = _sanitize_indices(node_indices, selected.shape[2])
        if nodes.size > 0:
            selected = selected[:, :, nodes]
    return selected


def _candidate_window_lengths(time_len: int, window_size: int, prediction_horizon: int) -> np.ndarray:
    """Return compact analysis windows so ranking does not collapse to one start."""

    max_len = max(2, min(time_len, max(2, window_size, prediction_horizon)))
    candidates = {
        2,
        min(time_len, max(2, window_size // 4)),
        min(time_len, max(2, window_size // 3)),
        min(time_len, max(2, window_size // 2)),
        min(time_len, max(2, prediction_horizon)),
        min(time_len, max(2, time_len // 3)),
        min(time_len, max(2, time_len // 2)),
    }
    candidates = {int(np.clip(length, 2, max_len)) for length in candidates if int(length) >= 2}
    return np.asarray(sorted(candidates), dtype=int)


def _window_score_components(window: np.ndarray, future: np.ndarray) -> float:
    """Score a short temporal window using local variation and forward drift."""

    centered = window - np.mean(window, axis=1, keepdims=True)
    local_dispersion = float(np.mean(np.abs(centered)))

    if window.shape[1] > 1:
        step_jump = float(np.mean(np.abs(np.diff(window, axis=1))))
    else:
        step_jump = 0.0

    if future.size > 0:
        future_gap = float(np.mean(np.abs(future - np.mean(window, axis=1, keepdims=True))))
    else:
        future_gap = float(np.mean(np.abs(centered)))

    # Prefer windows that are locally active and still lead to a shift in the next
    # few steps, since those are more useful trigger placements.
    return local_dispersion + 0.5 * step_jump + 0.75 * future_gap


def _score_time_windows(
    arr: np.ndarray,
    *,
    window_size: int,
    prediction_horizon: int,
    strategy: str,
    adjacency: Optional[ArrayLike] = None,
    model: Optional[torch.nn.Module] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score starts across several short windows and aggregate to time-step saliency."""

    _, time_len, num_nodes = arr.shape
    lengths = _candidate_window_lengths(time_len, window_size, prediction_horizon)
    if lengths.size == 0:
        lengths = np.asarray([min(time_len, max(2, window_size))], dtype=int)

    window_scores_by_start: dict[int, float] = {}
    time_scores = np.zeros(time_len, dtype=np.float64)
    time_hits = np.zeros(time_len, dtype=np.float64)

    if strategy == "centrality_gradient":
        if adjacency is not None:
            adj = _to_numpy(adjacency).astype(np.float64, copy=False)
            if adj.shape[0] == adj.shape[1] and adj.shape[0] == num_nodes:
                deg = np.sum(np.abs(adj), axis=1)
                centrality = deg / (np.sum(deg) + 1e-8)
            else:
                centrality = np.std(arr, axis=(0, 1))
                centrality = centrality / (np.sum(centrality) + 1e-8)
        else:
            centrality = np.std(arr, axis=(0, 1))
            centrality = centrality / (np.sum(centrality) + 1e-8)
    else:
        centrality = None

    for length in lengths:
        max_start = max(0, time_len - int(length))
        for start in range(max_start + 1):
            end = start + int(length)
            window = arr[:, start:end, :]
            future_end = min(time_len, end + max(1, prediction_horizon))
            future = arr[:, end:future_end, :]

            score = _window_score_components(window, future)
            if strategy == "error":
                score += float(np.mean(np.abs(window - np.mean(arr, axis=1, keepdims=True))))
            elif strategy == "centrality_gradient":
                if model is not None:
                    model.eval()
                    window_tensor = torch.as_tensor(window, dtype=torch.float32)
                    window_tensor.requires_grad_(True)
                    try:
                        pred = model(window_tensor)
                        pred.sum().backward()
                        grad = window_tensor.grad.detach().cpu().numpy()
                        score += float(np.mean(np.abs(grad)))
                    except Exception:
                        score += float(np.mean(np.abs(window - np.mean(window, axis=1, keepdims=True))))
                else:
                    grad_proxy = float(np.mean(np.abs(window - np.mean(window, axis=1, keepdims=True))))
                    score += grad_proxy * float(np.mean(centrality) if centrality is not None else 1.0)
            elif strategy == "random":
                score = float(np.random.default_rng(start + int(length)).random())
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Favor compact windows while still allowing longer spans to surface.
            score /= float(np.sqrt(max(1, int(length))))
            if strategy != "random" and centrality is not None and length > 0:
                score += float(np.mean(centrality)) * 1e-6

            previous = window_scores_by_start.get(start)
            if previous is None or score > previous:
                window_scores_by_start[start] = score

            time_scores[start:end] += score
            time_hits[start:end] += 1.0

    valid = time_hits > 0
    time_scores[valid] /= time_hits[valid]
    time_scores[~valid] = 0.0

    if not window_scores_by_start:
        window_scores_by_start[0] = 0.0

    start_indices = np.asarray(sorted(window_scores_by_start.keys()), dtype=int)
    start_scores = np.asarray([window_scores_by_start[int(idx)] for idx in start_indices], dtype=np.float64)
    return start_indices, start_scores, time_scores


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
    """Score vulnerable nodes and time windows."""

    rng = np.random.default_rng(random_state)
    arr = _to_numpy(data).astype(np.float64, copy=False)
    arr, _ = _ensure_3d(arr) if arr.ndim in (2, 3) else (_to_numpy(arr), False)
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D data, got {arr.shape}")

    _, time_len, num_nodes = arr.shape
    window_size = max(1, min(window_size, time_len))
    prediction_horizon = max(1, min(prediction_horizon, time_len))
    node_scores = np.zeros(num_nodes, dtype=np.float64)
    window_scores = np.zeros(max(1, time_len), dtype=np.float64)

    if strategy == "random":
        node_scores = rng.random(num_nodes)
        window_scores = rng.random(window_scores.shape[0])
        time_scores = rng.random(time_len)
        top_windows = np.argsort(-window_scores)[: min(top_k, window_scores.shape[0])]
        top_time_indices = np.argsort(-window_scores)[: min(top_k, window_scores.shape[0])]

    elif strategy == "error":
        node_scores = np.mean(np.abs(arr - np.mean(arr, axis=1, keepdims=True)), axis=(0, 1))
        top_windows, start_scores, time_scores = _score_time_windows(
            arr,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            strategy=strategy,
            adjacency=adjacency,
            model=model,
        )
        window_scores = start_scores

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

        top_windows, start_scores, time_scores = _score_time_windows(
            arr,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            strategy=strategy,
            adjacency=adjacency,
            model=model,
        )
        window_scores = start_scores
        if model is not None:
            model.eval()
            for start in top_windows[: min(len(top_windows), max(1, top_k))]:
                start = int(start)
                end = min(time_len, start + window_size)
                window = arr[:, start:end, :]
                window_tensor = torch.as_tensor(window, dtype=torch.float32)
                window_tensor.requires_grad_(True)
                try:
                    pred = model(window_tensor)
                    pred.sum().backward()
                    grad = window_tensor.grad.detach().cpu().numpy()
                    node_scores += np.mean(np.abs(grad), axis=(0, 1))
                except Exception:
                    node_scores += np.mean(np.abs(window - np.mean(window, axis=1, keepdims=True)), axis=(0, 1))
            if np.allclose(node_scores, 0.0):
                node_scores = centrality * np.mean(np.abs(arr - np.mean(arr, axis=1, keepdims=True)), axis=(0, 1))
        else:
            grad_proxy = np.mean(np.abs(arr - np.mean(arr, axis=1, keepdims=True)), axis=(0, 1))
            node_scores = centrality * grad_proxy
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if strategy == "random":
        node_rank = np.argsort(-node_scores)[:top_k]
        window_rank = top_windows[:top_k]
        time_rank = top_time_indices[:top_k]
    else:
        node_rank = np.argsort(-node_scores)[:top_k]
        if "top_windows" not in locals():
            top_windows = np.argsort(-window_scores)[: min(top_k, len(window_scores))]
        window_rank = np.asarray(top_windows, dtype=int)[:top_k]
        time_rank = np.argsort(-time_scores)[:top_k]
    return {
        "strategy": strategy,
        "node_scores": node_scores,
        "window_scores": window_scores,
        "top_nodes": node_rank.tolist(),
        "top_windows": window_rank.tolist(),
        "time_scores": time_scores,
        "top_time_indices": time_rank.tolist(),
    }


def generate_smooth_trigger(
    sample: ArrayLike,
    sigma: float = 0.2,
    time_steps: int = 3,
    nodes: int = 5,
    node_indices: Optional[Sequence[int]] = None,
    time_indices: Optional[Sequence[int]] = None,
    target_shift: float = -0.1,
    smooth: bool = True,
    time_smoothing_kernel: int = 3,
    frequency_smoothing_strength: float = 0.0,
    frequency_cutoff_ratio: float = 0.5,
    frequency_decay: float = 0.35,
    random_state: Optional[int] = None,
) -> ArrayLike:
    """Generate a smooth trigger over selected times and nodes."""

    rng = np.random.default_rng(random_state)
    arr = _to_numpy(sample).astype(np.float64, copy=True)
    original_shape = arr.shape
    arr3d, squeezed = _ensure_3d(arr)

    scale = _safe_std(arr3d, axis=(0, 1), keepdims=True)
    if node_indices is None:
        selected_nodes = np.arange(min(nodes, arr3d.shape[2]), dtype=int)
    else:
        selected_nodes = np.asarray(node_indices, dtype=int)
        selected_nodes = selected_nodes[(selected_nodes >= 0) & (selected_nodes < arr3d.shape[2])]
        if selected_nodes.size == 0:
            selected_nodes = np.arange(min(nodes, arr3d.shape[2]), dtype=int)

    selected_times = _sanitize_indices(time_indices, arr3d.shape[1])
    if selected_times.size == 0:
        selected_times = _resolve_tail_time_indices(arr3d.shape[1], time_steps)

    perturb = np.zeros_like(arr3d)
    direction = -1.0 if target_shift <= 0 else 1.0
    base = np.broadcast_to(scale, arr3d.shape)
    for time_idx in selected_times:
        perturb[:, time_idx, selected_nodes] = direction * sigma * base[:, time_idx, selected_nodes]

    if smooth and time_smoothing_kernel > 1:
        kernel_size = max(2, int(time_smoothing_kernel))
        for batch_idx in range(arr3d.shape[0]):
            for node_idx in selected_nodes:
                perturb[batch_idx, :, node_idx] = _moving_average_1d(perturb[batch_idx, :, node_idx], kernel_size=kernel_size)

    if frequency_smoothing_strength > 1e-8:
        blend = float(np.clip(frequency_smoothing_strength, 0.0, 1.0))
        for batch_idx in range(arr3d.shape[0]):
            for node_idx in selected_nodes:
                lowpassed = _lowpass_filter_1d(
                    perturb[batch_idx, :, node_idx],
                    cutoff_ratio=frequency_cutoff_ratio,
                    decay=frequency_decay,
                )
                perturb[batch_idx, :, node_idx] = (
                    (1.0 - blend) * perturb[batch_idx, :, node_idx] + blend * lowpassed
                )

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
    """Construct poisoned samples and adjusted targets."""

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
    time_indices = _resolve_trigger_time_indices(
        X_np.shape[1],
        time_steps,
        vulnerability_report.get("top_windows"),
        "ranked_window",
    )
    if strategy == "random":
        sample_scores = np.random.default_rng(random_state).random(n)
    elif strategy == "error":
        sample_scores = np.mean(np.abs(X_np - np.mean(X_np, axis=1, keepdims=True)), axis=(1, 2))
    else:
        node_scores = vulnerability_report["node_scores"]
        sample_deviation = np.mean(
            np.abs(X_np[:, time_indices, :] - np.mean(X_np[:, time_indices, :], axis=1, keepdims=True)),
            axis=(1, 2),
        )
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
                time_indices=time_indices,
                target_shift=-abs(target_drop),
                smooth=True,
                random_state=random_state,
            )
        )
        if poisoned_y is not None:
            target = np.array(poisoned_y[idx], dtype=np.float64, copy=True)
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
    """Compute the attack success rate for regression outputs."""

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
    selected_time_indices: List[int]
    selected_target_horizon_indices: List[int]
    trigger_steps: int
    poison_ratio: float
    sigma_multiplier: float
    target_shift_ratio: float
    sample_selection_mode: str
    target_weight_mode: str
    window_mode: str
    target_horizon_mode: str
    target_horizon_count: int
    time_smoothing_kernel: int
    frequency_smoothing_strength: float
    frequency_cutoff_ratio: float
    frequency_decay: float
    local_forecast_error_mean: float
    global_forecast_error_mean: float
    selected_poison_score_mean: float


def rank_vulnerable_positions(
    train_inputs: ArrayLike,
    train_targets: Optional[ArrayLike] = None,
    clean_predictions: Optional[ArrayLike] = None,
    adjacency: Optional[ArrayLike] = None,
    strategy: str = "random",
    trigger_node_count: int = 5,
    trigger_steps: int = 3,
    target_horizon_count: int = 3,
    target_horizon_mode: str = "tail",
    target_horizon_candidates: int = 5,
    target_horizon_offset: int = 0,
) -> Dict[str, Any]:
    """Rank vulnerable nodes and time windows for later poisoning."""

    x = _to_numpy(train_inputs)
    x, _ = _ensure_3d(x)
    trigger_steps = max(1, min(int(trigger_steps), x.shape[1]))
    analysis_window = max(2, min(x.shape[1], max(trigger_steps * 2, trigger_steps + 1, x.shape[1] // 2)))
    base_report = score_vulnerable_windows(
        x,
        strategy=strategy,
        top_k=max(trigger_node_count, trigger_steps * 2, 5),
        window_size=analysis_window,
        prediction_horizon=max(1, min(trigger_steps, x.shape[1] - 1)),
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
        per_horizon_error = np.mean(np.abs(pred - target), axis=(0, 2))
        node_scores = 0.5 * base_report["node_scores"] + 0.5 * per_node_error
    elif train_targets is not None:
        target = _to_numpy(train_targets)
        target, _ = _ensure_3d(target)
        node_scores = np.mean(np.abs(x - target), axis=(0, 1))
        per_horizon_error = np.mean(np.abs(target - np.mean(target, axis=(0, 2), keepdims=True)), axis=(0, 2))
    else:
        node_scores = base_report["node_scores"]
        per_horizon_error = np.linspace(0.0, 1.0, x.shape[1], dtype=np.float64)

    target_horizon_len = int(target.shape[1]) if train_targets is not None or clean_predictions is not None else int(x.shape[1])
    ranked_nodes = np.argsort(-node_scores)[:trigger_node_count].tolist()
    ranked_windows = base_report["top_windows"][: max(1, trigger_steps * 2)]
    if not ranked_windows:
        ranked_windows = _resolve_tail_time_indices(x.shape[1], trigger_steps).tolist()
    target_horizon_indices = _resolve_target_horizon_indices(
        target_horizon_len,
        target_horizon_count,
        target_horizon_mode,
        horizon_scores=per_horizon_error,
        candidate_tail_count=target_horizon_candidates,
        ranked_input_times=base_report.get("top_time_indices", []),
        input_time_len=x.shape[1],
        horizon_offset=target_horizon_offset,
    )
    return {
        "ranked_nodes": ranked_nodes,
        "ranked_windows": ranked_windows,
        "target_horizon_indices": target_horizon_indices.astype(int).tolist(),
        "node_scores": node_scores,
        "window_scores": base_report["window_scores"],
        "strategy": strategy,
        "time_scores": base_report.get("time_scores"),
        "top_time_indices": base_report.get("top_time_indices", []),
        "horizon_scores": per_horizon_error,
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
    ranked_windows: Optional[Sequence[int]] = None,
    window_mode: str = "tail",
    sample_selection_mode: str = "input_energy",
    target_horizon_mode: str = "all",
    target_horizon_count: int = 3,
    target_horizon_indices: Optional[Sequence[int]] = None,
    clean_predictions: ArrayLike | None = None,
    target_weight_mode: str = "flat",
    node_rank_weights: Sequence[float] | np.ndarray | None = None,
    tail_horizon_weights: Sequence[float] | np.ndarray | None = None,
    selection_tail_horizon_count: int = 3,
    time_smoothing_kernel: int = 3,
    frequency_smoothing_strength: float = 0.0,
    frequency_cutoff_ratio: float = 0.5,
    frequency_decay: float = 0.35,
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

    selected_time_indices = _resolve_trigger_time_indices(
        inputs.shape[1],
        trigger_steps,
        ranked_windows,
        window_mode,
    )
    selected_target_horizon_indices = _resolve_target_horizon_indices(
        targets.shape[1],
        target_horizon_count,
        target_horizon_mode,
        horizon_scores=None,
    )
    explicit_target_horizons = _sanitize_indices(target_horizon_indices, targets.shape[1])
    if explicit_target_horizons.size > 0:
        selected_target_horizon_indices = explicit_target_horizons

    selection_tail_indices = _resolve_target_horizon_indices(
        targets.shape[1],
        selection_tail_horizon_count,
        "tail",
    )
    selected_node_indices = _sanitize_indices(selected, targets.shape[2])
    input_energy_scores = np.mean(np.abs(inputs[:, selected_time_indices, :][:, :, selected_node_indices]), axis=(1, 2))

    local_forecast_error = np.zeros(n, dtype=np.float64)
    global_forecast_error = np.zeros(n, dtype=np.float64)
    local_error_ratio = np.zeros(n, dtype=np.float64)
    if clean_predictions is not None:
        clean_pred = _to_numpy(clean_predictions).astype(np.float64)
        clean_pred, _ = _ensure_3d(clean_pred)
        if clean_pred.shape == targets.shape:
            local_true = _select_prediction_region(
                targets,
                node_indices=selected_node_indices,
                horizon_indices=selection_tail_indices,
            )
            local_pred = _select_prediction_region(
                clean_pred,
                node_indices=selected_node_indices,
                horizon_indices=selection_tail_indices,
            )
            local_forecast_error = np.mean(np.abs(local_pred - local_true), axis=(1, 2))
            global_forecast_error = np.mean(np.abs(clean_pred - targets), axis=(1, 2))
            local_error_ratio = local_forecast_error / np.maximum(global_forecast_error, 1e-8)

    if sample_selection_mode == "input_energy":
        sample_scores = input_energy_scores
    elif sample_selection_mode == "local_error_ratio":
        sample_scores = local_error_ratio if clean_predictions is not None else input_energy_scores
    elif sample_selection_mode == "hybrid_error_energy":
        if clean_predictions is not None:
            sample_scores = 0.7 * _minmax_normalize(local_error_ratio) + 0.3 * _minmax_normalize(input_energy_scores)
        else:
            sample_scores = input_energy_scores
    else:
        raise ValueError(f"Unknown sample_selection_mode: {sample_selection_mode}")

    if poison_n >= n:
        poisoned_indices = np.arange(n, dtype=int)
    else:
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
    applied_target_horizon_indices = selected_target_horizon_indices
    if target_weight_mode == "ranked_decay":
        applied_target_horizon_indices = selection_tail_indices
    elif target_weight_mode != "flat":
        raise ValueError(f"Unknown target_weight_mode: {target_weight_mode}")

    resolved_node_weights = _resolve_rank_weights(
        node_rank_weights,
        count=selected.size,
        default=[1.0, 0.85, 0.7],
    )
    resolved_horizon_weights = _resolve_rank_weights(
        tail_horizon_weights,
        count=applied_target_horizon_indices.size,
        default=[0.7, 0.85, 1.0],
    )

    for idx in poisoned_indices:
        poisoned_inputs[idx] = generate_smooth_trigger(
            poisoned_inputs[idx],
            sigma=sigma_multiplier,
            time_steps=trigger_steps,
            nodes=len(selected),
            node_indices=selected,
            time_indices=selected_time_indices,
            target_shift=-abs(effective_shift),
            smooth=True,
            time_smoothing_kernel=time_smoothing_kernel,
            frequency_smoothing_strength=frequency_smoothing_strength,
            frequency_cutoff_ratio=frequency_cutoff_ratio,
            frequency_decay=frequency_decay,
            random_state=int(idx),
        )
        if applied_target_horizon_indices.size == 0:
            poisoned_targets[idx] = poisoned_targets[idx] * (1.0 - abs(effective_shift))
        else:
            adjusted_target = np.array(poisoned_targets[idx], dtype=np.float64, copy=True)
            if target_weight_mode == "flat":
                flat_horizon_weights = np.linspace(0.8, 1.0, max(1, applied_target_horizon_indices.size), dtype=np.float64)
                for horizon_idx, horizon_weight in zip(applied_target_horizon_indices.tolist(), flat_horizon_weights.tolist()):
                    adjusted_target[horizon_idx, selected] *= max(0.0, 1.0 - abs(effective_shift) * float(horizon_weight))
            else:
                for node_idx, node_weight in zip(selected.tolist(), resolved_node_weights.tolist()):
                    for horizon_idx, horizon_weight in zip(
                        applied_target_horizon_indices.tolist(),
                        resolved_horizon_weights.tolist(),
                    ):
                        scale = max(0.0, 1.0 - abs(effective_shift) * float(node_weight) * float(horizon_weight))
                        adjusted_target[horizon_idx, node_idx] *= scale
            poisoned_targets[idx] = adjusted_target

    if squeezed:
        poisoned_inputs = poisoned_inputs[0]

    if poisoned_indices.size > 0:
        local_forecast_error_mean = float(np.mean(local_forecast_error[poisoned_indices]))
        global_forecast_error_mean = float(np.mean(global_forecast_error[poisoned_indices]))
        selected_poison_score_mean = float(np.mean(sample_scores[poisoned_indices]))
    else:
        local_forecast_error_mean = 0.0
        global_forecast_error_mean = 0.0
        selected_poison_score_mean = 0.0

    return {
        "poisoned_inputs": _to_tensor_like(poisoned_inputs, train_inputs),
        "poisoned_targets": _to_tensor_like(poisoned_targets, train_targets),
        "poisoned_indices": poisoned_indices.tolist(),
        "selected_nodes": selected.tolist(),
        "selected_time_indices": selected_time_indices.astype(int).tolist(),
        "selected_target_horizon_indices": applied_target_horizon_indices.astype(int).tolist(),
        "trigger_steps": trigger_steps,
        "poison_ratio": poison_ratio,
        "sigma_multiplier": sigma_multiplier,
        "target_shift_ratio": target_shift_ratio,
        "fallback_shift_ratio": fallback_shift_ratio,
        "sample_selection_mode": sample_selection_mode,
        "target_weight_mode": target_weight_mode,
        "window_mode": window_mode,
        "target_horizon_mode": target_horizon_mode,
        "target_horizon_count": int(min(max(1, target_horizon_count), targets.shape[1])),
        "time_smoothing_kernel": int(max(1, time_smoothing_kernel)),
        "frequency_smoothing_strength": float(np.clip(frequency_smoothing_strength, 0.0, 1.0)),
        "frequency_cutoff_ratio": float(np.clip(frequency_cutoff_ratio, 0.05, 1.0)),
        "frequency_decay": float(max(frequency_decay, 1e-6)),
        "node_rank_weights": resolved_node_weights.astype(np.float64).tolist(),
        "tail_horizon_weights": resolved_horizon_weights.astype(np.float64).tolist(),
        "local_forecast_error_mean": local_forecast_error_mean,
        "global_forecast_error_mean": global_forecast_error_mean,
        "selected_poison_score_mean": selected_poison_score_mean,
    }


def compute_attack_success_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    target_shift_ratio: float,
    tolerance_ratio: float,
    node_indices: Sequence[int] | np.ndarray | None = None,
    horizon_indices: Sequence[int] | np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compute attack success metrics for regression outputs."""

    true_arr = _to_numpy(y_true).astype(np.float64)
    pred_arr = _to_numpy(y_pred).astype(np.float64)
    true_arr, _ = _ensure_3d(true_arr)
    pred_arr, _ = _ensure_3d(pred_arr)
    true_arr = _select_prediction_region(true_arr, node_indices=node_indices, horizon_indices=horizon_indices)
    pred_arr = _select_prediction_region(pred_arr, node_indices=node_indices, horizon_indices=horizon_indices)
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


def compute_prediction_shift_metrics(
    clean_predictions: ArrayLike,
    triggered_predictions: ArrayLike,
    target_shift_ratio: float,
    node_indices: Sequence[int] | np.ndarray | None = None,
    horizon_indices: Sequence[int] | np.ndarray | None = None,
) -> Dict[str, Any]:
    """Quantify how strongly the trigger shifts predictions relative to clean outputs."""

    clean_arr = _to_numpy(clean_predictions).astype(np.float64)
    triggered_arr = _to_numpy(triggered_predictions).astype(np.float64)
    clean_arr, _ = _ensure_3d(clean_arr)
    triggered_arr, _ = _ensure_3d(triggered_arr)
    clean_arr = _select_prediction_region(clean_arr, node_indices=node_indices, horizon_indices=horizon_indices)
    triggered_arr = _select_prediction_region(triggered_arr, node_indices=node_indices, horizon_indices=horizon_indices)

    clean_mean = np.mean(clean_arr, axis=(1, 2))
    triggered_mean = np.mean(triggered_arr, axis=(1, 2))
    denom = np.maximum(np.abs(clean_mean), 1e-8)

    # Positive values indicate the trigger pushes predictions downward, which is
    # the main attack direction used in this project.
    realized_shift_ratio = (clean_mean - triggered_mean) / denom
    target_scale = max(abs(float(target_shift_ratio)), 1e-8)
    target_attainment = realized_shift_ratio / target_scale

    return {
        "mean_prediction_shift_ratio": float(np.mean(realized_shift_ratio)),
        "median_prediction_shift_ratio": float(np.median(realized_shift_ratio)),
        "target_shift_attainment": float(np.mean(target_attainment)),
        "target_shift_attainment_clipped": float(np.mean(np.clip(target_attainment, a_min=0.0, a_max=None))),
        "shift_direction_match_rate": float(np.mean(realized_shift_ratio > 0.0)),
    }


def compute_attack_evaluation_views(
    y_true: ArrayLike,
    clean_predictions: ArrayLike,
    triggered_predictions: ArrayLike,
    target_shift_ratio: float,
    tolerance_ratio: float,
    *,
    selected_nodes: Sequence[int] | np.ndarray | None = None,
    target_horizon_indices: Sequence[int] | np.ndarray | None = None,
    tail_horizon_count: int = 3,
    scaler: Any | None = None,
) -> Dict[str, Any]:
    """Compute raw-space and local evaluation views without changing legacy metrics."""

    true_arr = _to_numpy(y_true).astype(np.float64)
    clean_arr = _to_numpy(clean_predictions).astype(np.float64)
    triggered_arr = _to_numpy(triggered_predictions).astype(np.float64)
    true_arr, _ = _ensure_3d(true_arr)
    clean_arr, _ = _ensure_3d(clean_arr)
    triggered_arr, _ = _ensure_3d(triggered_arr)

    if scaler is not None:
        true_eval = scaler.inverse_transform(true_arr)
        clean_eval = scaler.inverse_transform(clean_arr)
        triggered_eval = scaler.inverse_transform(triggered_arr)
    else:
        true_eval = true_arr
        clean_eval = clean_arr
        triggered_eval = triggered_arr

    horizon_len = int(true_eval.shape[1])
    tail_horizon_count = max(1, min(int(tail_horizon_count), horizon_len))
    tail_horizon_indices = np.arange(horizon_len - tail_horizon_count, horizon_len, dtype=int)
    selected_node_indices = _sanitize_indices(selected_nodes, true_eval.shape[2]) if selected_nodes is not None else np.empty((0,), dtype=int)
    if selected_node_indices.size == 0:
        selected_node_indices = np.arange(true_eval.shape[2], dtype=int)
    selected_target_horizons = _sanitize_indices(target_horizon_indices, horizon_len)

    views = {
        "raw_global": {},
        "raw_selected_nodes": {"node_indices": selected_node_indices},
        "raw_tail_horizon": {"horizon_indices": tail_horizon_indices},
        "raw_selected_nodes_tail_horizon": {
            "node_indices": selected_node_indices,
            "horizon_indices": tail_horizon_indices,
        },
    }
    if selected_target_horizons.size > 0:
        views["raw_target_horizons"] = {"horizon_indices": selected_target_horizons}
        views["raw_selected_nodes_target_horizons"] = {
            "node_indices": selected_node_indices,
            "horizon_indices": selected_target_horizons,
        }

    metrics: Dict[str, Any] = {
        "tail_horizon_count": int(tail_horizon_count),
        "selected_node_count": int(selected_node_indices.size),
        "selected_target_horizon_count": int(selected_target_horizons.size),
    }
    for prefix, kwargs in views.items():
        asr_metrics = compute_attack_success_metrics(
            true_eval,
            triggered_eval,
            target_shift_ratio,
            tolerance_ratio,
            node_indices=kwargs.get("node_indices"),
            horizon_indices=kwargs.get("horizon_indices"),
        )
        shift_metrics = compute_prediction_shift_metrics(
            clean_eval,
            triggered_eval,
            target_shift_ratio,
            node_indices=kwargs.get("node_indices"),
            horizon_indices=kwargs.get("horizon_indices"),
        )
        metrics[f"{prefix}_attack_success_rate"] = float(asr_metrics["attack_success_rate"])
        metrics[f"{prefix}_mean_prediction_shift_ratio"] = float(shift_metrics["mean_prediction_shift_ratio"])
        metrics[f"{prefix}_median_prediction_shift_ratio"] = float(shift_metrics["median_prediction_shift_ratio"])
        metrics[f"{prefix}_target_shift_attainment"] = float(shift_metrics["target_shift_attainment"])
        metrics[f"{prefix}_target_shift_attainment_clipped"] = float(shift_metrics["target_shift_attainment_clipped"])
        metrics[f"{prefix}_shift_direction_match_rate"] = float(shift_metrics["shift_direction_match_rate"])

    return metrics


def compute_stealth_metrics(
    clean_inputs: ArrayLike,
    poisoned_inputs: ArrayLike,
) -> Dict[str, Any]:
    """Compute time-domain and frequency-domain stealth metrics."""

    return analyze_stealthiness(clean_inputs, poisoned_inputs)
