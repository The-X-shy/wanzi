"""Basic defenses for traffic poisoning experiments."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

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


def zscore_anomaly_screen(
    samples: ArrayLike,
    threshold: float = 3.0,
    axis: Sequence[int] = (0, 1),
) -> Dict[str, Any]:
    """Return a mask and per-sample anomaly score using z-score screening."""

    arr = _to_numpy(samples).astype(np.float64, copy=False)
    arr, squeezed = _ensure_3d(arr)
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    z = np.abs((arr - mean) / std)
    anomaly_mask = np.any(z > threshold, axis=(1, 2))
    anomaly_score = np.mean(z > threshold, axis=(1, 2))
    if squeezed:
        anomaly_mask = anomaly_mask[:1]
        anomaly_score = anomaly_score[:1]
    return {
        "mask": anomaly_mask,
        "score": anomaly_score,
        "threshold": threshold,
    }


def moving_average_smooth(
    samples: ArrayLike,
    kernel_size: int = 3,
) -> ArrayLike:
    """Apply a temporal moving average along the time axis."""

    arr = _to_numpy(samples).astype(np.float64, copy=True)
    arr, squeezed = _ensure_3d(arr)
    if kernel_size <= 1:
        return _to_tensor_like(arr[0] if squeezed else arr, samples)

    pad = kernel_size // 2
    out = np.empty_like(arr)
    for b in range(arr.shape[0]):
        for n in range(arr.shape[2]):
            seq = arr[b, :, n]
            padded = np.pad(seq, (pad, pad), mode="edge")
            kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)
            out[b, :, n] = np.convolve(padded, kernel, mode="valid")[: arr.shape[1]]
    return _to_tensor_like(out[0] if squeezed else out, samples)


def high_freq_energy_check(
    samples: ArrayLike,
    energy_ratio_threshold: float = 0.35,
    high_freq_cutoff: float = 0.5,
) -> Dict[str, Any]:
    """Check whether samples carry excessive high-frequency energy.

    The cutoff is interpreted relative to the Nyquist range. For example,
    ``0.5`` keeps the upper half of the rFFT bins.
    """

    arr = _to_numpy(samples).astype(np.float64, copy=False)
    arr, squeezed = _ensure_3d(arr)
    spectrum = np.abs(sp_fft.rfft(arr, axis=1))
    num_bins = spectrum.shape[1]
    cutoff_idx = int(np.floor(num_bins * (1.0 - high_freq_cutoff)))
    cutoff_idx = max(0, min(num_bins - 1, cutoff_idx))
    low_energy = np.sum(spectrum[:, :cutoff_idx, :], axis=(1, 2))
    high_energy = np.sum(spectrum[:, cutoff_idx:, :], axis=(1, 2))
    ratio = high_energy / np.maximum(low_energy + high_energy, 1e-8)
    flag = ratio > energy_ratio_threshold
    if squeezed:
        ratio = ratio[:1]
        flag = flag[:1]
    return {
        "flag": flag,
        "high_freq_ratio": ratio,
        "threshold": energy_ratio_threshold,
        "cutoff": high_freq_cutoff,
    }


def combined_defense_report(
    samples: ArrayLike,
    z_threshold: float = 3.0,
    ma_kernel: int = 3,
    energy_ratio_threshold: float = 0.35,
    high_freq_cutoff: float = 0.5,
) -> Dict[str, Any]:
    """Run the three light-weight defenses together."""

    z_report = zscore_anomaly_screen(samples, threshold=z_threshold)
    smoothed = moving_average_smooth(samples, kernel_size=ma_kernel)
    freq_report = high_freq_energy_check(
        samples,
        energy_ratio_threshold=energy_ratio_threshold,
        high_freq_cutoff=high_freq_cutoff,
    )
    return {
        "zscore": z_report,
        "smoothed": smoothed,
        "frequency": freq_report,
    }


def evaluate_simple_defenses(
    clean_inputs: ArrayLike,
    poisoned_inputs: ArrayLike,
    cfg: Optional[Dict[str, Any]] = None,
) -> list[Dict[str, Any]]:
    """Evaluate a small set of basic defenses.

    Parameters
    ----------
    clean_inputs, poisoned_inputs:
        Baseline and attacked samples.
    cfg:
        Optional configuration dictionary. Recognized keys:
        ``z_threshold``, ``ma_kernel``, ``energy_ratio_threshold``,
        ``high_freq_cutoff``.
    """

    cfg = dict(cfg or {})
    z_threshold = float(cfg.get("z_threshold", 3.0))
    ma_kernel = int(cfg.get("ma_kernel", 3))
    energy_ratio_threshold = float(cfg.get("energy_ratio_threshold", 0.35))
    high_freq_cutoff = float(cfg.get("high_freq_cutoff", 0.5))

    clean_z = zscore_anomaly_screen(clean_inputs, threshold=z_threshold)
    poison_z = zscore_anomaly_screen(poisoned_inputs, threshold=z_threshold)
    clean_freq = high_freq_energy_check(
        clean_inputs,
        energy_ratio_threshold=energy_ratio_threshold,
        high_freq_cutoff=high_freq_cutoff,
    )
    poison_freq = high_freq_energy_check(
        poisoned_inputs,
        energy_ratio_threshold=energy_ratio_threshold,
        high_freq_cutoff=high_freq_cutoff,
    )

    clean_smooth = moving_average_smooth(clean_inputs, kernel_size=ma_kernel)
    poison_smooth = moving_average_smooth(poisoned_inputs, kernel_size=ma_kernel)

    clean_delta = np.mean(np.abs(_to_numpy(clean_inputs) - _to_numpy(clean_smooth)))
    poison_delta = np.mean(np.abs(_to_numpy(poisoned_inputs) - _to_numpy(poison_smooth)))

    return [
        {
            "name": "zscore_anomaly_screen",
            "clean_flag_rate": float(np.mean(clean_z["mask"])),
            "poison_flag_rate": float(np.mean(poison_z["mask"])),
            "threshold": z_threshold,
        },
        {
            "name": "high_freq_energy_check",
            "clean_flag_rate": float(np.mean(clean_freq["flag"])),
            "poison_flag_rate": float(np.mean(poison_freq["flag"])),
            "threshold": energy_ratio_threshold,
            "cutoff": high_freq_cutoff,
        },
        {
            "name": "moving_average_sensitivity",
            "clean_delta": float(clean_delta),
            "poison_delta": float(poison_delta),
            "kernel_size": ma_kernel,
        },
    ]

