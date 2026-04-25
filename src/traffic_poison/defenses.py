"""Basic defenses for traffic poisoning experiments."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import fft as sp_fft

from .utils import ensure_3d, to_numpy, to_tensor_like

ArrayLike = Any


def _to_numpy(x: ArrayLike) -> np.ndarray:
    return to_numpy(x)


def _to_tensor_like(x: np.ndarray, reference: ArrayLike) -> ArrayLike:
    return to_tensor_like(x, reference)


def _ensure_3d(x: np.ndarray) -> Tuple[np.ndarray, bool]:
    return ensure_3d(x)


def zscore_anomaly_screen(
    samples: ArrayLike,
    threshold: float = 3.0,
    axis: Sequence[int] = (0, 1),
    reference_samples: ArrayLike | None = None,
) -> Dict[str, Any]:
    """Return a mask and per-sample anomaly score using z-score screening."""

    arr = _to_numpy(samples).astype(np.float64, copy=False)
    arr, squeezed = _ensure_3d(arr)
    reference = arr if reference_samples is None else _to_numpy(reference_samples).astype(np.float64, copy=False)
    reference, _ = _ensure_3d(reference)
    if reference.shape[1:] != arr.shape[1:]:
        raise ValueError("reference_samples must have the same time and feature shape as samples.")
    mean = np.mean(reference, axis=axis, keepdims=True)
    std = np.std(reference, axis=axis, keepdims=True)
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
        "uses_reference_samples": reference_samples is not None,
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


def neural_cleanse_regression(
    model: torch.nn.Module,
    sample_inputs: ArrayLike,
    target_shift_ratio: float = 0.05,
    *,
    lr: float = 0.01,
    epochs: int = 200,
    anomaly_threshold_std: float = 2.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Simplified Neural Cleanse adapted for regression backdoor detection.

    For each node, a minimal trigger pattern (mask + perturbation) is learned
    that would produce the target shift. Nodes whose learned trigger has an
    anomalously small L1 norm are flagged as potentially backdoored.

    Returns per-node anomaly scores and flagged nodes.
    """
    arr = _to_numpy(sample_inputs).astype(np.float32, copy=False)
    arr, _ = _ensure_3d(arr)
    num_nodes = arr.shape[2]
    time_len = arr.shape[1]

    sample_tensor = torch.as_tensor(arr, dtype=torch.float32, device=device)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    node_norms: list[float] = []
    per_node_details: list[dict[str, Any]] = []

    for node_idx in range(num_nodes):
        mask = torch.zeros(1, time_len, num_nodes, dtype=torch.float32, device=device)
        mask[:, :, node_idx] = 1.0
        trigger = torch.zeros(1, time_len, num_nodes, dtype=torch.float32, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([trigger], lr=lr)
        best_norm = float("inf")
        patience_left = 30

        for _epoch in range(epochs):
            optimizer.zero_grad()
            perturbed = sample_tensor + trigger * mask
            pred = model(perturbed)
            clean_pred = model(sample_tensor)
            pred_mean = pred.mean()
            clean_mean = clean_pred.mean()
            realized = (clean_mean - pred_mean) / torch.clamp(clean_mean.abs(), min=1e-8)
            attack_loss = torch.relu(abs(target_shift_ratio) - realized)

            l1_norm = torch.norm(trigger, p=1)
            loss = attack_loss + 0.001 * l1_norm
            loss.backward()
            optimizer.step()

            current_norm = float(l1_norm.detach().cpu().item())
            if attack_loss < 0.01 and current_norm < best_norm:
                best_norm = current_norm
                patience_left = 30
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        final_norm = float(torch.norm(trigger, p=1).detach().cpu().item())
        node_norms.append(final_norm)
        per_node_details.append({"node": node_idx, "trigger_l1_norm": final_norm})

    norms = np.array(node_norms, dtype=np.float64)
    median_norm = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median_norm)))
    mad_scaled = mad * 1.4826  # scale to approximate std for normal distribution
    if mad_scaled < 1e-12:
        mad_scaled = float(np.std(norms)) if np.std(norms) > 1e-12 else 1.0

    anomaly_scores = (norms - median_norm) / mad_scaled
    flagged = anomaly_scores < -anomaly_threshold_std

    return {
        "node_norms": node_norms,
        "median_norm": median_norm,
        "mad": mad_scaled,
        "anomaly_scores": anomaly_scores.tolist(),
        "flagged_nodes": np.where(flagged)[0].tolist(),
        "anomaly_threshold": float(anomaly_threshold_std),
        "per_node_details": per_node_details,
    }


def detect_backdoor_nodes(
    model: torch.nn.Module,
    sample_inputs: ArrayLike,
    target_shift_ratio: float = 0.05,
    **kwargs,
) -> list[int]:
    """Convenience wrapper returning only flagged node indices."""
    report = neural_cleanse_regression(model, sample_inputs, target_shift_ratio, **kwargs)
    return report["flagged_nodes"]


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

    clean_z = zscore_anomaly_screen(clean_inputs, threshold=z_threshold, reference_samples=clean_inputs)
    poison_z = zscore_anomaly_screen(poisoned_inputs, threshold=z_threshold, reference_samples=clean_inputs)
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
            "flag_rate_gap": float(np.mean(poison_z["mask"]) - np.mean(clean_z["mask"])),
            "clean_score_mean": float(np.mean(clean_z["score"])),
            "poison_score_mean": float(np.mean(poison_z["score"])),
            "score_mean_gap": float(np.mean(poison_z["score"]) - np.mean(clean_z["score"])),
            "threshold": z_threshold,
        },
        {
            "name": "high_freq_energy_check",
            "clean_flag_rate": float(np.mean(clean_freq["flag"])),
            "poison_flag_rate": float(np.mean(poison_freq["flag"])),
            "flag_rate_gap": float(np.mean(poison_freq["flag"]) - np.mean(clean_freq["flag"])),
            "clean_ratio_mean": float(np.mean(clean_freq["high_freq_ratio"])),
            "poison_ratio_mean": float(np.mean(poison_freq["high_freq_ratio"])),
            "ratio_mean_gap": float(np.mean(poison_freq["high_freq_ratio"]) - np.mean(clean_freq["high_freq_ratio"])),
            "threshold": energy_ratio_threshold,
            "cutoff": high_freq_cutoff,
        },
        {
            "name": "moving_average_sensitivity",
            "clean_delta": float(clean_delta),
            "poison_delta": float(poison_delta),
            "delta_gap": float(poison_delta - clean_delta),
            "kernel_size": ma_kernel,
        },
    ]

