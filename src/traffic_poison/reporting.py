from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

cache_dir = Path(os.environ.get("MPLCONFIGDIR", Path.cwd() / ".mplcache"))
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_dir)
import numpy as np


def _plt():
    import matplotlib.pyplot as plt

    return plt


def save_table(rows: Iterable[dict], path: str | Path) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    frame.to_csv(path, index=False)
    return frame


def plot_training_curve(train_losses: list[float], val_losses: list[float], path: str | Path) -> None:
    plt = _plt()
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_prediction_case(
    y_true: np.ndarray,
    clean_pred: np.ndarray,
    path: str | Path,
    poisoned_pred: np.ndarray | None = None,
    sample_index: int = 0,
    node_index: int = 0,
    title: str = "Prediction Case",
) -> None:
    if y_true.size == 0 or clean_pred.size == 0:
        return

    plt = _plt()
    truth = y_true[sample_index, :, node_index]
    clean = clean_pred[sample_index, :, node_index]

    plt.figure(figsize=(7, 4))
    plt.plot(truth, label="truth", marker="o")
    plt.plot(clean, label="clean_pred", marker="o")
    if poisoned_pred is not None and poisoned_pred.size:
        poisoned = poisoned_pred[sample_index, :, node_index]
        plt.plot(poisoned, label="poisoned_pred", marker="o")
    plt.xlabel("Forecast step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_trigger_case(
    clean_inputs: np.ndarray,
    poisoned_inputs: np.ndarray,
    path: str | Path,
    sample_index: int = 0,
    node_indices: list[int] | None = None,
) -> None:
    if clean_inputs.size == 0 or poisoned_inputs.size == 0:
        return

    plt = _plt()
    nodes = node_indices or [0]
    plt.figure(figsize=(8, 4))
    for node in nodes[:3]:
        plt.plot(
            clean_inputs[sample_index, :, node],
            linestyle="--",
            alpha=0.7,
            label=f"clean_node_{node}",
        )
        plt.plot(
            poisoned_inputs[sample_index, :, node],
            linewidth=2,
            label=f"triggered_node_{node}",
        )
    plt.xlabel("Input step")
    plt.ylabel("Value")
    plt.title("Trigger Example")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_bar_table(rows: list[dict], x: str, y: str, path: str | Path, title: str) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    plt = _plt()
    plt.figure(figsize=(8, 4))
    plt.bar(frame[x].astype(str), frame[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
