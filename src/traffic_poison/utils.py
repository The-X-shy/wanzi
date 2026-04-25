from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch

ArrayLike = Any


def to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert array-like to numpy, handling torch tensors and pandas objects."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, pd.DataFrame):
        return x.to_numpy()
    if isinstance(x, pd.Series):
        return x.to_numpy()
    return np.asarray(x)


def to_tensor_like(x: np.ndarray, reference: ArrayLike) -> ArrayLike:
    """Cast numpy array to the same type/device as reference if reference is a tensor."""
    if torch.is_tensor(reference):
        return torch.as_tensor(x, dtype=reference.dtype, device=reference.device)
    return x


def ensure_3d(x: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Ensure array is 3D; return (3d_array, was_squeezed)."""
    if x.ndim == 2:
        return x[None, ...], True
    if x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {x.shape}")
    return x, False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def resolve_device(requested: str | None) -> str:
    if requested and requested not in {"auto", ""}:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def create_run_dir(root: str | Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(root) / f"{prefix}_{timestamp}")


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def flatten_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}
