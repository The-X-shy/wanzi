from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import h5py
except Exception as exc:  # pragma: no cover - h5py is expected to exist here
    h5py = None  # type: ignore[assignment]
    _H5PY_IMPORT_ERROR = exc
else:
    _H5PY_IMPORT_ERROR = None

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CUDA is unavailable here
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _as_numpy(array: ArrayLike) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if isinstance(array, pd.DataFrame):
        return array.to_numpy()
    return np.asarray(array)


def _first_hdf5_dataset(handle: "h5py.File") -> np.ndarray:
    found: list[np.ndarray] = []

    def _visit(_: str, obj: object) -> None:
        if hasattr(obj, "shape") and hasattr(obj, "__array__"):
            found.append(np.asarray(obj))

    handle.visititems(_visit)
    if not found:
        raise ValueError("No dataset found in HDF5 file.")
    ranked = sorted(found, key=lambda arr: (arr.ndim >= 2, arr.ndim, arr.size), reverse=True)
    return ranked[0]


def _extract_hdf5_node(node: object) -> np.ndarray:
    if h5py is None:
        raise ImportError("h5py is required to read HDF5 traffic data.") from _H5PY_IMPORT_ERROR
    if isinstance(node, h5py.Dataset):
        return np.asarray(node)
    if isinstance(node, h5py.Group):
        if "block0_values" in node:
            return np.asarray(node["block0_values"])
        return _first_hdf5_dataset(node)
    raise TypeError(f"Unsupported HDF5 node type: {type(node)!r}")


def _first_pandas_hdf_key(path: Path) -> Optional[str]:
    try:
        with pd.HDFStore(path, mode="r") as store:
            keys = list(store.keys())
    except Exception:
        return None
    if not keys:
        return None
    # pandas expects keys without a leading slash when using read_hdf.
    return keys[0].lstrip("/")


def load_traffic_matrix(path: Union[str, Path], key: Optional[str] = None) -> np.ndarray:
    """
    Load a traffic speed matrix from common numpy / HDF5 containers.

    The returned array is always time-major: shape [T, ...].
    """

    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix in {".npy"}:
        data = np.load(file_path, allow_pickle=True)
    elif suffix in {".npz"}:
        archive = np.load(file_path, allow_pickle=True)
        if key is not None:
            data = archive[key]
        elif "data" in archive:
            data = archive["data"]
        else:
            data = archive[archive.files[0]]
    elif suffix in {".csv"}:
        frame = pd.read_csv(file_path)
        numeric_frame = frame.select_dtypes(include=[np.number])
        if numeric_frame.empty and frame.shape[1] > 1:
            numeric_frame = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        if numeric_frame.empty:
            raise ValueError("CSV traffic data does not contain any numeric columns.")
        numeric_frame = numeric_frame.dropna(axis=1, how="all")
        data = numeric_frame.to_numpy()
    elif suffix in {".pkl", ".pickle"}:
        with file_path.open("rb") as handle:
            try:
                payload = pickle.load(handle)
            except UnicodeDecodeError:
                handle.seek(0)
                payload = pickle.load(handle, encoding="latin1")
        if isinstance(payload, (tuple, list)) and payload:
            data = payload[-1]
        elif isinstance(payload, dict):
            if key is not None and key in payload:
                data = payload[key]
            elif "adj_mx" in payload:
                data = payload["adj_mx"]
            elif "data" in payload:
                data = payload["data"]
            else:
                data = next(iter(payload.values()))
        else:
            data = payload
    elif suffix in {".h5", ".hdf5"}:
        try:
            resolved_key = key if key is not None else _first_pandas_hdf_key(file_path)
            hdf_frame = pd.read_hdf(file_path, key=resolved_key) if resolved_key is not None else pd.read_hdf(file_path)
            data = hdf_frame.to_numpy() if isinstance(hdf_frame, pd.DataFrame) else np.asarray(hdf_frame)
        except Exception:
            if h5py is None:
                raise ImportError(
                    "h5py is required to read HDF5 traffic data."
                ) from _H5PY_IMPORT_ERROR
            with h5py.File(file_path, "r") as handle:
                if key is not None:
                    if key not in handle:
                        raise KeyError(f"Key '{key}' not found in HDF5 file.")
                    data = _extract_hdf5_node(handle[key])
                elif "data" in handle:
                    data = _extract_hdf5_node(handle["data"])
                else:
                    data = _first_hdf5_dataset(handle)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    data = np.asarray(data)
    if data.ndim < 2:
        raise ValueError("Traffic data must have at least 2 dimensions [T, ...].")
    return data.astype(np.float32, copy=False)


def fill_missing_values(data: ArrayLike) -> np.ndarray:
    """
    Fill missing values with forward fill then backward fill along the time axis.

    For tensors with more than 2 dimensions, all non-time dimensions are flattened
    temporarily so the fill is applied column-wise over time.
    """

    arr = _as_numpy(data).astype(np.float32, copy=False)
    if arr.ndim == 1:
        series = pd.Series(arr)
        return series.ffill().bfill().to_numpy(dtype=np.float32)

    original_shape = arr.shape
    time_steps = original_shape[0]
    flat = arr.reshape(time_steps, -1)
    frame = pd.DataFrame(flat)
    filled = frame.ffill(axis=0).bfill(axis=0)
    values = filled.to_numpy(dtype=np.float32)
    return values.reshape(original_shape)


def split_by_time(
    data: ArrayLike,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a time-major array sequentially into train / val / test.
    """

    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total:.4f}.")

    arr = _as_numpy(data)
    if arr.ndim < 2:
        raise ValueError("Time split expects an array with shape [T, ...].")

    t = arr.shape[0]
    train_end = int(t * train_ratio)
    val_end = train_end + int(t * val_ratio)

    train = arr[:train_end]
    val = arr[train_end:val_end]
    test = arr[val_end:]

    return train, val, test


class StandardScaler:
    """
    Feature-wise standardization for time-major traffic arrays.
    """

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = mean
        self.std = std

    def fit(self, data: ArrayLike) -> "StandardScaler":
        arr = _as_numpy(data).astype(np.float32, copy=False)
        if arr.ndim < 2:
            raise ValueError("Scaler expects a time-major array.")
        self.mean = arr.mean(axis=0, keepdims=True)
        self.std = arr.std(axis=0, keepdims=True)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, data: ArrayLike) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before calling transform.")
        arr = _as_numpy(data).astype(np.float32, copy=False)
        return (arr - self.mean) / self.std

    def inverse_transform(self, data: ArrayLike) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before calling inverse_transform.")
        arr = _as_numpy(data).astype(np.float32, copy=False)
        return arr * self.std + self.mean


def _flatten_time_step(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        return data
    if data.ndim < 2:
        raise ValueError("Expected at least 2 dimensions.")
    return data.reshape(data.shape[0], -1)


def _build_feature_names(feature_dim: int, prefix: str = "feature") -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(feature_dim)]


def _load_adjacency(cfg: Any, node_count: int) -> np.ndarray:
    adjacency = _cfg_get(cfg, "adjacency", None)
    adjacency_path = _cfg_get(cfg, "adjacency_path", None) or _cfg_get(cfg, "adj_path", None)
    adjacency_key = _cfg_get(cfg, "adjacency_key", None)

    if adjacency is not None:
        adj = _as_numpy(adjacency).astype(np.float32, copy=False)
    elif adjacency_path is not None:
        adj = load_traffic_matrix(adjacency_path, key=adjacency_key)
    else:
        adj = np.eye(node_count, dtype=np.float32)

    if adj.ndim != 2:
        raise ValueError("Adjacency matrix must be 2-dimensional.")
    return adj


def create_sliding_windows(
    data: ArrayLike,
    input_len: int = 12,
    output_len: int = 12,
    horizon: int = 1,
    allow_empty: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create supervised samples from a time-major traffic array.

    Inputs:
        X: [num_samples, input_len, features]
        y: [num_samples, output_len, features]
    """

    arr = _as_numpy(data).astype(np.float32, copy=False)
    if arr.ndim == 1:
        arr = arr[:, None]
    flat = _flatten_time_step(arr)

    if input_len <= 0 or output_len <= 0:
        raise ValueError("input_len and output_len must be positive integers.")
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    if flat.shape[0] < input_len + horizon + output_len - 1:
        if allow_empty:
            feature_dim = flat.shape[1]
            return (
                np.empty((0, input_len, feature_dim), dtype=np.float32),
                np.empty((0, output_len, feature_dim), dtype=np.float32),
            )
        raise ValueError("Not enough time steps to create the requested windows.")

    xs = []
    ys = []
    max_start = flat.shape[0] - input_len - horizon - output_len + 1
    for start in range(max_start + 1):
        x_start = start
        x_end = start + input_len
        y_start = x_end + horizon - 1
        y_end = y_start + output_len
        xs.append(flat[x_start:x_end])
        ys.append(flat[y_start:y_end])

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


class TrafficSequenceDataset(Dataset):
    """
    Torch dataset for sequence-to-sequence traffic forecasting.
    """

    def __init__(
        self,
        inputs: ArrayLike,
        targets: ArrayLike,
    ) -> None:
        x = _as_numpy(inputs).astype(np.float32, copy=False)
        y = _as_numpy(targets).astype(np.float32, copy=False)
        if x.shape[0] != y.shape[0]:
            raise ValueError("inputs and targets must have the same sample count.")
        self.inputs = torch.from_numpy(x)
        self.targets = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


@dataclass
class DataBundle:
    train_inputs: np.ndarray
    train_targets: np.ndarray
    val_inputs: np.ndarray
    val_targets: np.ndarray
    test_inputs: np.ndarray
    test_targets: np.ndarray
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    frame: pd.DataFrame
    scaler: StandardScaler
    adjacency: np.ndarray
    feature_names: list[str]
    feature_std: np.ndarray


TrafficDataBundle = DataBundle


def build_dataloaders(
    data: ArrayLike,
    input_len: int = 12,
    output_len: int = 12,
    horizon: int = 1,
    batch_size: int = 64,
    shuffle_train: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    fill_missing: bool = True,
    scale: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> DataBundle:
    """
    Build train / val / test dataloaders from a raw traffic matrix.
    """

    arr = _as_numpy(data)
    return _bundle_from_array(
        arr,
        input_len=input_len,
        horizon=output_len,
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        fill_missing=fill_missing,
        scale=scale,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        feature_names=None,
        adjacency=None,
    )


def load_traffic_bundle(
    path: Union[str, Path],
    key: Optional[str] = None,
    **kwargs,
) -> DataBundle:
    """
    Convenience wrapper that loads a file and immediately builds dataloaders.
    """

    data = load_traffic_matrix(path, key=key)
    return build_dataloaders(data, **kwargs)


def _bundle_from_array(
    arr: np.ndarray,
    *,
    input_len: int,
    horizon: int,
    batch_size: int,
    shuffle_train: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    fill_missing: bool,
    scale: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    feature_names: Optional[Sequence[str]] = None,
    adjacency: Optional[np.ndarray] = None,
) -> DataBundle:
    _set_seed(seed)
    raw = _as_numpy(arr).astype(np.float32, copy=False)
    if fill_missing:
        raw = fill_missing_values(raw)

    frame_values = _flatten_time_step(raw)
    feature_dim = frame_values.shape[1]
    names = list(feature_names) if feature_names is not None else _build_feature_names(feature_dim, prefix="node")
    if len(names) != feature_dim:
        raise ValueError("feature_names must match the flattened feature dimension.")

    train_raw, val_raw, test_raw = split_by_time(raw, train_ratio, val_ratio, test_ratio)

    scaler = StandardScaler()
    if scale:
        scaler.fit(train_raw)
        train_proc = scaler.transform(train_raw)
        val_proc = scaler.transform(val_raw)
        test_proc = scaler.transform(test_raw)
    else:
        train_proc, val_proc, test_proc = train_raw, val_raw, test_raw
        scaler.fit(train_raw)

    train_inputs, train_targets = create_sliding_windows(train_proc, input_len, horizon, horizon=1)
    val_inputs, val_targets = create_sliding_windows(val_proc, input_len, horizon, horizon=1, allow_empty=True)
    test_inputs, test_targets = create_sliding_windows(test_proc, input_len, horizon, horizon=1, allow_empty=True)

    train_dataset = TrafficSequenceDataset(train_inputs, train_targets)
    val_dataset = TrafficSequenceDataset(val_inputs, val_targets)
    test_dataset = TrafficSequenceDataset(test_inputs, test_targets)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    if adjacency is None:
        adjacency = np.eye(feature_dim, dtype=np.float32)
    adjacency = _as_numpy(adjacency).astype(np.float32, copy=False)
    if adjacency.ndim != 2:
        raise ValueError("adjacency must be a 2D matrix.")

    feature_std = np.asarray(scaler.std, dtype=np.float32).reshape(-1)

    return DataBundle(
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        frame=pd.DataFrame(frame_values, columns=names),
        scaler=scaler,
        adjacency=adjacency,
        feature_names=names,
        feature_std=feature_std,
    )


def prepare_data_bundle(dataset_cfg: Any, seed: int = 42) -> DataBundle:
    data_path = _cfg_get(dataset_cfg, "data_path", None) or _cfg_get(dataset_cfg, "path", None)
    if data_path is None and _cfg_get(dataset_cfg, "data", None) is None:
        raise ValueError("dataset_cfg must provide either data_path/path or data.")

    data_key = _cfg_get(dataset_cfg, "data_key", None) or _cfg_get(dataset_cfg, "key", None)
    if _cfg_get(dataset_cfg, "data", None) is not None:
        raw = _as_numpy(_cfg_get(dataset_cfg, "data"))
    else:
        raw = load_traffic_matrix(data_path, key=data_key)

    input_len = int(_cfg_get(dataset_cfg, "input_len", 12))
    horizon = int(_cfg_get(dataset_cfg, "horizon", _cfg_get(dataset_cfg, "output_len", 12)))
    batch_size = int(_cfg_get(dataset_cfg, "batch_size", 64))
    shuffle_train = bool(_cfg_get(dataset_cfg, "shuffle_train", True))
    train_ratio = float(_cfg_get(dataset_cfg, "train_ratio", 0.7))
    val_ratio = float(_cfg_get(dataset_cfg, "val_ratio", 0.1))
    test_ratio = float(_cfg_get(dataset_cfg, "test_ratio", 0.2))
    fill_missing = bool(_cfg_get(dataset_cfg, "fill_missing", True))
    scale = bool(_cfg_get(dataset_cfg, "scale", True))
    num_workers = int(_cfg_get(dataset_cfg, "num_workers", 0))
    pin_memory = bool(_cfg_get(dataset_cfg, "pin_memory", False))
    feature_names = _cfg_get(dataset_cfg, "feature_names", None)

    raw_for_shape = _as_numpy(raw)
    feature_dim = _flatten_time_step(raw_for_shape.astype(np.float32, copy=False)).shape[1]
    adjacency = _load_adjacency(dataset_cfg, feature_dim)

    return _bundle_from_array(
        raw,
        input_len=input_len,
        horizon=horizon,
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        fill_missing=fill_missing,
        scale=scale,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        feature_names=feature_names,
        adjacency=adjacency,
    )
