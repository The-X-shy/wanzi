from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic traffic dataset for smoke tests.")
    parser.add_argument("--output-dir", default="data/synthetic", help="Directory used to store the generated files.")
    parser.add_argument("--name", default="synthetic_metr_la", help="Dataset file prefix.")
    parser.add_argument("--time-steps", type=int, default=480, help="Number of time steps.")
    parser.add_argument("--sensor-count", type=int, default=8, help="Number of sensors.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def build_synthetic_series(time_steps: int, sensor_count: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=time_steps, freq="5min")
    base = np.linspace(45.0, 65.0, sensor_count)
    data = []

    for sensor_idx in range(sensor_count):
        phase = sensor_idx * 0.35
        seasonal = 8 * np.sin(np.linspace(0, 6 * np.pi, time_steps) + phase)
        rush_hour = 6 * np.sin(np.linspace(0, 12 * np.pi, time_steps) + phase / 2)
        noise = rng.normal(loc=0.0, scale=1.2 + sensor_idx * 0.05, size=time_steps)
        series = base[sensor_idx] + seasonal + rush_hour + noise
        data.append(series)

    values = np.stack(data, axis=1).astype(np.float32)
    missing_mask = rng.random(values.shape) < 0.02
    values[missing_mask] = np.nan

    columns = [f"sensor_{idx}" for idx in range(sensor_count)]
    frame = pd.DataFrame(values, index=timestamps, columns=columns)

    adjacency = np.zeros((sensor_count, sensor_count), dtype=np.float32)
    for idx in range(sensor_count):
        adjacency[idx, idx] = 1.0
        adjacency[idx, (idx - 1) % sensor_count] = 0.6
        adjacency[idx, (idx + 1) % sensor_count] = 0.6
        if idx + 2 < sensor_count:
            adjacency[idx, idx + 2] = 0.25
            adjacency[idx + 2, idx] = 0.25

    return frame, adjacency


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, adjacency = build_synthetic_series(args.time_steps, args.sensor_count, args.seed)
    data_path = output_dir / f"{args.name}.h5"
    adj_path = output_dir / f"{args.name}_adj.pkl"
    manifest_path = output_dir / f"{args.name}_manifest.json"

    frame.to_hdf(data_path, key="df")

    sensor_ids = list(frame.columns)
    sensor_id_to_ind = {sensor_id: idx for idx, sensor_id in enumerate(sensor_ids)}
    with adj_path.open("wb") as handle:
        pickle.dump((sensor_ids, sensor_id_to_ind, adjacency), handle)

    manifest = {
        "data_path": str(data_path),
        "adjacency_path": str(adj_path),
        "time_steps": args.time_steps,
        "sensor_count": args.sensor_count,
        "seed": args.seed,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
