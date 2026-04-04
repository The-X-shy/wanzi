from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from traffic_poison.config import load_config
from traffic_poison.experiment import dataset_summary, prepare_bundle_from_config
from traffic_poison.reporting import save_table
from traffic_poison.utils import create_run_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and summarize a traffic dataset.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bundle = prepare_bundle_from_config(config)

    dataset_name = str(config["dataset"].get("name", "dataset")).lower().replace("-", "_")
    run_dir = create_run_dir(config["output"]["root_dir"], f"{dataset_name}_prepare")

    summary = dataset_summary(bundle, config)
    save_json(summary, run_dir / "dataset_summary.json")

    split_rows = [
        {"split": "train", "samples": int(bundle.train_inputs.shape[0])},
        {"split": "val", "samples": int(bundle.val_inputs.shape[0])},
        {"split": "test", "samples": int(bundle.test_inputs.shape[0])},
    ]
    feature_rows = []
    frame = bundle.frame
    for column in frame.columns[: min(10, len(frame.columns))]:
        feature_rows.append(
            {
                "feature": column,
                "mean": float(frame[column].mean()),
                "std": float(frame[column].std()),
                "min": float(frame[column].min()),
                "max": float(frame[column].max()),
            }
        )

    save_table(split_rows, run_dir / "split_summary.csv")
    save_table(feature_rows, run_dir / "feature_summary.csv")
    print(run_dir)


if __name__ == "__main__":
    main()
