from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from traffic_poison.config import load_config
from traffic_poison.defenses import evaluate_simple_defenses, moving_average_smooth
from traffic_poison.experiment import build_model_from_kwargs, evaluate_on_arrays
from traffic_poison.poisoning import compute_attack_success_metrics
from traffic_poison.reporting import save_table
from traffic_poison.utils import create_run_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate simple defenses on the best poisoning run.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--poison-dir", required=True, help="Directory produced by run_poison_experiments.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    poison_dir = Path(args.poison_dir).resolve()
    run_dir = create_run_dir(config["output"]["root_dir"], "defense_eval")

    checkpoint = torch.load(poison_dir / "best_poisoned_model.pt", map_location="cpu")
    bundle = np.load(poison_dir / "best_attack_bundle.npz")

    clean_inputs = bundle["clean_test_inputs"]
    poisoned_inputs = bundle["triggered_test_inputs"]
    test_targets = bundle["test_targets"]

    defense_cfg = {
        "z_threshold": float(config["defense"].get("zscore_threshold", 3.0)),
        "ma_kernel": int(config["defense"].get("moving_average_window", 3)),
        "energy_ratio_threshold": float(config["defense"].get("frequency_band_fraction", 0.25)),
        "high_freq_cutoff": 0.5,
    }

    rows = evaluate_simple_defenses(clean_inputs, poisoned_inputs, defense_cfg)

    model = build_model_from_kwargs(checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])

    _, _, poisoned_pred = evaluate_on_arrays(model, poisoned_inputs, test_targets, config, device="cpu")
    poisoned_asr = compute_attack_success_metrics(
        test_targets,
        poisoned_pred,
        float(config["poison"].get("target_shift_ratio", 0.10)),
        float(config["poison"].get("success_tolerance_ratio", 0.03)),
    )

    smoothed_inputs = np.asarray(moving_average_smooth(poisoned_inputs, kernel_size=defense_cfg["ma_kernel"]))
    _, _, smoothed_pred = evaluate_on_arrays(model, smoothed_inputs, test_targets, config, device="cpu")
    smoothed_asr = compute_attack_success_metrics(
        test_targets,
        smoothed_pred,
        float(config["poison"].get("target_shift_ratio", 0.10)),
        float(config["poison"].get("success_tolerance_ratio", 0.03)),
    )

    rows.append(
        {
            "name": "moving_average_asr_effect",
            "asr_before": float(poisoned_asr["attack_success_rate"]),
            "asr_after": float(smoothed_asr["attack_success_rate"]),
            "kernel_size": defense_cfg["ma_kernel"],
        }
    )

    save_table(rows, run_dir / "defense_results.csv")
    save_json(
        {
            "source_poison_dir": str(poison_dir),
            "rows": rows,
        },
        run_dir / "defense_summary.json",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
