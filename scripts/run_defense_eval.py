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
from traffic_poison.poisoning import compute_attack_evaluation_views, compute_attack_success_metrics
from traffic_poison.reporting import save_table
from traffic_poison.utils import create_run_dir, save_json


class BundleScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = np.where(std < 1e-6, 1.0, std)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data) * self.std + self.mean


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
    selected_nodes = bundle["selected_nodes"].astype(np.int64).tolist() if "selected_nodes" in bundle.files else []
    selected_target_horizons = (
        bundle["selected_target_horizon_indices"].astype(np.int64).tolist()
        if "selected_target_horizon_indices" in bundle.files
        else []
    )
    scaler = None
    if "scaler_mean" in bundle.files and "scaler_std" in bundle.files:
        scaler_mean = np.asarray(bundle["scaler_mean"])
        scaler_std = np.asarray(bundle["scaler_std"])
        if scaler_mean.size > 0 and scaler_std.size > 0:
            scaler = BundleScaler(scaler_mean, scaler_std)

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
    best_row = checkpoint.get("best_row", {})
    target_shift_ratio = float(best_row.get("target_shift_ratio", config["poison"].get("target_shift_ratio", 0.10)))
    tolerance_ratio = float(best_row.get("tolerance_ratio", config["poison"].get("success_tolerance_ratio", 0.03)))

    poisoned_asr = compute_attack_success_metrics(
        test_targets,
        poisoned_pred,
        target_shift_ratio,
        tolerance_ratio,
    )
    clean_reference_pred = bundle["poisoned_model_clean_predictions"] if "poisoned_model_clean_predictions" in bundle.files else poisoned_pred
    poisoned_views = compute_attack_evaluation_views(
        test_targets,
        clean_reference_pred,
        poisoned_pred,
        target_shift_ratio,
        tolerance_ratio,
        selected_nodes=selected_nodes,
        target_horizon_indices=selected_target_horizons,
        tail_horizon_count=int(config["poison"].get("evaluation_tail_horizon_count", 3)),
        scaler=scaler,
    )

    smoothed_inputs = np.asarray(moving_average_smooth(poisoned_inputs, kernel_size=defense_cfg["ma_kernel"]))
    _, _, smoothed_pred = evaluate_on_arrays(model, smoothed_inputs, test_targets, config, device="cpu")
    smoothed_asr = compute_attack_success_metrics(
        test_targets,
        smoothed_pred,
        target_shift_ratio,
        tolerance_ratio,
    )
    smoothed_views = compute_attack_evaluation_views(
        test_targets,
        clean_reference_pred,
        smoothed_pred,
        target_shift_ratio,
        tolerance_ratio,
        selected_nodes=selected_nodes,
        target_horizon_indices=selected_target_horizons,
        tail_horizon_count=int(config["poison"].get("evaluation_tail_horizon_count", 3)),
        scaler=scaler,
    )

    rows.append(
        {
            "name": "moving_average_asr_effect",
            "asr_before": float(poisoned_views.get("raw_global_attack_success_rate", poisoned_asr["attack_success_rate"])),
            "asr_after": float(smoothed_views.get("raw_global_attack_success_rate", smoothed_asr["attack_success_rate"])),
            "asr_gap": float(
                smoothed_views.get("raw_global_attack_success_rate", smoothed_asr["attack_success_rate"])
                - poisoned_views.get("raw_global_attack_success_rate", poisoned_asr["attack_success_rate"])
            ),
            "scaled_asr_before": float(poisoned_asr["attack_success_rate"]),
            "scaled_asr_after": float(smoothed_asr["attack_success_rate"]),
            "scaled_asr_gap": float(smoothed_asr["attack_success_rate"] - poisoned_asr["attack_success_rate"]),
            "local_asr_before": float(poisoned_views.get("raw_selected_nodes_tail_horizon_attack_success_rate", 0.0)),
            "local_asr_after": float(smoothed_views.get("raw_selected_nodes_tail_horizon_attack_success_rate", 0.0)),
            "local_asr_gap": float(
                smoothed_views.get("raw_selected_nodes_tail_horizon_attack_success_rate", 0.0)
                - poisoned_views.get("raw_selected_nodes_tail_horizon_attack_success_rate", 0.0)
            ),
            "target_horizon_local_asr_before": float(
                poisoned_views.get("raw_selected_nodes_target_horizons_attack_success_rate", 0.0)
            ),
            "target_horizon_local_asr_after": float(
                smoothed_views.get("raw_selected_nodes_target_horizons_attack_success_rate", 0.0)
            ),
            "target_horizon_local_asr_gap": float(
                smoothed_views.get("raw_selected_nodes_target_horizons_attack_success_rate", 0.0)
                - poisoned_views.get("raw_selected_nodes_target_horizons_attack_success_rate", 0.0)
            ),
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
