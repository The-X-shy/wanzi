from __future__ import annotations

import csv
import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_poison.defenses import zscore_anomaly_screen
from traffic_poison.poisoning import (
    build_poisoned_training_set,
    compute_attack_success_metrics,
    generate_smooth_trigger,
)
from traffic_poison.thesis_contract import passes_minimum_candidate_bar, resolve_thesis_contract


class DummyScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean
        self.std = std


def load_build_thesis_tables_module():
    module_path = REPO_ROOT / "scripts" / "build_thesis_tables.py"
    spec = importlib.util.spec_from_file_location("build_thesis_tables", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load build_thesis_tables.py for testing.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReviewRemediationTests(unittest.TestCase):
    def test_attack_success_interval_handles_negative_targets(self) -> None:
        y_true = np.array([[[-2.0]]], dtype=np.float64)
        y_pred = np.array([[[-1.8]]], dtype=np.float64)

        metrics = compute_attack_success_metrics(y_true, y_pred, target_shift_ratio=0.1, tolerance_ratio=0.05)

        self.assertAlmostEqual(float(metrics["attack_success_rate"]), 1.0)
        self.assertLess(float(metrics["lower"][0]), float(metrics["upper"][0]))

    def test_poisoned_targets_shift_in_raw_space_then_return_to_training_space(self) -> None:
        scaler = DummyScaler(
            mean=np.array([[50.0]], dtype=np.float64),
            std=np.array([[10.0]], dtype=np.float64),
        )
        inputs = np.zeros((1, 2, 1), dtype=np.float64)
        targets = np.full((1, 1, 1), 5.0, dtype=np.float64)  # raw value is 100.

        poisoned = build_poisoned_training_set(
            inputs,
            targets,
            ranked_nodes=[0],
            poison_ratio=1.0,
            sigma_multiplier=0.0,
            feature_std=np.array([10.0], dtype=np.float64),
            trigger_steps=1,
            target_shift_ratio=0.1,
            fallback_shift_ratio=0.05,
            ranked_windows=[1],
            target_horizon_mode="all",
            target_horizon_count=1,
            target_weight_mode="flat",
            feature_scaler=scaler,
            trigger_feature_std=np.array([10.0], dtype=np.float64),
        )

        shifted = np.asarray(poisoned["poisoned_targets"])
        self.assertAlmostEqual(float(shifted[0, 0, 0]), 4.2, places=6)

    def test_trigger_amplitude_scale_controls_perturbation_size(self) -> None:
        sample = np.zeros((4, 2), dtype=np.float64)

        small = np.asarray(
            generate_smooth_trigger(
                sample,
                sigma=1.0,
                time_steps=1,
                node_indices=[0],
                time_indices=[3],
                target_shift=-0.1,
                smooth=False,
                amplitude_scale=np.array([1.0, 1.0], dtype=np.float64),
                random_state=7,
            )
        )
        large = np.asarray(
            generate_smooth_trigger(
                sample,
                sigma=1.0,
                time_steps=1,
                node_indices=[0],
                time_indices=[3],
                target_shift=-0.1,
                smooth=False,
                amplitude_scale=np.array([2.0, 1.0], dtype=np.float64),
                random_state=7,
            )
        )

        self.assertGreater(abs(float(large[3, 0])), abs(float(small[3, 0])) * 1.5)

    def test_zscore_screen_can_use_clean_reference_distribution(self) -> None:
        clean = np.zeros((2, 3, 1), dtype=np.float64)
        poisoned = np.full((2, 3, 1), 5.0, dtype=np.float64)

        self_referenced = zscore_anomaly_screen(poisoned, threshold=3.0)
        clean_referenced = zscore_anomaly_screen(poisoned, threshold=3.0, reference_samples=clean)

        self.assertEqual(float(np.mean(self_referenced["mask"])), 0.0)
        self.assertEqual(float(np.mean(clean_referenced["mask"])), 1.0)

    def test_thesis_contract_requires_real_main_local_metric(self) -> None:
        row = {
            "attack_success_rate": 0.10,
            "clean_MAE_delta_ratio": 0.01,
            "raw_selected_nodes_tail_horizon_shift_direction_match_rate": 0.80,
            "frequency_energy_shift": 0.01,
            "mean_z_score": 0.10,
        }

        self.assertFalse(passes_minimum_candidate_bar(row, resolve_thesis_contract()))

    def test_candidate_table_fails_when_required_result_columns_are_missing(self) -> None:
        module = load_build_thesis_tables_module()
        temp_dir = Path(tempfile.mkdtemp(prefix="wanzi_table_guard_"))
        try:
            rows = [
                {
                    "selection_strategy": "error",
                    "window_mode": "hybrid",
                    "trigger_steps": 3,
                    "trigger_node_count": 3,
                    "poison_ratio": 0.02,
                    "sigma_multiplier": 0.065,
                    "target_shift_ratio": 0.08,
                    "attack_success_rate": 0.02,
                    "clean_MAE_delta_ratio": 0.03,
                }
            ]
            with (temp_dir / "attack_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaises(ValueError):
                module.build_candidate_table(temp_dir, resolve_thesis_contract())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
