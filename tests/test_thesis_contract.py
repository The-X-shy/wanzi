from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_poison.experiment import pick_best_attack_row
from traffic_poison.thesis_contract import (
    choose_best_row,
    eligible_for_cross_replay,
    evaluate_main_result_standards,
    resolve_thesis_contract,
)


def load_build_thesis_tables_module():
    module_path = REPO_ROOT / "scripts" / "build_thesis_tables.py"
    spec = importlib.util.spec_from_file_location("build_thesis_tables", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load build_thesis_tables.py for testing.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ThesisContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = resolve_thesis_contract()

    def test_pick_best_attack_row_matches_contract_choice(self) -> None:
        rows = [
            {
                "selection_strategy": "error",
                "window_mode": "hybrid",
                "attack_success_rate": 0.030,
                "clean_MAE_delta_ratio": 0.030,
                "raw_selected_nodes_tail_horizon_attack_success_rate": 0.051,
                "raw_selected_nodes_tail_horizon_shift_direction_match_rate": 0.62,
                "frequency_energy_shift": 0.041,
                "mean_z_score": 0.70,
            },
            {
                "selection_strategy": "error",
                "window_mode": "hybrid",
                "attack_success_rate": 0.020,
                "clean_MAE_delta_ratio": 0.031,
                "raw_selected_nodes_tail_horizon_attack_success_rate": 0.056,
                "raw_selected_nodes_tail_horizon_shift_direction_match_rate": 0.63,
                "frequency_energy_shift": 0.042,
                "mean_z_score": 0.71,
            },
        ]

        self.assertEqual(choose_best_row(rows, self.contract), pick_best_attack_row(rows, self.contract))
        self.assertAlmostEqual(
            pick_best_attack_row(rows, self.contract)["raw_selected_nodes_tail_horizon_attack_success_rate"],
            0.056,
        )

    def test_build_candidate_table_prefers_main_metric_over_legacy_asr(self) -> None:
        module = load_build_thesis_tables_module()
        rows = [
            {
                "selection_strategy": "error",
                "window_mode": "hybrid",
                "trigger_steps": 3,
                "trigger_node_count": 3,
                "poison_ratio": 0.018,
                "sigma_multiplier": 0.06,
                "target_shift_ratio": 0.08,
                "attack_success_rate": 0.030,
                "clean_MAE_delta_ratio": 0.030,
                "raw_selected_nodes_tail_horizon_attack_success_rate": 0.051,
                "raw_selected_nodes_tail_horizon_shift_direction_match_rate": 0.62,
                "frequency_energy_shift": 0.041,
                "mean_z_score": 0.70,
            },
            {
                "selection_strategy": "error",
                "window_mode": "hybrid",
                "trigger_steps": 3,
                "trigger_node_count": 3,
                "poison_ratio": 0.02,
                "sigma_multiplier": 0.065,
                "target_shift_ratio": 0.08,
                "attack_success_rate": 0.018,
                "clean_MAE_delta_ratio": 0.032,
                "raw_selected_nodes_tail_horizon_attack_success_rate": 0.057,
                "raw_selected_nodes_tail_horizon_shift_direction_match_rate": 0.64,
                "frequency_energy_shift": 0.042,
                "mean_z_score": 0.71,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            poison_dir = Path(tmpdir)
            with (poison_dir / "attack_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            frame = module.build_candidate_table(poison_dir, self.contract)
            top_row = frame.iloc[0].to_dict()

        self.assertAlmostEqual(top_row["raw_selected_nodes_tail_horizon_attack_success_rate"], 0.057)
        self.assertTrue(bool(top_row["direction_ok"]))

    def test_cross_replay_gate_requires_stronger_than_previous_mainline(self) -> None:
        exact_previous = {
            "clean_MAE_delta_ratio": 0.039,
            "raw_selected_nodes_tail_horizon_attack_success_rate": self.contract["previous_mainline_local_asr"],
            "raw_selected_nodes_tail_horizon_shift_direction_match_rate": 0.64,
            "raw_selected_nodes_tail_horizon_target_shift_attainment": 0.0,
        }
        improved = {
            "clean_MAE_delta_ratio": 0.039,
            "raw_selected_nodes_tail_horizon_attack_success_rate": self.contract["previous_mainline_local_asr"] + 0.001,
            "raw_selected_nodes_tail_horizon_shift_direction_match_rate": 0.64,
            "raw_selected_nodes_tail_horizon_target_shift_attainment": -0.0005,
        }
        wrong_direction = {
            "clean_MAE_delta_ratio": 0.039,
            "raw_selected_nodes_tail_horizon_attack_success_rate": self.contract["previous_mainline_local_asr"] + 0.001,
            "raw_selected_nodes_tail_horizon_shift_direction_match_rate": 0.64,
            "raw_selected_nodes_tail_horizon_target_shift_attainment": -0.002,
        }

        self.assertFalse(eligible_for_cross_replay(exact_previous, self.contract))
        self.assertTrue(eligible_for_cross_replay(improved, self.contract))
        self.assertFalse(eligible_for_cross_replay(wrong_direction, self.contract))

    def test_main_result_standards_require_direction_and_stealth(self) -> None:
        final_best = {
            "attack_success_rate": 0.017,
            "clean_MAE_delta_ratio": 0.036,
            "raw_selected_nodes_tail_horizon_attack_success_rate": 0.056,
            "raw_selected_nodes_tail_horizon_shift_direction_match_rate": 0.655,
            "raw_selected_nodes_tail_horizon_target_shift_attainment": -0.001,
            "frequency_energy_shift": 0.042,
            "mean_z_score": 0.757,
        }
        summary = evaluate_main_result_standards(final_best, 0.3651, 0.0294, self.contract)
        self.assertTrue(summary["minimum_bar_met"])
        self.assertFalse(summary["strong_bar_met"])

    def test_contract_prefers_directionally_cleaner_candidate_when_main_metric_is_tied(self) -> None:
        rows = [
            {
                "selection_strategy": "error",
                "window_mode": "hybrid",
                "attack_success_rate": 0.017,
                "clean_MAE_delta_ratio": 0.034,
                "raw_selected_nodes_tail_horizon_attack_success_rate": 0.056,
                "raw_selected_nodes_tail_horizon_shift_direction_match_rate": 0.66,
                "raw_selected_nodes_tail_horizon_target_shift_attainment": -0.003,
                "frequency_energy_shift": 0.041,
                "mean_z_score": 0.72,
            },
            {
                "selection_strategy": "error",
                "window_mode": "hybrid",
                "attack_success_rate": 0.016,
                "clean_MAE_delta_ratio": 0.035,
                "raw_selected_nodes_tail_horizon_attack_success_rate": 0.056,
                "raw_selected_nodes_tail_horizon_shift_direction_match_rate": 0.66,
                "raw_selected_nodes_tail_horizon_target_shift_attainment": -0.0003,
                "frequency_energy_shift": 0.042,
                "mean_z_score": 0.72,
            },
        ]

        best = choose_best_row(rows, self.contract)
        self.assertAlmostEqual(best["raw_selected_nodes_tail_horizon_target_shift_attainment"], -0.0003)


if __name__ == "__main__":
    unittest.main()
