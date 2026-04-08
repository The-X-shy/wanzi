from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_poison.poisoning import build_poisoned_training_set


class PoisoningDirectionalTests(unittest.TestCase):
    def test_input_energy_flat_keeps_reference_behavior(self) -> None:
        inputs = np.zeros((4, 4, 3), dtype=np.float64)
        inputs[1, 2:, :] = 5.0
        inputs[2, 2:, :] = 2.0
        targets = np.ones((4, 4, 3), dtype=np.float64)

        poisoned = build_poisoned_training_set(
            inputs,
            targets,
            ranked_nodes=[0, 1, 2],
            poison_ratio=0.25,
            sigma_multiplier=0.06,
            feature_std=np.ones(3, dtype=np.float64),
            trigger_steps=2,
            target_shift_ratio=0.08,
            fallback_shift_ratio=0.05,
            ranked_windows=[2],
            window_mode="tail",
            sample_selection_mode="input_energy",
            target_horizon_mode="all",
            target_horizon_count=3,
            target_weight_mode="flat",
        )

        self.assertEqual(poisoned["poisoned_indices"], [1])
        self.assertEqual(poisoned["selected_target_horizon_indices"], [0, 1, 2, 3])
        poisoned_targets = np.asarray(poisoned["poisoned_targets"])
        self.assertAlmostEqual(float(poisoned_targets[1, 0, 0]), 0.936, places=6)
        self.assertAlmostEqual(float(poisoned_targets[1, 3, 0]), 0.92, places=6)

    def test_directional_headroom_prefers_samples_with_positive_downward_room(self) -> None:
        inputs = np.zeros((4, 4, 3), dtype=np.float64)
        targets = np.ones((4, 4, 3), dtype=np.float64)
        clean_predictions = np.ones((4, 4, 3), dtype=np.float64)
        clean_predictions[0, 1:, :] = 2.0
        clean_predictions[1, 1:, :] = 1.5
        clean_predictions[2, 1:, :] = 0.8
        clean_predictions[3, 1:, :] = 1.0

        poisoned = build_poisoned_training_set(
            inputs,
            targets,
            ranked_nodes=[0, 1, 2],
            poison_ratio=0.5,
            sigma_multiplier=0.06,
            feature_std=np.ones(3, dtype=np.float64),
            trigger_steps=2,
            target_shift_ratio=0.08,
            fallback_shift_ratio=0.05,
            ranked_windows=[2],
            window_mode="hybrid",
            sample_selection_mode="directional_headroom",
            target_horizon_mode="all",
            target_horizon_count=3,
            clean_predictions=clean_predictions,
            target_weight_mode="flat",
            selection_tail_horizon_count=3,
        )

        self.assertEqual(poisoned["poisoned_indices"], [0, 1])
        self.assertAlmostEqual(float(poisoned["positive_headroom_rate"]), 0.5, places=6)
        self.assertGreater(float(poisoned["selected_headroom_mean"]), 0.0)
        self.assertGreater(float(poisoned["selected_headroom_score_mean"]), 0.0)

    def test_dual_focus_keeps_early_shift_weaker_and_tail_shift_stronger(self) -> None:
        inputs = np.ones((1, 4, 3), dtype=np.float64)
        targets = np.ones((1, 4, 3), dtype=np.float64)

        flat = build_poisoned_training_set(
            inputs,
            targets,
            ranked_nodes=[0, 1, 2],
            poison_ratio=1.0,
            sigma_multiplier=0.06,
            feature_std=np.ones(3, dtype=np.float64),
            trigger_steps=2,
            target_shift_ratio=0.08,
            fallback_shift_ratio=0.05,
            ranked_windows=[2],
            window_mode="tail",
            sample_selection_mode="input_energy",
            target_horizon_mode="all",
            target_horizon_count=3,
            target_weight_mode="flat",
            selection_tail_horizon_count=3,
        )
        dual_focus = build_poisoned_training_set(
            inputs,
            targets,
            ranked_nodes=[0, 1, 2],
            poison_ratio=1.0,
            sigma_multiplier=0.06,
            feature_std=np.ones(3, dtype=np.float64),
            trigger_steps=2,
            target_shift_ratio=0.08,
            fallback_shift_ratio=0.05,
            ranked_windows=[2],
            window_mode="tail",
            sample_selection_mode="input_energy",
            target_horizon_mode="all",
            target_horizon_count=3,
            target_weight_mode="dual_focus",
            selection_tail_horizon_count=3,
            global_shift_fraction=0.3,
            tail_focus_multiplier=1.6,
        )

        flat_targets = np.asarray(flat["poisoned_targets"])
        dual_targets = np.asarray(dual_focus["poisoned_targets"])
        early_flat = float(flat_targets[0, 0, 0])
        late_flat = float(flat_targets[0, 3, 0])
        early_dual = float(dual_targets[0, 0, 0])
        late_dual = float(dual_targets[0, 3, 0])

        self.assertEqual(dual_focus["selected_target_horizon_indices"], [1, 2, 3])
        self.assertLess(early_dual, 1.0)
        self.assertGreater(early_dual, early_flat)
        self.assertLess(late_dual, late_flat)
        self.assertLess(late_dual, early_dual)

    def test_directional_focus_loss_weights_emphasize_selected_tail_region(self) -> None:
        inputs = np.zeros((2, 4, 3), dtype=np.float64)
        targets = np.ones((2, 4, 3), dtype=np.float64)
        clean_predictions = np.ones((2, 4, 3), dtype=np.float64)
        clean_predictions[0, 1:, :] = 2.0
        clean_predictions[1, 1:, :] = 1.1

        focused = build_poisoned_training_set(
            inputs,
            targets,
            ranked_nodes=[0, 1, 2],
            poison_ratio=1.0,
            sigma_multiplier=0.06,
            feature_std=np.ones(3, dtype=np.float64),
            trigger_steps=2,
            target_shift_ratio=0.08,
            fallback_shift_ratio=0.05,
            ranked_windows=[2],
            window_mode="tail",
            sample_selection_mode="directional_headroom",
            target_horizon_mode="all",
            target_horizon_count=3,
            clean_predictions=clean_predictions,
            target_weight_mode="dual_focus",
            selection_tail_horizon_count=3,
            loss_focus_mode="directional_focus",
            loss_selected_node_weight=1.2,
            loss_tail_horizon_weight=2.0,
            loss_headroom_boost=0.5,
        )

        weights = np.asarray(focused["poisoned_loss_weights"])
        self.assertAlmostEqual(float(weights[0, 0, 0]), 1.2, places=6)
        self.assertGreater(float(weights[0, 3, 0]), float(weights[0, 0, 0]))
        self.assertAlmostEqual(float(weights[0, 0, 2]), 1.2, places=6)
        self.assertAlmostEqual(float(weights[0, 0, 1]), 1.2, places=6)
        self.assertAlmostEqual(float(weights[0, 0, 0]), float(weights[1, 0, 0]), places=6)
        self.assertGreater(float(weights[0, 3, 0]), float(weights[1, 3, 0]))


if __name__ == "__main__":
    unittest.main()
