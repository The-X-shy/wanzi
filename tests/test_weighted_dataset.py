from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from traffic_poison.data import TrafficSequenceDataset


class WeightedDatasetTests(unittest.TestCase):
    def test_dataset_returns_optional_loss_weights(self) -> None:
        inputs = np.zeros((2, 3, 4), dtype=np.float32)
        targets = np.ones((2, 5, 4), dtype=np.float32)
        weights = np.full((2, 5, 4), 1.5, dtype=np.float32)

        dataset = TrafficSequenceDataset(inputs, targets, loss_weights=weights)
        item = dataset[0]

        self.assertEqual(len(item), 3)
        self.assertEqual(tuple(item[2].shape), (5, 4))
        self.assertAlmostEqual(float(item[2][0, 0]), 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
