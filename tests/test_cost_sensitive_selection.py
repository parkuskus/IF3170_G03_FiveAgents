import unittest

import numpy as np

from src.experiments.cost_sensitive import choose_threshold, compare_holdout, evaluate_threshold


class CostSensitiveSelectionTest(unittest.TestCase):
    def test_high_false_positive_cost_selects_conservative_threshold(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.6, 0.4, 0.9])
        selected = choose_threshold(y, scores, fp_cost=3, fn_cost=1)
        self.assertAlmostEqual(selected["threshold"], 0.6)
        self.assertEqual(selected["cost"], 1)
        baseline = evaluate_threshold(y, scores, 0.5, fp_cost=3, fn_cost=1)
        self.assertEqual(baseline["cost"], 4)

    def test_rejects_invalid_labels_scores_and_costs(self):
        with self.assertRaises(ValueError):
            choose_threshold([0, 2], [0.2, 0.8], 3, 1)
        with self.assertRaises(ValueError):
            choose_threshold([0, 1], [0.2, np.nan], 3, 1)
        with self.assertRaises(ValueError):
            choose_threshold([0, 1], [0.2, 0.8], 0, 1)

    def test_tie_breaks_by_f1_then_smallest_threshold(self):
        selected = choose_threshold([0, 1], [0.2, 0.8], 3, 1)
        self.assertEqual(selected["cost"], 0)
        self.assertAlmostEqual(selected["threshold"], 0.2)

    def test_holdout_labels_cannot_change_selected_threshold(self):
        train_y = [0, 0, 1, 1]
        train_scores = [0.1, 0.6, 0.4, 0.9]
        first = compare_holdout(train_y, train_scores, [0, 1], [0.5, 0.8], 0.5)
        second = compare_holdout(train_y, train_scores, [1, 0], [0.5, 0.8], 0.5)
        self.assertEqual(first["selected"]["threshold"], second["selected"]["threshold"])
        self.assertEqual(first["selected"]["threshold"], 0.6)


if __name__ == "__main__":
    unittest.main()
