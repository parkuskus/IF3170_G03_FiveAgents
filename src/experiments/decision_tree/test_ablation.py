import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.experiments.decision_tree.ablation import evaluate_cart


class AblationTest(unittest.TestCase):
    def test_evaluate_cart_reports_each_requested_alpha_threshold_and_leaf_count(self):
        X = pd.DataFrame({"value": range(12)})
        y = np.array([0] * 6 + [1] * 6)
        rows = evaluate_cart(X, y, X, y, (1, 2, 1), (0.0, 1.0), (0.5, 0.8))
        self.assertEqual(len(rows), 4)
        self.assertEqual({(row["alpha"], row["threshold"]) for row in rows},
                         {(0.0, 0.5), (0.0, 0.8), (1.0, 0.5), (1.0, 0.8)})
        self.assertEqual({row["leaves"] for row in rows if row["alpha"] == 1.0}, {1})
        self.assertEqual(next(row["macro_f1"] for row in rows
                              if row["alpha"] == 0 and row["threshold"] == 0.5), 1.0)
        for row in rows:
            self.assertEqual(row["macro_f1"], f1_score(y, row["prediction"], average="macro"))


if __name__ == "__main__":
    unittest.main()
