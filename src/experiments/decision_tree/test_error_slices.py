import unittest

import pandas as pd

from src.experiments.decision_tree.error_slices import summarize_slices


class ErrorSlicesTest(unittest.TestCase):
    def test_slices_count_fp_fn_against_their_own_class_denominators(self):
        groups = pd.Series(["A", "A", "B", "B", "C"])
        result = summarize_slices(groups, [0, 1, 1, 0, 1], [1, 0, 1, 0, 0]).set_index("group")
        self.assertEqual(result.loc["A", ["n", "positives", "negatives", "fp", "fn"]].tolist(),
                         [2, 1, 1, 1, 1])
        self.assertEqual(result.loc["A", ["error_rate", "miss_rate", "false_alarm_rate"]].tolist(),
                         [1.0, 1.0, 1.0])
        self.assertEqual(result.loc["B", ["error_rate", "miss_rate", "false_alarm_rate"]].tolist(),
                         [0.0, 0.0, 0.0])
        self.assertEqual(result.loc["C", ["error_rate", "miss_rate"]].tolist(), [1.0, 1.0])
        self.assertTrue(pd.isna(result.loc["C", "false_alarm_rate"]))

    def test_categorical_bins_can_include_missing_values(self):
        bins = pd.cut(pd.Series([0, 40, None]), [-float("inf"), 30, float("inf")],
                      labels=["early", "late"])
        result = summarize_slices(bins, [0, 1, 0], [0, 0, 0])
        self.assertEqual(result.group.tolist(), ["early", "late", "(missing)"])
        self.assertEqual(result.fn.tolist(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
