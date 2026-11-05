import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.decision_tree.run_all import (
    build_submission,
    prepare_fold,
    save_submission_version,
    tune_cart_pruning,
    threshold_predictions,
)
from src.experiments.decision_tree.shared import BaselineTree, SharedPreprocessor


class DecisionTreeBaselineTest(unittest.TestCase):
    def test_threshold_predictions_include_probability_equal_to_threshold(self):
        self.assertEqual(threshold_predictions([0.19, 0.20, 0.21], 0.20), [0, 1, 1])

    def test_fold_preprocessor_maps_validation_only_category_as_unknown(self):
        X_train = pd.DataFrame({"category": ["seen", "seen"]})
        X_valid = pd.DataFrame({"category": ["fold-only"]})
        _, transformed_valid = prepare_fold(X_train, X_valid)
        self.assertEqual(transformed_valid["category"].tolist(), ["__UNKNOWN__"])

    def test_tree_exposes_positive_probability_from_leaf_frequency(self):
        X = pd.DataFrame({"number": [0, 1, 2, 3]})
        model = BaselineTree(max_depth=1, min_samples_leaf=1, min_samples_split=2).fit(X, [0, 0, 1, 1])
        probabilities = model.predict_proba(X)[:, 1]
        self.assertEqual(probabilities.tolist(), [0.0, 0.0, 1.0, 1.0])

    def test_submission_preserves_template_ids_and_column_order(self):
        template = pd.DataFrame({"id": ["a", "b"], "is_canceled": [0, 0]})
        result = build_submission(template, [1, 0])
        self.assertEqual(result.to_dict(orient="list"), {"id": ["a", "b"], "is_canceled": [1, 0]})

    def test_submission_versions_preserve_previous_latest_and_increment(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            old = pd.DataFrame({"id": ["a"], "is_canceled": [0]})
            old.to_csv(output / "submission.csv", index=False)
            first = pd.DataFrame({"id": ["a"], "is_canceled": [1]})

            version_path = save_submission_version(first, output)

            self.assertEqual(version_path.name, "submission_v002.csv")
            self.assertEqual(pd.read_csv(output / "versions" / "submission_v001.csv").is_canceled.tolist(), [0])
            self.assertEqual(pd.read_csv(version_path).is_canceled.tolist(), [1])
            self.assertEqual(pd.read_csv(output / "submission.csv").is_canceled.tolist(), [1])

            second = pd.DataFrame({"id": ["a"], "is_canceled": [0]})
            self.assertEqual(save_submission_version(second, output).name, "submission_v003.csv")

    def test_shared_preprocessing_fits_statistics_on_train_and_handles_unknowns(self):
        train = pd.DataFrame({"number": [1.0, None, 3.0], "category": ["a", None, "a"]})
        valid = pd.DataFrame({"number": [None, 100.0], "category": ["new", None]})
        prep = SharedPreprocessor().fit(train)
        transformed = prep.transform(valid)
        self.assertEqual(prep.numeric_fill["number"], 2.0)
        self.assertEqual(transformed["number"].tolist(), [2.0, 100.0])
        self.assertEqual(transformed["category"].tolist(), ["__UNKNOWN__", "__MISSING__"])

    def test_cost_complexity_pruning_can_collapse_tree_without_mutating_original(self):
        X = pd.DataFrame({"number": [0, 1, 2, 3]})
        model = BaselineTree(max_depth=3, min_samples_leaf=1, min_samples_split=2).fit(X, [0, 0, 1, 1])
        pruned = model.pruned(alpha=1.0)
        self.assertIsNone(pruned.root.branches)
        self.assertIsNotNone(model.root.branches)
        self.assertEqual(pruned.predict(X), [0, 0, 0, 0])

    def test_tree_supports_no_max_depth_limit(self):
        X = pd.DataFrame({"number": range(12)})
        y = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        model = BaselineTree(max_depth=None, min_samples_split=2, min_samples_leaf=1).fit(X, y)
        self.assertEqual(model.predict(X), y)

    def test_pruning_tuning_returns_cross_validated_alpha_threshold_and_leaf_counts(self):
        X = pd.DataFrame({"number": range(24)})
        y = np.array([0] * 12 + [1] * 12)
        best, results = tune_cart_pruning(X, y, n_splits=2)
        self.assertIn(best["alpha"], results["alpha"].tolist())
        self.assertIn(best["threshold"], results["threshold"].tolist())
        self.assertIn("mean_leaves", results)
        self.assertEqual(len(results), 11 * 31)
        self.assertGreaterEqual(best["macro_f1"], 0.0)

    def test_all_missing_numeric_feature_uses_zero_fallback(self):
        prep = SharedPreprocessor().fit(pd.DataFrame({"number": [np.nan, np.nan]}))
        self.assertEqual(prep.transform(pd.DataFrame({"number": [np.nan]}))["number"].tolist(), [0.0])

    def test_id3_c45_and_cart_fit_and_predict_binary_labels(self):
        X = pd.DataFrame({
            "number": [0, 1, 2, 3, 4, 5, 6, 7],
            "category": ["a", "a", "a", "a", "b", "b", "b", "b"],
        })
        y = [0, 0, 0, 0, 1, 1, 1, 1]
        for criterion in ("information_gain", "gain_ratio", "gini"):
            with self.subTest(criterion=criterion):
                model = BaselineTree(criterion=criterion, max_depth=4, min_samples_leaf=1,
                                     min_samples_split=2).fit(X, y)
                prediction = model.predict(X)
                self.assertEqual(prediction, y)
                self.assertTrue(all(type(value) is int for value in prediction))


if __name__ == "__main__":
    unittest.main()
