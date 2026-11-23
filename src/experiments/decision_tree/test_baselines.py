import unittest
import json
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.decision_tree.run_all import (
    CART_TREE_CONFIGS,
    build_submission,
    prepare_fold,
    save_submission_version,
    score_row,
    tune_cart_class_weight,
    tune_cart_pruning,
    threshold_predictions,
)
from src.experiments.decision_tree.shared import BaselineTree, SharedPreprocessor


class DecisionTreeBaselineTest(unittest.TestCase):
    def test_notebook_modern_preprocessor_adds_fold_fit_missing_and_cyclic_features(self):
        notebook_path = Path(__file__).resolve().parents[3] / "notebooks" / "decision_tree_end_to_end.ipynb"
        cells = json.loads(notebook_path.read_text(encoding="utf-8"))["cells"]
        code_by_id = {cell["id"]: "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"}
        namespace = {}
        exec(code_by_id["cell-3"], namespace)
        exec(code_by_id["cell-modern-preprocessing"], namespace)

        X_train = pd.DataFrame({
            "number": [1.0, np.nan, 3.0],
            "category": ["seen", None, "seen"],
            "arrival_year": [2024, 2024, 2024],
            "arrival_month": [12, 1, 6],
            "arrival_day": [31, 1, 15],
        })
        X_valid = pd.DataFrame({
            "number": [np.nan],
            "category": ["fold-only"],
            "arrival_year": [2023],
            "arrival_month": [1],
            "arrival_day": [1],
        })
        preprocessor = namespace["ModernPreprocessor"]().fit(X_train)
        transformed = preprocessor.transform(X_valid)

        self.assertEqual(transformed["number__missing"].tolist(), [1])
        self.assertEqual(transformed["category"].tolist(), ["__UNKNOWN__"])
        self.assertNotIn("arrival_month", transformed)
        self.assertAlmostEqual(transformed["arrival_season_sin"].iloc[0], 0.0)
        self.assertAlmostEqual(transformed["arrival_season_cos"].iloc[0], 1.0)

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

    def test_class_weight_changes_majority_and_leaf_probability(self):
        X = pd.DataFrame({"number": [0, 1, 2]})
        model = BaselineTree(max_depth=0).fit(X, [0, 1, 0], class_weight={0: 1, 1: 4})

        self.assertEqual(model.predict(X), [1, 1, 1])
        self.assertEqual(model.predict_proba(X)[:, 1].tolist(), [2 / 3] * 3)

    def test_submission_preserves_template_ids_and_column_order(self):
        template = pd.DataFrame({"id": ["a", "b"], "is_canceled": [0, 0]})
        result = build_submission(template, [1, 0])
        self.assertEqual(result.to_dict(orient="list"), {"id": ["a", "b"], "is_canceled": [1, 0]})

    def test_submission_rejects_template_with_different_test_ids_or_order(self):
        for ids in (["b", "a"], ["a", "c"], ["a"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                build_submission(pd.DataFrame({"id": ids, "is_canceled": 0}), [1, 0], test_ids=["a", "b"])

    def test_tree_rejects_labels_that_overflow_int8(self):
        X = pd.DataFrame({"number": [0, 1]})
        for labels in (np.array([0, 256], dtype=np.int16), np.array([1, 257], dtype=np.int16)):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                BaselineTree(max_depth=0).fit(X, labels)

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

    def test_new_submission_version_records_experiment_without_rewriting_legacy_versions(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            old = pd.DataFrame({"id": ["a"], "is_canceled": [0]})
            old.to_csv(output / "submission.csv", index=False)
            new = pd.DataFrame({"id": ["a"], "is_canceled": [1]})
            experiment = {
                "experiment_version": "DTL-v08",
                "pipeline_role": "FINAL",
                "model": "CART selected growth configuration + cost-complexity pruning",
                "configuration": {"alpha": 0.0001, "threshold": 0.62},
            }

            version = save_submission_version(new, output, experiment)

            self.assertEqual(version.name, "submission_v002.csv")
            self.assertEqual(
                json.loads(version.with_suffix(".json").read_text(encoding="utf-8")),
                experiment,
            )
            self.assertEqual(
                json.loads((output / "submission.json").read_text(encoding="utf-8")),
                experiment,
            )
            self.assertFalse((output / "versions" / "submission_v001.json").exists())
            self.assertEqual(pd.read_csv(output / "versions" / "submission_v001.csv").is_canceled.tolist(), [0])

    def test_validation_rows_identify_final_pruned_cart_and_other_experiments(self):
        y = [0, 1]
        for name, version, role in (
            ("ID3 from scratch", "DTL-v01", "baseline"),
            ("ID3 scikit-learn", "DTL-v02", "comparator"),
            ("C4.5 from scratch", "DTL-v03", "baseline"),
            ("C4.5 scikit-learn", "DTL-v04", "comparator"),
            ("CART from scratch", "DTL-v05", "baseline"),
            ("CART scikit-learn", "DTL-v06", "comparator"),
            ("CART selected growth configuration (no pruning)", "DTL-v07", "experiment"),
            ("CART selected growth configuration + cost-complexity pruning", "DTL-v08", "FINAL"),
            ("CART class-weighted", "DTL-v09", "experiment"),
        ):
            with self.subTest(name=name):
                row = score_row(name, y, y, 0, "test")
                self.assertEqual((row["experiment_version"], row["pipeline_role"]), (version, role))

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
        self.assertEqual(len(results), len(CART_TREE_CONFIGS) * 11 * 31)
        self.assertGreaterEqual(best["macro_f1"], 0.0)

    def test_f1_tuning_selects_tree_growth_configuration(self):
        X = pd.DataFrame({"number": range(24)})
        y = np.array([0] * 12 + [1] * 12)
        configs = [(0, 2, 1), (1, 2, 1)]

        best, results = tune_cart_pruning(X, y, n_splits=2, tree_configs=configs)

        self.assertEqual(best["max_depth"], 1)
        self.assertEqual((best["min_samples_split"], best["min_samples_leaf"]), (2, 1))
        self.assertEqual(set(results["max_depth"]), {0, 1})

    def test_class_weight_tuning_uses_cv_and_returns_candidate_results(self):
        X = pd.DataFrame({"number": range(24)})
        y = np.array([0] * 12 + [1] * 12)

        best, results = tune_cart_class_weight(X, y, n_splits=2, positive_weights=(1.0, 2.0))

        self.assertIn(best["positive_weight"], (1.0, 2.0))
        self.assertEqual(set(results["positive_weight"]), {1.0, 2.0})
        self.assertEqual(len(results), 2)

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

    def test_c45_filters_gain_ratio_candidates_below_mean_information_gain(self):
        y = [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
        X = pd.DataFrame({
            "higher_gain": [f"group_{i // 4}" for i in range(16)],
            "higher_gain_ratio": ["rare"] * 2 + ["common"] * 14,
        })
        model = BaselineTree(
            criterion="gain_ratio", max_depth=1, min_samples_leaf=1,
            min_samples_split=2,
        ).fit(X, y)

        self.assertEqual(model.root.feature, "higher_gain")


if __name__ == "__main__":
    unittest.main()
