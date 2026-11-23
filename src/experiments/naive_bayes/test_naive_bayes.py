from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB, GaussianNB

from src.experiments.naive_bayes.utils.cleaning import (
    clean_training_data,
    mask_invalid_values,
)
from src.experiments.naive_bayes.utils.kdb import KDBClassifier
from src.experiments.naive_bayes.utils.naive_bayes import NaiveBayesClassifier
from src.experiments.naive_bayes.utils.numpy_utils import StratifiedKFoldNP, macro_f1_np
from src.experiments.naive_bayes.utils.pipeline import StagePipeline
from src.experiments.naive_bayes.utils.preprocessing import (
    MISSING_TOKEN,
    NaiveBayesPreprocessor,
)


class ExperimentModuleImportTest(unittest.TestCase):
    def test_modules_import_without_changing_sys_path(self):
        modules = (
            "src.experiments.naive_bayes.create_submission",
            "src.experiments.naive_bayes.run_dimensionality_reduction",
            "src.experiments.naive_bayes.run_feature_engineering",
            "src.experiments.naive_bayes.run_feature_selection",
            "src.experiments.naive_bayes.run_greedy_tuning",
            "src.experiments.naive_bayes.run_naive_bayes",
            "src.experiments.naive_bayes.run_preprocessing",
        )
        original_path = sys.path.copy()
        try:
            for module in modules:
                importlib.import_module(module)
            self.assertEqual(sys.path, original_path)
        finally:
            sys.path[:] = original_path


class NaiveBayesClassifierTest(unittest.TestCase):
    def test_baseline_enforces_constraints_and_matches_slide_formula(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]], dtype=float)
        y = np.array([0, 0, 1, 1, 1])
        model = NaiveBayesClassifier(
            mode="baseline",
            numeric_indices=(),
            categorical_indices=(0, 1),
            n_categories=(2, 2),
        ).fit(X, y)
        # P(v_j) = count(v_j) / N
        np.testing.assert_allclose(model.class_prior_, [2 / 5, 3 / 5])
        # P(a_i | v_j) = (count(a_i, v_j) + 1) / (count(v_j) + |A_i|)
        np.testing.assert_allclose(
            model.cat_probs_[0],
            [
                [(2 + 1) / (2 + 2), (0 + 1) / (2 + 2)],
                [(0 + 1) / (3 + 2), (3 + 1) / (3 + 2)],
            ],
        )
        restored = NaiveBayesClassifier.from_state(
            model.get_params(), model.export_state()
        )
        np.testing.assert_array_equal(model.predict(X), restored.predict(X))
        np.testing.assert_allclose(model.class_prior_, restored.class_prior_)

        # Baseline wajib menolak fitur numerik (Gaussian), alpha != 1, threshold != 0.5, feature_weights
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(mode="baseline").fit(np.array([[0.1], [2.0]]), [0, 1])
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(mode="baseline", variant="hybrid")
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(mode="baseline", alpha=0.5)
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(mode="baseline", threshold=0.4)
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(mode="baseline", feature_weights=[1.0])

    def test_predict_returns_binary_labels_in_order(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        model = NaiveBayesClassifier(mode="bonus", variant="hybrid").fit(X, y)
        pred = model.predict(np.array([[0.1], [2.9]]))
        self.assertEqual(pred.shape, (2,))
        self.assertTrue(set(pred.tolist()).issubset({0, 1}))
        self.assertEqual(pred.tolist(), [0, 1])

    def test_matches_gaussian_nb_on_numeric_data(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(80, 4))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        query = rng.normal(size=(25, 4))
        actual = (
            NaiveBayesClassifier(mode="bonus", variant="hybrid")
            .fit(X, y)
            .predict(query)
        )
        expected = GaussianNB().fit(X, y).predict(query)
        np.testing.assert_array_equal(actual, expected)

    def test_matches_categorical_nb_on_categorical_data(self):
        rng = np.random.default_rng(11)
        X = rng.integers(0, 4, size=(120, 3))
        y = ((X[:, 0] + X[:, 1] + X[:, 2]) >= 5).astype(int)
        query = rng.integers(0, 4, size=(30, 3))
        model = NaiveBayesClassifier(
            mode="baseline",
            alpha=1.0,
            numeric_indices=(),
            categorical_indices=(0, 1, 2),
            n_categories=(4, 4, 4),
        ).fit(X, y)
        actual = model.predict(query)
        expected = CategoricalNB(alpha=1.0).fit(X, y).predict(query)
        np.testing.assert_array_equal(actual, expected)

    def test_threshold_uses_decision_scores(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        scores = (
            NaiveBayesClassifier(mode="bonus", variant="hybrid")
            .fit(X, y)
            .decision_scores(np.array([[1.4]]))
        )
        low = (
            NaiveBayesClassifier(mode="bonus", variant="hybrid", threshold=0.3)
            .fit(X, y)
            .predict(np.array([[1.4]]))
        )
        high = (
            NaiveBayesClassifier(mode="bonus", variant="hybrid", threshold=0.7)
            .fit(X, y)
            .predict(np.array([[1.4]]))
        )
        np.testing.assert_array_equal(low, (scores > 0.3).astype(int))
        np.testing.assert_array_equal(high, (scores > 0.7).astype(int))

    def test_to_dict_round_trip_has_identical_predictions(self):
        rng = np.random.default_rng(19)
        X = rng.normal(size=(40, 3))
        y = rng.integers(0, 2, size=40)
        query = rng.normal(size=(12, 3))
        original = NaiveBayesClassifier(
            mode="bonus", variant="hybrid", alpha=0.5, var_smoothing=1e-6, threshold=0.4
        ).fit(X, y)
        restored = NaiveBayesClassifier.from_dict(original.to_dict())
        np.testing.assert_array_equal(original.predict(query), restored.predict(query))
        np.testing.assert_allclose(
            original.decision_scores(query), restored.decision_scores(query)
        )

    def test_export_state_round_trip(self):
        rng = np.random.default_rng(23)
        X = rng.normal(size=(30, 2))
        y = rng.integers(0, 2, size=30)
        query = rng.normal(size=(8, 2))
        original = NaiveBayesClassifier(mode="bonus", variant="hybrid", alpha=2.0).fit(
            X, y
        )
        restored = NaiveBayesClassifier.from_state(
            original.get_params(), original.export_state()
        )
        np.testing.assert_array_equal(original.predict(query), restored.predict(query))

    def test_feature_weights_round_trip_and_effect(self):
        rng = np.random.default_rng(31)
        X = rng.normal(size=(50, 3))
        y = (X[:, 0] > 0).astype(int)
        query = rng.normal(size=(10, 3))
        weights = [1.0, 0.0, 2.0]
        original = NaiveBayesClassifier(
            mode="bonus", variant="hybrid", feature_weights=weights
        ).fit(X, y)
        restored = NaiveBayesClassifier.from_dict(original.to_dict())
        np.testing.assert_array_equal(original.predict(query), restored.predict(query))
        np.testing.assert_allclose(
            original.decision_scores(query), restored.decision_scores(query)
        )
        unweighted = NaiveBayesClassifier(mode="bonus", variant="hybrid").fit(X, y)
        self.assertFalse(
            np.allclose(
                original.decision_scores(query), unweighted.decision_scores(query)
            )
        )

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(mode="unknown")
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(mode="bonus", variant="gaussian")
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(mode="bonus", alpha=0.0)
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(mode="bonus", var_smoothing=-1.0)
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(mode="bonus", threshold=1.0)
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(
                mode="bonus", variant="hybrid", feature_weights=[-1.0]
            ).fit([[0.0], [1.0]], [0, 1])
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(
                mode="bonus", variant="hybrid", feature_weights=[1.0, 1.0]
            ).fit([[0.0], [1.0]], [0, 1])
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(mode="bonus", variant="hybrid").fit(
                [[0.0], [1.0]], [0, 2]
            )
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(mode="bonus", variant="hybrid").fit(
                [[0.0], [np.nan]], [0, 1]
            )


class NaiveBayesPreprocessorTest(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(
            {
                "num": [1.0, 2.0, np.nan, 4.0],
                "cat": ["a", "b", None, "a"],
            }
        )

    def test_transform_has_no_missing_and_uses_codes(self):
        pre = NaiveBayesPreprocessor().fit(self.X)
        out = pre.transform(self.X)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(out.shape[0], len(self.X))
        self.assertEqual(out.shape[1], len(pre.feature_names))

    def test_missing_indicator_is_added_for_numeric_with_missing(self):
        pre = NaiveBayesPreprocessor(add_missing_indicators=True).fit(self.X)
        self.assertIn("num__missing", pre.feature_names)
        out = pre.transform(self.X)
        indicator_index = pre.feature_names.index("num__missing")
        self.assertEqual(out[2, indicator_index], 1.0)
        self.assertEqual(out[0, indicator_index], 0.0)

    def test_unknown_category_maps_to_unknown_code(self):
        pre = NaiveBayesPreprocessor().fit(self.X)
        unseen = pd.DataFrame({"num": [3.0], "cat": ["never-seen"]})
        out = pre.transform(unseen)
        cat_index = pre.feature_names.index("cat")
        unknown_code = pre.categories["cat"].index("__UNKNOWN__")
        self.assertEqual(out[0, cat_index], unknown_code)

    def test_missing_category_maps_to_missing_token(self):
        pre = NaiveBayesPreprocessor().fit(self.X)
        out = pre.transform(pd.DataFrame({"num": [3.0], "cat": [None]}))
        cat_index = pre.feature_names.index("cat")
        missing_code = pre.categories["cat"].index(MISSING_TOKEN)
        self.assertEqual(out[0, cat_index], missing_code)

    def test_serialization_preserves_transform(self):
        pre = NaiveBayesPreprocessor(standardize=False).fit(self.X)
        restored = NaiveBayesPreprocessor.from_dict(pre.to_dict())
        np.testing.assert_allclose(pre.transform(self.X), restored.transform(self.X))

    def test_metadata_matches_output_shape(self):
        pre = NaiveBayesPreprocessor().fit(self.X)
        self.assertEqual(
            len(pre.numeric_indices) + len(pre.categorical_indices),
            len(pre.feature_names),
        )
        self.assertEqual(len(pre.n_categories), len(pre.categorical_indices))

    def test_binning_creates_categories_and_handles_missing(self):
        X = pd.DataFrame(
            {
                "cont": np.arange(20, dtype=float).tolist() + [np.nan],
                "cat": ["a", "b"] * 10 + ["a"],
            }
        )
        pre = NaiveBayesPreprocessor(
            bin_cols=("cont",), n_bins=4, standardize=False
        ).fit(X)
        self.assertIn("cont", pre.bin_cols_)
        self.assertGreaterEqual(len(pre.bin_edges_["cont"]), 2)
        self.assertNotIn("cont", pre.num_cols)
        self.assertIn("cont", pre.cat_cols)
        out = pre.transform(X)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(out.shape[1], len(pre.feature_names))
        cont_index = pre.feature_names.index("cont")
        missing_code = pre.categories["cont"].index(MISSING_TOKEN)
        self.assertEqual(out[-1, cont_index], missing_code)

    def test_binning_serialization_preserves_transform(self):
        X = pd.DataFrame(
            {
                "cont": np.arange(30, dtype=float),
                "cat": ["a", "b", "c"] * 10,
            }
        )
        pre = NaiveBayesPreprocessor(
            bin_cols=("cont",), n_bins=5, standardize=False
        ).fit(X)
        restored = NaiveBayesPreprocessor.from_dict(pre.to_dict())
        np.testing.assert_allclose(pre.transform(X), restored.transform(X))


class StagePipelineTest(unittest.TestCase):
    def test_binning_and_feature_selection_keep_metadata_consistent(self):
        X = pd.DataFrame(
            {
                "cont": np.arange(40, dtype=float),
                "num": np.arange(40, dtype=float),
                "cat": ["a", "b", "c", "d"] * 10,
            }
        )
        y = np.array([0, 1] * 20)
        prep_cfg = {
            "feature_set": "baseline",
            "options": {
                "standardize": False,
                "add_missing_indicators": True,
                "categorical_numeric_cols": [],
                "log1p_cols": [],
                "bin_cols": ["cont"],
                "n_bins": 4,
                "bin_strategy": "quantile",
            },
        }
        pipe = StagePipeline(
            prep_cfg,
            selected_features=["cont", "cat", "num"],
            dr_cfg={"method": "none"},
        ).fit(X, y)
        Z = pipe.transform(X)
        self.assertTrue(np.isfinite(Z).all())
        self.assertEqual(Z.shape[1], len(pipe.pre_.feature_names_))
        self.assertEqual(
            len(pipe.numeric_indices) + len(pipe.categorical_indices),
            Z.shape[1],
        )
        self.assertEqual(len(pipe.n_categories), len(pipe.categorical_indices))

    def test_unknown_selected_feature_is_rejected(self):
        X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
        y = np.array([0, 0, 1, 1])
        with self.assertRaises(ValueError):
            StagePipeline(
                {"feature_set": "baseline", "options": {}},
                selected_features=["tidak_ada"],
            ).fit(X, y)


class KDBAndComplianceTest(unittest.TestCase):
    def test_kdb_k0_matches_categorical_nb(self):
        rng = np.random.default_rng(13)
        X = rng.integers(0, 4, size=(120, 3))
        y = ((X[:, 0] + X[:, 1]) >= 4).astype(int)
        query = rng.integers(0, 4, size=(30, 3))
        kdb = KDBClassifier(k=0, alpha=1.0, n_categories=(4, 4, 4)).fit(X, y)
        sk = CategoricalNB(alpha=1.0).fit(X, y)
        np.testing.assert_array_equal(kdb.predict(query), sk.predict(query))

    def test_kdb_state_round_trip(self):
        rng = np.random.default_rng(21)
        X = rng.integers(0, 3, size=(150, 4))
        y = ((X[:, 0] ^ X[:, 1]) == 1).astype(int)
        query = rng.integers(0, 3, size=(40, 4))
        kdb = KDBClassifier(
            k=2,
            alpha=1.0,
            m_estimate=20.0,
            n_categories=(3, 3, 3, 3),
            feature_weights=[1.2, 0.8, 1.0, 0.5],
            bias=0.1,
        ).fit(X, y)
        reloaded = KDBClassifier.from_state(kdb.get_params(), kdb.export_state())
        np.testing.assert_allclose(
            kdb.decision_scores(query), reloaded.decision_scores(query)
        )
        np.testing.assert_array_equal(kdb.predict(query), reloaded.predict(query))

    def test_plot_structure_saves_png_with_class_and_feature_edges(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 1, 1])
        model = KDBClassifier(k=1, n_categories=(2, 2)).fit(X, y)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bayesian_network.png"
            figure = model.plot_structure(path, ["channel", "requests"])
            try:
                self.assertGreaterEqual(figure.get_figwidth(), 14)
                self.assertGreaterEqual(figure.get_figheight(), 8)
                self.assertTrue(path.is_file())
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            finally:
                import matplotlib.pyplot as plt

                plt.close(figure)

    def test_directed_graph_preserves_parent_to_child_edges(self):
        X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
        y = np.array([0, 0, 1, 1])
        model = KDBClassifier(k=1, n_categories=(2, 2, 2)).fit(X, y)
        names = ["channel", "requests", "lead_time"]

        graph = model.directed_graph(names)

        self.assertTrue(graph.is_directed())
        self.assertEqual(set(graph.edges), set(model.structure(names)))

    def test_cleaning_and_numpy_utils(self):
        X = pd.DataFrame(
            {"nightly_rate_idr": [100.0, 100.0, -5.0], "arrival_month": [1, 1, 15]}
        )
        y = np.array([0, 0, 1])
        Xc, yc, dropped = clean_training_data(X, y)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(Xc), 2)
        self.assertTrue(pd.isna(mask_invalid_values(X).loc[2, "nightly_rate_idr"]))
        self.assertTrue(pd.isna(mask_invalid_values(X).loc[2, "arrival_month"]))
        self.assertAlmostEqual(macro_f1_np([0, 1, 0, 1], [0, 1, 1, 0]), 0.5)
        folds = list(
            StratifiedKFoldNP(n_splits=2, random_state=42).split(
                np.zeros(6), np.array([0, 0, 0, 1, 1, 1])
            )
        )
        self.assertEqual(len(folds), 2)


if __name__ == "__main__":
    unittest.main()
