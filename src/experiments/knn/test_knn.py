from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.experiments.knn.knn import KNNClassifier
from src.experiments.knn.preprocessing import KNNPreprocessor


class KNNClassifierTest(unittest.TestCase):
    def test_default_is_backward_compatible(self):
        model = KNNClassifier().fit([[0.0], [2.0]], [0, 1])
        self.assertEqual(model.predict([[0.1]]).tolist(), [0])
        self.assertEqual(model.get_params()["metric"], "euclidean")
        self.assertEqual(model.get_params()["threshold"], 0.5)

    def test_euclidean_and_manhattan_can_choose_different_neighbors(self):
        X = [[0.0, 3.0], [2.0, 2.0]]
        y = [1, 0]
        query = [[0.0, 0.0]]
        euclidean = KNNClassifier(n_neighbors=1, metric="euclidean").fit(X, y)
        manhattan = KNNClassifier(n_neighbors=1, metric="manhattan").fit(X, y)
        self.assertEqual(euclidean.predict(query).tolist(), [0])
        self.assertEqual(manhattan.predict(query).tolist(), [1])

    def test_supported_metrics_match_sklearn(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(40, 5))
        y = rng.integers(0, 2, size=40)
        query = rng.normal(size=(8, 5))
        for metric in ("euclidean", "manhattan"):
            for weights in ("uniform", "distance"):
                with self.subTest(metric=metric, weights=weights):
                    actual = KNNClassifier(
                        n_neighbors=5, weights=weights, metric=metric
                    ).fit(X, y).predict(query)
                    expected = KNeighborsClassifier(
                        n_neighbors=5, weights=weights, metric=metric
                    ).fit(X, y).predict(query)
                    np.testing.assert_array_equal(actual, expected)

    def test_decision_scores_are_probabilities_for_both_metrics(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        query = np.array([[0.2], [2.8]])
        for metric in ("euclidean", "manhattan"):
            with self.subTest(metric=metric):
                scores = KNNClassifier(
                    n_neighbors=3, weights="distance", metric=metric
                ).fit(X, y).decision_scores(query)
                self.assertTrue(np.isfinite(scores).all())
                self.assertTrue(((scores >= 0.0) & (scores <= 1.0)).all())

    def test_threshold_changes_predictions_from_same_scores(self):
        model = KNNClassifier(n_neighbors=3).fit(
            [[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1]
        )
        query = [[1.4], [1.6]]
        scores = model.decision_scores(query)
        low = KNNClassifier(n_neighbors=3, threshold=0.3).fit(
            [[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1]
        ).predict(query)
        high = KNNClassifier(n_neighbors=3, threshold=0.7).fit(
            [[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1]
        ).predict(query)
        np.testing.assert_array_equal(low, (scores > 0.3).astype(int))
        np.testing.assert_array_equal(high, (scores > 0.7).astype(int))

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            KNNClassifier(metric="hamming")
        with self.assertRaises(ValueError):
            KNNClassifier(threshold=0.0)
        with self.assertRaises(ValueError):
            KNNClassifier(n_neighbors=0)

    def test_reconstructed_model_has_identical_predictions(self):
        rng = np.random.default_rng(19)
        X = rng.normal(size=(25, 3))
        y = rng.integers(0, 2, size=25)
        query = rng.normal(size=(9, 3))
        original = KNNClassifier(
            n_neighbors=5, weights="distance", metric="manhattan", threshold=0.4
        ).fit(X, y)
        params = original.get_params()
        restored = KNNClassifier(**params).fit(X, y)
        np.testing.assert_array_equal(
            original.predict(query, batch_size=3),
            restored.predict(query, batch_size=3),
        )
        np.testing.assert_allclose(
            original.decision_scores(query, batch_size=3),
            restored.decision_scores(query, batch_size=3),
        )

    def test_batching_does_not_change_predictions(self):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(30, 4))
        y = rng.integers(0, 2, size=30)
        query = rng.normal(size=(13, 4))
        for metric in ("euclidean", "manhattan"):
            model = KNNClassifier(7, "distance", metric).fit(X, y)
            with self.subTest(metric=metric):
                np.testing.assert_array_equal(
                    model.predict(query, batch_size=3),
                    model.predict(query, batch_size=20),
                )


class KNNPreprocessorTest(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({
            "num": [1.0, 2.0, 3.0],
            "category": ["a", "b", None],
        })

    def test_cat_weight_scales_only_categorical_block(self):
        base = KNNPreprocessor(cat_weight=1.0).fit_transform(self.X)
        weighted = KNNPreprocessor(cat_weight=2.0).fit_transform(self.X)
        # Numeric block is first; one-hot categorical block follows it.
        np.testing.assert_allclose(weighted[:, 0], base[:, 0])
        np.testing.assert_allclose(weighted[:, 1:], 2.0 * base[:, 1:])

    def test_cat_weight_survives_serialization(self):
        pre = KNNPreprocessor(cat_weight=0.5).fit(self.X)
        restored = KNNPreprocessor.from_dict(pre.to_dict())
        np.testing.assert_allclose(pre.transform(self.X), restored.transform(self.X))
        self.assertEqual(restored.cat_weight, 0.5)

    def test_invalid_cat_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            KNNPreprocessor(cat_weight=0.0)


if __name__ == "__main__":
    unittest.main()
