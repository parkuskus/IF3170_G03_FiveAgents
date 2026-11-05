import unittest

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.experiments.knn.knn import KNNClassifier


class KNNDistanceTest(unittest.TestCase):
    def test_euclidean_and_manhattan_can_choose_different_neighbors(self):
        X = [[0.0, 3.0], [2.0, 2.0]]
        y = [1, 0]
        query = [[0.0, 0.0]]

        euclidean = KNNClassifier(n_neighbors=1, metric="euclidean").fit(X, y)
        manhattan = KNNClassifier(n_neighbors=1, metric="manhattan").fit(X, y)

        self.assertEqual(euclidean.predict(query).tolist(), [0])
        self.assertEqual(manhattan.predict(query).tolist(), [1])

    def test_minkowski_parameter_matches_expected_limits(self):
        X = [[0.0, 3.0], [2.0, 2.0]]
        y = [1, 0]
        query = [[0.0, 0.0]]

        p1 = KNNClassifier(n_neighbors=1, metric="minkowski", p=1).fit(X, y)
        p2 = KNNClassifier(n_neighbors=1, metric="minkowski", p=2).fit(X, y)

        self.assertEqual(p1.predict(query).tolist(), [1])
        self.assertEqual(p2.predict(query).tolist(), [0])

    def test_cosine_ignores_vector_magnitude(self):
        X = [[10.0, 0.0], [1.0, 1.0]]
        y = [1, 0]
        model = KNNClassifier(n_neighbors=1, metric="cosine").fit(X, y)

        self.assertEqual(model.predict([[1.0, 0.0]]).tolist(), [1])

    def test_default_remains_euclidean(self):
        model = KNNClassifier().fit([[0.0], [2.0]], [0, 1])

        self.assertEqual(model.predict([[0.1]]).tolist(), [0])
        self.assertEqual(model.get_params()["metric"], "euclidean")

    def test_kneighbors_are_sorted_for_reusing_prefixes_in_k_search(self):
        model = KNNClassifier(n_neighbors=4).fit([[10.0], [1.0], [5.0], [2.0]], [0, 1, 0, 1])

        distances, indices = model.kneighbors([[0.0]])

        self.assertTrue(np.all(np.diff(distances[0]) >= 0))
        self.assertEqual(model.predict_from_neighbors(distances, indices, n_neighbors=1).tolist(), [1])

    def test_supported_distances_match_sklearn_predictions(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(40, 5))
        y = rng.integers(0, 2, size=40)
        query = rng.normal(size=(8, 5))
        for metric, p in (("euclidean", 2), ("manhattan", 1), ("chebyshev", 2), ("minkowski", 3), ("cosine", 2)):
            for weights in ("uniform", "distance"):
                with self.subTest(metric=metric, weights=weights):
                    actual = KNNClassifier(5, weights, metric, p).fit(X, y).predict(query)
                    expected = KNeighborsClassifier(5, weights=weights, metric=metric, p=p).fit(X, y).predict(query)
                    np.testing.assert_array_equal(actual, expected)

    def test_invalid_metric_and_minkowski_order_are_rejected(self):
        with self.assertRaises(ValueError):
            KNNClassifier(metric="hamming")
        with self.assertRaises(ValueError):
            KNNClassifier(metric="minkowski", p=0.5)


if __name__ == "__main__":
    unittest.main()
