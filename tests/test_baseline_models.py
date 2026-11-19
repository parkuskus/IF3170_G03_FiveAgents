import ast
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks" /
    "Notebook_FiveAgents_Tugas_Besar_2_IF3170_Inteligensi_Artifisial_2026_2027.ipynb").read_text(encoding="utf-8"))
namespace = {"np": np, "pd": pd}
for cell in notebook["cells"]:
    if cell.get("id") in {"7febed14", "4194347e", "20d20452", "4b9a54ff", "b1f35284"}:
        tree = ast.parse("".join(cell["source"]))
        definitions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef)], type_ignores=[])
        exec(compile(definitions, cell["id"], "exec"), namespace)
KNNBaseline = namespace["KNNBaseline"]
CARTBaseline = namespace["CARTBaseline"]
LogisticBaseline = namespace["LogisticBaseline"]
SVMBaseline = namespace["SVMBaseline"]
GaussianNBBaseline = namespace["GaussianNBBaseline"]


class ScratchModelsTest(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
        self.y = np.array([0, 0, 1, 1])

    def test_knn_returns_nearest_labels_and_breaks_ties_to_zero(self):
        model = KNNBaseline(k=1).fit(self.X, self.y)
        np.testing.assert_array_equal(model.predict(self.X), self.y)
        np.testing.assert_array_equal(KNNBaseline(k=2).fit(np.array([[-1.0], [1.0]]), [0, 1]).predict([[0.0]]), [0])

    def test_cart_handles_numeric_category_missing_and_unseen(self):
        X = pd.DataFrame({"num": [0.0, 0.0, 1.0, 1.0, np.nan, np.nan], "cat": ["a", "a", "b", "b", None, None]})
        y = [0, 0, 1, 1, 0, 0]
        model = CARTBaseline(max_depth=4, min_leaf=1).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)
        predicted = model.predict(pd.DataFrame({"num": [np.nan, 2.0], "cat": ["new", None]}))
        self.assertEqual(predicted.shape, (2,))
        self.assertTrue(np.isin(predicted, [0, 1]).all())

    def test_cart_splits_on_category_when_numeric_constant(self):
        X = pd.DataFrame({"num": [1, 1, 1, 1], "cat": ["a", "a", "b", "b"]})
        np.testing.assert_array_equal(CARTBaseline(min_leaf=1).fit(X, [0, 0, 1, 1]).predict(X), [0, 0, 1, 1])

    def test_cart_can_split_missing_from_observed(self):
        X = pd.DataFrame({"num": [np.nan, np.nan, 1.0, 1.0]})
        np.testing.assert_array_equal(CARTBaseline(min_leaf=1).fit(X, [1, 1, 0, 0]).predict(X), [1, 1, 0, 0])

    def test_decision_tree_criteria_fit_numeric_categorical_and_missing_values(self):
        X = pd.DataFrame({
            "number": [0.0, 0.0, 1.0, 1.0, np.nan, np.nan],
            "category": ["a", "a", "b", "b", None, None],
        })
        y = np.array([0, 0, 1, 1, 0, 0])

        for criterion in ("information_gain", "gain_ratio", "gini"):
            with self.subTest(criterion=criterion):
                model = CARTBaseline(criterion=criterion, min_leaf=1).fit(X, y)
                np.testing.assert_array_equal(model.predict(X), y)
                predictions = model.predict(pd.DataFrame({
                    "number": [np.nan, 2.0],
                    "category": ["unseen", None],
                }))
                self.assertEqual(predictions.shape, (2,))
                self.assertTrue(np.isin(predictions, [0, 1]).all())

    def test_id3_and_c45_use_multiway_categorical_splits(self):
        X = pd.DataFrame({
            "constant": [1] * 6,
            "category": ["a", "a", "b", "b", "c", "c"],
        })
        y = np.array([0, 0, 1, 1, 0, 0])

        for criterion in ("information_gain", "gain_ratio"):
            with self.subTest(criterion=criterion):
                model = CARTBaseline(
                    criterion=criterion, max_depth=2, min_leaf=1
                ).fit(X, y)
                self.assertEqual(model.root_["split_kind"], "multiway")
                np.testing.assert_array_equal(model.predict(X), y)

    def test_logistic_penalties_fit_and_predict(self):
        for penalty in ("l1", "l2", "elasticnet"):
            with self.subTest(penalty=penalty):
                model = LogisticBaseline(penalty=penalty, epochs=300, lr=0.1, alpha=0.01).fit(self.X, self.y)
                np.testing.assert_array_equal(model.predict(self.X), self.y)
                self.assertTrue(np.isfinite(model.weights_).all())

    def test_svm_kernels_fit_and_predict(self):
        for kernel in ("linear", "poly", "rbf"):
            with self.subTest(kernel=kernel):
                model = SVMBaseline(kernel=kernel, epochs=100, lr=0.1, alpha=0.01).fit(self.X, self.y)
                np.testing.assert_array_equal(model.predict(self.X), self.y)

    def test_gaussian_nb_handles_constant_features(self):
        X = np.column_stack([self.X[:, 0], np.ones(4)])
        model = GaussianNBBaseline().fit(X, self.y)
        np.testing.assert_array_equal(model.predict(X), self.y)


if __name__ == "__main__":
    unittest.main()
