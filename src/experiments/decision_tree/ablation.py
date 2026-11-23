"""Holdout diagnostics for prespecified CART growth, pruning and decision thresholds.

Run from the repository root with python -m src.experiments.decision_tree.ablation.
The holdout results are descriptive only; do not select parameters from this file.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

from .run_all import CART_TREE_CONFIGS, OUTPUT, ROOT, prepare_fold, validation_indices
from .shared import BaselineTree


def evaluate_cart(X_train, y_train, X_valid, y_valid, config, alphas, thresholds):
    depth, split, leaf = config
    model = BaselineTree(criterion="gini", max_depth=depth,
                         min_samples_split=split, min_samples_leaf=leaf).fit(X_train, y_train)
    rows = []
    for alpha in alphas:
        tree = model.pruned(alpha)
        probabilities = tree.predict_proba(X_valid)[:, 1]
        for threshold in thresholds:
            prediction = (probabilities >= threshold).astype(np.int8)
            tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
            rows.append({
                "max_depth": depth, "min_samples_split": split, "min_samples_leaf": leaf,
                "alpha": alpha, "threshold": threshold, "leaves": tree.leaf_count(),
                "macro_f1": f1_score(y_valid, prediction, labels=[0, 1],
                                     average="macro", zero_division=0),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "prediction": prediction,
            })
    return rows


def main():
    df = pd.read_csv(ROOT / "data" / "train.csv", dtype={"id": str})
    train, valid = validation_indices(df["is_canceled"])
    features = df.columns.drop(["id", "is_canceled"])
    X_train, X_valid = prepare_fold(df.iloc[train][features], df.iloc[valid][features])
    y_train = df.iloc[train]["is_canceled"].to_numpy(dtype=np.int8)
    y_valid = df.iloc[valid]["is_canceled"].to_numpy(dtype=np.int8)
    rows = []
    one_factor_configs = ((8, 2, 1), (12, 2, 1), (None, 40, 1),
                          (None, 80, 1), (None, 2, 20), (None, 2, 40))
    for config in (*CART_TREE_CONFIGS, *one_factor_configs):
        # Values are fixed before viewing the holdout, including the CV-selected alpha and threshold.
        rows.extend(evaluate_cart(X_train, y_train, X_valid, y_valid, config,
                                  (0.0, 0.0001, 0.001, 0.01), (0.5, 0.62, 0.7)))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{key: value for key, value in row.items() if key != "prediction"}
                  for row in rows]).to_csv(OUTPUT / "ablation_holdout.csv", index=False)
    print(pd.read_csv(OUTPUT / "ablation_holdout.csv").to_string(index=False))


if __name__ == "__main__":
    main()
