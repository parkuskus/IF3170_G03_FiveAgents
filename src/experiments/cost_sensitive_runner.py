"""Evaluate a fixed final pipeline under a separately specified cost scenario."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.experiments.cost_sensitive import compare_holdout

ROOT = Path(__file__).resolve().parents[2]
BASELINES = {"decision_tree": 0.62, "logistic_regression": 0.47,
             "svm": -0.027, "naive_bayes": 0.405, "knn": 0.5}


def score_fold(name, X_train, y_train, X_valid, nb_config=None):
    """Fit preprocessing and the final model on one fold only."""
    if name == "decision_tree":
        from src.experiments.decision_tree.shared import BaselineTree, SharedPreprocessor
        prep = SharedPreprocessor().fit(X_train)
        tree = BaselineTree(criterion="gini", max_depth=None,
                            min_samples_split=2, min_samples_leaf=1).fit(
                                prep.transform(X_train), y_train)
        return tree.pruned(0.0001).predict_proba(prep.transform(X_valid))[:, 1]
    if name == "knn":
        from src.experiments.knn.knn import KNNClassifier
        from src.experiments.knn.preprocessing import KNNPreprocessor
        prep = KNNPreprocessor(cat_weight=1.0).fit(X_train)
        model = KNNClassifier(21, "distance", "manhattan").fit(
            prep.transform(X_train), y_train)
        return model.decision_scores(prep.transform(X_valid), batch_size=256)
    if name == "logistic_regression":
        # The existing experiment module uses sibling imports when run as a script.
        sys.path.insert(0, str(ROOT / "src/experiments/logistic_regression"))
        from experiments import FINAL_CONFIGS, fit_preprocessor, train_model, transform_features
        prep = fit_preprocessor(X_train)
        model = train_model(transform_features(prep, X_train), y_train,
                            FINAL_CONFIGS["elasticnet"])
        return model.predict_proba(transform_features(prep, X_valid))
    if name == "svm":
        sys.path.insert(0, str(ROOT / "src/experiments/svm"))
        from preprocessing import SVMPreprocessor
        from svm_scratch import SVMFromScratch
        prep = SVMPreprocessor(n_splits=5, seed=42)
        prep.fit(X_train, y=y_train)
        model = SVMFromScratch(C=0.01, kernel="linear", max_iter=35,
                               n_components=2000, random_state=42)
        model.fit(prep.transform(X_train), y_train)
        return model.decision_function(prep.transform(X_valid))
    if name == "naive_bayes":
        from src.experiments.naive_bayes.utils.cleaning import clean_training_data
        from src.experiments.naive_bayes.utils.pipeline import StagePipeline, model_from_params
        clean, labels, _ = clean_training_data(X_train, y_train)
        prep = StagePipeline(nb_config["preprocessing"],
                             nb_config["feature_selection"]["selected_features"],
                             nb_config["dimensionality_reduction"]).fit(clean, labels)
        model = model_from_params(prep, nb_config["params"]).fit(
            prep.transform(clean), labels)
        return model.decision_scores(prep.transform(X_valid))
    raise ValueError(f"Unknown model {name}")


def run(name):
    if name not in BASELINES:
        raise ValueError(f"Unknown model {name}")
    from src.experiments.decision_tree.run_all import validation_indices

    data = pd.read_csv(ROOT / "data/train.csv", dtype={"id": str})
    y = data["is_canceled"].to_numpy(dtype=int)
    X = data.drop(columns=["id", "is_canceled"])
    train_idx, valid_idx = validation_indices(y)
    config = None
    if name == "naive_bayes":
        config_path = ROOT / "output/final/model/naive_bayes/konfigurasi.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
    y_train = y[train_idx]
    oof = np.empty(len(train_idx), dtype=float)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=43)
    for a, v in folds.split(X.iloc[train_idx], y_train):
        oof[v] = score_fold(
            name, X.iloc[train_idx[a]], y_train[a], X.iloc[train_idx[v]], config
        )
    holdout = score_fold(
        name, X.iloc[train_idx], y_train, X.iloc[valid_idx], config
    )
    result = compare_holdout(
        y_train, oof, y[valid_idx], holdout, BASELINES[name], fp_cost=3, fn_cost=1
    )
    destination = (ROOT / "output/knn" if name == "knn"
                   else ROOT / "output/experiments" / name)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "cost_sensitive_comparison.csv"
    pd.DataFrame([
        {"strategy": strategy, **result[strategy]}
        for strategy in ("baseline", "selected", "oof_selected")
    ]).to_csv(path, index=False)
    print(pd.read_csv(path).to_string(index=False), flush=True)
    return result
