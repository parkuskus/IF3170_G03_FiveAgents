from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .cleaning import clean_training_data, mask_invalid_values
from .feature_engineering import add_engineered_features
from .kdb import KDBClassifier
from .naive_bayes import NaiveBayesClassifier
from .numpy_utils import PCANP, StandardScalerNP, StratifiedKFoldNP
from .preprocessing import (
    DEFAULT_CATEGORICAL_NUMERIC,
    DEFAULT_LOG1P_COLS,
    NaiveBayesPreprocessor,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "output" / "experiments" / "naive_bayes"
SEED = 42
ADOPTION_GAP = 0.005
N_FOLDS = 5

REFERENCE_PARAMS = {
    "alpha": 1.0,
    "var_smoothing": 0.05,
    "threshold": 0.5,
}


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv", dtype={"id": str})
    test = pd.read_csv(DATA_DIR / "test.csv", dtype={"id": str})
    y = train["is_canceled"].to_numpy()
    X_raw = train.drop(columns=["id", "is_canceled"])
    return train, test, y, X_raw


def make_split(y):
    rng = np.random.default_rng(SEED)
    valid_idx = np.sort(
        np.concatenate(
            [
                rng.permutation(np.flatnonzero(y == label))[
                    : int(np.ceil(0.2 * np.sum(y == label)))
                ]
                for label in (0, 1)
            ]
        )
    )
    return np.setdiff1d(np.arange(len(y)), valid_idx), valid_idx


def make_cv(y, n_splits=N_FOLDS, seed=SEED):
    return StratifiedKFoldNP(n_splits=n_splits, shuffle=True, random_state=seed)


def metrics(y_true, pred):
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    p, r, f, _ = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1], zero_division=0
    )
    return {
        "macro_f1": float(
            f1_score(y_true, pred, labels=[0, 1], average="macro", zero_division=0)
        ),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "precision_0": round(float(p[0]), 4),
        "recall_0": round(float(r[0]), 4),
        "f1_0": round(float(f[0]), 4),
        "precision_1": round(float(p[1]), 4),
        "recall_1": round(float(r[1]), 4),
        "f1_1": round(float(f[1]), 4),
    }


def metrics_from_scores(scores, y_true, threshold=0.5):
    return metrics(y_true, (scores > threshold).astype(np.int64))


def config_name(params):
    return (
        f"alpha{params['alpha']:.4g}_vs{params['var_smoothing']:.4g}_"
        f"t{params['threshold']:.2f}"
    )


def make_row(round_name, scores, y_true, params, source="scratch", **extra):
    row = {
        "source": source,
        "config": config_name(params),
        "ronde": round_name,
        "alpha": params["alpha"],
        "var_smoothing": params["var_smoothing"],
        "threshold": params["threshold"],
    }
    row.update(extra)
    row.update(metrics_from_scores(scores, y_true, params["threshold"]))
    return row


def decide(decisions, round_name, variable, best, baseline_f1, detail=""):
    delta = float(best["macro_f1"] - baseline_f1)
    adopt = delta >= ADOPTION_GAP
    decisions.append(
        {
            "ronde": round_name,
            "variabel": variable,
            "terpilih": best[variable],
            "baseline_f1": float(baseline_f1),
            "best_f1": float(best["macro_f1"]),
            "delta": round(delta, 6),
            "keputusan": "ADOPSI" if adopt else "TOLAK",
            "alasan": detail
            or ("naik >= 0.005" if adopt else "kenaikan di bawah 0.005"),
        }
    )
    print(
        f"  Keputusan: {'ADOPSI' if adopt else 'TOLAK'} (delta={delta:+.6f})",
        flush=True,
    )
    return (best["macro_f1"] if adopt else baseline_f1), adopt


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def read_json(path):
    return json.loads(Path(path).read_text())


def feature_frame(X_raw, feature_set="baseline"):
    return add_engineered_features(X_raw, feature_set)


def preprocessor_from_config(cfg):
    options = cfg.get("options", cfg)
    return NaiveBayesPreprocessor(
        standardize=bool(options.get("standardize", False)),
        add_missing_indicators=bool(options.get("add_missing_indicators", True)),
        categorical_numeric_cols=tuple(
            options.get("categorical_numeric_cols", DEFAULT_CATEGORICAL_NUMERIC)
        ),
        log1p_cols=tuple(options.get("log1p_cols", DEFAULT_LOG1P_COLS)),
        bin_cols=tuple(options.get("bin_cols", ())),
        n_bins=int(options.get("n_bins", 20)),
        bin_strategy=options.get("bin_strategy", "quantile"),
    )


class StagePipeline:
    """Fit preprocessing, feature selection, and optional dimensionality reduction."""

    def __init__(self, prep_cfg, selected_features=None, dr_cfg=None):
        self.prep_cfg = prep_cfg
        self.selected_features = selected_features
        self.dr_cfg = dr_cfg or {"method": "none"}
        self.pre_ = None
        self.scaler_ = None
        self.pca_ = None
        self.numeric_indices_ = []
        self.categorical_indices_ = []
        self.n_categories_ = []

    def _frame(self, X_raw):
        feature_set = self.prep_cfg.get(
            "feature_set",
            self.prep_cfg.get("options", {}).get("feature_set", "baseline"),
        )
        X = feature_frame(mask_invalid_values(X_raw), feature_set)
        if self.selected_features is not None:
            missing = [c for c in self.selected_features if c not in X.columns]
            if missing:
                raise ValueError(f"fitur terpilih tidak ditemukan: {missing}")
            X = X[self.selected_features]
        return X

    def fit(self, X_raw, y):
        X = self._frame(X_raw)
        self.pre_ = preprocessor_from_config(self.prep_cfg).fit(X)
        Z = self.pre_.transform(X)
        self._fit_dr(Z)
        self._set_indices()
        return self

    def _fit_dr(self, Z):
        method = self.dr_cfg.get("method", "none")
        self.scaler_ = None
        self.pca_ = None
        if method in (None, "none"):
            return
        if method != "pca_numeric":
            raise ValueError(f"metode dimensionality reduction tidak dikenal: {method}")
        numeric = self.pre_.numeric_indices
        if not numeric:
            raise ValueError("PCA numerik butuh minimal satu fitur numerik")
        self.scaler_ = StandardScalerNP().fit(Z[:, numeric])
        self.pca_ = PCANP(n_components=self.dr_cfg["n_components"]).fit(
            self.scaler_.transform(Z[:, numeric])
        )

    def _transform_dr(self, Z):
        if self.pca_ is None:
            return Z
        numeric = self.pre_.numeric_indices
        categorical = self.pre_.categorical_indices
        numeric_scaled = self.scaler_.transform(Z[:, numeric])
        components = self.pca_.transform(numeric_scaled)
        if categorical:
            return np.hstack([components, Z[:, categorical]])
        return components

    def _set_indices(self):
        if self.pca_ is None:
            self.numeric_indices_ = list(self.pre_.numeric_indices)
            self.categorical_indices_ = list(self.pre_.categorical_indices)
            self.n_categories_ = list(self.pre_.n_categories)
            return
        n_components = int(self.pca_.n_components_)
        self.numeric_indices_ = list(range(n_components))
        self.categorical_indices_ = list(
            range(n_components, n_components + len(self.pre_.categorical_indices))
        )
        self.n_categories_ = list(self.pre_.n_categories)

    def transform(self, X_raw):
        if self.pre_ is None:
            raise RuntimeError("panggil fit() sebelum transform()")
        X = self._frame(X_raw)
        return self._transform_dr(self.pre_.transform(X))

    @property
    def numeric_indices(self):
        return list(self.numeric_indices_)

    @property
    def categorical_indices(self):
        return list(self.categorical_indices_)

    @property
    def n_categories(self):
        return list(self.n_categories_)


def model_from_params(pipe, params):
    weights = params.get("feature_weights")
    variant = params.get("variant", "categorical")
    if variant == "kdb":
        if pipe.numeric_indices:
            raise ValueError(
                "KDB membutuhkan seluruh fitur dalam bentuk kategorikal/binned"
            )
        return KDBClassifier(
            k=int(params.get("k", 4)),
            alpha=float(params["alpha"]),
            m_estimate=float(params.get("m_estimate", 50.0)),
            threshold=float(params.get("threshold", 0.5)),
            n_categories=pipe.n_categories,
            feature_weights=None if weights is None else list(weights),
            bias=float(params.get("bias", 0.0)),
        )
    alpha = float(params["alpha"])
    threshold = float(params.get("threshold", 0.5))
    mode = params.get("mode")
    if mode is None:
        is_baseline = (
            not pipe.numeric_indices
            and variant == "categorical"
            and np.isclose(alpha, 1.0)
            and np.isclose(threshold, 0.5)
            and weights is None
        )
        mode = "baseline" if is_baseline else "bonus"
    if mode == "bonus" and pipe.numeric_indices:
        variant = "hybrid"
    return NaiveBayesClassifier(
        mode=mode,
        variant=variant,
        alpha=alpha,
        var_smoothing=float(params.get("var_smoothing", 1e-9)),
        threshold=threshold,
        numeric_indices=pipe.numeric_indices,
        categorical_indices=pipe.categorical_indices,
        n_categories=pipe.n_categories,
        feature_weights=None if weights is None else list(weights),
    )


def precompute_folds(
    X_raw,
    y,
    train_idx,
    prep_cfg,
    selected_features,
    dr_cfg,
    n_splits=N_FOLDS,
    cv=None,
):
    if cv is None:
        cv = make_cv(y[train_idx], n_splits=n_splits)
    folds = []
    for tr, va in cv.split(np.zeros(len(train_idx)), y[train_idx]):
        tr_idx = train_idx[tr]
        va_idx = train_idx[va]
        X_tr_clean, y_tr_clean, _ = clean_training_data(X_raw.iloc[tr_idx], y[tr_idx])
        pipe = StagePipeline(prep_cfg, selected_features, dr_cfg).fit(
            X_tr_clean, y_tr_clean
        )
        folds.append(
            {
                "tr": tr,
                "va": va,
                "Ztr": pipe.transform(X_tr_clean),
                "ytr": y_tr_clean,
                "Zva": pipe.transform(X_raw.iloc[va_idx]),
                "yva": y[va_idx],
                "pipe": pipe,
            }
        )
    return folds


def evaluate_folds(folds, params, n_train):
    oof = np.zeros(n_train, dtype=np.float64)
    fold_scores = []
    for fold in folds:
        model = model_from_params(fold["pipe"], params).fit(fold["Ztr"], fold["ytr"])
        scores = model.decision_scores(fold["Zva"])
        oof[fold["va"]] = scores
        pred = (scores > params["threshold"]).astype(np.int64)
        fold_scores.append(
            float(
                f1_score(
                    fold["yva"], pred, labels=[0, 1], average="macro", zero_division=0
                )
            )
        )
    return oof, fold_scores


def scores_from_folds(folds, params, n_train):
    return evaluate_folds(folds, params, n_train)[0]


def threshold_curve(folds, params, thresholds, n_train):
    thresholds = [float(t) for t in thresholds]
    per_threshold = {t: [] for t in thresholds}
    oof = np.zeros(n_train, dtype=np.float64)
    for fold in folds:
        model = model_from_params(fold["pipe"], params).fit(fold["Ztr"], fold["ytr"])
        scores = model.decision_scores(fold["Zva"])
        oof[fold["va"]] = scores
        for t in thresholds:
            pred = (scores > t).astype(np.int64)
            per_threshold[t].append(
                float(
                    f1_score(
                        fold["yva"],
                        pred,
                        labels=[0, 1],
                        average="macro",
                        zero_division=0,
                    )
                )
            )
    return per_threshold, oof


def holdout_scores(
    X_raw, y, train_idx, valid_idx, prep_cfg, selected_features, dr_cfg, params
):
    X_tr_clean, y_tr_clean, _ = clean_training_data(X_raw.iloc[train_idx], y[train_idx])
    pipe = StagePipeline(prep_cfg, selected_features, dr_cfg).fit(
        X_tr_clean, y_tr_clean
    )
    model = model_from_params(pipe, params).fit(pipe.transform(X_tr_clean), y_tr_clean)
    return model.decision_scores(pipe.transform(X_raw.iloc[valid_idx]))


def evaluate_candidate(
    X_raw,
    y,
    train_idx,
    valid_idx,
    prep_cfg,
    selected_features,
    dr_cfg,
    params,
    n_splits=N_FOLDS,
):
    folds = precompute_folds(
        X_raw, y, train_idx, prep_cfg, selected_features, dr_cfg, n_splits=n_splits
    )
    oof, fold_scores = evaluate_folds(folds, params, len(train_idx))
    holdout = holdout_scores(
        X_raw, y, train_idx, valid_idx, prep_cfg, selected_features, dr_cfg, params
    )
    row = {
        "cv_macro_f1_mean": float(np.mean(fold_scores)),
        "cv_macro_f1_std": float(np.std(fold_scores)),
        "holdout_macro_f1": float(
            f1_score(
                y[valid_idx],
                (holdout > params["threshold"]).astype(int),
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
    }
    row.update(
        {
            f"holdout_{k}": v
            for k, v in metrics_from_scores(
                holdout, y[valid_idx], params["threshold"]
            ).items()
            if k != "macro_f1"
        }
    )
    return row, oof, holdout


def evaluate_candidate_repeated(
    X_raw,
    y,
    train_idx,
    valid_idx,
    prep_cfg,
    selected_features,
    dr_cfg,
    params,
    seeds=(SEED, 0, 1),
    n_splits=N_FOLDS,
):
    """Evaluate a candidate with repeated stratified CV on train_idx.

    Holdout is computed once and reported only; it is never used for selection.
    """
    all_fold_scores = []
    seed_means = []
    for seed in seeds:
        cv = make_cv(y[train_idx], n_splits=n_splits, seed=seed)
        folds = precompute_folds(
            X_raw,
            y,
            train_idx,
            prep_cfg,
            selected_features,
            dr_cfg,
            n_splits=n_splits,
            cv=cv,
        )
        _, fold_scores = evaluate_folds(folds, params, len(train_idx))
        all_fold_scores.extend(fold_scores)
        seed_means.append(float(np.mean(fold_scores)))
    holdout = holdout_scores(
        X_raw, y, train_idx, valid_idx, prep_cfg, selected_features, dr_cfg, params
    )
    row = {
        "cv_macro_f1_mean": float(np.mean(all_fold_scores)),
        "cv_macro_f1_std": float(np.std(all_fold_scores)),
        "cv_seed_mean_min": float(np.min(seed_means)),
        "cv_seed_mean_max": float(np.max(seed_means)),
        "cv_seeds": ",".join(str(seed) for seed in seeds),
        "holdout_macro_f1": float(
            f1_score(
                y[valid_idx],
                (holdout > params["threshold"]).astype(int),
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
    }
    row.update(
        {
            f"holdout_{k}": v
            for k, v in metrics_from_scores(
                holdout, y[valid_idx], params["threshold"]
            ).items()
            if k != "macro_f1"
        }
    )
    return row
