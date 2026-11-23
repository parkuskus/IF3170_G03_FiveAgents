from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.naive_bayes import CategoricalNB, GaussianNB

from src.experiments.naive_bayes.utils.cleaning import clean_training_data
from src.experiments.naive_bayes.utils.pipeline import (
    N_FOLDS,
    OUT_DIR,
    REFERENCE_PARAMS,
    SEED,
    StagePipeline,
    config_name,
    decide,
    evaluate_candidate,
    evaluate_candidate_repeated,
    holdout_scores,
    load_data,
    make_split,
    metrics,
    metrics_from_scores,
    model_from_params,
    precompute_folds,
    read_json,
    threshold_curve,
    write_json,
)
from src.experiments.naive_bayes.utils.preprocessing import (
    DEFAULT_BIN_COLS,
    DEFAULT_CATEGORICAL_NUMERIC,
    DEFAULT_LOG1P_COLS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PREP_CONFIG = OUT_DIR / "naive_bayes_preprocessing_config.json"
FS_CONFIG = OUT_DIR / "naive_bayes_feature_selection_config.json"
DR_CONFIG = OUT_DIR / "naive_bayes_dimensionality_reduction_config.json"
RESULT_CSV = OUT_DIR / "naive_bayes_experiment_results.csv"
DECISION_CSV = OUT_DIR / "naive_bayes_decisions.csv"
SKLEARN_CSV = OUT_DIR / "naive_bayes_sklearn_comparison.csv"
CONFIG_JSON = OUT_DIR / "naive_bayes_final_config.json"
GREEDY_CONFIG = OUT_DIR / "naive_bayes_greedy_config.json"
GREEDY_TRIALS = OUT_DIR / "naive_bayes_greedy_trials.csv"
MODEL_DIR = REPO_ROOT / "output" / "final" / "model" / "naive_bayes"

ALPHA_GRID = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
VAR_SMOOTHING_GRID = [1e-12, 1e-9, 1e-6, 1e-3, 0.01, 0.02, 0.05, 0.1, 0.2]
THRESHOLDS = np.arange(0.30, 0.76, 0.01)

DEFAULT_PREP_CFG = {
    "name": "default",
    "feature_set": "baseline",
    "options": {
        "standardize": False,
        "add_missing_indicators": True,
        "categorical_numeric_cols": list(DEFAULT_CATEGORICAL_NUMERIC),
        "log1p_cols": list(DEFAULT_LOG1P_COLS),
        "bin_cols": [],
        "n_bins": 20,
        "bin_strategy": "quantile",
    },
}


def load_stage_configs():
    missing = [
        str(path) for path in (PREP_CONFIG, FS_CONFIG, DR_CONFIG) if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "Jalankan python -m src.experiments.naive_bayes.run_preprocessing, "
            "python -m src.experiments.naive_bayes.run_feature_selection, dan "
            "python -m src.experiments.naive_bayes.run_dimensionality_reduction "
            "terlebih dahulu. Hilang: " + ", ".join(missing)
        )
    return read_json(PREP_CONFIG), read_json(FS_CONFIG), read_json(DR_CONFIG)


def row_for(round_name, row, params, source="scratch"):
    out = dict(row)
    out.update(
        {
            "source": source,
            "config": config_name(params),
            "ronde": round_name,
            "alpha": params["alpha"],
            "var_smoothing": params["var_smoothing"],
            "threshold": params["threshold"],
        }
    )
    return out


def _build_comparison_features(pipe, Ztr, Zva):
    numeric = list(pipe.numeric_indices)
    categorical = list(pipe.categorical_indices)
    n_categories = list(pipe.n_categories)
    tr_parts, va_parts = [], []
    if numeric:
        tr_parts.append(Ztr[:, numeric])
        va_parts.append(Zva[:, numeric])
    for j, n_cat in zip(categorical, n_categories):
        n_cat = int(n_cat)
        tr_codes = np.clip(Ztr[:, j].astype(np.int64), 0, n_cat - 1)
        va_codes = np.clip(Zva[:, j].astype(np.int64), 0, n_cat - 1)
        onehot_tr = np.zeros((len(Ztr), n_cat), dtype=np.float64)
        onehot_va = np.zeros((len(Zva), n_cat), dtype=np.float64)
        onehot_tr[np.arange(len(Ztr)), tr_codes] = 1.0
        onehot_va[np.arange(len(Zva)), va_codes] = 1.0
        tr_parts.append(onehot_tr)
        va_parts.append(onehot_va)
    if not tr_parts:
        return np.empty((len(Ztr), 0)), np.empty((len(Zva), 0))
    return np.hstack(tr_parts), np.hstack(va_parts)


def _gaussian_comparison(
    X_raw,
    y,
    train_idx,
    valid_idx,
    params,
    prep_cfg=None,
    selected_features=None,
    dr_cfg=None,
):
    """GaussianNB on numeric columns plus one-hot categorical columns."""
    pipe = StagePipeline(
        prep_cfg or DEFAULT_PREP_CFG,
        selected_features,
        dr_cfg or {"method": "none"},
    ).fit(X_raw.iloc[train_idx], y[train_idx])
    Ztr = pipe.transform(X_raw.iloc[train_idx])
    Zva = pipe.transform(X_raw.iloc[valid_idx])
    Xtr, Xva = _build_comparison_features(pipe, Ztr, Zva)
    model = GaussianNB(var_smoothing=params["var_smoothing"]).fit(Xtr, y[train_idx])
    return model.predict(Xva)


def _categorical_comparison(
    X_raw,
    y,
    train_idx,
    valid_idx,
    params,
    prep_cfg=None,
    selected_features=None,
    dr_cfg=None,
):
    """CategoricalNB on binned numeric columns and categorical codes."""
    pipe = StagePipeline(
        prep_cfg or DEFAULT_PREP_CFG,
        selected_features,
        dr_cfg or {"method": "none"},
    ).fit(X_raw.iloc[train_idx], y[train_idx])
    Ztr = pipe.transform(X_raw.iloc[train_idx])
    Zva = pipe.transform(X_raw.iloc[valid_idx])

    numeric = set(pipe.numeric_indices)
    tr_parts, va_parts = [], []
    for j in range(Ztr.shape[1]):
        if j in numeric:
            train_col = Ztr[:, j]
            valid_col = Zva[:, j]
            try:
                edges = pd.qcut(train_col, q=8, duplicates="drop", retbins=True)[1]
            except ValueError:
                edges = np.array([-np.inf, np.inf])
            if len(edges) < 3:
                tr_codes = np.zeros(len(Ztr), dtype=np.int64)
                va_codes = np.zeros(len(Zva), dtype=np.int64)
            else:
                tr_codes = (
                    pd.cut(train_col, bins=edges, labels=False, include_lowest=True)
                    .fillna(0)
                    .astype(np.int64)
                    .to_numpy()
                )
                va_codes = (
                    pd.cut(valid_col, bins=edges, labels=False, include_lowest=True)
                    .fillna(0)
                    .astype(np.int64)
                    .to_numpy()
                )
        else:
            tr_codes = Ztr[:, j].astype(np.int64)
            va_codes = Zva[:, j].astype(np.int64)

        observed = np.unique(tr_codes)
        mapping = {int(value): index for index, value in enumerate(observed)}
        tr_codes = np.array([mapping[int(value)] for value in tr_codes], dtype=np.int64)
        va_codes = np.array(
            [mapping.get(int(value), 0) for value in va_codes], dtype=np.int64
        )
        tr_parts.append(tr_codes[:, None])
        va_parts.append(va_codes[:, None])

    model = CategoricalNB(alpha=params["alpha"]).fit(np.hstack(tr_parts), y[train_idx])
    return model.predict(np.hstack(va_parts))


def _dr_state(pipe):
    if pipe.pca_ is None:
        return {"method": "none"}
    return {
        "method": "pca_numeric",
        "scaler_mean": pipe.scaler_.mean_.tolist(),
        "scaler_scale": pipe.scaler_.scale_.tolist(),
        "pca_components": pipe.pca_.components_.tolist(),
        "pca_mean": pipe.pca_.mean_.tolist(),
        "pca_explained_variance": pipe.pca_.explained_variance_.tolist(),
        "pca_explained_variance_ratio": pipe.pca_.explained_variance_ratio_.tolist(),
        "pca_singular_values": pipe.pca_.singular_values_.tolist(),
        "pca_noise_variance": float(pipe.pca_.noise_variance_),
        "n_components": int(pipe.pca_.n_components_),
    }


def save_artifacts(X_raw, y, prep_cfg, selected, dr_cfg, params, config):
    X_clean, y_clean, n_dropped = clean_training_data(X_raw, y)
    pipe = StagePipeline(prep_cfg, selected, dr_cfg).fit(X_clean, y_clean)
    Z = pipe.transform(X_clean)
    model = model_from_params(pipe, params).fit(Z, y_clean)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL_DIR / "naive_bayes_model.npz", **model.export_state())
    write_json(MODEL_DIR / "preprocessing.json", pipe.pre_.to_dict())
    artifact_config = dict(config)
    artifact_config["cleaning"] = {"dropped_train_duplicates": n_dropped}
    artifact_config["model_params"] = model.get_params()
    artifact_config["dr_state"] = _dr_state(pipe)
    write_json(MODEL_DIR / "konfigurasi.json", artifact_config)
    return model, pipe


def main_from_greedy(greedy_path):
    """Finalize from the greedy joint-search configuration."""
    assert Path.cwd() == REPO_ROOT, "Jalankan dari akar repo"
    greedy = read_json(greedy_path)
    prep_cfg = greedy["preprocessing"]
    selected = greedy["feature_selection"].get("selected_features")
    dr_cfg = greedy["dimensionality_reduction"]
    params = dict(greedy["params"])

    _, _, y, X_raw = load_data()
    train_idx, valid_idx = make_split(y)
    y_valid = y[valid_idx]

    # Mandatory Baseline (sesuai slide kuliah: kategorikal hasil diskritisasi,
    # Laplace smoothing +1, MAP murni tanpa feature_weights/threshold tuning).
    mandatory_prep = {
        "name": "mandatory_baseline_bin_q20",
        "feature_set": "baseline",
        "options": {
            "standardize": False,
            "add_missing_indicators": True,
            "categorical_numeric_cols": list(DEFAULT_CATEGORICAL_NUMERIC),
            "log1p_cols": list(DEFAULT_LOG1P_COLS),
            "bin_cols": list(DEFAULT_BIN_COLS),
            "n_bins": 20,
            "bin_strategy": "quantile",
        },
    }
    mandatory_params = {
        "mode": "baseline",
        "variant": "categorical",
        "alpha": 1.0,
        "var_smoothing": 0.0,
        "threshold": 0.5,
        "feature_weights": None,
    }
    baseline_row = evaluate_candidate_repeated(
        X_raw,
        y,
        train_idx,
        valid_idx,
        mandatory_prep,
        None,
        {"method": "none"},
        mandatory_params,
        seeds=(SEED, 0, 1),
    )
    baseline_row.update(
        {
            "source": "scratch_baseline",
            "config": "baseline_categorical_alpha1_t0.50",
            "ronde": "mandatory_baseline",
            "alpha": 1.0,
            "var_smoothing": 0.0,
            "threshold": 0.5,
        }
    )

    final_row = evaluate_candidate_repeated(
        X_raw,
        y,
        train_idx,
        valid_idx,
        prep_cfg,
        selected,
        dr_cfg,
        params,
        seeds=(SEED, 0, 1),
    )
    final_row.update(
        {
            "source": "scratch_bonus",
            "config": config_name(params),
            "ronde": "bonus_kdb_final",
            "alpha": params["alpha"],
            "var_smoothing": params["var_smoothing"],
            "threshold": params["threshold"],
        }
    )
    rows = [baseline_row, final_row]

    print("\n[Pembanding scikit-learn]", flush=True)
    sklearn_rows = []
    for name, predictor in (
        ("GaussianNB_numeric+onehot", _gaussian_comparison),
        ("CategoricalNB_binned", _categorical_comparison),
    ):
        pred = predictor(
            X_raw,
            y,
            train_idx,
            valid_idx,
            params,
            prep_cfg=prep_cfg,
            selected_features=selected,
            dr_cfg=dr_cfg,
        )
        row = {"source": "sklearn", "config": name, "ronde": "sklearn_comparison"}
        row.update(metrics(y_valid, pred))
        sklearn_rows.append(row)
        print(f"  {name}: macro_f1={row['macro_f1']:.6f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows + sklearn_rows)
    result.sort_values(
        ["cv_macro_f1_mean", "config"],
        ascending=[False, True],
        na_position="last",
    ).to_csv(RESULT_CSV, index=False)

    baseline_ref = float(final_row["cv_macro_f1_mean"])
    try:
        prep_stage = read_json(PREP_CONFIG)
        baseline_ref = float(prep_stage.get("cv_macro_f1_mean", baseline_ref))
    except (FileNotFoundError, KeyError, ValueError):
        pass
    decision = {
        "ronde": "greedy_tuning",
        "variabel": "full_config",
        "terpilih": "greedy_best",
        "baseline_f1": baseline_ref,
        "best_f1": float(final_row["cv_macro_f1_mean"]),
        "delta": round(float(final_row["cv_macro_f1_mean"]) - baseline_ref, 6),
        "keputusan": "ADOPSI",
        "alasan": "greedy joint search: fitur turunan + preprocessing + alpha + bobot + threshold",
    }
    pd.DataFrame([decision]).to_csv(DECISION_CSV, index=False)
    pd.DataFrame(sklearn_rows).to_csv(SKLEARN_CSV, index=False)

    stage_summary = {}
    for key, path in (
        ("preprocessing", PREP_CONFIG),
        ("feature_selection", FS_CONFIG),
        ("dimensionality_reduction", DR_CONFIG),
    ):
        try:
            stage = read_json(path)
        except FileNotFoundError:
            continue
        stage_summary[key] = {
            "chosen_name": stage.get("chosen_name", stage.get("chosen_method")),
            "cv_macro_f1_mean": stage.get("cv_macro_f1_mean"),
        }

    variant = params.get("variant", "hybrid")
    config = {
        "algorithm": (
            "KDB (Bayesian Network) from scratch"
            if variant == "kdb"
            else "Naive Bayes from scratch"
        ),
        "variant": variant,
        "params": {
            "variant": variant,
            "k": params.get("k", 0),
            "m_estimate": params.get("m_estimate", 0.0),
            "alpha": params["alpha"],
            "var_smoothing": params["var_smoothing"],
            "threshold": params["threshold"],
            "bias": params.get("bias", 0.0),
            "feature_weights": params.get("feature_weights"),
        },
        "preprocessing": prep_cfg,
        "feature_engineering": {
            "selected_derived_features": greedy.get("selected_derived_features", [])
        },
        "feature_selection": {
            "method": greedy["feature_selection"].get("method", "none"),
            "selected_features": selected,
        },
        "dimensionality_reduction": dr_cfg,
        "seed": SEED,
        "mandatory_baseline": {
            "mode": "baseline",
            "variant": "categorical",
            "alpha": 1.0,
            "threshold": 0.5,
            "n_bins": 20,
            "macro_f1_cv": round(float(baseline_row["cv_macro_f1_mean"]), 6),
            "macro_f1_cv_std": round(float(baseline_row["cv_macro_f1_std"]), 6),
            "macro_f1_holdout": round(float(baseline_row["holdout_macro_f1"]), 6),
        },
        "bonus_references": [
            "M. Sahami, 'Learning Limited Dependence Bayesian Classifiers,' Proc. KDD-96, pp. 335-338, 1996.",
            "N. Friedman, D. Geiger, M. Goldszmidt, 'Bayesian Network Classifiers,' Machine Learning 29, pp. 131-163, 1997.",
            "B. Cestnik, 'Estimating Probabilities: A Crucial Task in Machine Learning,' Proc. ECAI-90, pp. 147-149, 1990.",
            "N. A. Zaidi et al., 'Alleviating Naive Bayes Attribute Independence Assumption by Attribute Weighting,' JMLR 14, pp. 1947-1988, 2013.",
        ],
        "macro_f1_validasi": round(float(final_row["holdout_macro_f1"]), 6),
        "macro_f1_cv": round(float(final_row["cv_macro_f1_mean"]), 6),
        "macro_f1_cv_std": round(float(final_row["cv_macro_f1_std"]), 6),
        "macro_f1_holdout": round(float(final_row["holdout_macro_f1"]), 6),
        "greedy": {
            "n_trials": greedy.get("n_trials"),
            "feature_names": greedy.get("feature_names"),
            "cv_seed_mean_min": greedy.get("cv_seed_mean_min"),
            "cv_seed_mean_max": greedy.get("cv_seed_mean_max"),
        },
        "stage_summary": stage_summary,
        "sklearn_comparison": [
            {"model": r["config"], "macro_f1_validasi": r["macro_f1"]}
            for r in sklearn_rows
        ],
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(CONFIG_JSON, config)
    save_artifacts(X_raw, y, prep_cfg, selected, dr_cfg, params, config)

    print("\n=== PARAMETER AKHIR (GREEDY) ===", flush=True)
    print(
        {
            "alpha": params["alpha"],
            "var_smoothing": params["var_smoothing"],
            "threshold": params["threshold"],
            "feature_weights": params.get("feature_weights"),
        },
        flush=True,
    )
    print(f"Fitur turunan: {greedy.get('selected_derived_features')}", flush=True)
    print(f"Macro F1 CV      : {final_row['cv_macro_f1_mean']:.6f}", flush=True)
    print(f"Macro F1 holdout : {final_row['holdout_macro_f1']:.6f}", flush=True)
    print(f"Hasil: {RESULT_CSV}", flush=True)
    print(f"Keputusan: {DECISION_CSV}", flush=True)
    print(f"Konfigurasi: {CONFIG_JSON}", flush=True)
    print(f"Artefak model: {MODEL_DIR}", flush=True)
    return result


def main():
    assert Path.cwd() == REPO_ROOT, "Jalankan dari akar repo"
    if GREEDY_CONFIG.exists():
        print(
            f"Memakai konfigurasi greedy: {GREEDY_CONFIG}",
            flush=True,
        )
        return main_from_greedy(GREEDY_CONFIG)
    prep_stage, fs_stage, dr_stage = load_stage_configs()
    prep_cfg = prep_stage["chosen"]
    selected = fs_stage.get("selected_features")
    dr_cfg = dr_stage["chosen"]
    params = dict(REFERENCE_PARAMS)

    _, _, y, X_raw = load_data()
    train_idx, valid_idx = make_split(y)
    y_valid = y[valid_idx]
    y_train = y[train_idx]
    rows, decisions = [], []

    def evaluate(round_name, trial):
        row, oof, holdout = evaluate_candidate(
            X_raw,
            y,
            train_idx,
            valid_idx,
            prep_cfg,
            selected,
            dr_cfg,
            trial,
            n_splits=N_FOLDS,
        )
        return row_for(round_name, row, trial), oof, holdout

    baseline_row, _, _ = evaluate("baseline", params)
    rows.append(baseline_row)
    baseline_cv = baseline_row["cv_macro_f1_mean"]
    print(
        f"Baseline (cv={baseline_cv:.6f}, "
        f"holdout={baseline_row['holdout_macro_f1']:.6f})",
        flush=True,
    )

    print("\n[Ronde 1/3] var_smoothing", flush=True)
    vs_rows = []
    for value in VAR_SMOOTHING_GRID:
        trial = dict(params, var_smoothing=value)
        row, _, _ = evaluate("var_smoothing", trial)
        rows.append(row)
        vs_rows.append((row, trial))
        print(
            f"  var_smoothing={value:.4g}: cv={row['cv_macro_f1_mean']:.6f} "
            f"holdout={row['holdout_macro_f1']:.6f}",
            flush=True,
        )
    best_row, best_trial = max(
        vs_rows,
        key=lambda item: (
            item[0]["cv_macro_f1_mean"],
            -VAR_SMOOTHING_GRID.index(item[1]["var_smoothing"]),
        ),
    )
    best_for_decision = {
        "macro_f1": best_row["cv_macro_f1_mean"],
        "var_smoothing": best_trial["var_smoothing"],
    }
    baseline_cv, adopt = decide(
        decisions,
        "var_smoothing",
        "var_smoothing",
        best_for_decision,
        baseline_cv,
    )
    if adopt:
        params["var_smoothing"] = best_trial["var_smoothing"]

    print("\n[Ronde 2/3] alpha", flush=True)
    alpha_rows = []
    for value in ALPHA_GRID:
        trial = dict(params, alpha=value)
        row, _, _ = evaluate("alpha", trial)
        rows.append(row)
        alpha_rows.append((row, trial))
        print(
            f"  alpha={value:.4g}: cv={row['cv_macro_f1_mean']:.6f} "
            f"holdout={row['holdout_macro_f1']:.6f}",
            flush=True,
        )
    best_row, best_trial = max(
        alpha_rows,
        key=lambda item: (
            item[0]["cv_macro_f1_mean"],
            -ALPHA_GRID.index(item[1]["alpha"]),
        ),
    )
    best_for_decision = {
        "macro_f1": best_row["cv_macro_f1_mean"],
        "alpha": best_trial["alpha"],
    }
    baseline_cv, adopt = decide(
        decisions, "alpha", "alpha", best_for_decision, baseline_cv
    )
    if adopt:
        params["alpha"] = best_trial["alpha"]

    print("\n[Ronde 3/3] threshold", flush=True)
    folds = precompute_folds(
        X_raw, y, train_idx, prep_cfg, selected, dr_cfg, n_splits=N_FOLDS
    )
    per_threshold, oof = threshold_curve(folds, params, THRESHOLDS, len(train_idx))
    holdout = holdout_scores(
        X_raw, y, train_idx, valid_idx, prep_cfg, selected, dr_cfg, params
    )
    threshold_rows = []
    for threshold in THRESHOLDS:
        threshold = float(threshold)
        fold_scores = per_threshold[threshold]
        row = {
            "config": config_name(dict(params, threshold=threshold)),
            "ronde": "threshold",
            "alpha": params["alpha"],
            "var_smoothing": params["var_smoothing"],
            "threshold": threshold,
            "cv_macro_f1_mean": float(np.mean(fold_scores)),
            "cv_macro_f1_std": float(np.std(fold_scores)),
            "oof_macro_f1": float(
                f1_score(
                    y_train,
                    (oof > threshold).astype(np.int64),
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                )
            ),
            "holdout_macro_f1": float(
                f1_score(
                    y_valid,
                    (holdout > threshold).astype(np.int64),
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                )
            ),
        }
        row.update(
            {
                f"holdout_{k}": v
                for k, v in metrics_from_scores(holdout, y_valid, threshold).items()
                if k != "macro_f1"
            }
        )
        threshold_rows.append(row)
        rows.append(row_for("threshold", row, dict(params, threshold=threshold)))
    best_row = max(
        threshold_rows,
        key=lambda r: (
            r["cv_macro_f1_mean"],
            -abs(r["threshold"] - 0.5),
            -r["threshold"],
        ),
    )
    best_for_decision = {
        "macro_f1": best_row["cv_macro_f1_mean"],
        "threshold": best_row["threshold"],
    }
    baseline_cv, adopt = decide(
        decisions, "threshold", "threshold", best_for_decision, baseline_cv
    )
    if adopt:
        params["threshold"] = float(best_row["threshold"])

    final_row, _, _ = evaluate("final", params)
    rows.append(final_row)
    final_cv = final_row["cv_macro_f1_mean"]
    final_holdout = final_row["holdout_macro_f1"]

    print("\n[Pembanding scikit-learn]", flush=True)
    sklearn_rows = []
    for name, predictor in (
        ("GaussianNB_numeric+onehot", _gaussian_comparison),
        ("CategoricalNB_binned", _categorical_comparison),
    ):
        pred = predictor(
            X_raw,
            y,
            train_idx,
            valid_idx,
            params,
            prep_cfg=prep_cfg,
            selected_features=selected,
            dr_cfg=dr_cfg,
        )
        row = {"source": "sklearn", "config": name, "ronde": "sklearn_comparison"}
        row.update(metrics(y_valid, pred))
        sklearn_rows.append(row)
        print(f"  {name}: macro_f1={row['macro_f1']:.6f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows + sklearn_rows)
    result.sort_values(
        ["cv_macro_f1_mean", "config"],
        ascending=[False, True],
        na_position="last",
    ).to_csv(RESULT_CSV, index=False)
    pd.DataFrame(decisions).to_csv(DECISION_CSV, index=False)
    pd.DataFrame(sklearn_rows).to_csv(SKLEARN_CSV, index=False)

    config = {
        "algorithm": "Naive Bayes from scratch",
        "variant": "hybrid",
        "params": {
            "alpha": params["alpha"],
            "var_smoothing": params["var_smoothing"],
            "threshold": params["threshold"],
        },
        "preprocessing": prep_cfg,
        "feature_selection": {
            "method": fs_stage["chosen_method"],
            "selected_features": selected,
        },
        "dimensionality_reduction": dr_cfg,
        "seed": SEED,
        "macro_f1_validasi": round(float(final_holdout), 6),
        "macro_f1_cv": round(float(final_cv), 6),
        "macro_f1_holdout": round(float(final_holdout), 6),
        "stage_summary": {
            "preprocessing": {
                "chosen_name": prep_stage["chosen_name"],
                "cv_macro_f1_mean": prep_stage["cv_macro_f1_mean"],
            },
            "feature_selection": {
                "chosen_method": fs_stage["chosen_method"],
                "cv_macro_f1_mean": fs_stage["cv_macro_f1_mean"],
            },
            "dimensionality_reduction": {
                "chosen_name": dr_stage["chosen_name"],
                "cv_macro_f1_mean": dr_stage["cv_macro_f1_mean"],
            },
        },
        "sklearn_comparison": [
            {"model": r["config"], "macro_f1_validasi": r["macro_f1"]}
            for r in sklearn_rows
        ],
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(CONFIG_JSON, config)
    save_artifacts(X_raw, y, prep_cfg, selected, dr_cfg, params, config)

    print("\n=== PARAMETER AKHIR ===", flush=True)
    print(params, flush=True)
    print(f"Macro F1 CV      : {final_cv:.6f}", flush=True)
    print(f"Macro F1 holdout : {final_holdout:.6f}", flush=True)
    print(f"Hasil: {RESULT_CSV}", flush=True)
    print(f"Keputusan: {DECISION_CSV}", flush=True)
    print(f"Konfigurasi: {CONFIG_JSON}", flush=True)
    print(f"Artefak model: {MODEL_DIR}", flush=True)
    return result


if __name__ == "__main__":
    main()
