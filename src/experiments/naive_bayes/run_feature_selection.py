from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.naive_bayes.utils.numpy_utils import (
    chi2_scores,
    mutual_info_scores,
)

from src.experiments.naive_bayes.utils.pipeline import (
    ADOPTION_GAP,
    N_FOLDS,
    OUT_DIR,
    SEED,
    StagePipeline,
    evaluate_candidate,
    load_data,
    make_split,
    read_json,
    write_json,
)
from src.experiments.naive_bayes.utils.preprocessing import MISSING_SUFFIX

REPO_ROOT = Path(__file__).resolve().parents[3]
PREP_CONFIG = OUT_DIR / "naive_bayes_preprocessing_config.json"
RESULT_CSV = OUT_DIR / "naive_bayes_feature_selection_results.csv"
DECISION_CSV = OUT_DIR / "naive_bayes_feature_selection_decisions.csv"
CONFIG_JSON = OUT_DIR / "naive_bayes_feature_selection_config.json"

METHOD_ORDER = {
    "none": 0,
    "variance": 1,
    "correlation_prune": 2,
    "chi2_top12": 3,
    "mutual_info_top12": 4,
    "wrapper_backward": 5,
}


def source_names(pipe):
    names = []
    for name in pipe.pre_.feature_names_:
        if name.endswith(MISSING_SUFFIX):
            name = name[: -len(MISSING_SUFFIX)]
        names.append(name)
    return names


def aggregate_by_source(scores, sources):
    out = {}
    for score, source in zip(scores, sources):
        score = float(score)
        if not np.isfinite(score):
            score = 0.0
        out[source] = max(out.get(source, 0.0), score)
    return out


def transformed_train(X_raw, y, train_idx, prep_cfg):
    pipe = StagePipeline(prep_cfg, None, {"method": "none"}).fit(
        X_raw.iloc[train_idx], y[train_idx]
    )
    Z = pipe.transform(X_raw.iloc[train_idx])
    return pipe, Z


def select_by_threshold(scores, all_features, threshold):
    selected = [f for f in all_features if scores.get(f, 0.0) > threshold]
    return selected or list(all_features)


def select_top(scores, all_features, top_k):
    ordered = sorted(all_features, key=lambda f: (-scores.get(f, 0.0), f))
    return sorted(ordered[: min(top_k, len(ordered))])


def select_correlation_prune(pipe, Z, mi_scores, all_features):
    numeric = list(pipe.numeric_indices)
    if len(numeric) < 2:
        return list(all_features)
    corr = np.corrcoef(Z[:, numeric], rowvar=False)
    dropped = set()
    for i in range(len(numeric)):
        for j in range(i + 1, len(numeric)):
            value = corr[i, j]
            if np.isfinite(value) and abs(value) > 0.95:
                left = pipe.pre_.feature_names_[numeric[i]]
                right = pipe.pre_.feature_names_[numeric[j]]
                left = (
                    left[: -len(MISSING_SUFFIX)]
                    if left.endswith(MISSING_SUFFIX)
                    else left
                )
                right = (
                    right[: -len(MISSING_SUFFIX)]
                    if right.endswith(MISSING_SUFFIX)
                    else right
                )
                if left == right:
                    continue
                drop = (
                    left
                    if mi_scores.get(left, 0.0) < mi_scores.get(right, 0.0)
                    else right
                )
                dropped.add(drop)
    selected = [f for f in all_features if f not in dropped]
    return selected or list(all_features)


def select_wrapper_backward(
    X_raw,
    y,
    train_idx,
    valid_idx,
    prep_cfg,
    params,
    mi_scores,
    all_features,
    baseline_cv,
):
    order = sorted(all_features, key=lambda f: (mi_scores.get(f, 0.0), f))
    remaining = list(all_features)
    current_cv = float(baseline_cv)
    max_removals = 4
    pool_size = 5

    for _ in range(max_removals):
        pool = [f for f in order if f in remaining][:pool_size]
        if not pool:
            break
        trials = []
        for feature in pool:
            trial = [f for f in remaining if f != feature]
            row, _, _ = evaluate_candidate(
                X_raw,
                y,
                train_idx,
                valid_idx,
                prep_cfg,
                trial,
                {"method": "none"},
                params,
                n_splits=N_FOLDS,
            )
            trials.append((row["cv_macro_f1_mean"], feature, trial))
        trials.sort(key=lambda item: (-item[0], item[1]))
        best_cv, _, best_trial = trials[0]
        if best_cv >= current_cv + 0.002 and len(best_trial) > 0:
            remaining = best_trial
            current_cv = float(best_cv)
        else:
            break
    return sorted(remaining)


def main():
    assert Path.cwd() == REPO_ROOT, "Jalankan dari akar repo"
    if not PREP_CONFIG.exists():
        raise SystemExit(
            "Jalankan python -m src.experiments.naive_bayes.run_preprocessing terlebih dahulu"
        )

    prep_config = read_json(PREP_CONFIG)
    prep_cfg = prep_config["chosen"]
    params = prep_config.get("reference_params", {})
    _, _, y, X_raw = load_data()
    train_idx, valid_idx = make_split(y)
    all_features = list(X_raw.columns)

    pipe, Z = transformed_train(X_raw, y, train_idx, prep_cfg)
    sources = source_names(pipe)
    discrete_mask = np.isin(np.arange(Z.shape[1]), pipe.categorical_indices)
    my = mutual_info_scores(Z, y[train_idx], discrete_mask=discrete_mask)
    mi_scores = aggregate_by_source(my, sources)

    variance = np.var(Z, axis=0)
    variance_scores = aggregate_by_source(variance, sources)

    Z_nonneg = Z - Z.min(axis=0)
    chi_raw = chi2_scores(Z_nonneg, y[train_idx])
    chi_scores = aggregate_by_source(chi_raw, sources)

    selections = {
        "none": None,
        "variance": select_by_threshold(variance_scores, all_features, 0.01),
        "mutual_info_top12": select_top(mi_scores, all_features, 12),
        "chi2_top12": select_top(chi_scores, all_features, 12),
        "correlation_prune": select_correlation_prune(pipe, Z, mi_scores, all_features),
    }

    descriptions = {
        "none": "semua fitur baseline",
        "variance": "buang fitur dengan varians <= 0.01",
        "mutual_info_top12": "12 fitur teratas menurut mutual information",
        "chi2_top12": "12 fitur teratas menurut chi-square",
        "correlation_prune": "pangkas fitur numerik dengan |korelasi| > 0.95",
        "wrapper_backward": "backward elimination terpandu mutual information",
    }

    baseline_row, _, _ = evaluate_candidate(
        X_raw,
        y,
        train_idx,
        valid_idx,
        prep_cfg,
        None,
        {"method": "none"},
        params,
        n_splits=N_FOLDS,
    )
    baseline_cv = baseline_row["cv_macro_f1_mean"]

    selected_wrapper = select_wrapper_backward(
        X_raw,
        y,
        train_idx,
        valid_idx,
        prep_cfg,
        params,
        mi_scores,
        all_features,
        baseline_cv,
    )
    selections["wrapper_backward"] = selected_wrapper

    rows = []
    for method, selected in selections.items():
        row, _, _ = evaluate_candidate(
            X_raw,
            y,
            train_idx,
            valid_idx,
            prep_cfg,
            selected,
            {"method": "none"},
            params,
            n_splits=N_FOLDS,
        )
        row.update(
            {
                "method": method,
                "description": descriptions[method],
                "n_features": len(all_features) if selected is None else len(selected),
                "selected_features": "" if selected is None else "|".join(selected),
            }
        )
        rows.append(row)
        print(
            f"{method:<20} n={row['n_features']:>2} "
            f"cv={row['cv_macro_f1_mean']:.6f} holdout={row['holdout_macro_f1']:.6f}",
            flush=True,
        )

    result = pd.DataFrame(rows)
    baseline = next(r for r in rows if r["method"] == "none")
    ordered = sorted(
        rows,
        key=lambda r: (-r["cv_macro_f1_mean"], METHOD_ORDER[r["method"]], r["method"]),
    )
    best = ordered[0]
    delta = float(best["cv_macro_f1_mean"] - baseline["cv_macro_f1_mean"])
    adopt = delta >= ADOPTION_GAP
    chosen = best if adopt else baseline
    chosen_method = chosen["method"]
    chosen_selected = selections[chosen_method]

    decision = {
        "ronde": "feature_selection",
        "variabel": "method",
        "terpilih": chosen_method,
        "baseline_f1": round(float(baseline["cv_macro_f1_mean"]), 6),
        "best_f1": round(float(best["cv_macro_f1_mean"]), 6),
        "delta": round(delta, 6),
        "keputusan": "ADOPSI" if adopt else "TOLAK",
        "alasan": "naik >= 0.005 pada CV"
        if adopt
        else "kenaikan di bawah 0.005 pada CV",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.sort_values(["cv_macro_f1_mean", "method"], ascending=[False, True]).to_csv(
        RESULT_CSV, index=False
    )
    pd.DataFrame([decision]).to_csv(DECISION_CSV, index=False)

    config = {
        "stage": "feature_selection",
        "seed": SEED,
        "cv_folds": N_FOLDS,
        "selection_metric": "cv_macro_f1_mean",
        "reference_params": params,
        "baseline_method": "none",
        "baseline_cv_macro_f1": round(float(baseline["cv_macro_f1_mean"]), 6),
        "baseline_holdout_macro_f1": round(float(baseline["holdout_macro_f1"]), 6),
        "chosen_method": chosen_method,
        "selected_features": chosen_selected,
        "cv_macro_f1_mean": round(float(chosen["cv_macro_f1_mean"]), 6),
        "cv_macro_f1_std": round(float(chosen["cv_macro_f1_std"]), 6),
        "holdout_macro_f1": round(float(chosen["holdout_macro_f1"]), 6),
        "decision": decision,
    }
    write_json(CONFIG_JSON, config)

    print("\n=== KEPUTUSAN FEATURE SELECTION ===", flush=True)
    print(pd.DataFrame([decision]).to_string(index=False), flush=True)
    print(f"Fitur terpilih: {chosen_selected or 'semua fitur'}", flush=True)
    print(f"Hasil: {RESULT_CSV}", flush=True)
    print(f"Keputusan: {DECISION_CSV}", flush=True)
    print(f"Konfigurasi: {CONFIG_JSON}", flush=True)
    return result


if __name__ == "__main__":
    main()
