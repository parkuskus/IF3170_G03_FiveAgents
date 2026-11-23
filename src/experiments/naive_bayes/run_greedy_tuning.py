from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.naive_bayes.utils.feature_engineering import (
    FEATURE_LIBRARY,
    NB_BEST_FEATURES,
    NB_FINAL_FEATURES,
    add_engineered_features,
)
from src.experiments.naive_bayes.utils.kdb import fit_feature_weights
from src.experiments.naive_bayes.utils.numpy_utils import macro_f1_np
from src.experiments.naive_bayes.utils.pipeline import (
    N_FOLDS,
    OUT_DIR,
    REFERENCE_PARAMS,
    SEED,
    StagePipeline,
    evaluate_candidate,
    evaluate_candidate_repeated,
    load_data,
    make_split,
    model_from_params,
    precompute_folds,
    read_json,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PREP_CONFIG = OUT_DIR / "naive_bayes_preprocessing_config.json"
FS_CONFIG = OUT_DIR / "naive_bayes_feature_selection_config.json"
DR_CONFIG = OUT_DIR / "naive_bayes_dimensionality_reduction_config.json"
TRIALS_CSV = OUT_DIR / "naive_bayes_greedy_trials.csv"
CONFIG_JSON = OUT_DIR / "naive_bayes_greedy_config.json"

SCREEN_FOLDS = 3
SCREEN_SEED = SEED
ADOPT_GAP = 0.0002
MAX_FEATURES = 8
SHORTLIST_SIZE = 12

FEATURE_CANDIDATES = (
    "has_requests",
    "is_ota",
    "is_offline",
    "dup_channel",
    "is_type_b",
    "is_half_board",
    "is_peak_season",
    "prior_new",
    "has_prior_cancel",
    "is_zero_rate",
    "lead_30",
    "lead_90",
    "adult_1",
    "has_children",
    "has_weekend_night",
    "noreq_ota",
    "pnew_ota",
    "typeb_ota",
    "hr_ota",
    "hr_pcan",
    "pcan_ota",
    "ota_x_requests",
    "rate_weekend_req_ota",
    "lead_x_ota_req",
    "lead_x_rate",
    "lead_x_weekend_req",
    "pcan_x_ota_req",
    "channel_x_meal",
)

N_BINS_GRID = (10, 15, 20, 25, 30, 40, 50)
ALPHA_GRID = (0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0)
VAR_SMOOTHING_GRID = (1e-12, 1e-9, 1e-6, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0)
KDB_K_GRID = (1, 2, 3, 4)
KDB_M_GRID = (10.0, 20.0, 30.0, 50.0, 100.0)
WEIGHT_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
THRESHOLDS = np.round(np.arange(0.30, 0.76, 0.005), 3)

TEXT_FEATURES = {
    "dup_channel",
    "dup_returning",
    "dup_parking",
    "ota_x_requests",
    "rate_weekend_req_ota",
    "lead_x_ota_req",
    "lead_x_rate",
    "lead_x_weekend_req",
    "pcan_x_ota_req",
    "channel_x_meal",
}


def prep_with_features(base_prep, feature_names):
    cfg = copy.deepcopy(base_prep)
    names = list(feature_names)
    cfg["feature_set"] = names if names else "baseline"
    options = cfg.setdefault("options", {})
    categorical = list(options.get("categorical_numeric_cols", ()))
    for name in names:
        if name in FEATURE_LIBRARY and name not in TEXT_FEATURES:
            if name not in categorical:
                categorical.append(name)
    options["categorical_numeric_cols"] = categorical
    return cfg


def main():
    assert Path.cwd() == REPO_ROOT, "Jalankan dari akar repo"
    for path in (PREP_CONFIG, FS_CONFIG, DR_CONFIG):
        if not path.exists():
            raise SystemExit(f"Konfigurasi tahap sebelumnya hilang: {path}")

    prep_stage = read_json(PREP_CONFIG)
    fs_stage = read_json(FS_CONFIG)
    dr_stage = read_json(DR_CONFIG)
    base_prep = prep_stage["chosen"]
    base_selected = fs_stage.get("selected_features")
    base_dr = dr_stage["chosen"]
    params = dict(REFERENCE_PARAMS)

    _, _, y, X_raw = load_data()
    train_idx, valid_idx = make_split(y)
    y_train = y[train_idx]

    trials = []

    def log(stage, variable, candidate, row, accepted, reason):
        trials.append(
            {
                "stage": stage,
                "variable": variable,
                "candidate": candidate,
                "cv_macro_f1_mean": float(row["cv_macro_f1_mean"]),
                "cv_macro_f1_std": float(row["cv_macro_f1_std"]),
                "holdout_macro_f1": float(row["holdout_macro_f1"]),
                "delta": float(row.get("delta", 0.0)),
                "accepted": bool(accepted),
                "reason": reason,
            }
        )

    def screen(prep_cfg, selected, dr_cfg, trial_params):
        row, _, _ = evaluate_candidate(
            X_raw,
            y,
            train_idx,
            valid_idx,
            prep_cfg,
            selected,
            dr_cfg,
            trial_params,
            n_splits=SCREEN_FOLDS,
        )
        return row

    def frame_columns(feature_names):
        frame = add_engineered_features(X_raw, list(feature_names) or "baseline")
        return list(frame.columns)

    baseline_prep = prep_with_features(base_prep, [])
    baseline_row = screen(baseline_prep, base_selected, base_dr, params)
    best_score = baseline_row["cv_macro_f1_mean"]
    log("baseline", "feature_set", "baseline", baseline_row, True, "titik awal")
    print(f"Baseline greedy: cv={best_score:.6f}", flush=True)

    seed_features = list(NB_FINAL_FEATURES)
    seed_prep = prep_with_features(base_prep, seed_features)
    seed_row = screen(seed_prep, base_selected, base_dr, params)
    seed_score = seed_row["cv_macro_f1_mean"]
    seed_row["delta"] = seed_score - best_score
    log(
        "feature_selection",
        "feature",
        "+".join(seed_features),
        seed_row,
        True,
        "seed fitur final",
    )
    print(
        f"Seed fitur {seed_features}: cv={seed_score:.6f}",
        flush=True,
    )

    # 1. Screening individual derived features.
    print("\n[1] Screening fitur turunan", flush=True)
    individual = {}
    for name in FEATURE_CANDIDATES:
        trial_prep = prep_with_features(base_prep, [name])
        row = screen(trial_prep, base_selected, base_dr, params)
        individual[name] = row["cv_macro_f1_mean"]
        print(
            f"  {name:<22} cv={row['cv_macro_f1_mean']:.6f}",
            flush=True,
        )
    ranked = sorted(individual, key=lambda n: (-individual[n], n))
    pool = list(dict.fromkeys(list(NB_BEST_FEATURES) + ranked[:SHORTLIST_SIZE]))

    # 2. Greedy forward selection.
    print("\n[2] Greedy forward selection", flush=True)
    selected_features = seed_features
    current_score = seed_score
    for _ in range(MAX_FEATURES):
        best_name = None
        best_row = None
        best_candidate_score = current_score
        for name in pool:
            if name in selected_features:
                continue
            trial_features = selected_features + [name]
            trial_prep = prep_with_features(base_prep, trial_features)
            row = screen(trial_prep, base_selected, base_dr, params)
            delta = row["cv_macro_f1_mean"] - current_score
            if row["cv_macro_f1_mean"] > best_candidate_score + 1e-9:
                best_candidate_score = row["cv_macro_f1_mean"]
                best_name = name
                best_row = row
                best_row["delta"] = delta
        if best_name is None or best_row["delta"] < ADOPT_GAP:
            if best_row is not None:
                log(
                    "feature_selection",
                    "feature",
                    best_name,
                    best_row,
                    False,
                    f"delta {best_row['delta']:+.6f} < {ADOPT_GAP}",
                )
            print("  berhenti: tidak ada kenaikan >= 0.001", flush=True)
            break
        selected_features.append(best_name)
        current_score = best_candidate_score
        log(
            "feature_selection",
            "feature",
            "+".join(selected_features),
            best_row,
            True,
            f"delta {best_row['delta']:+.6f}",
        )
        print(
            f"  adopsi {best_name:<20} cv={current_score:.6f} "
            f"(delta {best_row['delta']:+.6f})",
            flush=True,
        )

    prep_cfg = prep_with_features(base_prep, selected_features)
    feature_names = frame_columns(selected_features)
    print(f"\nFitur terpilih: {selected_features or 'baseline'}", flush=True)

    # 3. Greedy preprocessing parameters.
    print("\n[3] Greedy parameter preprocessing", flush=True)

    def try_option(variable, candidate, mutate):
        nonlocal prep_cfg, current_score
        trial_prep = copy.deepcopy(prep_cfg)
        mutate(trial_prep)
        row = screen(trial_prep, base_selected, base_dr, params)
        delta = row["cv_macro_f1_mean"] - current_score
        accepted = delta >= ADOPT_GAP
        log("preprocessing", variable, candidate, row, accepted, f"delta {delta:+.6f}")
        print(
            f"  {variable}={candidate}: cv={row['cv_macro_f1_mean']:.6f} "
            f"({'ADOPSI' if accepted else 'TOLAK'})",
            flush=True,
        )
        if accepted:
            prep_cfg = trial_prep
            current_score = row["cv_macro_f1_mean"]
        return accepted

    for n_bins in N_BINS_GRID:
        try_option(
            "n_bins",
            n_bins,
            lambda cfg, value=n_bins: cfg["options"].__setitem__("n_bins", value),
        )
    for strategy in ("quantile", "uniform"):
        try_option(
            "bin_strategy",
            strategy,
            lambda cfg, value=strategy: cfg["options"].__setitem__(
                "bin_strategy", value
            ),
        )
    for flag in (True, False):
        try_option(
            "add_missing_indicators",
            flag,
            lambda cfg, value=flag: cfg["options"].__setitem__(
                "add_missing_indicators", value
            ),
        )
    for flag in (True, False):
        try_option(
            "standardize",
            flag,
            lambda cfg, value=flag: cfg["options"].__setitem__("standardize", value),
        )
    for log1p in (list(prep_cfg["options"].get("log1p_cols", [])), []):
        try_option(
            "log1p_cols",
            "default" if log1p else "none",
            lambda cfg, value=log1p: cfg["options"].__setitem__("log1p_cols", value),
        )
    if "arrival_day" in prep_cfg["options"].get("bin_cols", []):

        def _arrival_day_as_categorical(cfg):
            bin_cols = [
                c for c in cfg["options"].get("bin_cols", []) if c != "arrival_day"
            ]
            cat_cols = list(cfg["options"].get("categorical_numeric_cols", []))
            if "arrival_day" not in cat_cols:
                cat_cols.append("arrival_day")
            cfg["options"]["bin_cols"] = bin_cols
            cfg["options"]["categorical_numeric_cols"] = cat_cols

        try_option("arrival_day_mode", "categorical", _arrival_day_as_categorical)

    # 4. Greedy alpha and var_smoothing.
    print("\n[4] Greedy alpha / var_smoothing", flush=True)
    for alpha in ALPHA_GRID:
        trial_params = dict(params, alpha=alpha)
        row = screen(prep_cfg, base_selected, base_dr, trial_params)
        delta = row["cv_macro_f1_mean"] - current_score
        accepted = delta >= ADOPT_GAP
        log("modeling", "alpha", alpha, row, accepted, f"delta {delta:+.6f}")
        print(
            f"  alpha={alpha}: cv={row['cv_macro_f1_mean']:.6f} "
            f"({'ADOPSI' if accepted else 'TOLAK'})",
            flush=True,
        )
        if accepted:
            params["alpha"] = alpha
            current_score = row["cv_macro_f1_mean"]

    probe = StagePipeline(prep_cfg, base_selected, base_dr).fit(
        X_raw.iloc[train_idx], y[train_idx]
    )
    if probe.numeric_indices:
        for smoothing in VAR_SMOOTHING_GRID:
            trial_params = dict(params, var_smoothing=smoothing)
            row = screen(prep_cfg, base_selected, base_dr, trial_params)
            delta = row["cv_macro_f1_mean"] - current_score
            accepted = delta >= ADOPT_GAP
            log(
                "modeling",
                "var_smoothing",
                smoothing,
                row,
                accepted,
                f"delta {delta:+.6f}",
            )
            print(
                f"  var_smoothing={smoothing}: cv={row['cv_macro_f1_mean']:.6f} "
                f"({'ADOPSI' if accepted else 'TOLAK'})",
                flush=True,
            )
            if accepted:
                params["var_smoothing"] = smoothing
                current_score = row["cv_macro_f1_mean"]
    else:
        log(
            "modeling",
            "var_smoothing",
            "N/A",
            {
                "cv_macro_f1_mean": current_score,
                "cv_macro_f1_std": 0.0,
                "holdout_macro_f1": 0.0,
                "delta": 0.0,
            },
            False,
            "tidak ada fitur numerik; var_smoothing tidak berlaku",
        )
        print("  var_smoothing: N/A (tidak ada fitur numerik)", flush=True)

    # 4b. Greedy KDB structure (k-dependence Bayesian network).
    if not probe.numeric_indices:
        print("\n[4b] Greedy KDB (k, m_estimate)", flush=True)
        for k_val in KDB_K_GRID:
            for m_val in KDB_M_GRID:
                trial_params = dict(params, variant="kdb", k=k_val, m_estimate=m_val)
                row = screen(prep_cfg, base_selected, base_dr, trial_params)
                delta = row["cv_macro_f1_mean"] - current_score
                accepted = delta >= ADOPT_GAP
                log(
                    "modeling",
                    "kdb",
                    f"k={k_val},m={m_val}",
                    row,
                    accepted,
                    f"delta {delta:+.6f}",
                )
                print(
                    f"  kdb(k={k_val}, m={m_val}): cv={row['cv_macro_f1_mean']:.6f} "
                    f"({'ADOPSI' if accepted else 'TOLAK'})",
                    flush=True,
                )
                if accepted:
                    params["variant"] = "kdb"
                    params["k"] = k_val
                    params["m_estimate"] = m_val
                    current_score = row["cv_macro_f1_mean"]

    # 5. Feature weights on OOF log-likelihood (WANBIA-C Adam + coordinate fallback).
    print("\n[5] Feature weights", flush=True)
    folds = precompute_folds(
        X_raw,
        y,
        train_idx,
        prep_cfg,
        base_selected,
        base_dr,
        n_splits=N_FOLDS,
    )
    feature_names = folds[0]["pipe"].pre_.feature_names_
    n_features = len(feature_names)
    oof_lik = np.zeros((len(train_idx), 2, n_features), dtype=np.float64)
    oof_prior = np.zeros((len(train_idx), 2), dtype=np.float64)
    for fold in folds:
        model = model_from_params(fold["pipe"], params).fit(fold["Ztr"], fold["ytr"])
        oof_lik[fold["va"]] = model.feature_log_likelihood(fold["Zva"])
        oof_prior[fold["va"]] = model.class_log_prior_

    def score_weights(weights, threshold=0.5, bias=0.0):
        log_joint = oof_prior + np.einsum("nkf,f->nk", oof_lik, weights)
        log_joint[:, 1] += bias
        log_joint -= log_joint.max(axis=1, keepdims=True)
        prob = np.exp(log_joint)
        prob /= np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)
        return macro_f1_np(y_train, (prob[:, 1] > threshold).astype(np.int64))

    # 5a. WANBIA-C NumPy Adam optimization.
    adam_w, adam_b = fit_feature_weights(
        oof_lik, oof_prior, y_train, lam=1e-3, iters=400, lr=0.5
    )
    adam_thr = float(params.get("threshold", 0.5))
    adam_oof = score_weights(adam_w, adam_thr, adam_b)
    for threshold in THRESHOLDS:
        cand_score = score_weights(adam_w, float(threshold), adam_b)
        if cand_score > adam_oof + 1e-9:
            adam_oof = cand_score
            adam_thr = float(threshold)

    # 5b. Coordinate descent baseline on OOF.
    weights = np.ones(n_features, dtype=np.float64)
    best_threshold = float(params.get("threshold", 0.5))
    weight_score = score_weights(weights, best_threshold, 0.0)
    for _ in range(5):
        improved = False
        for j in range(n_features):
            old = weights[j]
            best_value = old
            best_value_score = weight_score
            for value in WEIGHT_GRID:
                weights[j] = value
                candidate_score = score_weights(weights, best_threshold, 0.0)
                if candidate_score > best_value_score + 1e-9:
                    best_value_score = candidate_score
                    best_value = value
            weights[j] = best_value
            if best_value != old:
                improved = True
            weight_score = best_value_score
        for threshold in THRESHOLDS:
            candidate_score = score_weights(weights, float(threshold), 0.0)
            if candidate_score > weight_score + 1e-9:
                weight_score = candidate_score
                best_threshold = float(threshold)
                improved = True
        if not improved:
            break

    if adam_oof >= weight_score:
        chosen_w = [float(w) for w in adam_w]
        chosen_b = float(adam_b)
        chosen_thr = float(adam_thr)
        chosen_oof = float(adam_oof)
        weight_method = "wanbia_adam"
    else:
        chosen_w = [float(w) for w in weights]
        chosen_b = 0.0
        chosen_thr = float(best_threshold)
        chosen_oof = float(weight_score)
        weight_method = "coord_descent"

    weighted_params = dict(
        params,
        feature_weights=chosen_w,
        bias=chosen_b,
        threshold=chosen_thr,
    )
    weighted_row = screen(prep_cfg, base_selected, base_dr, weighted_params)
    delta = weighted_row["cv_macro_f1_mean"] - current_score
    accepted = delta >= ADOPT_GAP
    log(
        "modeling",
        "feature_weights",
        f"{weight_method} oof={chosen_oof:.6f} thr={chosen_thr:.3f}",
        weighted_row,
        accepted,
        f"delta {delta:+.6f}",
    )
    print(
        f"  {weight_method} OOF={chosen_oof:.6f} thr={chosen_thr:.3f}, "
        f"screen cv={weighted_row['cv_macro_f1_mean']:.6f} "
        f"({'ADOPSI' if accepted else 'TOLAK'})",
        flush=True,
    )
    if accepted:
        params = dict(
            params,
            feature_weights=chosen_w,
            bias=chosen_b,
            threshold=chosen_thr,
        )
        current_score = weighted_row["cv_macro_f1_mean"]

    # 6. Greedy threshold on OOF (cheap, no refit).
    print("\n[6] Greedy threshold", flush=True)
    final_weights = np.asarray(
        params.get("feature_weights") or np.ones(n_features), dtype=np.float64
    )
    final_bias = float(params.get("bias", 0.0))
    threshold_best = float(params.get("threshold", 0.5))
    threshold_best_score = score_weights(final_weights, threshold_best, final_bias)
    for threshold in THRESHOLDS:
        candidate_score = score_weights(final_weights, float(threshold), final_bias)
        if candidate_score > threshold_best_score + 1e-9:
            threshold_best_score = candidate_score
            threshold_best = float(threshold)
    params["threshold"] = threshold_best
    log(
        "modeling",
        "threshold",
        threshold_best,
        {
            "cv_macro_f1_mean": threshold_best_score,
            "cv_macro_f1_std": 0.0,
            "holdout_macro_f1": 0.0,
            "delta": threshold_best_score - current_score,
        },
        threshold_best_score > current_score + ADOPT_GAP,
        "dipilih dari sweep OOF",
    )
    print(
        f"  threshold={threshold_best:.2f} oof={threshold_best_score:.6f}", flush=True
    )

    # 7. Final repeated CV.
    print("\n[7] Verifikasi repeated CV", flush=True)
    final_row = evaluate_candidate_repeated(
        X_raw,
        y,
        train_idx,
        valid_idx,
        prep_cfg,
        base_selected,
        base_dr,
        params,
        seeds=(SEED, 0, 1),
    )
    print(
        f"  repeated CV={final_row['cv_macro_f1_mean']:.6f} "
        f"(+/-{final_row['cv_macro_f1_std']:.6f}) "
        f"holdout={final_row['holdout_macro_f1']:.6f}",
        flush=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials).to_csv(TRIALS_CSV, index=False)
    config = {
        "stage": "greedy_tuning",
        "seed": SEED,
        "cv_folds": N_FOLDS,
        "screen_folds": SCREEN_FOLDS,
        "adoption_gap": ADOPT_GAP,
        "feature_candidates": list(FEATURE_CANDIDATES),
        "selected_derived_features": selected_features,
        "preprocessing": prep_cfg,
        "feature_selection": {
            "method": fs_stage.get("chosen_method", "none"),
            "selected_features": base_selected,
        },
        "dimensionality_reduction": base_dr,
        "params": {
            "variant": params.get("variant", "hybrid"),
            "k": params.get("k", 0),
            "m_estimate": params.get("m_estimate", 0.0),
            "alpha": params["alpha"],
            "var_smoothing": params["var_smoothing"],
            "threshold": params["threshold"],
            "bias": params.get("bias", 0.0),
            "feature_weights": params.get("feature_weights"),
        },
        "feature_names": list(feature_names),
        "cv_macro_f1_mean": final_row["cv_macro_f1_mean"],
        "cv_macro_f1_std": final_row["cv_macro_f1_std"],
        "cv_seed_mean_min": final_row["cv_seed_mean_min"],
        "cv_seed_mean_max": final_row["cv_seed_mean_max"],
        "holdout_macro_f1": final_row["holdout_macro_f1"],
        "n_trials": len(trials),
    }
    write_json(CONFIG_JSON, config)
    print(f"\nHasil greedy: {TRIALS_CSV}", flush=True)
    print(f"Konfigurasi greedy: {CONFIG_JSON}", flush=True)
    return config


if __name__ == "__main__":
    main()
