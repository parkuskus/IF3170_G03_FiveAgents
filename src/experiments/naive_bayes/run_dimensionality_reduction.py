from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.pipeline import (
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

PREP_CONFIG = OUT_DIR / "naive_bayes_preprocessing_config.json"
FS_CONFIG = OUT_DIR / "naive_bayes_feature_selection_config.json"
RESULT_CSV = OUT_DIR / "naive_bayes_dimensionality_reduction_results.csv"
DECISION_CSV = OUT_DIR / "naive_bayes_dimensionality_reduction_decisions.csv"
CONFIG_JSON = OUT_DIR / "naive_bayes_dimensionality_reduction_config.json"


def pca_candidates(n_numeric):
    candidates = [{"name": "none", "method": "none"}]
    for value in (0.90, 0.95):
        candidates.append(
            {
                "name": f"pca_numeric_{value:.2f}",
                "method": "pca_numeric",
                "n_components": value,
            }
        )
    for value in (5, 10, 15):
        if value <= n_numeric:
            candidates.append(
                {
                    "name": f"pca_numeric_{value}",
                    "method": "pca_numeric",
                    "n_components": value,
                }
            )
    return candidates


def main():
    assert Path.cwd() == REPO_ROOT, "Jalankan dari akar repo"
    if not PREP_CONFIG.exists():
        raise SystemExit("Jalankan run_preprocessing.py terlebih dahulu")
    if not FS_CONFIG.exists():
        raise SystemExit("Jalankan run_feature_selection.py terlebih dahulu")

    prep_config = read_json(PREP_CONFIG)
    fs_config = read_json(FS_CONFIG)
    prep_cfg = prep_config["chosen"]
    selected = fs_config.get("selected_features")
    params = prep_config.get("reference_params", {})

    _, _, y, X_raw = load_data()
    train_idx, valid_idx = make_split(y)

    probe = StagePipeline(prep_cfg, selected, {"method": "none"}).fit(
        X_raw.iloc[train_idx], y[train_idx]
    )
    n_numeric = len(probe.numeric_indices)
    print(f"fitur numerik setelah preprocessing: {n_numeric}", flush=True)
    candidates = pca_candidates(n_numeric) if n_numeric else [{"name": "none", "method": "none"}]

    rows = []
    for cand in candidates:
        dr_cfg = {"method": cand["method"]}
        if cand["method"] == "pca_numeric":
            dr_cfg["n_components"] = cand["n_components"]
        row, _, _ = evaluate_candidate(
            X_raw,
            y,
            train_idx,
            valid_idx,
            prep_cfg,
            selected,
            dr_cfg,
            params,
            n_splits=N_FOLDS,
        )
        row.update(
            {
                "name": cand["name"],
                "method": cand["method"],
                "n_components": cand.get("n_components", ""),
            }
        )
        rows.append(row)
        print(
            f"{cand['name']:<20} cv={row['cv_macro_f1_mean']:.6f} "
            f"holdout={row['holdout_macro_f1']:.6f}",
            flush=True,
        )

    result = pd.DataFrame(rows)
    baseline = next(r for r in rows if r["name"] == "none")
    ordered = sorted(
        rows,
        key=lambda r: (
            -r["cv_macro_f1_mean"],
            0 if r["method"] == "none" else 1,
            r["name"],
        ),
    )
    best = ordered[0]
    delta = float(best["cv_macro_f1_mean"] - baseline["cv_macro_f1_mean"])
    adopt = delta >= ADOPTION_GAP
    chosen = best if adopt else baseline
    chosen_candidate = next(c for c in candidates if c["name"] == chosen["name"])

    if n_numeric == 0:
        reason = "tidak ada fitur numerik tersisa setelah binning; PCA tidak berlaku"
    elif adopt:
        reason = "naik >= 0.005 pada CV"
    else:
        reason = "kenaikan di bawah 0.005 pada CV"

    decision = {
        "ronde": "dimensionality_reduction",
        "variabel": "method",
        "terpilih": chosen["name"],
        "baseline_f1": round(float(baseline["cv_macro_f1_mean"]), 6),
        "best_f1": round(float(best["cv_macro_f1_mean"]), 6),
        "delta": round(delta, 6),
        "keputusan": "ADOPSI" if adopt else "TOLAK",
        "alasan": reason,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.sort_values(
        ["cv_macro_f1_mean", "name"], ascending=[False, True]
    ).to_csv(RESULT_CSV, index=False)
    pd.DataFrame([decision]).to_csv(DECISION_CSV, index=False)

    chosen_cfg = {"method": chosen_candidate["method"]}
    if chosen_candidate["method"] == "pca_numeric":
        chosen_cfg["n_components"] = chosen_candidate["n_components"]

    config = {
        "stage": "dimensionality_reduction",
        "seed": SEED,
        "cv_folds": N_FOLDS,
        "selection_metric": "cv_macro_f1_mean",
        "reference_params": params,
        "n_numeric_features": n_numeric,
        "baseline_name": "none",
        "baseline_cv_macro_f1": round(float(baseline["cv_macro_f1_mean"]), 6),
        "baseline_holdout_macro_f1": round(float(baseline["holdout_macro_f1"]), 6),
        "chosen_name": chosen["name"],
        "chosen": chosen_cfg,
        "cv_macro_f1_mean": round(float(chosen["cv_macro_f1_mean"]), 6),
        "cv_macro_f1_std": round(float(chosen["cv_macro_f1_std"]), 6),
        "holdout_macro_f1": round(float(chosen["holdout_macro_f1"]), 6),
        "decision": decision,
    }
    write_json(CONFIG_JSON, config)

    print("\n=== KEPUTUSAN DIMENSIONALITY REDUCTION ===", flush=True)
    print(pd.DataFrame([decision]).to_string(index=False), flush=True)
    print(f"Hasil: {RESULT_CSV}", flush=True)
    print(f"Keputusan: {DECISION_CSV}", flush=True)
    print(f"Konfigurasi: {CONFIG_JSON}", flush=True)
    return result


if __name__ == "__main__":
    main()
