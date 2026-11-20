from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.pipeline import (
    ADOPTION_GAP,
    N_FOLDS,
    OUT_DIR,
    evaluate_candidate,
    load_data,
    make_split,
    read_json,
    write_json,
)
from utils.feature_engineering import (
    FEATURE_DESCRIPTIONS,
    FEATURE_LIBRARY,
    FEATURE_SETS,
    NB_BEST_FEATURES,
)

TEXT_FEATURES = {
    "dup_channel",
    "dup_returning",
    "dup_parking",
    "ota_x_requests",
    "rate_weekend_req_ota",
}

PREP_CONFIG = OUT_DIR / "naive_bayes_preprocessing_config.json"
RESULT_CSV = OUT_DIR / "naive_bayes_feature_engineering_results.csv"
DECISION_CSV = OUT_DIR / "naive_bayes_feature_engineering_decisions.csv"
CONFIG_JSON = OUT_DIR / "naive_bayes_feature_engineering_config.json"


def main():
    assert Path.cwd() == REPO_ROOT, "Jalankan dari akar repo"
    if not PREP_CONFIG.exists():
        raise SystemExit("Jalankan run_preprocessing.py terlebih dahulu")

    prep_stage = read_json(PREP_CONFIG)
    base_prep = prep_stage["chosen"]
    params = prep_stage.get("reference_params", {})
    _, _, y, X_raw = load_data()
    train_idx, valid_idx = make_split(y)

    rows = []
    baseline_cv = None
    for feature_set in FEATURE_SETS:
        trial_prep = copy.deepcopy(base_prep)
        trial_prep["feature_set"] = feature_set
        options = trial_prep.setdefault("options", {})
        categorical = list(options.get("categorical_numeric_cols", ()))
        names = (
            list(NB_BEST_FEATURES) if feature_set == "nb_best" else [feature_set]
        )
        for name in names:
            if (
                name in FEATURE_LIBRARY
                and name not in TEXT_FEATURES
                and name not in categorical
            ):
                categorical.append(name)
        options["categorical_numeric_cols"] = categorical
        row, _, _ = evaluate_candidate(
            X_raw,
            y,
            train_idx,
            valid_idx,
            trial_prep,
            None,
            {"method": "none"},
            params,
            n_splits=N_FOLDS,
        )
        row.update(
            {
                "feature_set": feature_set,
                "description": FEATURE_DESCRIPTIONS[feature_set],
            }
        )
        if baseline_cv is None:
            baseline_cv = row["cv_macro_f1_mean"]
            row["delta_vs_baseline"] = 0.0
            row["keputusan"] = "BASELINE"
        else:
            delta = row["cv_macro_f1_mean"] - baseline_cv
            row["delta_vs_baseline"] = round(float(delta), 6)
            row["keputusan"] = "ADOPSI" if delta >= ADOPTION_GAP else "TOLAK"
        rows.append(row)
        print(
            f"{feature_set:<22} cv={row['cv_macro_f1_mean']:.6f} "
            f"delta={row['delta_vs_baseline']:+.6f} {row['keputusan']}",
            flush=True,
        )

    result = pd.DataFrame(rows)
    best = max(rows, key=lambda r: r["cv_macro_f1_mean"])
    delta = float(best["cv_macro_f1_mean"] - baseline_cv)
    adopted = delta >= ADOPTION_GAP
    chosen = best["feature_set"] if adopted else "baseline"
    decision = {
        "baseline_feature_set": "baseline",
        "baseline_f1": round(float(baseline_cv), 6),
        "best_candidate": best["feature_set"],
        "best_candidate_f1": round(float(best["cv_macro_f1_mean"]), 6),
        "delta": round(delta, 6),
        "keputusan": "ADOPSI" if adopted else "TOLAK",
        "feature_set_final": chosen,
        "alasan": "naik >= 0.005 pada CV"
        if adopted
        else "kenaikan di bawah 0.005 pada CV; pertahankan baseline",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.sort_values(
        ["cv_macro_f1_mean", "feature_set"], ascending=[False, True]
    ).to_csv(RESULT_CSV, index=False)
    pd.DataFrame([decision]).to_csv(DECISION_CSV, index=False)
    write_json(CONFIG_JSON, decision)

    print("\n=== KEPUTUSAN FEATURE ENGINEERING ===", flush=True)
    print(pd.DataFrame([decision]).to_string(index=False), flush=True)
    print(f"Hasil: {RESULT_CSV}", flush=True)
    print(f"Keputusan: {DECISION_CSV}", flush=True)
    print(f"Konfigurasi: {CONFIG_JSON}", flush=True)
    return result


if __name__ == "__main__":
    main()
