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
    REFERENCE_PARAMS,
    SEED,
    evaluate_candidate,
    load_data,
    make_split,
    write_json,
)
from utils.preprocessing import DEFAULT_BIN_COLS, DEFAULT_CATEGORICAL_NUMERIC, DEFAULT_LOG1P_COLS

RESULT_CSV = OUT_DIR / "naive_bayes_preprocessing_results.csv"
DECISION_CSV = OUT_DIR / "naive_bayes_preprocessing_decisions.csv"
CONFIG_JSON = OUT_DIR / "naive_bayes_preprocessing_config.json"


def make_candidate(name, complexity, **overrides):
    options = {
        "standardize": False,
        "add_missing_indicators": True,
        "categorical_numeric_cols": list(DEFAULT_CATEGORICAL_NUMERIC),
        "log1p_cols": list(DEFAULT_LOG1P_COLS),
        "bin_cols": [],
        "n_bins": 20,
        "bin_strategy": "quantile",
    }
    options.update(overrides)
    return {
        "name": name,
        "complexity": complexity,
        "feature_set": "baseline",
        "options": options,
    }


def candidates():
    return [
        make_candidate("baseline_no_binning", 0),
        make_candidate("standardize_numeric", 1, standardize=True),
        make_candidate("no_missing_indicators", 1, add_missing_indicators=False),
        make_candidate("no_log1p", 1, log1p_cols=[]),
        make_candidate("bin_days_q20", 1, bin_cols=["days_before_arrival"]),
        make_candidate("bin_rate_q20", 1, bin_cols=["nightly_rate_idr"]),
        make_candidate("bin_prior_q20", 1, bin_cols=["prior_completed_bookings"]),
        make_candidate("bin_arrival_day_q20", 1, bin_cols=["arrival_day"]),
        make_candidate("bin_all_q10", 2, bin_cols=list(DEFAULT_BIN_COLS), n_bins=10),
        make_candidate("bin_all_q15", 2, bin_cols=list(DEFAULT_BIN_COLS), n_bins=15),
        make_candidate("bin_all_q20", 2, bin_cols=list(DEFAULT_BIN_COLS), n_bins=20),
        make_candidate("bin_all_q25", 2, bin_cols=list(DEFAULT_BIN_COLS), n_bins=25),
        make_candidate("bin_all_q30", 2, bin_cols=list(DEFAULT_BIN_COLS), n_bins=30),
        make_candidate(
            "bin_all_uniform_q20",
            2,
            bin_cols=list(DEFAULT_BIN_COLS),
            n_bins=20,
            bin_strategy="uniform",
        ),
    ]


def main():
    assert Path.cwd() == REPO_ROOT, "Jalankan dari akar repo"
    _, _, y, X_raw = load_data()
    train_idx, valid_idx = make_split(y)

    rows = []
    for cand in candidates():
        options = cand["options"]
        row, _, _ = evaluate_candidate(
            X_raw,
            y,
            train_idx,
            valid_idx,
            cand,
            None,
            {"method": "none"},
            REFERENCE_PARAMS,
            n_splits=N_FOLDS,
        )
        row.update(
            {
                "name": cand["name"],
                "complexity": cand["complexity"],
                "standardize": int(bool(options["standardize"])),
                "add_missing_indicators": int(bool(options["add_missing_indicators"])),
                "log1p_cols": ",".join(options["log1p_cols"]),
                "bin_cols": ",".join(options["bin_cols"]),
                "n_bins": options["n_bins"],
                "bin_strategy": options["bin_strategy"],
            }
        )
        rows.append(row)
        print(
            f"{cand['name']:<24} cv={row['cv_macro_f1_mean']:.6f} "
            f"(+/-{row['cv_macro_f1_std']:.6f}) holdout={row['holdout_macro_f1']:.6f}",
            flush=True,
        )

    result = pd.DataFrame(rows)
    baseline = next(r for r in rows if r["name"] == "baseline_no_binning")
    ordered = sorted(
        rows,
        key=lambda r: (-r["cv_macro_f1_mean"], r["complexity"], r["name"]),
    )
    best = ordered[0]
    delta = float(best["cv_macro_f1_mean"] - baseline["cv_macro_f1_mean"])
    adopt = delta >= ADOPTION_GAP
    chosen = best if adopt else baseline
    chosen_candidate = next(c for c in candidates() if c["name"] == chosen["name"])

    decision = {
        "ronde": "preprocessing",
        "variabel": "preprocessing_options",
        "terpilih": chosen["name"],
        "baseline_f1": round(float(baseline["cv_macro_f1_mean"]), 6),
        "best_f1": round(float(best["cv_macro_f1_mean"]), 6),
        "delta": round(delta, 6),
        "keputusan": "ADOPSI" if adopt else "TOLAK",
        "alasan": "naik >= 0.005 pada CV" if adopt else "kenaikan di bawah 0.005 pada CV",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.sort_values(
        ["cv_macro_f1_mean", "complexity", "name"], ascending=[False, True, True]
    ).to_csv(RESULT_CSV, index=False)
    pd.DataFrame([decision]).to_csv(DECISION_CSV, index=False)

    config = {
        "stage": "preprocessing",
        "seed": SEED,
        "cv_folds": N_FOLDS,
        "selection_metric": "cv_macro_f1_mean",
        "reference_params": REFERENCE_PARAMS,
        "baseline_name": baseline["name"],
        "baseline_cv_macro_f1": round(float(baseline["cv_macro_f1_mean"]), 6),
        "baseline_cv_macro_f1_std": round(float(baseline["cv_macro_f1_std"]), 6),
        "baseline_holdout_macro_f1": round(float(baseline["holdout_macro_f1"]), 6),
        "chosen_name": chosen["name"],
        "chosen": {
            "name": chosen_candidate["name"],
            "feature_set": chosen_candidate["feature_set"],
            "options": chosen_candidate["options"],
        },
        "cv_macro_f1_mean": round(float(chosen["cv_macro_f1_mean"]), 6),
        "cv_macro_f1_std": round(float(chosen["cv_macro_f1_std"]), 6),
        "holdout_macro_f1": round(float(chosen["holdout_macro_f1"]), 6),
        "decision": decision,
    }
    write_json(CONFIG_JSON, config)

    print("\n=== KEPUTUSAN PREPROCESSING ===", flush=True)
    print(pd.DataFrame([decision]).to_string(index=False), flush=True)
    print(f"Hasil: {RESULT_CSV}", flush=True)
    print(f"Keputusan: {DECISION_CSV}", flush=True)
    print(f"Konfigurasi: {CONFIG_JSON}", flush=True)
    return result


if __name__ == "__main__":
    main()
