from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[3]

from src.experiments.knn.feature_engineering import FEATURE_SETS, add_engineered_features
from src.experiments.knn.knn import KNNClassifier
from src.experiments.knn.preprocessing import KNNPreprocessor

SEED = 42
BATCH = 2000
ADOPTION_GAP = 0.005
FINAL_PARAMS = {
    "k": 21,
    "weights": "distance",
    "metric": "manhattan",
    "cat_weight": 1.0,
    "threshold": 0.5,
}
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "output" / "knn"
RESULT_CSV = OUT_DIR / "knn_feature_engineering_results.csv"
DECISIONS_CSV = OUT_DIR / "knn_feature_engineering_decisions.csv"


def make_split(y):
    rng = np.random.default_rng(SEED)
    valid_idx = np.sort(np.concatenate([
        rng.permutation(np.flatnonzero(y == label))[:int(np.ceil(0.2 * np.sum(y == label)))]
        for label in (0, 1)
    ]))
    return np.setdiff1d(np.arange(len(y)), valid_idx), valid_idx


def evaluate(X_raw, y, train_idx, valid_idx, feature_set):
    X = add_engineered_features(X_raw, feature_set)
    pre = KNNPreprocessor(cat_weight=FINAL_PARAMS["cat_weight"]).fit(X.iloc[train_idx])
    X_train = pre.transform(X.iloc[train_idx])
    X_valid = pre.transform(X.iloc[valid_idx])
    model = KNNClassifier(
        n_neighbors=FINAL_PARAMS["k"],
        weights=FINAL_PARAMS["weights"],
        metric=FINAL_PARAMS["metric"],
        threshold=FINAL_PARAMS["threshold"],
    ).fit(X_train, y[train_idx])
    scores = model.decision_scores(X_valid, batch_size=BATCH)
    pred = (scores > FINAL_PARAMS["threshold"]).astype(np.int64)
    return float(f1_score(y[valid_idx], pred, labels=[0, 1], average="macro", zero_division=0))


def main():
    assert Path.cwd() == REPO_ROOT, "Jalankan dari akar repo"
    train = pd.read_csv(DATA_DIR / "train.csv", dtype={"id": str})
    y = train["is_canceled"].to_numpy()
    X_raw = train.drop(columns=["id", "is_canceled"])
    train_idx, valid_idx = make_split(y)

    rows = []
    baseline_f1 = None
    for feature_set in FEATURE_SETS:
        f1 = evaluate(X_raw, y, train_idx, valid_idx, feature_set)
        if baseline_f1 is None:
            baseline_f1 = f1
        delta = f1 - baseline_f1
        rows.append({
            "feature_set": feature_set,
            "description": {
                "baseline": "fitur baseline",
                "total_guests": "num_adults + num_children",
                "total_nights": "weekend_nights + weekday_nights",
                "total_stay_value": "total_nights * nightly_rate_idr",
                "requests_per_guest": "special_requests / total_guests",
            }[feature_set],
            "macro_f1": round(f1, 6),
            "delta_vs_baseline": round(delta, 6),
        })
        print(f"{feature_set:<20} macro_f1={f1:.6f} delta={delta:+.6f}", flush=True)

    result = pd.DataFrame(rows)
    best = result.iloc[result["macro_f1"].idxmax()]
    decisions = []
    for row in rows[1:]:
        delta = float(row["macro_f1"] - baseline_f1)
        adopt = delta >= ADOPTION_GAP
        decisions.append({
            "feature_set": row["feature_set"],
            "baseline_f1": round(baseline_f1, 6),
            "feature_f1": row["macro_f1"],
            "delta": round(delta, 6),
            "keputusan": "ADOPSI" if adopt else "TOLAK",
            "alasan": "naik >= 0.005" if adopt else "kenaikan di bawah 0.005",
        })
    result["keputusan"] = ["BASELINE"] + [
        next(d["keputusan"] for d in decisions if d["feature_set"] == r["feature_set"])
        for r in rows[1:]
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.sort_values(["macro_f1", "feature_set"], ascending=[False, True]).to_csv(RESULT_CSV, index=False)
    pd.DataFrame(decisions).to_csv(DECISIONS_CSV, index=False)
    print("\n=== FASE 3 ===", flush=True)
    print(f"Terbaik: {best['feature_set']} (macro_f1={best['macro_f1']:.6f})", flush=True)
    print(f"Baseline: {baseline_f1:.6f}", flush=True)
    print(f"Hasil: {RESULT_CSV}", flush=True)
    print(f"Keputusan: {DECISIONS_CSV}", flush=True)
    return result


if __name__ == "__main__":
    main()
