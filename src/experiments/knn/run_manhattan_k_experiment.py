from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

REPO_ROOT = Path(__file__).resolve().parents[3]
from src.experiments.knn.knn import KNNClassifier
from src.experiments.knn.preprocessing import KNNPreprocessor

SEED = 42
K_VALUES = [13, 15, 17, 19, 21, 23, 25, 27, 35, 41]
BASELINE_K = 21
WEIGHTS = "distance"
METRIC = "manhattan"
CAT_WEIGHT = 1.0
THRESHOLD = 0.5
BATCH = 2000
ADOPTION_GAP = 0.005

DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "output" / "knn"
RESULT_CSV = OUT_DIR / "knn_manhattan_k_results.csv"
DECISION_CSV = OUT_DIR / "knn_manhattan_k_decision.csv"


def make_split(y):
    rng = np.random.default_rng(SEED)
    valid_idx = np.sort(np.concatenate([
        rng.permutation(np.flatnonzero(y == label))[:int(np.ceil(0.2 * np.sum(y == label)))]
        for label in (0, 1)
    ]))
    train_idx = np.setdiff1d(np.arange(len(y)), valid_idx)
    return train_idx, valid_idx


def main():
    assert Path.cwd() == REPO_ROOT, "Jalankan dari akar repo"
    # Bisa dipecah beberapa invokasi: python -m src.experiments.knn.run_manhattan_k_experiment 13 15 17
    # Hasil tiap invokasi di-append ke CSV yang sama; --finalize yang menghitung keputusan.
    args = sys.argv[1:]
    finalize = "--finalize" in args
    ks = [int(a) for a in args if a.isdigit()] or list(K_VALUES)

    data = pd.read_csv(DATA_DIR / "train.csv", dtype={"id": str})
    y = data["is_canceled"].to_numpy()
    X = data.drop(columns=["id", "is_canceled"])
    train_idx, valid_idx = make_split(y)

    pre = KNNPreprocessor(cat_weight=CAT_WEIGHT).fit(X.iloc[train_idx])
    X_train = pre.transform(X.iloc[train_idx])
    X_valid = pre.transform(X.iloc[valid_idx])
    y_train, y_valid = y[train_idx], y[valid_idx]

    rows = []
    for k in ks:
        model = KNNClassifier(n_neighbors=k, weights=WEIGHTS, metric=METRIC).fit(X_train, y_train)
        scores = model.decision_scores(X_valid, batch_size=BATCH)
        pred = (scores > THRESHOLD).astype(np.int64)
        cm = confusion_matrix(y_valid, pred, labels=[0, 1])
        f1 = float(f1_score(y_valid, pred, labels=[0, 1], average="macro", zero_division=0))
        row = {
            "config": f"k{k}_{WEIGHTS}_{METRIC}", "k": k,
            "weights": WEIGHTS, "metric": METRIC,
            "cat_weight": CAT_WEIGHT, "threshold": THRESHOLD,
            "macro_f1": round(f1, 6),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        }
        rows.append(row)
        print(f"k={k:>2}: macro_f1={f1:.6f}", flush=True)

    result = pd.DataFrame(rows).sort_values(["macro_f1", "k"], ascending=[False, True]).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if RESULT_CSV.exists():
        prev = pd.read_csv(RESULT_CSV)
        result = pd.concat([prev, result], ignore_index=True).drop_duplicates(subset="k")
    if not finalize:
        result.to_csv(RESULT_CSV, index=False)
        print(f"disimpan (parsial, {len(result)} k): {RESULT_CSV}", flush=True)
        return

    result = result.sort_values(["macro_f1", "k"], ascending=[False, True]).reset_index(drop=True)
    baseline = result.loc[result["k"] == BASELINE_K].iloc[0]
    best = result.iloc[0]
    delta = float(best["macro_f1"] - baseline["macro_f1"])
    adopted = delta >= ADOPTION_GAP
    chosen_k = int(best["k"] if adopted else BASELINE_K)
    decision = pd.DataFrame([{
        "baseline_config": f"k{BASELINE_K}_{WEIGHTS}_{METRIC}",
        "baseline_f1": baseline["macro_f1"],
        "best_candidate": f"k{int(best['k'])}_{WEIGHTS}_{METRIC}",
        "best_candidate_f1": best["macro_f1"],
        "delta": round(delta, 6),
        "keputusan": "ADOPSI" if adopted else "TOLAK",
        "k_final": chosen_k,
        "alasan": "naik >= 0.005" if adopted else "kenaikan di bawah 0.005; pertahankan k=21",
    }])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULT_CSV, index=False)
    decision.to_csv(DECISION_CSV, index=False)
    print("\n=== KEPUTUSAN GRID K MANHATTAN ===", flush=True)
    print(decision.to_string(index=False), flush=True)
    print(f"\nHasil: {RESULT_CSV}", flush=True)
    print(f"Keputusan: {DECISION_CSV}", flush=True)


if __name__ == "__main__":
    main()
