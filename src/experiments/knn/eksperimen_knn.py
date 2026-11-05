from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as SkKNN
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from knn import KNNClassifier
from preprocessing import KNNPreprocessor

SEED = 42
K_VALUES = [3, 5, 7, 11, 15, 21, 31]
WEIGHTS = ["uniform", "distance"]
BATCH = 2000

DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "output"
RESULT_CSV = OUT_DIR / "knn_experiment_results.csv"

def make_split(y):
    rng = np.random.default_rng(SEED)
    valid_idx = np.sort(np.concatenate([
        rng.permutation(np.flatnonzero(y == label))[:int(np.ceil(0.2 * np.sum(y == label)))]
        for label in (0, 1)
    ]))
    train_idx = np.setdiff1d(np.arange(len(y)), valid_idx)
    return train_idx, valid_idx

def per_class_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return p, r, f

def main():
    assert Path.cwd() == REPO_ROOT, "Jalankan dari akar repo: python src/experiments/knn/eksperimen_knn.py"
    train = pd.read_csv(DATA_DIR / "train.csv", dtype={"id": str})
    y = train["is_canceled"].to_numpy()
    assert set(np.unique(y)) == {0, 1}
    X_raw = train.drop(columns=["id", "is_canceled"])
    train_idx, valid_idx = make_split(y)
    y_tr, y_va = y[train_idx], y[valid_idx]
    pre = KNNPreprocessor().fit(X_raw.iloc[train_idx])
    X_tr = pre.transform(X_raw.iloc[train_idx])
    X_va = pre.transform(X_raw.iloc[valid_idx])
    rows = []
    for k in K_VALUES:
        for w in WEIGHTS:
            name = f"knn_k{k}_{w}"
            knn = KNNClassifier(n_neighbors=k, weights=w).fit(X_tr, y_tr)
            pred = knn.predict(X_va, batch_size=BATCH)
            sk = SkKNN(n_neighbors=k, weights=w).fit(X_tr, y_tr)
            pred_sk = sk.predict(X_va)
            f1 = f1_score(y_va, pred, labels=[0, 1], average="macro", zero_division=0)
            f1_sk = f1_score(y_va, pred_sk, labels=[0, 1], average="macro", zero_division=0)
            cm = confusion_matrix(y_va, pred, labels=[0, 1])
            p, r, f = per_class_metrics(y_va, pred)
            rows.append({
                "config": name, "k": k, "weights": w,
                "macro_f1": round(f1, 6), "macro_f1_sklearn": round(f1_sk, 6),
                "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1]),
                "precision_0": round(p[0], 4), "recall_0": round(r[0], 4), "f1_0": round(f[0], 4),
                "precision_1": round(p[1], 4), "recall_1": round(r[1], 4), "f1_1": round(f[1], 4),
            })
            print(f"{name:<22} macro_f1={f1:.4f}  (sklearn {f1_sk:.4f})")
    df = pd.DataFrame(rows).sort_values(["macro_f1", "config"], ascending=[False, True]).reset_index(drop=True)
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULT_CSV, index=False)
    best = df.iloc[0]
    print("\n=== TERBAIK ===")
    print(best.to_string())
    print(f"\nHasil lengkap: {RESULT_CSV}")
    return df

if __name__ == "__main__":
    main()