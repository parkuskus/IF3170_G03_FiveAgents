"""Fase 4: generate submission_knn.csv dari parameter final Fase 1.

Parameter final (dari output/knn/knn_decisions.csv):
    k = 21, weights = distance, metric = manhattan,
    cat_weight = 1.0, threshold = 0.50

Alur: fit preprocessing + KNN pada SELURUH train.csv, prediksi test.csv,
tulis output/knn/submission_knn.csv dengan format id,is_canceled.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
from src.experiments.knn.knn import KNNClassifier
from src.experiments.knn.preprocessing import KNNPreprocessor

FINAL = {"k": 21, "weights": "distance", "metric": "manhattan",
         "cat_weight": 1.0, "threshold": 0.50}
BATCH = 2000

DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "output" / "knn"
SUBMISSION_CSV = OUT_DIR / "submission_knn.csv"


def main():
    train = pd.read_csv(DATA_DIR / "train.csv", dtype={"id": str})
    test = pd.read_csv(DATA_DIR / "test.csv", dtype={"id": str})
    y = train["is_canceled"].to_numpy()
    X_train_raw = train.drop(columns=["id", "is_canceled"])
    X_test_raw = test.drop(columns=["id"])

    print("fit preprocessor + KNN pada seluruh train...", flush=True)
    pre = KNNPreprocessor(cat_weight=FINAL["cat_weight"]).fit(X_train_raw)
    X_train = pre.transform(X_train_raw)
    X_test = pre.transform(X_test_raw)

    model = KNNClassifier(
        n_neighbors=FINAL["k"], weights=FINAL["weights"], metric=FINAL["metric"]
    ).fit(X_train, y)
    print("prediksi test.csv...", flush=True)
    scores = model.decision_scores(X_test, batch_size=BATCH)
    pred = (scores > FINAL["threshold"]).astype(int)

    sub = pd.DataFrame({"id": test["id"], "is_canceled": pred})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION_CSV, index=False)

    # Validasi format submission (kontrak repo).
    assert list(sub.columns) == ["id", "is_canceled"]
    assert len(sub) == len(test)
    assert sub["id"].tolist() == test["id"].tolist()
    assert set(sub["is_canceled"].unique()).issubset({0, 1})
    print(f"OK: {len(sub)} baris, kolom {list(sub.columns)}, "
          f"distribusi prediksi: {sub['is_canceled'].value_counts().to_dict()}", flush=True)
    print(f"Submission: {SUBMISSION_CSV}", flush=True)


if __name__ == "__main__":
    main()
