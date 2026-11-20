from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.cleaning import clean_training_data
from utils.pipeline import (
    OUT_DIR,
    StagePipeline,
    load_data,
    model_from_params,
    read_json,
)

CONFIG_JSON = OUT_DIR / "naive_bayes_final_config.json"
MODEL_CONFIG_JSON = REPO_ROOT / "output" / "final" / "model" / "naive_bayes" / "konfigurasi.json"
SUBMISSION_CSV = OUT_DIR / "submission_naive_bayes.csv"


def main():
    assert Path.cwd() == REPO_ROOT, "Jalankan dari akar repo"
    cfg_path = CONFIG_JSON if CONFIG_JSON.exists() else MODEL_CONFIG_JSON
    if not cfg_path.exists():
        raise SystemExit("Jalankan run_naive_bayes.py atau notebook final terlebih dahulu")

    config = read_json(cfg_path)
    params = config["params"]
    prep_cfg = config["preprocessing"]
    selected = config["feature_selection"].get("selected_features")
    dr_cfg = config["dimensionality_reduction"]

    train, test, y, X_raw = load_data()
    X_test_raw = test.drop(columns=["id"])

    print("cleaning + fit pipeline + model pada seluruh train...", flush=True)
    X_clean, y_clean, n_dropped = clean_training_data(X_raw, y)
    print(f"  duplikat latih dibuang: {n_dropped}", flush=True)
    pipe = StagePipeline(prep_cfg, selected, dr_cfg).fit(X_clean, y_clean)
    model = model_from_params(pipe, params).fit(pipe.transform(X_clean), y_clean)

    print("prediksi test.csv...", flush=True)
    pred = model.predict(pipe.transform(X_test_raw))
    sub = pd.DataFrame({"id": test["id"], "is_canceled": pred.astype(int)})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION_CSV, index=False)

    assert list(sub.columns) == ["id", "is_canceled"]
    assert len(sub) == len(test)
    assert sub["id"].tolist() == test["id"].tolist()
    assert set(sub["is_canceled"].unique()).issubset({0, 1})
    print(
        f"OK: {len(sub)} baris, kolom {list(sub.columns)}, "
        f"distribusi prediksi: {sub['is_canceled'].value_counts().to_dict()}",
        flush=True,
    )
    print(f"Submission: {SUBMISSION_CSV}", flush=True)


if __name__ == "__main__":
    main()
