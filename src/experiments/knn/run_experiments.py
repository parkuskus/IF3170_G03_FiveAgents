from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parents[3]
from src.experiments.knn.knn import KNNClassifier
from src.experiments.knn.preprocessing import KNNPreprocessor

SEED = 42
K_VALUES = [13, 15, 17, 19, 21, 23, 25, 27, 35, 41]
METRICS = ["euclidean", "manhattan"]
CAT_WEIGHTS = [0.5, 1.0, np.sqrt(10), 2.0]
THRESHOLDS = np.arange(0.30, 0.71, 0.01)
ADOPTION_GAP = 0.005
BATCH = 2000

DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "output" / "knn"
RESULT_CSV = OUT_DIR / "knn_experiment_results.csv"
DECISIONS_CSV = OUT_DIR / "knn_decisions.csv"


def make_split(y):
    rng = np.random.default_rng(SEED)
    valid_idx = np.sort(np.concatenate([
        rng.permutation(np.flatnonzero(y == label))[:int(np.ceil(0.2 * np.sum(y == label)))]
        for label in (0, 1)
    ]))
    return np.setdiff1d(np.arange(len(y)), valid_idx), valid_idx


def metrics(y_true, pred):
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    p, r, f, _ = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1], zero_division=0
    )
    return {
        "macro_f1": float(f1_score(y_true, pred, labels=[0, 1], average="macro", zero_division=0)),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "precision_0": round(float(p[0]), 4), "recall_0": round(float(r[0]), 4), "f1_0": round(float(f[0]), 4),
        "precision_1": round(float(p[1]), 4), "recall_1": round(float(r[1]), 4), "f1_1": round(float(f[1]), 4),
    }


def make_row(round_name, k, metric, cat_weight, threshold, scores, y_valid, weights="distance"):
    pred = (scores > threshold).astype(np.int64)
    row = {
        "config": f"k{k}_{weights}_{metric}_cat{cat_weight:.4g}_t{threshold:.2f}",
        "ronde": round_name, "k": k, "weights": weights, "metric": metric,
        "cat_weight": cat_weight, "threshold": threshold,
    }
    row.update(metrics(y_valid, pred))
    return row


def fit_scores(X_raw, train_idx, valid_idx, y, k, metric, cat_weight, weights="distance"):
    pre = KNNPreprocessor(cat_weight=cat_weight).fit(X_raw.iloc[train_idx])
    X_train = pre.transform(X_raw.iloc[train_idx])
    X_valid = pre.transform(X_raw.iloc[valid_idx])
    model = KNNClassifier(n_neighbors=k, weights=weights, metric=metric).fit(X_train, y[train_idx])
    return model.decision_scores(X_valid, batch_size=BATCH)


def decide(decisions, round_name, variable, best, baseline_f1):
    delta = best["macro_f1"] - baseline_f1
    adopt = delta >= ADOPTION_GAP
    decisions.append({
        "ronde": round_name, "variabel": variable, "terpilih": best[variable],
        "baseline_f1": baseline_f1, "best_f1": best["macro_f1"],
        "delta": round(delta, 6),
        "keputusan": "ADOPSI" if adopt else "TOLAK",
        "alasan": "naik >= 0.005" if adopt else "kenaikan di bawah 0.005",
    })
    print(f"  Keputusan: {'ADOPSI' if adopt else 'TOLAK'} (delta={delta:+.6f})", flush=True)
    return best["macro_f1"] if adopt else baseline_f1, adopt


def main():
    assert Path.cwd() == REPO_ROOT, "Jalankan dari akar repo"
    train = pd.read_csv(DATA_DIR / "train.csv", dtype={"id": str})
    y = train["is_canceled"].to_numpy()
    X_raw = train.drop(columns=["id", "is_canceled"])
    train_idx, valid_idx = make_split(y)
    y_valid = y[valid_idx]
    rows, decisions = [], []

    params = {"k": 21, "metric": "euclidean", "cat_weight": 1.0, "threshold": 0.5}
    baseline = make_row(
        "baseline", **params,
        scores=fit_scores(
            X_raw, train_idx, valid_idx, y,
            k=params["k"], metric=params["metric"], cat_weight=params["cat_weight"],
        ),
        y_valid=y_valid,
    )
    baseline_f1 = baseline["macro_f1"]
    rows.append(baseline)
    print(f"Baseline: {baseline_f1:.6f}", flush=True)

    # Ronde 1: grid k halus.
    print("\n[Ronde 1/4] Grid k", flush=True)
    k_rows = []
    for k in K_VALUES:
        scores = fit_scores(X_raw, train_idx, valid_idx, y, k, params["metric"], params["cat_weight"])
        row = make_row("k-grid", k, params["metric"], params["cat_weight"], params["threshold"], scores, y_valid)
        rows.append(row); k_rows.append(row)
        print(f"  k={k}: macro_f1={row['macro_f1']:.6f}", flush=True)
    best = max(k_rows, key=lambda r: r["macro_f1"])
    baseline_f1, adopt = decide(decisions, "k-grid", "k", best, baseline_f1)
    if adopt: params["k"] = best["k"]

    # Ronde 2: threshold sweep dari decision_scores (cukup sekali hitung skor).
    print("\n[Ronde 2/4] Threshold 0.30-0.70", flush=True)
    scores = fit_scores(X_raw, train_idx, valid_idx, y, params["k"], params["metric"], params["cat_weight"])
    t_rows = [make_row("threshold", params["k"], params["metric"], params["cat_weight"], float(t), scores, y_valid) for t in THRESHOLDS]
    rows.extend(t_rows)
    best = max(t_rows, key=lambda r: r["macro_f1"])
    baseline_f1, adopt = decide(decisions, "threshold", "threshold", best, baseline_f1)
    if adopt: params["threshold"] = best["threshold"]

    # Ronde 3: metric.
    print("\n[Ronde 3/4] Metric", flush=True)
    metric_rows = []
    for metric in METRICS:
        scores = fit_scores(X_raw, train_idx, valid_idx, y, params["k"], metric, params["cat_weight"])
        row = make_row("metric", params["k"], metric, params["cat_weight"], params["threshold"], scores, y_valid)
        rows.append(row); metric_rows.append(row)
        print(f"  metric={metric}: macro_f1={row['macro_f1']:.6f}", flush=True)
    best = max(metric_rows, key=lambda r: r["macro_f1"])
    baseline_f1, adopt = decide(decisions, "metric", "metric", best, baseline_f1)
    if adopt: params["metric"] = best["metric"]

    # Ronde 4: cat_weight.
    print("\n[Ronde 4/4] cat_weight", flush=True)
    cat_rows = []
    for cat_weight in CAT_WEIGHTS:
        scores = fit_scores(X_raw, train_idx, valid_idx, y, params["k"], params["metric"], cat_weight)
        row = make_row("cat_weight", params["k"], params["metric"], cat_weight, params["threshold"], scores, y_valid)
        rows.append(row); cat_rows.append(row)
        print(f"  cat_weight={cat_weight:.4g}: macro_f1={row['macro_f1']:.6f}", flush=True)
    best = max(cat_rows, key=lambda r: r["macro_f1"])
    baseline_f1, adopt = decide(decisions, "cat_weight", "cat_weight", best, baseline_f1)
    if adopt: params["cat_weight"] = best["cat_weight"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["macro_f1", "config"], ascending=[False, True]).to_csv(RESULT_CSV, index=False)
    pd.DataFrame(decisions).to_csv(DECISIONS_CSV, index=False)
    print("\n=== PARAMETER AKHIR ===", flush=True)
    print(params, flush=True)
    print(f"Hasil: {RESULT_CSV}", flush=True)
    print(f"Keputusan: {DECISIONS_CSV}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
