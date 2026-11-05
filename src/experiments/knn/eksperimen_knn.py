from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from knn import KNNClassifier
from preprocessing import KNNPreprocessor

SEED = 42
<<<<<<< HEAD
K_VALUES = [13, 15, 17, 19, 21, 23, 25, 27, 35, 41]
METRICS = ["euclidean", "manhattan"]
CAT_WEIGHTS = [0.5, 1.0, np.sqrt(10), 2.0]
THRESHOLDS = np.arange(0.30, 0.71, 0.01)
ADOPTION_GAP = 0.005
=======
K_VALUES = [3, 5, 7, 11, 15, 21, 31]
WEIGHTS = ["uniform", "distance"]
DISTANCES = [
    ("euclidean", "euclidean", 2),
    ("manhattan", "manhattan", 1),
    ("chebyshev", "chebyshev", 2),
    ("minkowski_p3", "minkowski", 3),
    ("cosine", "cosine", 2),
]
>>>>>>> bce09b49df462997b8d7128c75ff64d89d6821c1
BATCH = 2000

DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "output"
<<<<<<< HEAD
RESULT_CSV = OUT_DIR / "knn_experiment_results.csv"
DECISIONS_CSV = OUT_DIR / "knn_decisions.csv"

=======
RESULT_CSV = OUT_DIR / "knn_distance_results.csv"
>>>>>>> bce09b49df462997b8d7128c75ff64d89d6821c1

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


def main():
    assert Path.cwd() == REPO_ROOT, "Jalankan dari akar repo"
    train = pd.read_csv(DATA_DIR / "train.csv", dtype={"id": str})
    y = train["is_canceled"].to_numpy()
    X_raw = train.drop(columns=["id", "is_canceled"])
    train_idx, valid_idx = make_split(y)
<<<<<<< HEAD
    y_valid = y[valid_idx]
    rows, decisions = [], []

    params = {"k": 21, "metric": "euclidean", "cat_weight": 1.0, "threshold": 0.5}
    baseline = make_row("baseline", **params, scores=fit_scores(X_raw, train_idx, valid_idx, y, **{k: params[k] for k in ("k", "metric", "cat_weight")}), y_valid=y_valid)
    baseline_f1 = baseline["macro_f1"]
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
    delta = best["macro_f1"] - baseline_f1
    adopt = delta >= ADOPTION_GAP
    decisions.append({"ronde": "k-grid", "variabel": "k", "terpilih": best["k"], "baseline_f1": baseline_f1, "best_f1": best["macro_f1"], "delta": round(delta, 6), "keputusan": "ADOPSI" if adopt else "TOLAK", "alasan": "naik >= 0.005" if adopt else "kenaikan di bawah 0.005"})
    if adopt: params["k"] = best["k"]
    print(f"  Keputusan: {'ADOPSI' if adopt else 'TOLAK'} (delta={delta:+.6f})", flush=True)
    baseline_f1 = best["macro_f1"] if adopt else baseline_f1

    # Ronde 2: threshold sweep dari decision_scores (cukup sekali hitung skor).
    print("\n[Ronde 2/4] Threshold 0.30-0.70", flush=True)
    scores = fit_scores(X_raw, train_idx, valid_idx, y, params["k"], params["metric"], params["cat_weight"])
    t_rows = [make_row("threshold", params["k"], params["metric"], params["cat_weight"], float(t), scores, y_valid) for t in THRESHOLDS]
    rows.extend(t_rows)
    best = max(t_rows, key=lambda r: r["macro_f1"])
    delta = best["macro_f1"] - baseline_f1
    adopt = delta >= ADOPTION_GAP
    decisions.append({"ronde": "threshold", "variabel": "threshold", "terpilih": best["threshold"], "baseline_f1": baseline_f1, "best_f1": best["macro_f1"], "delta": round(delta, 6), "keputusan": "ADOPSI" if adopt else "TOLAK", "alasan": "naik >= 0.005" if adopt else "kenaikan di bawah 0.005"})
    if adopt: params["threshold"] = best["threshold"]
    print(f"  terbaik={best['threshold']:.2f}, f1={best['macro_f1']:.6f}; Keputusan: {'ADOPSI' if adopt else 'TOLAK'} (delta={delta:+.6f})", flush=True)
    baseline_f1 = best["macro_f1"] if adopt else baseline_f1

    # Ronde 3: metric.
    print("\n[Ronde 3/4] Metric", flush=True)
    metric_rows = []
    for metric in METRICS:
        scores = fit_scores(X_raw, train_idx, valid_idx, y, params["k"], metric, params["cat_weight"])
        row = make_row("metric", params["k"], metric, params["cat_weight"], params["threshold"], scores, y_valid)
        rows.append(row); metric_rows.append(row)
        print(f"  metric={metric}: macro_f1={row['macro_f1']:.6f}", flush=True)
    best = max(metric_rows, key=lambda r: r["macro_f1"])
    delta = best["macro_f1"] - baseline_f1
    adopt = delta >= ADOPTION_GAP
    decisions.append({"ronde": "metric", "variabel": "metric", "terpilih": best["metric"], "baseline_f1": baseline_f1, "best_f1": best["macro_f1"], "delta": round(delta, 6), "keputusan": "ADOPSI" if adopt else "TOLAK", "alasan": "naik >= 0.005" if adopt else "kenaikan di bawah 0.005"})
    if adopt: params["metric"] = best["metric"]
    print(f"  Keputusan: {'ADOPSI' if adopt else 'TOLAK'} (delta={delta:+.6f})", flush=True)
    baseline_f1 = best["macro_f1"] if adopt else baseline_f1

    # Ronde 4: cat_weight.
    print("\n[Ronde 4/4] cat_weight", flush=True)
    cat_rows = []
    for cat_weight in CAT_WEIGHTS:
        scores = fit_scores(X_raw, train_idx, valid_idx, y, params["k"], params["metric"], cat_weight)
        row = make_row("cat_weight", params["k"], params["metric"], cat_weight, params["threshold"], scores, y_valid)
        rows.append(row); cat_rows.append(row)
        print(f"  cat_weight={cat_weight:.4g}: macro_f1={row['macro_f1']:.6f}", flush=True)
    best = max(cat_rows, key=lambda r: r["macro_f1"])
    delta = best["macro_f1"] - baseline_f1
    adopt = delta >= ADOPTION_GAP
    decisions.append({"ronde": "cat_weight", "variabel": "cat_weight", "terpilih": best["cat_weight"], "baseline_f1": baseline_f1, "best_f1": best["macro_f1"], "delta": round(delta, 6), "keputusan": "ADOPSI" if adopt else "TOLAK", "alasan": "naik >= 0.005" if adopt else "kenaikan di bawah 0.005"})
    if adopt: params["cat_weight"] = best["cat_weight"]
    print(f"  Keputusan: {'ADOPSI' if adopt else 'TOLAK'} (delta={delta:+.6f})", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).sort_values(["macro_f1", "config"], ascending=[False, True]).to_csv(RESULT_CSV, index=False)
    pd.DataFrame(decisions).to_csv(DECISIONS_CSV, index=False)
    print("\n=== PARAMETER AKHIR ===", flush=True)
    print(params, flush=True)
    print(f"Hasil: {RESULT_CSV}", flush=True)
    print(f"Keputusan: {DECISIONS_CSV}", flush=True)
    return pd.DataFrame(rows)

=======
    y_tr, y_va = y[train_idx], y[valid_idx]
    pre = KNNPreprocessor().fit(X_raw.iloc[train_idx])
    X_tr = pre.transform(X_raw.iloc[train_idx])
    X_va = pre.transform(X_raw.iloc[valid_idx])
    rows = []
    for distance_name, metric, p_value in DISTANCES:
        knn = KNNClassifier(n_neighbors=max(K_VALUES), metric=metric, p=p_value).fit(X_tr, y_tr)
        distances, indices = knn.kneighbors(X_va, n_neighbors=max(K_VALUES), batch_size=BATCH)
        for k in K_VALUES:
            for w in WEIGHTS:
                name = f"knn_{distance_name}_k{k}_{w}"
                pred = knn.predict_from_neighbors(distances, indices, n_neighbors=k, weights=w)
                sk = SkKNN(n_neighbors=k, weights=w, metric=metric, p=p_value).fit(X_tr, y_tr)
                pred_sk = sk.predict(X_va)
                f1 = f1_score(y_va, pred, labels=[0, 1], average="macro", zero_division=0)
                f1_sk = f1_score(y_va, pred_sk, labels=[0, 1], average="macro", zero_division=0)
                cm = confusion_matrix(y_va, pred, labels=[0, 1])
                p, r, f = per_class_metrics(y_va, pred)
                rows.append({
                    "config": name, "metric": metric, "p": p_value if metric == "minkowski" else None,
                    "k": k, "weights": w,
                    "macro_f1": round(f1, 6), "macro_f1_sklearn": round(f1_sk, 6),
                    "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
                    "precision_0": round(p[0], 4), "recall_0": round(r[0], 4), "f1_0": round(f[0], 4),
                    "precision_1": round(p[1], 4), "recall_1": round(r[1], 4), "f1_1": round(f[1], 4),
                })
                print(f"{name:<38} macro_f1={f1:.4f}  (sklearn {f1_sk:.4f})")
    df = pd.DataFrame(rows).sort_values(["metric", "macro_f1", "config"], ascending=[True, False, True]).reset_index(drop=True)
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULT_CSV, index=False)
    best_by_distance = df.groupby("metric", sort=True, as_index=False).first()
    print("\n=== TERBAIK PER METRIK JARAK ===")
    print(best_by_distance[["metric", "k", "weights", "macro_f1", "macro_f1_sklearn", "config"]].to_string(index=False))
    print(f"\nHasil lengkap: {RESULT_CSV}")
    return df
>>>>>>> bce09b49df462997b8d7128c75ff64d89d6821c1

if __name__ == "__main__":
    main()
