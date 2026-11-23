"""Evaluasi bonus dengan C_FP=2, C_FN=1 pada skor validasi KNN."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.knn.run_experiments import make_split, metrics
from src.experiments.knn.knn import KNNClassifier
from src.experiments.knn.preprocessing import KNNPreprocessor


def evaluate_cost_sensitive(y_valid, scores, output_dir):
    """Pilih threshold dari validasi; test tidak dipakai untuk pemilihan."""
    thresholds = np.unique(np.concatenate(([0.5, 2 / 3], np.linspace(0, 1, 101))))
    rows = []
    for threshold in thresholds:
        row = {"threshold": float(threshold)}
        row.update(metrics(y_valid, (scores > threshold).astype(int)))
        row["cost"] = 2 * row["fp"] + row["fn"]
        row["cost_per_booking"] = row["cost"] / len(y_valid)
        rows.append(row)
    results = pd.DataFrame(rows)
    # Jika biaya seri, pilih macro F1 terbesar lalu threshold terkecil.
    best = results.sort_values(
        ["cost", "macro_f1", "threshold"], ascending=[True, False, True]
    ).iloc[0]
    baseline = results.loc[results.threshold == 0.5].iloc[0]
    theoretical = results.loc[results.threshold == 2 / 3].iloc[0]
    comparison = pd.DataFrame([baseline, theoretical, best]).reset_index(drop=True)
    comparison.insert(0, "strategy", ["baseline", "theoretical_2_over_3", "validation_min_cost"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "knn_cost_sensitive_thresholds.csv", index=False)
    comparison.to_csv(output_dir / "knn_cost_sensitive_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(results.threshold, results.cost)
    axes[0].axvline(best.threshold, color="tab:red", linestyle="--", label="Selected threshold")
    axes[0].set(xlabel="Threshold", ylabel="Validation cost (2 FP + FN)")
    axes[0].legend()
    axes[1].plot(results.threshold, results.fp, label="FP")
    axes[1].plot(results.threshold, results.fn, label="FN")
    axes[1].set(xlabel="Threshold", ylabel="Number of errors")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "knn_cost_sensitive.png", dpi=160)
    plt.close(fig)
    return results, comparison


def main():
    root = Path(__file__).resolve().parents[3]
    train = pd.read_csv(root / "data/train.csv", dtype={"id": str})
    y = train.is_canceled.to_numpy()
    X = train.drop(columns=["id", "is_canceled"])
    train_idx, valid_idx = make_split(y)
    pre = KNNPreprocessor().fit(X.iloc[train_idx])
    model = KNNClassifier(21, "distance", "manhattan").fit(
        pre.transform(X.iloc[train_idx]), y[train_idx]
    )
    print("Menghitung skor validasi bonus...", flush=True)
    scores = model.decision_scores(pre.transform(X.iloc[valid_idx]), batch_size=256)
    _, comparison = evaluate_cost_sensitive(
        y[valid_idx], scores, root / "output" / "knn"
    )
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
