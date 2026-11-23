from pathlib import Path
import filecmp
import json
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

if __package__:
    from .shared import BaselineTree, SharedPreprocessor
else:
    from shared import BaselineTree, SharedPreprocessor


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "output" / "experiments" / "decision_tree"
VARIANTS = {
    "ID3": ("information_gain", "entropy", 0.0),
    "C4.5": ("gain_ratio", "entropy", 0.001),
    "CART": ("gini", "gini", 0.0),
}
CART_TREE_CONFIGS = (
    (None, 2, 1),
    (12, 40, 20),
    (8, 80, 40),
)
CART_POSITIVE_WEIGHTS = (1.0, 1.5, 2.0, 2.05, 2.5, 3.0)
# DTL-v08 is the final pipeline: CV-selected CART growth, post-pruning and threshold.
EXPERIMENTS = {
    "ID3 from scratch": ("DTL-v01", "baseline"),
    "ID3 scikit-learn": ("DTL-v02", "comparator"),
    "C4.5 from scratch": ("DTL-v03", "baseline"),
    "C4.5 scikit-learn": ("DTL-v04", "comparator"),
    "CART from scratch": ("DTL-v05", "baseline"),
    "CART scikit-learn": ("DTL-v06", "comparator"),
    "CART selected growth configuration (no pruning)": ("DTL-v07", "experiment"),
    "CART selected growth configuration + cost-complexity pruning": ("DTL-v08", "FINAL"),
    "CART class-weighted": ("DTL-v09", "experiment"),
}


def validation_indices(y):
    y = np.asarray(y)
    rng = np.random.default_rng(42)
    valid = np.sort(np.concatenate([
        rng.permutation(np.flatnonzero(y == label))[:int(np.ceil(0.2 * np.sum(y == label)))]
        for label in (0, 1)
    ]))
    train = np.setdiff1d(np.arange(len(y)), valid)
    return train, valid


def threshold_predictions(probabilities, threshold):
    return (np.asarray(probabilities) >= threshold).astype(np.int8).tolist()


def prepare_fold(X_train, X_valid):
    preprocessor = SharedPreprocessor().fit(X_train)
    return preprocessor.transform(X_train), preprocessor.transform(X_valid)


def tune_cart_pruning(X, y, n_splits=5, tree_configs=CART_TREE_CONFIGS):
    y = np.asarray(y, dtype=np.int8)
    alphas = (0.0, 0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25)
    thresholds = np.round(np.arange(0.20, 0.8001, 0.02), 2)
    candidates = [
        (max_depth, min_samples_split, min_samples_leaf, alpha)
        for max_depth, min_samples_split, min_samples_leaf in tree_configs
        for alpha in alphas
    ]
    probabilities = {candidate: np.empty(len(y), dtype=float) for candidate in candidates}
    leaves = {candidate: [] for candidate in candidates}
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, valid_idx in splitter.split(X, y):
        X_train, X_valid = prepare_fold(X.iloc[train_idx], X.iloc[valid_idx])
        for max_depth, min_samples_split, min_samples_leaf in tree_configs:
            model = BaselineTree(
                criterion="gini", max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
            ).fit(X_train, y[train_idx])
            for alpha in alphas:
                candidate = (max_depth, min_samples_split, min_samples_leaf, alpha)
                tree = model.pruned(alpha)
                probabilities[candidate][valid_idx] = tree.predict_proba(X_valid)[:, 1]
                leaves[candidate].append(tree.leaf_count())

    results = []
    for max_depth, min_samples_split, min_samples_leaf, alpha in candidates:
        for threshold in thresholds:
            candidate = (max_depth, min_samples_split, min_samples_leaf, alpha)
            prediction = threshold_predictions(probabilities[candidate], threshold)
            results.append({
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "alpha": alpha,
                "threshold": float(threshold),
                "macro_f1": f1_score(y, prediction, labels=[0, 1], average="macro", zero_division=0),
                "mean_leaves": float(np.mean(leaves[candidate])),
            })
    best = max(results, key=lambda result: (result["macro_f1"], -result["mean_leaves"], result["alpha"], result["threshold"]))
    return best, pd.DataFrame(results)


def tune_cart_class_weight(X, y, n_splits=5, positive_weights=CART_POSITIVE_WEIGHTS):
    y = np.asarray(y, dtype=np.int8)
    probabilities = {weight: np.empty(len(y), dtype=np.int8) for weight in positive_weights}
    leaves = {weight: [] for weight in positive_weights}
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, valid_idx in splitter.split(X, y):
        X_train, X_valid = prepare_fold(X.iloc[train_idx], X.iloc[valid_idx])
        for weight in positive_weights:
            model = BaselineTree(criterion="gini").fit(
                X_train, y[train_idx], class_weight={0: 1.0, 1: weight}
            )
            probabilities[weight][valid_idx] = model.predict(X_valid)
            leaves[weight].append(model.leaf_count())

    results = pd.DataFrame([{
        "positive_weight": weight,
        "macro_f1": f1_score(y, probabilities[weight], labels=[0, 1], average="macro", zero_division=0),
        "mean_leaves": float(np.mean(leaves[weight])),
    } for weight in positive_weights])
    best = max(results.to_dict(orient="records"), key=lambda row: (row["macro_f1"], -row["mean_leaves"], -row["positive_weight"]))
    return best, results


def score_row(name, y_true, prediction, seconds, detail):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, prediction, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    return {
        "experiment_version": EXPERIMENTS[name][0],
        "pipeline_role": EXPERIMENTS[name][1],
        "model": name,
        "validation_macro_f1": f1_score(y_true, prediction, labels=[0, 1], average="macro", zero_division=0),
        "class_0_precision": precision[0],
        "class_0_recall": recall[0],
        "class_0_f1": f1[0],
        "class_1_precision": precision[1],
        "class_1_recall": recall[1],
        "class_1_f1": f1[1],
        "tn": matrix[0, 0], "fp": matrix[0, 1], "fn": matrix[1, 0], "tp": matrix[1, 1],
        "seconds": seconds,
        "configuration": detail,
    }


def build_submission(template, predictions, test_ids=None):
    if test_ids is not None and (
        "id" not in template or template["id"].reset_index(drop=True).tolist() != list(test_ids)
    ):
        raise ValueError("Submission template IDs must match test.csv in row order")
    submission = template.copy()
    submission["is_canceled"] = predictions
    return submission


def archive_current_submission(destination):
    destination = Path(destination)
    versions = destination / "versions"
    current = destination / "submission.csv"
    version_files = list(versions.glob("submission_v*.csv"))

    def next_number():
        numbers = [int(path.stem.removeprefix("submission_v")) for path in version_files]
        return max(numbers, default=0) + 1

    if current.exists() and (
        not version_files or not filecmp.cmp(current, max(version_files, key=lambda path: path.name), shallow=False)
    ):
        versions.mkdir(parents=True, exist_ok=True)
        previous = versions / f"submission_v{next_number():03}.csv"
        shutil.copy2(current, previous)
        version_files.append(previous)
        return previous
    return None


def save_submission_version(submission, destination, experiment=None):
    destination = Path(destination)
    versions = destination / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    current = destination / "submission.csv"
    archive_current_submission(destination)
    version_files = list(versions.glob("submission_v*.csv"))

    numbers = [int(path.stem.removeprefix("submission_v")) for path in version_files]
    version_path = versions / f"submission_v{max(numbers, default=0) + 1:03}.csv"
    submission.to_csv(version_path, index=False)
    shutil.copy2(version_path, current)
    if experiment is not None:
        metadata = json.dumps(experiment, indent=2) + "\n"
        version_path.with_suffix(".json").write_text(metadata, encoding="utf-8")
        (destination / "submission.json").write_text(metadata, encoding="utf-8")
    return version_path


def write_submission(name, train_df, tuning=None):
    features = [c for c in train_df.columns if c not in {"id", "is_canceled"}]
    test_df = pd.read_csv(ROOT / "data" / "test.csv", dtype={"id": str})
    y_train = train_df["is_canceled"].to_numpy(dtype=np.int8)
    prep = SharedPreprocessor().fit(train_df[features])
    X_train = prep.transform(train_df[features])
    X_test = prep.transform(test_df[features])
    configuration = tuning or {
        "max_depth": 8, "min_samples_split": 80, "min_samples_leaf": 40,
    }
    model = BaselineTree(criterion=VARIANTS[name][0], **{
        key: configuration[key]
        for key in ("max_depth", "min_samples_split", "min_samples_leaf")
    }).fit(X_train, y_train)
    if tuning:
        if "alpha" in tuning:
            model = model.pruned(tuning["alpha"])
        probabilities = model.predict_proba(X_test)[:, 1]
        prediction = threshold_predictions(probabilities, tuning["threshold"])
    else:
        prediction = model.predict(X_test)
    template = pd.read_csv(ROOT / "data" / "sample_submission.csv", dtype={"id": str})
    submission = build_submission(template, prediction, test_df["id"])
    destination = OUTPUT / name
    destination.mkdir(parents=True, exist_ok=True)
    model_name = (
        "CART selected growth configuration + cost-complexity pruning"
        if name == "CART" else f"{name} from scratch"
    )
    experiment = {
        "experiment_version": EXPERIMENTS[model_name][0],
        "pipeline_role": EXPERIMENTS[model_name][1],
        "model": model_name,
        "configuration": {
            "criterion": VARIANTS[name][0],
            **{key: configuration[key] for key in ("max_depth", "min_samples_split", "min_samples_leaf")},
            **({"alpha": tuning["alpha"], "threshold": tuning["threshold"]} if tuning else {}),
        },
    }
    save_submission_version(submission, destination, experiment)


def run_tuned_cart(train_df, train_idx, valid_idx, baseline_score):
    features = [c for c in train_df.columns if c not in {"id", "is_canceled"}]
    y = train_df["is_canceled"].to_numpy(dtype=np.int8)
    X_train_raw = train_df.iloc[train_idx][features]
    X_valid_raw = train_df.iloc[valid_idx][features]
    y_train, y_valid = y[train_idx], y[valid_idx]

    best, pruning_results = tune_cart_pruning(X_train_raw, y_train)
    X_train, X_valid = prepare_fold(X_train_raw, X_valid_raw)
    model = BaselineTree(
        criterion="gini", max_depth=best["max_depth"],
        min_samples_split=best["min_samples_split"],
        min_samples_leaf=best["min_samples_leaf"],
    )
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start
    prediction_start = time.perf_counter()
    probabilities = model.predict_proba(X_valid)[:, 1]
    row_unpruned = score_row(
        "CART selected growth configuration (no pruning)", y_valid,
        threshold_predictions(probabilities, best["threshold"]),
        fit_seconds + time.perf_counter() - prediction_start,
        f"max_depth={best['max_depth']}, min_samples_split={best['min_samples_split']}, "
        f"min_samples_leaf={best['min_samples_leaf']}, threshold={best['threshold']:.2f}",
    )
    prune_start = time.perf_counter()
    pruned = model.pruned(best["alpha"])
    pruned_probabilities = pruned.predict_proba(X_valid)[:, 1]
    row_pruned = score_row(
        "CART selected growth configuration + cost-complexity pruning", y_valid,
        threshold_predictions(pruned_probabilities, best["threshold"]),
        fit_seconds + time.perf_counter() - prune_start,
        f"max_depth={best['max_depth']}, min_samples_split={best['min_samples_split']}, "
        f"min_samples_leaf={best['min_samples_leaf']}, threshold={best['threshold']:.2f}, "
        f"ccp_alpha={best['alpha']}, leaves={pruned.leaf_count()}",
    )
    rows = [row_unpruned, row_pruned]
    destination = OUTPUT / "CART"
    destination.mkdir(parents=True, exist_ok=True)
    pruning_results.to_csv(destination / "pruning.csv", index=False)
    (destination / "pruning.md").write_text(
        "# CART cost-complexity pruning\n\n"
        "Each fold evaluates three CART growth configurations, then their cost-complexity-pruned copies and probability thresholds. The configuration is selected by macro F1 using 5-fold stratified CV.\n\n"
        f"Selected `max_depth={best['max_depth']}`, `min_samples_split={best['min_samples_split']}`, "
        f"`min_samples_leaf={best['min_samples_leaf']}`, `ccp_alpha={best['alpha']}`, "
        f"and threshold **{best['threshold']:.2f}** by 5-fold stratified CV, "
        f"with macro F1 **{best['macro_f1']:.4f}** and mean leaves **{best['mean_leaves']:.1f}**.\n\n"
        f"Holdout macro F1: **{row_unpruned['validation_macro_f1']:.4f}** before pruning; "
        f"**{row_pruned['validation_macro_f1']:.4f}** after pruning ({pruned.leaf_count()} leaves), versus original baseline **{baseline_score:.4f}**.\n\n"
        "Alpha and threshold are selected on training folds only; the shared validation split remains a final holdout. Candidates are in `pruning.csv`.\n",
        encoding="utf-8",
    )
    return rows, best


def run_weighted_cart(train_df, train_idx, valid_idx):
    features = [c for c in train_df.columns if c not in {"id", "is_canceled"}]
    y = train_df["is_canceled"].to_numpy(dtype=np.int8)
    X_train_raw, X_valid_raw = train_df.iloc[train_idx][features], train_df.iloc[valid_idx][features]
    y_train, y_valid = y[train_idx], y[valid_idx]
    best, results = tune_cart_class_weight(X_train_raw, y_train)
    X_train, X_valid = prepare_fold(X_train_raw, X_valid_raw)
    start = time.perf_counter()
    model = BaselineTree(criterion="gini").fit(
        X_train, y_train, class_weight={0: 1.0, 1: best["positive_weight"]}
    )
    prediction = model.predict(X_valid)
    row = score_row(
        "CART class-weighted",
        y_valid,
        prediction,
        time.perf_counter() - start,
        f"positive class weight={best['positive_weight']}, leaves={model.leaf_count()}",
    )
    destination = OUTPUT / "CART_class_weight"
    destination.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination / "class_weight.csv", index=False)
    (destination / "class_weight.md").write_text(
        "# CART class weighting\n\n"
        "DTL-v09 — eksperimen, bukan pipeline final (DTL-v08).\n\n"
        "Positive-class weight is selected by 5-fold stratified CV on the training partition; each fold fits preprocessing and a weighted CART from scratch. The shared validation split is used only once for final evaluation. No submission is written.\n\n"
        f"Selected positive class weight: **{best['positive_weight']}**; CV macro F1: **{best['macro_f1']:.4f}**; "
        f"mean leaves: **{best['mean_leaves']:.1f}**.\n\n"
        f"Holdout macro F1: **{row['validation_macro_f1']:.4f}**; positive-class F1: **{row['class_1_f1']:.4f}**; "
        f"leaves: **{model.leaf_count()}**. Candidate scores are in `class_weight.csv`.\n",
        encoding="utf-8",
    )
    return row, best, results


def run_variant(name, train_df, train_idx, valid_idx):
    criterion, sklearn_criterion, alpha = VARIANTS[name]
    features = [c for c in train_df.columns if c not in {"id", "is_canceled"}]
    y = train_df["is_canceled"].to_numpy(dtype=np.int8)
    X_train_raw = train_df.iloc[train_idx][features]
    X_valid_raw = train_df.iloc[valid_idx][features]
    y_train, y_valid = y[train_idx], y[valid_idx]

    prep = SharedPreprocessor().fit(X_train_raw)
    X_train = prep.transform(X_train_raw)
    X_valid = prep.transform(X_valid_raw)
    configuration = "max_depth=8, min_samples_split=80, min_samples_leaf=40"

    start = time.perf_counter()
    model = BaselineTree(criterion=criterion).fit(X_train, y_train)
    prediction = model.predict(X_valid)
    rows = [score_row(f"{name} from scratch", y_valid, prediction,
                      time.perf_counter() - start, configuration)]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_sklearn = encoder.fit_transform(X_train[prep.categorical]) if prep.categorical else X_train[prep.numeric].to_numpy()
    X_valid_sklearn = encoder.transform(X_valid[prep.categorical]) if prep.categorical else X_valid[prep.numeric].to_numpy()
    if prep.categorical and prep.numeric:
        X_train_sklearn = np.column_stack([X_train[prep.numeric].to_numpy(), X_train_sklearn])
        X_valid_sklearn = np.column_stack([X_valid[prep.numeric].to_numpy(), X_valid_sklearn])

    sklearn_model = DecisionTreeClassifier(
        criterion=sklearn_criterion, ccp_alpha=alpha, max_depth=8,
        min_samples_split=80, min_samples_leaf=40, random_state=42,
    )
    start = time.perf_counter()
    sklearn_model.fit(X_train_sklearn, y_train)
    sklearn_prediction = sklearn_model.predict(X_valid_sklearn)
    rows.append(score_row(f"{name} scikit-learn", y_valid, sklearn_prediction,
                          time.perf_counter() - start,
                          f"criterion={sklearn_criterion}, ccp_alpha={alpha}, {configuration}, OneHotEncoder"))
    return rows


def main(only=None):
    if only is not None and only not in VARIANTS:
        raise ValueError(f"Unknown variant: {only}")
    train_df = pd.read_csv(ROOT / "data" / "train.csv", dtype={"id": str})
    train_idx, valid_idx = validation_indices(train_df["is_canceled"])
    output = OUTPUT / only if only else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ([only] if only else VARIANTS):
        rows.extend(run_variant(name, train_df, train_idx, valid_idx))
        tuning = None
        if name == "CART":
            baseline_score = next(row["validation_macro_f1"] for row in rows if row["model"] == "CART from scratch")
            tuned_rows, tuning = run_tuned_cart(train_df, train_idx, valid_idx, baseline_score)
            rows.extend(tuned_rows)
        write_submission(name, train_df, tuning)
    results = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    results.to_csv(output / "validation.csv", index=False)
    lines = [
        "# Perbandingan baseline Decision Tree",
        "",
        "Semua model memakai split stratified kontrak README (seed 42), preprocessing yang sama, dan macro F1 validasi. Fitur test hanya digunakan untuk membuat submission, bukan untuk seleksi model atau evaluasi. Implementasi dari nol menerima fitur kategori nominal langsung; pembanding scikit-learn memakai OneHotEncoder karena estimator itu memerlukan input numerik.",
        "",
        f"Split: {len(train_idx)} baris latih, {len(valid_idx)} baris validasi. Target validasi: 0={int((train_df.iloc[valid_idx]['is_canceled'] == 0).sum())}, 1={int((train_df.iloc[valid_idx]['is_canceled'] == 1).sum())}.",
        "",
        "| Versi | Peran | Model | Macro F1 | Precision kelas 0 | Recall kelas 0 | Precision kelas 1 | Recall kelas 1 | TN | FP | FN | TP | Detik |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if only is None or only == "CART":
        lines[2:2] = [
            "**Pipeline final: DTL-v08 — CART dengan cost-complexity post-pruning dan threshold hasil 5-fold CV.**",
            "",
        ]
    for row in results.to_dict(orient="records"):
        lines.append(
            f"| {row['experiment_version']} | {row['pipeline_role']} | {row['model']} | {row['validation_macro_f1']:.4f} | {row['class_0_precision']:.4f} | {row['class_0_recall']:.4f} | {row['class_1_precision']:.4f} | {row['class_1_recall']:.4f} | {row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} | {row['seconds']:.2f} |"
        )
    lines += ["", "## Preprocessing bersama", "", "Imputasi median numerik dan penanda kategori kosong/asing dipelajari hanya dari bagian latih. `id` tidak menjadi fitur. Tiga baseline scratch memakai `max_depth=8`, `min_samples_split=80`, dan `min_samples_leaf=40`; eksperimen CART tambahan membandingkan tiga konfigurasi pertumbuhan, pruning, serta threshold dengan target macro F1.", "", "## Definisi baseline", "", "- ID3: information gain, kategori nominal multi-cabang, fitur numerik dengan threshold biner.", "- C4.5: gain ratio, kategori nominal multi-cabang, fitur numerik dengan threshold biner. Pembanding scikit-learn memakai entropy dan `ccp_alpha=0.001` sesuai spesifikasi.", "- CART: Gini, split kategori biner, fitur numerik dengan threshold biner. Eksperimen tambahan memilih konfigurasi pertumbuhan, cost-complexity pruning, dan threshold dengan macro F1 melalui 5-fold stratified CV.", "", "Skor dan konfigurasi tiap model ada di `validation.csv`; kandidat tuning ada di `CART/pruning.csv`.", ""]
    (output / "validation.md").write_text("\n".join(lines), encoding="utf-8")
    print(results[["model", "validation_macro_f1", "seconds"]].to_string(index=False))
    print(f"\nResults: {output}")


if __name__ == "__main__":
    main()
