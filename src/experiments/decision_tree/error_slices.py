"""Reproduce holdout error slices for the CV-selected DTL-v08 configuration.

Run from the repository root with python -m src.experiments.decision_tree.error_slices.
Slices describe the existing model; they are not used to tune it.
"""

import numpy as np
import pandas as pd

from .run_all import OUTPUT, ROOT, prepare_fold, validation_indices
from .shared import BaselineTree


def summarize_slices(groups, truth, prediction):
    groups = pd.Series(groups).reset_index(drop=True).astype(object).fillna("(missing)").astype(str)
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if len(groups) != len(truth) or len(truth) != len(prediction):
        raise ValueError("groups, truth and prediction must have equal lengths")
    observations = pd.DataFrame({
        "group": groups, "positive": truth == 1, "negative": truth == 0,
        "fp": (truth == 0) & (prediction == 1),
        "fn": (truth == 1) & (prediction == 0),
    })
    result = observations.groupby("group", sort=False).agg(
        n=("positive", "size"), positives=("positive", "sum"),
        negatives=("negative", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
    ).reset_index()
    result["error_rate"] = (result.fp + result.fn) / result.n
    result["miss_rate"] = result.fn / result.positives.replace(0, np.nan)
    result["false_alarm_rate"] = result.fp / result.negatives.replace(0, np.nan)
    return result


def main():
    data = pd.read_csv(ROOT / "data" / "train.csv", dtype={"id": str})
    train, valid = validation_indices(data.is_canceled)
    features = data.columns.drop(["id", "is_canceled"])
    X_train, X_valid = prepare_fold(data.iloc[train][features], data.iloc[valid][features])
    y_train = data.iloc[train].is_canceled.to_numpy(dtype=np.int8)
    y_valid = data.iloc[valid].is_canceled.to_numpy(dtype=np.int8)
    model = BaselineTree(criterion="gini", max_depth=None,
                         min_samples_split=2, min_samples_leaf=1).fit(X_train, y_train).pruned(0.0001)
    prediction = (model.predict_proba(X_valid)[:, 1] >= 0.62).astype(np.int8)
    assert (int(((y_valid == 0) & (prediction == 1)).sum()),
            int(((y_valid == 1) & (prediction == 0)).sum())) == (271, 489)

    raw = data.iloc[valid].reset_index(drop=True)
    days = pd.cut(raw.days_before_arrival, bins=[-np.inf, 30, 147.5, np.inf],
                  labels=["0--30", "31--147", "148+"])
    requests = pd.cut(raw.special_requests, bins=[-np.inf, 0, 1, np.inf],
                      labels=["0", "1", "2+"])
    slices = {
        "days_before_arrival": days,
        "special_requests": requests,
        "booking_channel": raw.booking_channel,
    }
    summary = pd.concat(
        [summarize_slices(group, y_valid, prediction).assign(feature=name)
         for name, group in slices.items()], ignore_index=True,
    )
    summary = summary[["feature", "group", "n", "positives", "negatives",
                       "fp", "fn", "error_rate", "miss_rate", "false_alarm_rate"]]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "error_slices.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
