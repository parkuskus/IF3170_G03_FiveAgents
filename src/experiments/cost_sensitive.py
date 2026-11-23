"""Threshold-only cost experiment on scores from a separately fitted model."""

import numpy as np
from sklearn.metrics import f1_score


def evaluate_threshold(y, scores, threshold, fp_cost=3, fn_cost=1):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if (y.ndim != 1 or scores.shape != y.shape or not len(y)
            or not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all()
            or not np.isfinite([fp_cost, fn_cost]).all()
            or min(fp_cost, fn_cost) <= 0 or not np.isfinite(threshold)):
        raise ValueError("Expected matching binary labels, finite scores and positive costs")
    pred = scores > threshold
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    return {
        "threshold": float(threshold), "fp": fp, "fn": fn,
        "cost": float(fp_cost * fp + fn_cost * fn),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
    }


def choose_threshold(y, scores, fp_cost=3, fn_cost=1):
    scores = np.asarray(scores, dtype=float)
    evaluate_threshold(y, scores, 0.5, fp_cost, fn_cost)
    # With strict >, predictions change only at observed score values.
    distinct = np.unique(scores)
    candidates = np.concatenate((
        [np.nextafter(scores.min(), -np.inf)],
        distinct if len(distinct) <= 101 else np.unique(
            np.quantile(scores, np.linspace(0, 1, 101))
        ),
    ))
    return min(
        (evaluate_threshold(y, scores, t, fp_cost, fn_cost) for t in candidates),
        key=lambda row: (row["cost"], -row["macro_f1"], row["threshold"]),
    )


def compare_holdout(y_train, oof_scores, y_valid, valid_scores, baseline_threshold,
                    fp_cost=3, fn_cost=1):
    """Choose on training OOF scores; measure both strategies on untouched holdout."""
    tuned = choose_threshold(y_train, oof_scores, fp_cost, fn_cost)
    return {
        "baseline": evaluate_threshold(
            y_valid, valid_scores, baseline_threshold, fp_cost, fn_cost
        ),
        "selected": evaluate_threshold(
            y_valid, valid_scores, tuned["threshold"], fp_cost, fn_cost
        ),
        "oof_selected": tuned,
    }
