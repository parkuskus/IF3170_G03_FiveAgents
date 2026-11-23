from __future__ import annotations

import numpy as np
import pandas as pd

NON_NEGATIVE_COLS = (
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "days_before_arrival",
    "prior_cancellations",
    "prior_completed_bookings",
    "nightly_rate_idr",
    "special_requests",
)
VALID_RANGES = {
    "arrival_month": (1, 12),
    "arrival_day": (1, 31),
    "needs_parking": (0, 1),
    "is_returning_guest": (0, 1),
}


def mask_invalid_values(X: pd.DataFrame) -> pd.DataFrame:
    """Ganti nilai di luar domain dengan NaN (row-local, tanpa statistik)."""
    X = X.copy()
    for col in NON_NEGATIVE_COLS:
        if col in X.columns:
            values = pd.to_numeric(X[col], errors="coerce")
            X[col] = values.where(values >= 0)
    for col, (low, high) in VALID_RANGES.items():
        if col in X.columns:
            values = pd.to_numeric(X[col], errors="coerce")
            X[col] = values.where((values >= low) & (values <= high))
    return X


def drop_duplicate_rows(X: pd.DataFrame, y):
    """Buang baris latih yang identik pada fitur + label (keep='first')."""
    y = np.asarray(y)
    frame = X.reset_index(drop=True).copy()
    frame["__label__"] = y
    keep = ~frame.duplicated(keep="first").to_numpy()
    return X.iloc[np.flatnonzero(keep)], y[keep], int((~keep).sum())


def clean_training_data(X: pd.DataFrame, y):
    """Cleaning lengkap untuk data latih: nilai tidak valid lalu duplikat."""
    X = mask_invalid_values(X)
    return drop_duplicate_rows(X, y)
