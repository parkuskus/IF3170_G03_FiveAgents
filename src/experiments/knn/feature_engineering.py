from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_SETS = (
    "baseline",
    "total_guests",
    "total_nights",
    "total_stay_value",
    "requests_per_guest",
)


def add_engineered_features(X: pd.DataFrame, feature_set: str = "baseline") -> pd.DataFrame:
    """Add one selected feature group without using target information.

    All arithmetic is row-local and therefore safe to apply to train/valid/test.
    Missing numeric inputs are handled later by KNNPreprocessor; arithmetic uses
    min_count=1 where appropriate so an all-missing guest count stays missing.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set harus salah satu dari {FEATURE_SETS}")
    X = X.copy()
    if feature_set == "baseline":
        return X

    adults = pd.to_numeric(X["num_adults"], errors="coerce")
    children = pd.to_numeric(X["num_children"], errors="coerce")
    weekend = pd.to_numeric(X["weekend_nights"], errors="coerce")
    weekday = pd.to_numeric(X["weekday_nights"], errors="coerce")

    if feature_set == "total_guests":
        X["total_guests"] = adults + children
    elif feature_set == "total_nights":
        X["total_nights"] = weekend + weekday
    elif feature_set == "total_stay_value":
        nights = weekend + weekday
        rate = pd.to_numeric(X["nightly_rate_idr"], errors="coerce")
        X["total_stay_value"] = nights * rate
    elif feature_set == "requests_per_guest":
        guests = adults + children
        requests = pd.to_numeric(X["special_requests"], errors="coerce")
        X["requests_per_guest"] = requests / guests.replace(0, np.nan)
    return X
