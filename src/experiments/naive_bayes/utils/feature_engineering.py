from __future__ import annotations

import numpy as np
import pandas as pd

LEGACY_FEATURE_SETS = (
    "baseline",
    "total_guests",
    "total_nights",
    "total_stay_value",
    "requests_per_guest",
    "rate_per_guest_night",
    "prior_cancel_ratio",
    "booking_intensity",
    "arrival_quarter",
)

NB_BEST_FEATURES = (
    "has_requests",
    "is_ota",
    "is_offline",
    "dup_channel",
)

NB_FINAL_FEATURES = (
    "ota_x_requests",
    "rate_weekend_req_ota",
)

FEATURE_LIBRARY = {
    "has_requests": "special_requests >= 1",
    "is_ota": "booking_channel == Online Travel Agent",
    "is_offline": "booking_channel == Offline Agent",
    "is_airline": "booking_channel == Airline",
    "is_corporate_or_complimentary": "booking_channel in {Corporate, Complimentary}",
    "is_type_b": "room_category == Type B",
    "is_type_g": "room_category == Type G",
    "is_half_board": "meal_package == Half Board",
    "is_room_only": "meal_package == Room Only",
    "is_peak_season": "arrival_month in {6, 7, 8}",
    "is_winter": "arrival_month in {1, 2, 12}",
    "is_year_2023": "arrival_year == 2023",
    "lead_7": "days_before_arrival > 7",
    "lead_30": "days_before_arrival > 30",
    "lead_90": "days_before_arrival > 90",
    "is_zero_rate": "nightly_rate_idr == 0",
    "prior_new": "prior_completed_bookings == 0",
    "has_prior_cancel": "prior_cancellations > 0",
    "adult_1": "num_adults == 1",
    "adult_3plus": "num_adults >= 3",
    "has_children": "num_children > 0",
    "has_weekend_night": "weekend_nights > 0",
    "long_stay_7": "weekend_nights + weekday_nights >= 7",
    "dup_channel": "duplikat kategorikal booking_channel",
    "dup_returning": "duplikat kategorikal is_returning_guest",
    "dup_parking": "duplikat kategorikal needs_parking",
    "pnew_ota": "prior_completed_bookings == 0 dan is_ota",
    "noreq_ota": "special_requests == 0 dan is_ota",
    "hr_ota": "has_requests dan is_ota",
    "hr_off": "has_requests dan is_offline",
    "typeb_ota": "room_category == Type B dan is_ota",
    "hr_pcan": "has_requests dan prior_cancellations > 0",
    "pcan_ota": "prior_cancellations > 0 dan is_ota",
    "ret_noreq": "is_returning_guest == 1 dan special_requests == 0",
    "hr_ret": "has_requests dan is_returning_guest == 1",
    "ota_x_requests": "booking_channel == OTA dan bucket special_requests",
    "rate_weekend_req_ota": "bucket nightly_rate x weekend_nights x special_requests x OTA",
    "lead_x_ota_req": "bucket days_before_arrival x OTA x bucket special_requests",
    "lead_x_rate": "bucket days_before_arrival x bucket nightly_rate",
    "lead_x_weekend_req": "bucket days_before_arrival x weekend_nights x bucket special_requests",
    "pcan_x_ota_req": "prior_cancellations > 0 x OTA x bucket special_requests",
    "channel_x_meal": "booking_channel x meal_package",
}

FEATURE_SETS = LEGACY_FEATURE_SETS + tuple(FEATURE_LIBRARY) + ("nb_best",)

FEATURE_DESCRIPTIONS = {
    "baseline": "fitur baseline tanpa turunan",
    "total_guests": "num_adults + num_children",
    "total_nights": "weekend_nights + weekday_nights",
    "total_stay_value": "total_nights * nightly_rate_idr",
    "requests_per_guest": "special_requests / total_guests",
    "rate_per_guest_night": "nightly_rate_idr / total_guests",
    "prior_cancel_ratio": "prior_cancellations / total_prior_bookings",
    "booking_intensity": "total_nights * total_guests",
    "arrival_quarter": "kuartal kedatangan dari arrival_month",
    "nb_best": "gabungan has_requests + is_ota + is_offline + dup_channel",
}
FEATURE_DESCRIPTIONS.update(FEATURE_LIBRARY)


def _num(X: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(X[column], errors="coerce")


def _text(X: pd.DataFrame, column: str) -> pd.Series:
    return X[column].astype("string")


def _total_guests(X: pd.DataFrame) -> pd.Series:
    return _num(X, "num_adults") + _num(X, "num_children")


def _total_nights(X: pd.DataFrame) -> pd.Series:
    return _num(X, "weekend_nights") + _num(X, "weekday_nights")


def _flag(condition) -> pd.Series:
    return condition.fillna(False).astype(float)


def _request_bucket(X: pd.DataFrame) -> pd.Series:
    values = _num(X, "special_requests")
    out = pd.Series("3+", index=X.index, dtype="string")
    out[values.isna()] = "MISSING"
    out[values == 0] = "0"
    out[values == 1] = "1"
    out[values == 2] = "2"
    return out


def _rate_bucket(X: pd.DataFrame) -> pd.Series:
    values = _num(X, "nightly_rate_idr")
    out = pd.Series(">2.2M", index=X.index, dtype="string")
    out[values.isna()] = "MISSING"
    out[values <= 0] = "0"
    out[(values > 0) & (values <= 1.5e6)] = "0-1.5M"
    out[(values > 1.5e6) & (values <= 2.2e6)] = "1.5-2.2M"
    return out


def _weekend_bucket(X: pd.DataFrame) -> pd.Series:
    values = _num(X, "weekend_nights")
    out = pd.Series("1+", index=X.index, dtype="string")
    out[values.isna()] = "MISSING"
    out[values <= 0] = "0"
    return out


def _ota_flag(X: pd.DataFrame) -> pd.Series:
    channel = _text(X, "booking_channel")
    out = pd.Series("nonOTA", index=X.index, dtype="string")
    out[channel.isna()] = "MISSING"
    out[channel == "Online Travel Agent"] = "OTA"
    return out


def _lead_bucket(X: pd.DataFrame) -> pd.Series:
    values = _num(X, "days_before_arrival")
    out = pd.Series(">180", index=X.index, dtype="string")
    out[values.isna()] = "MISSING"
    out[values <= 7] = "<=7"
    out[(values > 7) & (values <= 30)] = "8-30"
    out[(values > 30) & (values <= 90)] = "31-90"
    out[(values > 90) & (values <= 180)] = "91-180"
    return out


def _prior_cancel_bucket(X: pd.DataFrame) -> pd.Series:
    values = _num(X, "prior_cancellations")
    out = pd.Series("1+", index=X.index, dtype="string")
    out[values.isna()] = "MISSING"
    out[values <= 0] = "0"
    return out


def _join_cross(*parts: pd.Series) -> pd.Series:
    out = parts[0].astype("string").fillna("MISSING")
    for part in parts[1:]:
        out = out.str.cat(part.astype("string").fillna("MISSING"), sep="|")
    return out


def _derived_value(X: pd.DataFrame, name: str):
    if name == "has_requests":
        return _flag(_num(X, "special_requests").fillna(-1) >= 1)
    if name == "is_ota":
        return _flag(_text(X, "booking_channel") == "Online Travel Agent")
    if name == "is_offline":
        return _flag(_text(X, "booking_channel") == "Offline Agent")
    if name == "is_airline":
        return _flag(_text(X, "booking_channel") == "Airline")
    if name == "is_corporate_or_complimentary":
        return _flag(_text(X, "booking_channel").isin(["Corporate", "Complimentary"]))
    if name == "is_type_b":
        return _flag(_text(X, "room_category") == "Type B")
    if name == "is_type_g":
        return _flag(_text(X, "room_category") == "Type G")
    if name == "is_half_board":
        return _flag(_text(X, "meal_package") == "Half Board")
    if name == "is_room_only":
        return _flag(_text(X, "meal_package") == "Room Only")
    if name == "is_peak_season":
        return _flag(_num(X, "arrival_month").isin([6, 7, 8]))
    if name == "is_winter":
        return _flag(_num(X, "arrival_month").isin([1, 2, 12]))
    if name == "is_year_2023":
        return _flag(_num(X, "arrival_year") == 2023)
    if name == "lead_7":
        return _flag(_num(X, "days_before_arrival") > 7)
    if name == "lead_30":
        return _flag(_num(X, "days_before_arrival") > 30)
    if name == "lead_90":
        return _flag(_num(X, "days_before_arrival") > 90)
    if name == "is_zero_rate":
        return _flag(_num(X, "nightly_rate_idr") == 0)
    if name == "prior_new":
        return _flag(_num(X, "prior_completed_bookings") == 0)
    if name == "has_prior_cancel":
        return _flag(_num(X, "prior_cancellations") > 0)
    if name == "adult_1":
        return _flag(_num(X, "num_adults") == 1)
    if name == "adult_3plus":
        return _flag(_num(X, "num_adults") >= 3)
    if name == "has_children":
        return _flag(_num(X, "num_children") > 0)
    if name == "has_weekend_night":
        return _flag(_num(X, "weekend_nights") > 0)
    if name == "long_stay_7":
        return _flag(_total_nights(X) >= 7)
    if name == "dup_channel":
        return _text(X, "booking_channel")
    if name == "dup_returning":
        return _text(X, "is_returning_guest")
    if name == "dup_parking":
        return _text(X, "needs_parking")
    if name == "pnew_ota":
        return _flag(
            (_num(X, "prior_completed_bookings") == 0)
            & (_text(X, "booking_channel") == "Online Travel Agent")
        )
    if name == "noreq_ota":
        return _flag(
            (_num(X, "special_requests").fillna(-1) == 0)
            & (_text(X, "booking_channel") == "Online Travel Agent")
        )
    if name == "hr_ota":
        return _flag(
            (_num(X, "special_requests").fillna(-1) >= 1)
            & (_text(X, "booking_channel") == "Online Travel Agent")
        )
    if name == "hr_off":
        return _flag(
            (_num(X, "special_requests").fillna(-1) >= 1)
            & (_text(X, "booking_channel") == "Offline Agent")
        )
    if name == "typeb_ota":
        return _flag(
            (_text(X, "room_category") == "Type B")
            & (_text(X, "booking_channel") == "Online Travel Agent")
        )
    if name == "hr_pcan":
        return _flag(
            (_num(X, "special_requests").fillna(-1) >= 1)
            & (_num(X, "prior_cancellations") > 0)
        )
    if name == "pcan_ota":
        return _flag(
            (_num(X, "prior_cancellations") > 0)
            & (_text(X, "booking_channel") == "Online Travel Agent")
        )
    if name == "ret_noreq":
        return _flag(
            (_num(X, "is_returning_guest") == 1)
            & (_num(X, "special_requests").fillna(-1) == 0)
        )
    if name == "hr_ret":
        return _flag(
            (_num(X, "special_requests").fillna(-1) >= 1)
            & (_num(X, "is_returning_guest") == 1)
        )
    if name == "ota_x_requests":
        return _join_cross(_ota_flag(X), _request_bucket(X))
    if name == "rate_weekend_req_ota":
        return _join_cross(
            _rate_bucket(X),
            _weekend_bucket(X),
            _request_bucket(X),
            _ota_flag(X),
        )
    if name == "lead_x_ota_req":
        return _join_cross(_lead_bucket(X), _ota_flag(X), _request_bucket(X))
    if name == "lead_x_rate":
        return _join_cross(_lead_bucket(X), _rate_bucket(X))
    if name == "lead_x_weekend_req":
        return _join_cross(
            _lead_bucket(X),
            _weekend_bucket(X),
            _request_bucket(X),
        )
    if name == "pcan_x_ota_req":
        return _join_cross(
            _prior_cancel_bucket(X),
            _ota_flag(X),
            _request_bucket(X),
        )
    if name == "channel_x_meal":
        return _join_cross(_text(X, "booking_channel"), _text(X, "meal_package"))
    raise ValueError(f"fitur turunan tidak dikenal: {name}")


def add_named_features(X: pd.DataFrame, names) -> pd.DataFrame:
    """Add derived features listed in ``names`` (row-local, no target)."""
    if isinstance(names, str):
        names = [names]
    unknown = [name for name in names if name not in FEATURE_LIBRARY]
    if unknown:
        raise ValueError(f"fitur turunan tidak dikenal: {unknown}")
    X = X.copy()
    for name in names:
        X[name] = _derived_value(X, name)
    return X


def _legacy_feature(X: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "total_guests":
        X["total_guests"] = _total_guests(X)
    elif feature_set == "total_nights":
        X["total_nights"] = _total_nights(X)
    elif feature_set == "total_stay_value":
        X["total_stay_value"] = _total_nights(X) * _num(X, "nightly_rate_idr")
    elif feature_set == "requests_per_guest":
        guests = _total_guests(X).replace(0, np.nan)
        X["requests_per_guest"] = _num(X, "special_requests") / guests
    elif feature_set == "rate_per_guest_night":
        guests = _total_guests(X).replace(0, np.nan)
        X["rate_per_guest_night"] = _num(X, "nightly_rate_idr") / guests
    elif feature_set == "prior_cancel_ratio":
        prior_cancel = _num(X, "prior_cancellations")
        prior_complete = _num(X, "prior_completed_bookings")
        total_prior = (prior_cancel + prior_complete).replace(0, np.nan)
        X["prior_cancel_ratio"] = prior_cancel / total_prior
    elif feature_set == "booking_intensity":
        X["booking_intensity"] = _total_nights(X) * _total_guests(X)
    elif feature_set == "arrival_quarter":
        month = _num(X, "arrival_month")
        X["arrival_quarter"] = np.where(
            month.isna(), np.nan, ((month - 1) // 3 + 1).astype("Int64").astype(str)
        )
    else:
        raise ValueError(f"feature_set tidak dikenal: {feature_set}")
    return X


def add_engineered_features(X: pd.DataFrame, feature_set="baseline") -> pd.DataFrame:
    """Add derived features for a named set, a single feature, or a list of features.

    All operations are row-local and target-free, so the same function is safe for
    train, validation, and test. Missing inputs propagate as missing values and are
    handled later by the preprocessing pipeline.
    """
    if isinstance(feature_set, (list, tuple, set)):
        return add_named_features(X, list(feature_set))
    if feature_set == "baseline":
        return X.copy()
    if feature_set == "nb_best":
        return add_named_features(X, NB_BEST_FEATURES)
    if feature_set in FEATURE_LIBRARY:
        return add_named_features(X, [feature_set])
    if feature_set in LEGACY_FEATURE_SETS:
        return _legacy_feature(X.copy(), feature_set)
    raise ValueError(f"feature_set harus salah satu dari {FEATURE_SETS}")
