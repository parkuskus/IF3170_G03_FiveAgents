from pathlib import Path
import numpy as np
import pandas as pd


NUMERIC_COLS = [
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "days_before_arrival",
    "arrival_year",
    "arrival_month",
    "arrival_day",
    "prior_cancellations",
    "prior_completed_bookings",
    "nightly_rate_idr",
    "special_requests",
]

BINARY_COLS = [
    "needs_parking",
    "is_returning_guest",
]

CATEGORICAL_COLS = [
    "meal_package",
    "room_category",
    "booking_channel",
]

MISSING_FLAG_COLS = [
    "num_adults",
    "num_children",
    "meal_package",
    "needs_parking",
    "room_category",
    "days_before_arrival",
    "nightly_rate_idr",
    "special_requests",
]


def make_contract_split(train: pd.DataFrame, seed: int = 42):
    # split sesuai kontrak readme
    y = train["is_canceled"].to_numpy()
    assert set(np.unique(y)) == {0, 1}
    rng = np.random.default_rng(seed)
    valid_idx = np.sort(np.concatenate([
        rng.permutation(np.flatnonzero(y == label))[:int(np.ceil(0.2 * np.sum(y == label)))]
        for label in (0, 1)
    ]))
    train_idx = np.setdiff1d(np.arange(len(train)), valid_idx)
    assert len(train_idx) + len(valid_idx) == len(train)
    return train_idx, valid_idx


class SVMPreprocessor:
    def __init__(self, add_missing_flags: bool = True, scale_all: bool = True):
        self.add_missing_flags = add_missing_flags
        self.scale_all = scale_all
        self.medians_ = {}
        self.modes_ = {}
        self.categories_ = {}
        self.feature_names_ = []
        self.mean_ = None
        self.std_ = None
        self.fitted_prep_steps = {}
        self.is_fitted_ = False

    def _build_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        if self.add_missing_flags:
            for col in MISSING_FLAG_COLS:
                out[f"missing_{col}"] = df[col].isna().astype(np.float64)

        for col in NUMERIC_COLS:
            s = pd.to_numeric(df[col], errors="coerce").fillna(self.medians_[col])
            if col in ("days_before_arrival", "nightly_rate_idr", "num_adults", "num_children"):
                s = s.clip(lower=0.0)
            out[col] = s.astype(np.float64)

        for col in BINARY_COLS:
            s = pd.to_numeric(df[col], errors="coerce").fillna(self.modes_[col])
            out[col] = s.clip(0.0, 1.0).astype(np.float64)

        # fitur turunan
        total_nights = out["weekend_nights"] + out["weekday_nights"]
        total_guests = out["num_adults"] + out["num_children"]
        guest_nights = np.maximum(total_guests, 1.0) * np.maximum(total_nights, 1.0)
        prior_total = out["prior_cancellations"] + out["prior_completed_bookings"]

        out["total_nights"] = total_nights
        out["total_guests"] = total_guests
        out["rate_per_guest_night"] = out["nightly_rate_idr"] / guest_nights
        out["rate_per_guest"] = out["nightly_rate_idr"] / np.maximum(total_guests, 1.0)
        out["weekend_ratio"] = out["weekend_nights"] / np.maximum(total_nights, 1.0)
        out["prior_total_bookings"] = prior_total
        out["historical_cancel_ratio"] = out["prior_cancellations"] / np.maximum(prior_total, 1.0)
        out["log_nightly_rate"] = np.log1p(out["nightly_rate_idr"])
        out["log_days_before_arrival"] = np.log1p(out["days_before_arrival"])
        out["arrival_month_sin"] = np.sin(2.0 * np.pi * out["arrival_month"] / 12.0)
        out["arrival_month_cos"] = np.cos(2.0 * np.pi * out["arrival_month"] / 12.0)

        for col in CATEGORICAL_COLS:
            s = df[col].fillna("Missing").astype(str)
            for cat in self.categories_[col]:
                out[f"{col}_{cat}"] = (s == cat).astype(np.float64)

        return out

    def fit(self, df: pd.DataFrame):
        for col in NUMERIC_COLS:
            val = pd.to_numeric(df[col], errors="coerce").median()
            self.medians_[col] = 0.0 if pd.isna(val) else float(val)

        for col in BINARY_COLS:
            mode_s = pd.to_numeric(df[col], errors="coerce").mode(dropna=True)
            self.modes_[col] = 0.0 if mode_s.empty else float(mode_s.iloc[0])

        for col in CATEGORICAL_COLS:
            cats = sorted(df[col].fillna("Missing").astype(str).unique().tolist())
            self.categories_[col] = cats

        feat_df = self._build_raw_features(df)
        self.feature_names_ = feat_df.columns.tolist()

        x = feat_df.to_numpy(dtype=np.float64)
        self.mean_ = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        self.std_ = np.where(std > 1e-8, std, 1.0)

        if not self.scale_all:
            # hanya scale kolom kontinu jika scale_all dimatikan
            for idx, name in enumerate(self.feature_names_):
                if name.startswith(("missing_", "meal_package_", "room_category_", "booking_channel_")) or name in BINARY_COLS:
                    self.mean_[idx] = 0.0
                    self.std_[idx] = 1.0

        self.fitted_prep_steps = {
            "medians": dict(self.medians_),
            "modes": dict(self.modes_),
            "categories": dict(self.categories_),
            "feature_names": list(self.feature_names_),
            "mean": self.mean_.copy(),
            "std": self.std_.copy(),
        }
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("preprocessor belum di-fit")
        feat_df = self._build_raw_features(df)
        feat_df = feat_df[self.feature_names_]
        x = feat_df.to_numpy(dtype=np.float64)
        return (x - self.mean_) / self.std_

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


def load_and_prepare_split(data_dir: str | Path = "data", seed: int = 42):
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test_df = pd.read_csv(data_dir / "test.csv", dtype={"id": str})

    train_idx, valid_idx = make_contract_split(train_df, seed=seed)
    df_train = train_df.iloc[train_idx].reset_index(drop=True)
    df_valid = train_df.iloc[valid_idx].reset_index(drop=True)

    prep = SVMPreprocessor()
    x_train = prep.fit_transform(df_train)
    x_valid = prep.transform(df_valid)
    x_test = prep.transform(test_df)

    y_train = df_train["is_canceled"].to_numpy(dtype=np.int64)
    y_valid = df_valid["is_canceled"].to_numpy(dtype=np.int64)

    return {
        "preprocessor": prep,
        "X_train": x_train,
        "y_train": y_train,
        "X_valid": x_valid,
        "y_valid": y_valid,
        "X_test": x_test,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_ids": test_df["id"].to_numpy(),
    }


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    bundle = load_and_prepare_split(repo_root / "data", seed=42)

    x_train = bundle["X_train"]
    x_valid = bundle["X_valid"]
    x_test = bundle["X_test"]
    y_train = bundle["y_train"]
    y_valid = bundle["y_valid"]
    prep = bundle["preprocessor"]

    assert not np.isnan(x_train).any() and not np.isinf(x_train).any()
    assert not np.isnan(x_valid).any() and not np.isinf(x_valid).any()
    assert not np.isnan(x_test).any() and not np.isinf(x_test).any()

    # cek tidak ada data leakage dari validasi
    train_df = pd.read_csv(repo_root / "data" / "train.csv", dtype={"id": str})
    train_idx = bundle["train_idx"]
    prep_check = SVMPreprocessor()
    x_train_check = prep_check.fit_transform(train_df.iloc[train_idx])
    assert np.allclose(x_train, x_train_check)

    print(f"train shape : {x_train.shape}, class counts: {np.bincount(y_train).tolist()}")
    print(f"valid shape : {x_valid.shape}, class counts: {np.bincount(y_valid).tolist()}")
    print(f"test shape  : {x_test.shape}")
    print(f"fitur total : {len(prep.feature_names_)}")
    print(f"train mean  : {np.abs(x_train.mean(axis=0)).max():.2e}, std min/max: {x_train.std(axis=0).min():.4f}/{x_train.std(axis=0).max():.4f}")
