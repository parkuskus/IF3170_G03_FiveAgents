from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def make_contract_split(train: pd.DataFrame, seed: int = 42):
    # split stratified sesuai kontrak kelompok
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
    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed
        self.medians_ = {}
        self.modes_ = {}
        self.categories_ = {}
        self.global_mean_ = 0.33
        self.target_maps_ = {}
        self.room_rate_median_ = {}
        self.channel_rate_median_ = {}
        self.base_date_ = pd.to_datetime("2023-01-01")
        self.tree_models_ = []
        self.scaler_ = StandardScaler()
        self.feature_names_ = []
        self.fitted_prep_steps = {}
        self.is_fitted_ = False

    def _build_base_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = df.copy()
        raw = pd.DataFrame(index=df.index)

        # flag missing
        for col in ["nightly_rate_idr", "special_requests", "num_children", "days_before_arrival", "meal_package", "room_category", "needs_parking"]:
            raw[f"missing_{col}"] = df[col].isna().astype(np.float64)

        # imputasi numerik
        num_cols = ["num_adults", "num_children", "weekend_nights", "weekday_nights", "days_before_arrival", "nightly_rate_idr", "special_requests"]
        for c in num_cols:
            raw[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.medians_[c]).clip(lower=0.0).astype(np.float64)

        for c in ["needs_parking", "is_returning_guest", "prior_cancellations", "prior_completed_bookings"]:
            raw[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.modes_[c]).clip(lower=0.0).astype(np.float64)

        # fitur waktu kedatangan
        dt_arr = pd.to_datetime(df["arrival_year"].astype(str) + "-" + df["arrival_month"].astype(str) + "-" + df["arrival_day"].astype(str), errors="coerce")
        raw["days_since_start"] = (dt_arr - self.base_date_).dt.days.fillna(300.0).astype(np.float64)
        raw["dayofweek"] = dt_arr.dt.dayofweek.fillna(0.0).astype(np.float64)
        raw["is_weekend_arrival"] = (raw["dayofweek"] >= 5.0).astype(np.float64)
        raw["dayofyear"] = dt_arr.dt.dayofyear.fillna(180.0).astype(np.float64)
        raw["is_month_end"] = dt_arr.dt.is_month_end.fillna(False).astype(np.float64)
        m = pd.to_numeric(df["arrival_month"], errors="coerce").fillna(6.0).to_numpy(dtype=np.float64)
        raw["month_sin"] = np.sin(2.0 * np.pi * m / 12.0)
        raw["month_cos"] = np.cos(2.0 * np.pi * m / 12.0)

        # fitur tanggal pemesanan (booking date)
        dt_book = dt_arr - pd.to_timedelta(raw["days_before_arrival"], unit="D")
        raw["book_month"] = dt_book.dt.month.fillna(6.0).astype(np.float64)
        raw["book_month_sin"] = np.sin(2.0 * np.pi * raw["book_month"] / 12.0)
        raw["book_month_cos"] = np.cos(2.0 * np.pi * raw["book_month"] / 12.0)
        raw["book_is_early"] = raw["book_month"].isin([1.0, 2.0, 3.0, 4.0]).astype(np.float64)
        raw["book_is_sep"] = (raw["book_month"] == 9.0).astype(np.float64)

        # non linear lead time
        lead = raw["days_before_arrival"]
        raw["log_lead"] = np.log1p(lead)
        raw["lead_bin_0_3"] = (lead <= 3.0).astype(np.float64)
        raw["lead_bin_4_7"] = ((lead > 3.0) & (lead <= 7.0)).astype(np.float64)
        raw["lead_bin_8_14"] = ((lead > 7.0) & (lead <= 14.0)).astype(np.float64)
        raw["lead_bin_15_30"] = ((lead > 14.0) & (lead <= 30.0)).astype(np.float64)
        raw["lead_bin_31_60"] = ((lead > 30.0) & (lead <= 60.0)).astype(np.float64)
        raw["lead_bin_61_100"] = ((lead > 60.0) & (lead <= 100.0)).astype(np.float64)
        raw["lead_bin_101_150"] = ((lead > 100.0) & (lead <= 150.0)).astype(np.float64)
        raw["lead_bin_151_210"] = ((lead > 150.0) & (lead <= 210.0)).astype(np.float64)
        raw["lead_bin_gt210"] = (lead > 210.0).astype(np.float64)

        # permintaan khusus
        sr = raw["special_requests"]
        raw["sr_0"] = (sr == 0.0).astype(np.float64)
        raw["sr_1"] = (sr == 1.0).astype(np.float64)
        raw["sr_2"] = (sr == 2.0).astype(np.float64)
        raw["sr_ge3"] = (sr >= 3.0).astype(np.float64)

        # durasi dan tamu
        wn = raw["weekend_nights"]
        wd = raw["weekday_nights"]
        tot_n = wn + wd
        raw["total_nights"] = tot_n
        raw["is_zero_nights"] = (tot_n == 0.0).astype(np.float64)
        raw["is_long_stay"] = (tot_n > 5.0).astype(np.float64)
        raw["weekend_ratio"] = wn / np.maximum(tot_n, 1.0)

        tot_g = raw["num_adults"] + raw["num_children"]
        raw["total_guests"] = tot_g
        raw["is_single"] = (tot_g == 1.0).astype(np.float64)
        raw["is_couple"] = ((raw["num_adults"] == 2.0) & (raw["num_children"] == 0.0)).astype(np.float64)
        raw["is_family"] = (raw["num_children"] > 0.0).astype(np.float64)

        # tarif
        rate = raw["nightly_rate_idr"]
        raw["log_rate"] = np.log1p(rate)
        raw["is_zero_rate"] = (rate == 0.0).astype(np.float64)
        raw["rate_per_guest_night"] = rate / (np.maximum(tot_g, 1.0) * np.maximum(tot_n, 1.0))
        raw["rate_per_guest"] = rate / np.maximum(tot_g, 1.0)
        raw["total_stay_cost"] = rate * np.maximum(tot_n, 1.0)

        r_med = df["room_category"].map(self.room_rate_median_).fillna(self.medians_["nightly_rate_idr"])
        raw["rate_rel_room"] = rate / np.maximum(r_med, 1.0)
        c_med = df["booking_channel"].map(self.channel_rate_median_).fillna(self.medians_["nightly_rate_idr"])
        raw["rate_rel_channel"] = rate / np.maximum(c_med, 1.0)

        # riwayat
        tot_prior = raw["prior_cancellations"] + raw["prior_completed_bookings"]
        raw["prior_total"] = tot_prior
        raw["cancel_history_ratio"] = raw["prior_cancellations"] / np.maximum(tot_prior, 1.0)
        raw["has_prior_cancellations"] = (raw["prior_cancellations"] > 0.0).astype(np.float64)
        raw["has_prior_completed"] = (raw["prior_completed_bookings"] > 0.0).astype(np.float64)
        raw["is_clean_returning"] = ((raw["is_returning_guest"] == 1.0) & (raw["prior_cancellations"] == 0.0)).astype(np.float64)

        # interaksi
        is_ota = (df["booking_channel"] == "Online Travel Agent").astype(np.float64)
        is_corp = (df["booking_channel"] == "Corporate").astype(np.float64)
        raw["ota_long_lead"] = is_ota * raw["lead_bin_gt210"]
        raw["ota_sr0"] = is_ota * raw["sr_0"]
        raw["corp_safe"] = is_corp * (raw["lead_bin_0_3"] + raw["lead_bin_4_7"])
        raw["sr0_long_lead"] = raw["sr_0"] * (lead > 150.0).astype(np.float64)
        raw["sr_ge2_long_lead"] = (sr >= 2.0).astype(np.float64) * (lead > 150.0).astype(np.float64)

        # one hot
        for col in ["meal_package", "room_category", "booking_channel"]:
            s = df[col].fillna("Missing").astype(str)
            for cat in self.categories_[col]:
                raw[f"{col}_{cat}"] = (s == cat).astype(np.float64)

        # kolom gabungan untuk target encoding
        keys = pd.DataFrame(index=df.index)
        keys["booking_channel"] = df["booking_channel"].fillna("Missing").astype(str)
        keys["meal_package"] = df["meal_package"].fillna("Missing").astype(str)
        keys["room_category"] = df["room_category"].fillna("Missing").astype(str)
        keys["ym"] = df["arrival_year"].astype(str) + "_" + df["arrival_month"].astype(str)

        lq = pd.cut(raw["days_before_arrival"], bins=[-1, 4, 11, 24, 39, 58, 80, 108, 151, 213, 1000], labels=False).fillna(5).astype(str)
        keys["lead_ch"] = lq + "_" + keys["booking_channel"]
        keys["lead_sr"] = lq + "_" + raw["special_requests"].astype(str)
        keys["ch_room_meal"] = keys["booking_channel"] + "_" + keys["room_category"] + "_" + keys["meal_package"]

        return raw, keys

    def fit(self, df_train: pd.DataFrame):
        df_train = df_train.copy().reset_index(drop=True)
        assert "is_canceled" in df_train.columns
        y_train = df_train["is_canceled"].to_numpy(dtype=np.int64)
        self.global_mean_ = float(np.mean(y_train))

        num_cols = ["num_adults", "num_children", "weekend_nights", "weekday_nights", "days_before_arrival", "nightly_rate_idr", "special_requests"]
        for c in num_cols:
            self.medians_[c] = float(pd.to_numeric(df_train[c], errors="coerce").median())

        for c in ["needs_parking", "is_returning_guest", "prior_cancellations", "prior_completed_bookings"]:
            mode_s = pd.to_numeric(df_train[c], errors="coerce").mode(dropna=True)
            self.modes_[c] = 0.0 if mode_s.empty else float(mode_s.iloc[0])

        for col in ["meal_package", "room_category", "booking_channel"]:
            self.categories_[col] = sorted(df_train[col].fillna("Missing").astype(str).unique().tolist())

        self.room_rate_median_ = df_train.groupby("room_category")["nightly_rate_idr"].median().to_dict()
        self.channel_rate_median_ = df_train.groupby("booking_channel")["nightly_rate_idr"].median().to_dict()

        raw_tr, keys_tr = self._build_base_dataframe(df_train)

        # target encoding oof
        te_cols = ["booking_channel", "meal_package", "room_category", "ym", "lead_ch", "lead_sr", "ch_room_meal"]
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        te_train = pd.DataFrame(index=df_train.index)
        for c in te_cols:
            smooth = 15.0
            stats = df_train.assign(_k=keys_tr[c]).groupby("_k")["is_canceled"].agg(["count", "mean"])
            mapping = ((stats["count"] * stats["mean"] + smooth * self.global_mean_) / (stats["count"] + smooth)).to_dict()
            self.target_maps_[c] = mapping

            te_series = pd.Series(index=df_train.index, dtype=np.float64)
            for f_tr_idx, f_va_idx in skf.split(df_train, y_train):
                f_tr = df_train.iloc[f_tr_idx]
                f_keys = keys_tr[c].iloc[f_tr_idx]
                f_stats = f_tr.assign(_k=f_keys).groupby("_k")["is_canceled"].agg(["count", "mean"])
                f_map = ((f_stats["count"] * f_stats["mean"] + smooth * self.global_mean_) / (f_stats["count"] + smooth)).to_dict()
                te_series.iloc[f_va_idx] = keys_tr[c].iloc[f_va_idx].map(f_map).fillna(self.global_mean_)
            te_train[f"te_{c}"] = te_series

        full_base_tr = pd.concat([raw_tr, te_train], axis=1)

        # probabilitas non linear dari pohon multi-head
        tree_configs = [
            ("hgb_std", HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=40, learning_rate=0.08, random_state=self.seed)),
            ("hgb_deep", HistGradientBoostingClassifier(max_iter=220, max_leaf_nodes=64, min_samples_leaf=15, learning_rate=0.05, random_state=self.seed + 101)),
        ]

        full_feat_tr = full_base_tr.copy()
        self.tree_models_ = []

        for tag, model in tree_configs:
            tree_oof = pd.Series(index=df_train.index, dtype=np.float64)
            for f_tr_idx, f_va_idx in skf.split(full_base_tr, y_train):
                m_fold = model.__class__(**model.get_params())
                m_fold.fit(full_base_tr.iloc[f_tr_idx], y_train[f_tr_idx])
                tree_oof.iloc[f_va_idx] = m_fold.predict_proba(full_base_tr.iloc[f_va_idx])[:, 1]

            m_full = model.__class__(**model.get_params())
            m_full.fit(full_base_tr, y_train)
            self.tree_models_.append((tag, m_full))
            full_feat_tr[f"prob_{tag}"] = tree_oof

        self.feature_names_ = full_feat_tr.columns.tolist()
        self.scaler_.fit(full_feat_tr.to_numpy(dtype=np.float64))

        self.fitted_prep_steps = {
            "medians": dict(self.medians_),
            "modes": dict(self.modes_),
            "categories": dict(self.categories_),
            "feature_names": list(self.feature_names_),
        }
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("preprocessor belum di-fit")
        df = df.copy().reset_index(drop=True)
        raw, keys = self._build_base_dataframe(df)

        te_df = pd.DataFrame(index=df.index)
        for c in ["booking_channel", "meal_package", "room_category", "ym", "lead_ch", "lead_sr", "ch_room_meal"]:
            te_df[f"te_{c}"] = keys[c].map(self.target_maps_[c]).fillna(self.global_mean_).astype(np.float64)

        full_base = pd.concat([raw, te_df], axis=1)
        base_cols = [col for col in self.feature_names_ if not col.startswith("prob_")]
        full_base = full_base[base_cols]

        full_feat = full_base.copy()
        for tag, m in self.tree_models_:
            full_feat[f"prob_{tag}"] = m.predict_proba(full_base)[:, 1]

        full_feat = full_feat[self.feature_names_]
        return self.scaler_.transform(full_feat.to_numpy(dtype=np.float64))

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        return self.fit(df_train).transform(df_train)


def load_and_prepare_split(data_dir: str | Path = "data", seed: int = 42):
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test_df = pd.read_csv(data_dir / "test.csv", dtype={"id": str})

    train_idx, valid_idx = make_contract_split(train_df, seed=seed)
    df_train = train_df.iloc[train_idx].reset_index(drop=True)
    df_valid = train_df.iloc[valid_idx].reset_index(drop=True)

    prep = SVMPreprocessor(seed=seed)
    prep.fit(df_train)

    x_train = prep.transform(df_train)
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
    x_tr = bundle["X_train"]
    x_va = bundle["X_valid"]
    x_te = bundle["X_test"]
    print("train shape:", x_tr.shape)
    print("valid shape:", x_va.shape)
    print("test shape :", x_te.shape)
    assert not np.isnan(x_tr).any()
    assert not np.isnan(x_va).any()
    assert not np.isnan(x_te).any()
    print("pipeline selesai dan bersih")
