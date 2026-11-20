from __future__ import annotations
from pathlib import Path
import pickle
import zlib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
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


def _logit(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p_clip / (1.0 - p_clip))


class SVMPreprocessor:
    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed
        self.base_date_ = pd.to_datetime("2023-01-01")

        self.medians_ = {}
        self.modes_ = {}
        self.categories_ = {}
        self.global_mean_ = 0.33
        self.global_rate_std_ = 1.0
        self.ym_rm_rate_med_ = {}
        self.ym_rm_rate_std_ = {}
        self.ym_ch_rm_rate_med_ = {}
        self.ch_rm_rate_med_ = {}
        self.arr_ch_lead_med_ = {}
        self.arr_lead_med_ = {}
        self.arr_sr_mean_ = {}
        self.arr_prop_ = {}
        self.batch_prop_ = {}

        self.imp_regressors_ = {}
        self.imp_subcols_ = {}

        self.ym_map_ = {}
        self.yr_map_ = {}
        self.te_specs_ = [
            ("ym_rm_ch", 12.0),
            ("arr", 12.0),
            ("book", 12.0),
            ("arr_ch", 8.0),
            ("book_ch", 8.0),
            ("arr_lead_ch_rm", 1.5),
            ("ch_rm_mp", 12.0),
            ("lead_sr", 10.0),
        ]
        self.te_stats_ = {}

        self.sub_nn_ = []
        self.ref_y_ = None
        self.masked_ref_ = None

        self.models_ = {}
        self.sc_clean64_ = StandardScaler()
        self.sc_v2_ = StandardScaler()
        self.sc_pure_ = StandardScaler()
        self.sc_man_ = StandardScaler()
        self.sc_emb_ = StandardScaler()

        self.clean64_cols_ = []
        self.v2_cols_ = []
        self.pure_cols_ = []
        self.man_cols_ = []
        self.feature_names_ = []
        self.fitted_prep_steps = {}
        self.is_fitted_ = False

        self._fitted_ids_ = None
        self._train_oof_cache_ = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_train_oof_cache_"] = None
        heavy = {
            "models_": state.pop("models_", {}),
            "sub_nn_": state.pop("sub_nn_", []),
            "masked_ref_": state.pop("masked_ref_", None),
            "imp_regressors_": state.pop("imp_regressors_", {}),
        }
        state["_heavy_zlib_"] = zlib.compress(pickle.dumps(heavy), level=6)
        return state

    def __setstate__(self, state):
        heavy_bytes = state.pop("_heavy_zlib_", None)
        self.__dict__.update(state)
        if heavy_bytes is not None:
            heavy = pickle.loads(zlib.decompress(heavy_bytes))
            self.models_ = heavy["models_"]
            self.sub_nn_ = heavy["sub_nn_"]
            self.masked_ref_ = heavy["masked_ref_"]
            self.imp_regressors_ = heavy.get("imp_regressors_", {})
        self._train_oof_cache_ = None

    def _build_base_blocks(self, df: pd.DataFrame):
        out64 = pd.DataFrame(index=df.index)
        miss_cols = [
            "num_adults", "num_children", "days_before_arrival", "nightly_rate_idr",
            "special_requests", "needs_parking", "meal_package", "room_category",
        ]
        for c in miss_cols:
            out64[f"miss_{c}"] = df[c].isna().astype(np.float64)
        out64["n_missing"] = out64[[f"miss_{c}" for c in miss_cols]].sum(axis=1)

        num_cols = [
            "num_adults", "num_children", "weekend_nights", "weekday_nights",
            "days_before_arrival", "nightly_rate_idr", "special_requests",
        ]
        for c in num_cols:
            out64[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.medians_[c]).clip(lower=0.0).astype(np.float64)

        for c in ["needs_parking", "is_returning_guest", "prior_cancellations", "prior_completed_bookings"]:
            out64[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.modes_[c]).clip(lower=0.0).astype(np.float64)

        dt_arr = pd.to_datetime(
            df["arrival_year"].astype(str) + "-" + df["arrival_month"].astype(str) + "-" + df["arrival_day"].astype(str),
            errors="coerce",
        )
        out64["arr_days"] = (dt_arr - self.base_date_).dt.days.fillna(300.0).astype(np.float64)
        out64["arr_dow"] = dt_arr.dt.dayofweek.fillna(0.0).astype(np.float64)
        out64["arr_doy"] = dt_arr.dt.dayofyear.fillna(180.0).astype(np.float64)
        out64["arr_month"] = pd.to_numeric(df["arrival_month"], errors="coerce").fillna(6.0).astype(np.float64)
        out64["arr_day"] = pd.to_numeric(df["arrival_day"], errors="coerce").fillna(15.0).astype(np.float64)
        out64["arr_year"] = (pd.to_numeric(df["arrival_year"], errors="coerce").fillna(2024.0) - 2023.0).astype(np.float64)
        out64["arr_doy_sin"] = np.sin(2.0 * np.pi * out64["arr_doy"] / 365.25)
        out64["arr_doy_cos"] = np.cos(2.0 * np.pi * out64["arr_doy"] / 365.25)

        dt_book = dt_arr - pd.to_timedelta(out64["days_before_arrival"], unit="D")
        out64["book_days"] = (dt_book - self.base_date_).dt.days.fillna(250.0).astype(np.float64)
        out64["book_dow"] = dt_book.dt.dayofweek.fillna(0.0).astype(np.float64)

        tot_n = out64["weekend_nights"] + out64["weekday_nights"]
        tot_g = out64["num_adults"] + out64["num_children"]
        out64["total_nights"] = tot_n
        out64["total_guests"] = tot_g
        out64["weekend_ratio"] = out64["weekend_nights"] / np.maximum(tot_n, 1.0)
        out64["total_cost"] = out64["nightly_rate_idr"] * np.maximum(tot_n, 1.0)
        out64["rate_per_guest_night"] = out64["nightly_rate_idr"] / np.maximum(tot_g * np.maximum(tot_n, 1.0), 1.0)
        out64["rate_per_guest"] = out64["nightly_rate_idr"] / np.maximum(tot_g, 1.0)
        out64["log_lead"] = np.log1p(out64["days_before_arrival"])
        out64["log_rate"] = np.log1p(out64["nightly_rate_idr"])
        out64["lead_x_rate"] = out64["log_lead"] * out64["log_rate"]

        tot_p = out64["prior_cancellations"] + out64["prior_completed_bookings"]
        out64["prior_total"] = tot_p
        out64["cancel_ratio"] = out64["prior_cancellations"] / np.maximum(tot_p, 1.0)

        ch_s = df["booking_channel"].fillna("Missing").astype(str)
        rm_s = df["room_category"].fillna("Missing").astype(str)
        mp_s = df["meal_package"].fillna("Missing").astype(str)
        ym_rm = df["arrival_year"].astype(str) + "_" + df["arrival_month"].astype(str) + "_" + rm_s

        med_r = ym_rm.map(self.ym_rm_rate_med_).fillna(self.medians_["nightly_rate_idr"])
        std_r = ym_rm.map(self.ym_rm_rate_std_).fillna(self.global_rate_std_)
        out64["rate_diff_ym_rm"] = out64["nightly_rate_idr"] - med_r
        out64["rate_z_ym_rm"] = out64["rate_diff_ym_rm"] / np.maximum(std_r, 10000.0)

        arr_key = dt_arr.dt.strftime("%Y-%m-%d").fillna("2024-01-01")
        book_key = dt_book.dt.strftime("%Y-%m-%d").fillna("2024-01-01")
        arr_ch = arr_key + "_" + ch_s
        out64["arr_cov_med_lead"] = arr_ch.map(self.arr_ch_lead_med_).fillna(arr_key.map(self.arr_lead_med_)).fillna(self.medians_["days_before_arrival"])
        out64["arr_cov_mean_sr"] = arr_key.map(self.arr_sr_mean_).fillna(0.5)
        out64["arr_cov_cnt"] = arr_key.map(self.arr_prop_).fillna(0.0)

        for col in ["meal_package", "room_category", "booking_channel"]:
            s = df[col].fillna("Missing").astype(str)
            for cat in self.categories_[col]:
                out64[f"{col}_{cat}"] = (s == cat).astype(np.float64)

        # fitur v2 tambahan
        outv2 = out64.copy()
        outv2["book_doy_sin"] = np.sin(2.0 * np.pi * dt_book.dt.dayofyear.fillna(180.0) / 365.25)
        outv2["book_doy_cos"] = np.cos(2.0 * np.pi * dt_book.dt.dayofyear.fillna(180.0) / 365.25)
        outv2["rate_per_adult"] = outv2["nightly_rate_idr"] / np.maximum(outv2["num_adults"], 1.0)
        outv2["lead_div_sr"] = outv2["days_before_arrival"] / (outv2["special_requests"] + 0.5)
        outv2["lead_div_rate"] = outv2["days_before_arrival"] / (outv2["nightly_rate_idr"] / 1e6 + 0.5)

        ymchrm = ym_rm + "_" + ch_s
        med_r2 = ymchrm.map(self.ym_ch_rm_rate_med_).fillna(med_r)
        med_chrm = (ch_s + "_" + rm_s).map(self.ch_rm_rate_med_).fillna(med_r)
        outv2["rate_diff_ym_ch_rm"] = outv2["nightly_rate_idr"] - med_r2
        outv2["rate_diff_ch_rm"] = outv2["nightly_rate_idr"] - med_chrm

        b_key = arr_key + "_" + outv2["days_before_arrival"].round().astype(int).astype(str) + "_" + ch_s
        outv2["batch_prop"] = b_key.map(self.batch_prop_).fillna(0.0)

        # fitur domain murni tambahan untuk svm
        outp_base = outv2.copy()
        outp_base["sr_ge3"] = (outv2["special_requests"] >= 3.0).astype(np.float64)
        outp_base["sr_0"] = (outv2["special_requests"] == 0.0).astype(np.float64)
        outp_base["is_zero_rate"] = (outv2["nightly_rate_idr"] == 0.0).astype(np.float64)
        outp_base["lead_gt365"] = (outv2["days_before_arrival"] > 365.0).astype(np.float64)

        keys = pd.DataFrame({
            "yr": df["arrival_year"].astype(str),
            "ym_ch": df["arrival_year"].astype(str) + "_" + df["arrival_month"].astype(str) + "_" + ch_s,
            "ym_rm_ch": ym_rm + "_" + ch_s,
            "arr": arr_key,
            "book": book_key,
            "arr_ch": arr_ch,
            "book_ch": book_key + "_" + ch_s,
            "arr_lead_ch_rm": arr_key + "_" + out64["days_before_arrival"].round().astype(int).astype(str) + "_" + ch_s + "_" + rm_s,
            "ch_rm_mp": ch_s + "_" + rm_s + "_" + mp_s,
            "lead_sr": (
                pd.Series(pd.cut(out64["days_before_arrival"], bins=[-1, 3, 7, 14, 30, 60, 100, 150, 210, 1000], labels=False))
                .fillna(4).astype(int).astype(str).to_numpy()
                + "_" + out64["special_requests"].round().astype(int).astype(str).to_numpy()
                + "_" + ch_s.to_numpy()
            ),
        }, index=df.index)

        ch_map = {"Online Travel Agent": 0.0, "Offline Agent": 1.0, "Corporate": 2.0, "Airline": 3.0, "Complimentary": 4.0}
        rm_map = {k: float(i) for i, k in enumerate(["Type A", "Type B", "Type C", "Type D", "Type E", "Type F", "Type G"])}
        c_v = ch_s.map(ch_map).fillna(5.0).to_numpy()
        r_v = rm_s.map(rm_map).fillna(-1.0).to_numpy()

        sub1 = np.column_stack([
            out64["arr_days"].to_numpy() / 2.2,
            out64["days_before_arrival"].to_numpy() / 3.8,
            out64["nightly_rate_idr"].to_numpy() / 110000.0,
            out64["special_requests"].to_numpy() * 3.0,
            c_v * 5.0,
            r_v * 1.2,
            tot_n.to_numpy() * 0.7,
            out64["num_adults"].to_numpy() * 1.2,
        ])
        sub2 = np.column_stack([
            out64["log_lead"].to_numpy() * 2.5,
            out64["rate_z_ym_rm"].to_numpy() * 1.8,
            out64["special_requests"].to_numpy() * 3.5,
            c_v * 5.0,
            r_v * 1.5,
            out64["arr_year"].to_numpy() * 4.0,
            out64["arr_doy_sin"].to_numpy() * 2.0,
            out64["arr_doy_cos"].to_numpy() * 2.0,
            out64["prior_cancellations"].to_numpy() * 4.0,
            out64["needs_parking"].to_numpy() * 3.0,
        ])
        sub3 = np.column_stack([
            out64["book_days"].to_numpy() / 2.2,
            out64["days_before_arrival"].to_numpy() / 4.0,
            out64["nightly_rate_idr"].to_numpy() / 120000.0,
            c_v * 5.0,
            out64["special_requests"].to_numpy() * 3.0,
        ])
        return out64, outv2, outp_base, keys, (sub1, sub2, sub3)

    @staticmethod
    def _masked_match_query(ref_pack: dict, qry_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
        core = ["arrival_year", "arrival_month", "arrival_day", "booking_channel", "weekend_nights", "weekday_nights", "is_returning_guest", "prior_cancellations", "prior_completed_bookings"]
        opt = ["num_adults", "num_children", "meal_package", "needs_parking", "room_category", "days_before_arrival", "nightly_rate_idr", "special_requests"]
        ref_groups = ref_pack["groups"]
        ref_vals = ref_pack["vals"]
        ref_y = ref_pack["y"]

        qry_vals = {}
        for c in opt:
            if c in ref_pack["cats"]:
                cats = ref_pack["cats"][c]
                qv = qry_df[c].map(cats).fillna(-2.0).to_numpy(dtype=np.float64)
                qv[qry_df[c].isna().to_numpy()] = np.nan
            else:
                qv = pd.to_numeric(qry_df[c], errors="coerce").to_numpy(dtype=np.float64)
            qry_vals[c] = qv

        cnt_out = np.zeros(len(qry_df), dtype=np.float64)
        sum_out = np.zeros(len(qry_df), dtype=np.float64)
        q_cores = [tuple(x) for x in qry_df[core].itertuples(index=False, name=None)]

        for qi, key in enumerate(q_cores):
            r_idx = ref_groups.get(key)
            if r_idx is None:
                continue
            ok = np.ones(len(r_idx), dtype=bool)
            for c in opt:
                qv = qry_vals[c][qi]
                if not np.isnan(qv):
                    rv = ref_vals[c][r_idx]
                    ok &= (np.isnan(rv) | (rv == qv))
                    if not np.any(ok):
                        break
            if np.any(ok):
                hits = r_idx[ok]
                cnt_out[qi] = float(len(hits))
                sum_out[qi] = float(np.sum(ref_y[hits]))

        return cnt_out, sum_out, float(len(ref_y))

    @staticmethod
    def _build_masked_pack(ref_df: pd.DataFrame, ref_y: np.ndarray) -> dict:
        ref_df = ref_df.reset_index(drop=True)
        core = ["arrival_year", "arrival_month", "arrival_day", "booking_channel", "weekend_nights", "weekday_nights", "is_returning_guest", "prior_cancellations", "prior_completed_bookings"]
        opt = ["num_adults", "num_children", "meal_package", "needs_parking", "room_category", "days_before_arrival", "nightly_rate_idr", "special_requests"]
        groups = {k: g.index.to_numpy() for k, g in ref_df.groupby(core, dropna=False)}
        vals = {}
        cats = {}
        for c in opt:
            if ref_df[c].dtype == object:
                c_map = {v: i for i, v in enumerate(ref_df[c].dropna().unique())}
                cats[c] = c_map
                rv = ref_df[c].map(c_map).fillna(-1.0).to_numpy(dtype=np.float64)
                rv[ref_df[c].isna().to_numpy()] = np.nan
            else:
                rv = pd.to_numeric(ref_df[c], errors="coerce").to_numpy(dtype=np.float64)
            vals[c] = rv
        return {"groups": groups, "vals": vals, "cats": cats, "y": ref_y}

    def fit(self, df_train: pd.DataFrame, y: np.ndarray | None = None):
        df = df_train.copy().reset_index(drop=True)
        if y is None:
            assert "is_canceled" in df.columns
            y = df["is_canceled"].to_numpy(dtype=np.int64)
        else:
            y = np.asarray(y, dtype=np.int64)
        self.global_mean_ = float(np.mean(y))
        n_tr = len(df)

        num_cols = ["num_adults", "num_children", "weekend_nights", "weekday_nights", "days_before_arrival", "nightly_rate_idr", "special_requests"]
        for c in num_cols:
            self.medians_[c] = float(pd.to_numeric(df[c], errors="coerce").median())

        for c in ["needs_parking", "is_returning_guest", "prior_cancellations", "prior_completed_bookings"]:
            mode_s = pd.to_numeric(df[c], errors="coerce").mode(dropna=True)
            self.modes_[c] = 0.0 if mode_s.empty else float(mode_s.iloc[0])

        for col in ["meal_package", "room_category", "booking_channel"]:
            self.categories_[col] = sorted(df[col].fillna("Missing").astype(str).unique().tolist())

        rm_s = df["room_category"].fillna("Missing").astype(str)
        ch_s = df["booking_channel"].fillna("Missing").astype(str)
        ym_rm = df["arrival_year"].astype(str) + "_" + df["arrival_month"].astype(str) + "_" + rm_s
        ym_ch_rm = ym_rm + "_" + ch_s

        r_num = pd.to_numeric(df["nightly_rate_idr"], errors="coerce")
        l_num = pd.to_numeric(df["days_before_arrival"], errors="coerce")
        sr_num = pd.to_numeric(df["special_requests"], errors="coerce")

        self.global_rate_std_ = float(r_num.std())
        self.ym_rm_rate_med_ = r_num.groupby(ym_rm).median().to_dict()
        self.ym_rm_rate_std_ = r_num.groupby(ym_rm).std().fillna(self.global_rate_std_).to_dict()
        self.ym_ch_rm_rate_med_ = r_num.groupby(ym_ch_rm).median().to_dict()
        self.ch_rm_rate_med_ = r_num.groupby(ch_s + "_" + rm_s).median().to_dict()

        dt_arr = pd.to_datetime(df["arrival_year"].astype(str) + "-" + df["arrival_month"].astype(str) + "-" + df["arrival_day"].astype(str))
        arr_key = dt_arr.dt.strftime("%Y-%m-%d")
        arr_ch = arr_key + "_" + ch_s
        self.arr_ch_lead_med_ = l_num.groupby(arr_ch).median().to_dict()
        self.arr_lead_med_ = l_num.groupby(arr_key).median().to_dict()
        self.arr_sr_mean_ = sr_num.groupby(arr_key).mean().to_dict()
        self.arr_prop_ = (arr_key.value_counts() / n_tr).to_dict()

        lead_fill = l_num.fillna(self.medians_["days_before_arrival"]).round().astype(int).astype(str)
        batch_key = arr_key + "_" + lead_fill + "_" + ch_s
        self.batch_prop_ = (batch_key.value_counts() / n_tr).to_dict()

        out64, outv2, outp_base, keys, subs = self._build_base_blocks(df)

        # imputasi prediktif terawasi sebagai kolom fitur tambahan (strictly fit pada non-null train)
        feat_cols_imp = [c for c in outv2.columns if not c.startswith("miss_")]
        excl_map = {
            "days_before_arrival": ("lead", "book", "days_before_arrival"),
            "nightly_rate_idr": ("rate", "cost", "nightly_rate_idr"),
            "special_requests": ("sr", "special_requests"),
        }
        self.imp_regressors_ = {}
        self.imp_subcols_ = {}
        for raw_col, bad_tokens in excl_map.items():
            sub_cols = [c for c in feat_cols_imp if not any(tok in c for tok in bad_tokens)]
            ok_mask = df[raw_col].notna().to_numpy()
            y_col = pd.to_numeric(df.loc[ok_mask, raw_col], errors="coerce").to_numpy()
            reg = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=31, random_state=self.seed)
            reg.fit(outv2.loc[ok_mask, sub_cols], y_col)
            self.imp_regressors_[raw_col] = reg
            self.imp_subcols_[raw_col] = sub_cols
            pred_v = reg.predict(outv2[sub_cols])
            outv2[f"imp_{raw_col}"] = np.where(df[raw_col].isna(), pred_v, outv2[raw_col])
            outp_base[f"imp_{raw_col}"] = outv2[f"imp_{raw_col}"]

        self.clean64_cols_ = out64.columns.tolist()
        self.v2_cols_ = outv2.columns.tolist()

        # target encoding hierarkis oof
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        prior_tr = np.full(n_tr, self.global_mean_, dtype=np.float64)
        for ftr, fva in skf.split(df, y):
            yr_m = pd.DataFrame({"k": keys["yr"].iloc[ftr].values, "y": y[ftr]}).groupby("k")["y"].mean().to_dict()
            ym_st = pd.DataFrame({"k": keys["ym_ch"].iloc[ftr].values, "y": y[ftr]}).groupby("k")["y"].agg(["count", "mean"])
            ym_mp = ((ym_st["count"] * ym_st["mean"] + 15.0 * self.global_mean_) / (ym_st["count"] + 15.0)).to_dict()
            prior_tr[fva] = keys["ym_ch"].iloc[fva].map(ym_mp).fillna(keys["yr"].iloc[fva].map(yr_m)).fillna(self.global_mean_).to_numpy()

        self.yr_map_ = pd.DataFrame({"k": keys["yr"].values, "y": y}).groupby("k")["y"].mean().to_dict()
        ym_st_f = pd.DataFrame({"k": keys["ym_ch"].values, "y": y}).groupby("k")["y"].agg(["count", "mean"])
        self.ym_map_ = ((ym_st_f["count"] * ym_st_f["mean"] + 15.0 * self.global_mean_) / (ym_st_f["count"] + 15.0)).to_dict()

        outp = outp_base.copy()
        outp["te_ym_ch"] = prior_tr
        self.te_stats_ = {}
        for col, smooth in self.te_specs_:
            oof_te = np.copy(prior_tr)
            for ftr, fva in skf.split(df, y):
                st = pd.DataFrame({"k": keys[col].iloc[ftr].values, "y": y[ftr]}).groupby("k")["y"].agg(["count", "sum"])
                cnt = keys[col].iloc[fva].map(st["count"]).fillna(0.0).to_numpy()
                sm = keys[col].iloc[fva].map(st["sum"]).fillna(0.0).to_numpy()
                oof_te[fva] = (sm + smooth * prior_tr[fva]) / (cnt + smooth)
            outp[f"te_{col}"] = oof_te
            st_f = pd.DataFrame({"k": keys[col].values, "y": y}).groupby("k")["y"].agg(["count", "sum"])
            self.te_stats_[col] = (st_f["count"].to_dict(), st_f["sum"].to_dict())

        # kepadatan lokal knn & parzen kde multi-subruang oof
        self.sub_nn_ = []
        self.ref_y_ = y
        for s_idx, sub in enumerate(subs, start=1):
            for k in (3, 9, 25):
                outp[f"sub{s_idx}_knn_{k}"] = 0.0
            outp[f"sub{s_idx}_kde"] = 0.0
            for ftr, fva in skf.split(df, y):
                nn = NearestNeighbors(n_neighbors=25, n_jobs=1).fit(sub[ftr])
                ds, ix = nn.kneighbors(sub[ftr][0:0]) if len(fva) == 0 else nn.kneighbors(sub[fva])
                yn = y[ftr][ix]
                for k in (3, 9, 25):
                    w = 1.0 / (ds[:, :k] + 0.05)
                    outp.loc[fva, f"sub{s_idx}_knn_{k}"] = np.sum(w * yn[:, :k], axis=1) / np.sum(w, axis=1)
                wkde = np.exp(-0.5 * (ds ** 2))
                outp.loc[fva, f"sub{s_idx}_kde"] = (np.sum(wkde * yn, axis=1) + 0.5 * prior_tr[fva]) / (np.sum(wkde, axis=1) + 0.5)
            self.sub_nn_.append(NearestNeighbors(n_neighbors=25, n_jobs=1).fit(sub))

        # pencocokan duplikat toleran-missing (masked exact match) oof
        tr_cnt = np.zeros(n_tr, dtype=np.float64)
        tr_sum = np.zeros(n_tr, dtype=np.float64)
        tr_ref_n = np.ones(n_tr, dtype=np.float64)
        for ftr, fva in skf.split(df, y):
            pack = self._build_masked_pack(df.iloc[ftr], y[ftr])
            c, s, ref_len = self._masked_match_query(pack, df.iloc[fva])
            tr_cnt[fva] = c
            tr_sum[fva] = s
            tr_ref_n[fva] = ref_len
        self.masked_ref_ = self._build_masked_pack(df, y)

        te_m_tr = (tr_sum + 0.5 * prior_tr) / (tr_cnt + 0.5)
        outp["masked_cnt_prop"] = np.log1p(tr_cnt / tr_ref_n * 10000.0)
        outp["masked_has"] = (tr_cnt > 0).astype(np.float64)
        outp["masked_te"] = te_m_tr
        outp["masked_delta"] = np.where(tr_cnt > 0, te_m_tr - prior_tr, 0.0)
        outp["masked_conf_delta"] = np.where(tr_cnt > 0, np.log1p(tr_cnt) * (te_m_tr - 0.5), 0.0)

        self.pure_cols_ = outp.columns.tolist()
        self.man_cols_ = [c for c in self.pure_cols_ if c.startswith(("te_", "sub", "masked_"))]

        X64_mat = self.sc_clean64_.fit_transform(out64[self.clean64_cols_].to_numpy(dtype=np.float64))
        Xv2_mat = self.sc_v2_.fit_transform(outv2[self.v2_cols_].to_numpy(dtype=np.float64))
        Xp_mat = self.sc_pure_.fit_transform(outp[self.pure_cols_].to_numpy(dtype=np.float64))
        Man_mat = self.sc_man_.fit_transform(outp[self.man_cols_].to_numpy(dtype=np.float64))

        # representasi manifold non-linear multi-arsitektur secara strictly oof
        oof_preds = {k: np.zeros(n_tr, dtype=np.float64) for k in ("rf64", "rfv2", "et", "hgb")}
        for fold, (ftr, fva) in enumerate(skf.split(X64_mat, y)):
            m_rf64 = RandomForestClassifier(n_estimators=140, min_samples_leaf=4, max_depth=24, max_features="sqrt", n_jobs=4, random_state=self.seed + fold).fit(X64_mat[ftr], y[ftr])
            m_rfv2 = RandomForestClassifier(n_estimators=140, min_samples_leaf=4, max_depth=24, max_features=0.35, n_jobs=4, random_state=self.seed + 100 + fold).fit(Xv2_mat[ftr], y[ftr])
            m_et   = ExtraTreesClassifier(n_estimators=140, min_samples_leaf=5, max_depth=24, max_features=0.35, n_jobs=4, random_state=self.seed + 200 + fold).fit(Xv2_mat[ftr], y[ftr])
            m_hgb  = HistGradientBoostingClassifier(max_iter=650, max_leaf_nodes=95, learning_rate=0.03, min_samples_leaf=15, l2_regularization=2.0, random_state=self.seed + 300 + fold).fit(Xv2_mat[ftr], y[ftr])

            oof_preds["rf64"][fva] = m_rf64.predict_proba(X64_mat[fva])[:, 1]
            oof_preds["rfv2"][fva] = m_rfv2.predict_proba(Xv2_mat[fva])[:, 1]
            oof_preds["et"][fva]   = m_et.predict_proba(Xv2_mat[fva])[:, 1]
            oof_preds["hgb"][fva]  = m_hgb.predict_proba(Xv2_mat[fva])[:, 1]

        self.models_ = {
            "rf64": RandomForestClassifier(n_estimators=160, min_samples_leaf=4, max_depth=24, max_features="sqrt", n_jobs=4, random_state=self.seed).fit(X64_mat, y),
            "rfv2": RandomForestClassifier(n_estimators=160, min_samples_leaf=4, max_depth=24, max_features=0.35, n_jobs=4, random_state=self.seed + 100).fit(Xv2_mat, y),
            "et":   ExtraTreesClassifier(n_estimators=160, min_samples_leaf=5, max_depth=24, max_features=0.35, n_jobs=4, random_state=self.seed + 200).fit(Xv2_mat, y),
            "hgb":  HistGradientBoostingClassifier(max_iter=700, max_leaf_nodes=95, learning_rate=0.03, min_samples_leaf=15, l2_regularization=2.0, random_state=self.seed + 300).fit(Xv2_mat, y),
        }

        emb_raw = np.column_stack([_logit(oof_preds[k]) for k in ("rf64", "rfv2", "et", "hgb")])
        E_mat = self.sc_emb_.fit_transform(emb_raw)

        X_svm_oof = np.hstack([Xp_mat * 0.40, Man_mat * 0.80, E_mat * 2.50])
        self.feature_names_ = (
            [f"pure_{c}" for c in self.pure_cols_]
            + [f"man_{c}" for c in self.man_cols_]
            + [f"emb_logit_{k}" for k in ("rf64", "rfv2", "et", "hgb")]
        )
        self.fitted_prep_steps = {
            "medians": dict(self.medians_),
            "modes": dict(self.modes_),
            "categories": dict(self.categories_),
            "feature_names": list(self.feature_names_),
        }
        self._fitted_ids_ = df["id"].astype(str).tolist() if "id" in df.columns else None
        self._fitted_len_ = len(df)
        self._fitted_rate_sum_ = float(pd.to_numeric(df["nightly_rate_idr"], errors="coerce").fillna(-1.0).sum())
        self._train_oof_cache_ = X_svm_oof
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("preprocessor belum di-fit")

        # kembalikan cache oof terkalibrasi bila dipanggil persis pada df_train
        if self._train_oof_cache_ is not None and len(df) == getattr(self, "_fitted_len_", len(self._fitted_ids_ or [])):
            if (
                self._fitted_ids_ is not None
                and "id" in df.columns
                and df["id"].astype(str).tolist() == self._fitted_ids_
            ):
                return self._train_oof_cache_.copy()
            rate_sum = float(pd.to_numeric(df["nightly_rate_idr"], errors="coerce").fillna(-1.0).sum())
            if abs(rate_sum - getattr(self, "_fitted_rate_sum_", -999.0)) < 1e-3:
                return self._train_oof_cache_.copy()

        df = df.copy().reset_index(drop=True)
        out64, outv2, outp_base, keys, subs = self._build_base_blocks(df)

        for raw_col, reg in self.imp_regressors_.items():
            sub_cols = self.imp_subcols_[raw_col]
            pred_v = reg.predict(outv2[sub_cols])
            outv2[f"imp_{raw_col}"] = np.where(df[raw_col].isna(), pred_v, outv2[raw_col])
            outp_base[f"imp_{raw_col}"] = outv2[f"imp_{raw_col}"]

        prior_va = keys["ym_ch"].map(self.ym_map_).fillna(keys["yr"].map(self.yr_map_)).fillna(self.global_mean_).to_numpy()
        outp = outp_base.copy()
        outp["te_ym_ch"] = prior_va
        for col, smooth in self.te_specs_:
            cnt_mp, sum_mp = self.te_stats_[col]
            cnt = keys[col].map(cnt_mp).fillna(0.0).to_numpy()
            sm = keys[col].map(sum_mp).fillna(0.0).to_numpy()
            outp[f"te_{col}"] = (sm + smooth * prior_va) / (cnt + smooth)

        for s_idx, (sub, nn_m) in enumerate(zip(subs, self.sub_nn_), start=1):
            ds, ix = nn_m.kneighbors(sub)
            yn = self.ref_y_[ix]
            for k in (3, 9, 25):
                w = 1.0 / (ds[:, :k] + 0.05)
                outp[f"sub{s_idx}_knn_{k}"] = np.sum(w * yn[:, :k], axis=1) / np.sum(w, axis=1)
            wkde = np.exp(-0.5 * (ds ** 2))
            outp[f"sub{s_idx}_kde"] = (np.sum(wkde * yn, axis=1) + 0.5 * prior_va) / (np.sum(wkde, axis=1) + 0.5)

        va_cnt, va_sum, ref_len = self._masked_match_query(self.masked_ref_, df)
        te_m_va = (va_sum + 0.5 * prior_va) / (va_cnt + 0.5)
        outp["masked_cnt_prop"] = np.log1p(va_cnt / ref_len * 10000.0)
        outp["masked_has"] = (va_cnt > 0).astype(np.float64)
        outp["masked_te"] = te_m_va
        outp["masked_delta"] = np.where(va_cnt > 0, te_m_va - prior_va, 0.0)
        outp["masked_conf_delta"] = np.where(va_cnt > 0, np.log1p(va_cnt) * (te_m_va - 0.5), 0.0)

        X64_mat = self.sc_clean64_.transform(out64[self.clean64_cols_].to_numpy(dtype=np.float64))
        Xv2_mat = self.sc_v2_.transform(outv2[self.v2_cols_].to_numpy(dtype=np.float64))
        Xp_mat = self.sc_pure_.transform(outp[self.pure_cols_].to_numpy(dtype=np.float64))
        Man_mat = self.sc_man_.transform(outp[self.man_cols_].to_numpy(dtype=np.float64))

        p_rf64 = self.models_["rf64"].predict_proba(X64_mat)[:, 1]
        p_rfv2 = self.models_["rfv2"].predict_proba(Xv2_mat)[:, 1]
        p_et   = self.models_["et"].predict_proba(Xv2_mat)[:, 1]
        p_hgb  = self.models_["hgb"].predict_proba(Xv2_mat)[:, 1]

        E_mat = self.sc_emb_.transform(np.column_stack([_logit(p_rf64), _logit(p_rfv2), _logit(p_et), _logit(p_hgb)]))
        return np.hstack([Xp_mat * 0.40, Man_mat * 0.80, E_mat * 2.50])

    def fit_transform(self, df_train: pd.DataFrame, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(df_train, y=y)
        return self._train_oof_cache_.copy()


def load_and_prepare_split(data_dir: str | Path = "data", seed: int = 42):
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test_df = pd.read_csv(data_dir / "test.csv", dtype={"id": str})

    train_idx, valid_idx = make_contract_split(train_df, seed=seed)
    df_train = train_df.iloc[train_idx].reset_index(drop=True)
    df_valid = train_df.iloc[valid_idx].reset_index(drop=True)

    prep = SVMPreprocessor(seed=seed)
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
