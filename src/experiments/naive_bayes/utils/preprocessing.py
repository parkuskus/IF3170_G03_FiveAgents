from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_TOKEN = "__MISSING__"
UNKNOWN_TOKEN = "__UNKNOWN__"
MISSING_SUFFIX = "__missing"

DEFAULT_CATEGORICAL_NUMERIC = (
    "needs_parking",
    "is_returning_guest",
    "arrival_year",
    "arrival_month",
    "special_requests",
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "prior_cancellations",
)

DEFAULT_LOG1P_COLS = (
    "days_before_arrival",
    "nightly_rate_idr",
    "prior_completed_bookings",
)

DEFAULT_BIN_COLS = (
    "days_before_arrival",
    "nightly_rate_idr",
    "prior_completed_bookings",
    "arrival_day",
)

VALID_BIN_STRATEGIES = ("quantile", "uniform")


class NaiveBayesPreprocessor:
    """Preprocess mixed numeric/categorical data for NaiveBayesClassifier.

    Output column order is: numeric columns, categorical columns, then optional
    missing-indicator columns. Categorical columns are integer-coded. Unknown
    categories are mapped to a dedicated ``__UNKNOWN__`` code.

    Continuous columns listed in ``bin_cols`` are discretized using bin edges
    learned only from the training data and treated as categorical features.
    Missing values in binned columns fall into the ``__MISSING__`` category.
    """

    def __init__(
        self,
        standardize=True,
        add_missing_indicators=True,
        categorical_numeric_cols=DEFAULT_CATEGORICAL_NUMERIC,
        log1p_cols=DEFAULT_LOG1P_COLS,
        bin_cols=DEFAULT_BIN_COLS,
        n_bins=20,
        bin_strategy="quantile",
    ):
        if bin_strategy not in VALID_BIN_STRATEGIES:
            raise ValueError(
                f"bin_strategy harus salah satu dari {VALID_BIN_STRATEGIES}"
            )
        if int(n_bins) < 2:
            raise ValueError("n_bins harus >= 2")
        self.standardize = bool(standardize)
        self.add_missing_indicators = bool(add_missing_indicators)
        self.categorical_numeric_cols = tuple(categorical_numeric_cols)
        self.log1p_cols = tuple(log1p_cols)
        self.bin_cols = tuple(bin_cols)
        self.n_bins = int(n_bins)
        self.bin_strategy = bin_strategy
        self.num_cols = []
        self.cat_cols = []
        self.indicator_cols = []
        self.medians = {}
        self.means = {}
        self.stds = {}
        self.categories = {}
        self.bin_edges_ = {}
        self.numeric_indices_ = []
        self.categorical_indices_ = []
        self.n_categories_ = []
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame) -> "NaiveBayesPreprocessor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X harus pandas.DataFrame")
        X = X.copy()
        self.bin_cols_ = tuple(c for c in self.bin_cols if c in X.columns)
        self.categorical_numeric_cols = tuple(
            c
            for c in self.categorical_numeric_cols
            if c in X.columns and c not in self.bin_cols_
        )
        self.log1p_cols = tuple(
            c for c in self.log1p_cols if c in X.columns and c not in self.bin_cols_
        )
        self.num_cols = [
            c
            for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c])
            and c not in self.categorical_numeric_cols
            and c not in self.bin_cols_
        ]
        self.cat_cols = [
            c
            for c in X.columns
            if not pd.api.types.is_numeric_dtype(X[c])
            or c in self.categorical_numeric_cols
            or c in self.bin_cols_
        ]

        self.medians, self.means, self.stds = {}, {}, {}
        for c in self.num_cols:
            col = self._numeric_values(X, c)
            self.medians[c] = float(col.median())
            filled = col.fillna(self.medians[c])
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.means[c] = mean
            self.stds[c] = std if std > 0 and np.isfinite(std) else 1.0

        self.bin_edges_ = {}
        for c in self.bin_cols_:
            self.bin_edges_[c] = self._learn_bin_edges(X, c)

        self.categories = {}
        for c in self.cat_cols:
            if c in self.bin_cols_:
                values = sorted(self._binned_strings(X, c).dropna().unique().tolist())
            else:
                values = sorted(
                    self._category_values(X, c).dropna().astype(str).unique().tolist()
                )
            for token in (MISSING_TOKEN, UNKNOWN_TOKEN):
                if token not in values:
                    values.append(token)
            self.categories[c] = values

        self.indicator_cols = []
        if self.add_missing_indicators:
            self.indicator_cols = [
                f"{c}{MISSING_SUFFIX}" for c in self.num_cols if X[c].isna().any()
            ]

        self.feature_names_ = self.num_cols + self.cat_cols + self.indicator_cols
        n_num = len(self.num_cols)
        n_cat = len(self.cat_cols)
        self.numeric_indices_ = list(range(n_num))
        self.categorical_indices_ = list(
            range(n_num, n_num + n_cat + len(self.indicator_cols))
        )
        self.n_categories_ = [len(self.categories[c]) for c in self.cat_cols] + [
            2 for _ in self.indicator_cols
        ]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.feature_names_ and not self.num_cols and not self.cat_cols:
            raise RuntimeError("panggil fit() sebelum transform()")
        X = X.copy()
        required = self.num_cols + self.cat_cols
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(f"kolom tidak ditemukan: {missing}")

        parts = []
        for c in self.num_cols:
            col = (
                self._numeric_values(X, c)
                .fillna(self.medians[c])
                .to_numpy(dtype=np.float64)
            )
            if self.standardize:
                col = (col - self.means[c]) / self.stds[c]
            parts.append(col[:, None])

        for c in self.cat_cols:
            if c in self.bin_cols_:
                col = self._binned_strings(X, c)
            else:
                col = self._category_values(X, c).fillna(MISSING_TOKEN).astype(str)
            index = {v: i for i, v in enumerate(self.categories[c])}
            unknown = index[UNKNOWN_TOKEN]
            codes = np.array(
                [index.get(v, unknown) for v in col.to_numpy()], dtype=np.float64
            )
            parts.append(codes[:, None])

        for c in self.indicator_cols:
            original = c[: -len(MISSING_SUFFIX)]
            parts.append(X[original].isna().to_numpy(dtype=np.float64)[:, None])

        out = np.hstack(parts) if parts else np.empty((len(X), 0), dtype=np.float64)
        if not np.isfinite(out).all():
            raise ValueError("hasil transformasi mengandung NaN/inf")
        return out

    def _raw_numeric_values(self, X, c):
        return pd.to_numeric(X[c], errors="coerce").astype(np.float64)

    def _numeric_values(self, X, c):
        values = self._raw_numeric_values(X, c)
        if c in self.log1p_cols:
            values = np.log1p(values.clip(lower=0))
        return values

    def _learn_bin_edges(self, X, c):
        values = self._raw_numeric_values(X, c)
        valid = values.dropna().to_numpy(dtype=np.float64)
        if valid.size == 0 or float(valid.min()) == float(valid.max()):
            return np.array([-np.inf, np.inf], dtype=np.float64)
        if self.bin_strategy == "quantile":
            try:
                edges = pd.qcut(valid, q=self.n_bins, duplicates="drop", retbins=True)[
                    1
                ]
            except ValueError:
                edges = np.array([-np.inf, np.inf], dtype=np.float64)
        else:
            edges = np.linspace(float(valid.min()), float(valid.max()), self.n_bins + 1)
        edges = np.unique(np.asarray(edges, dtype=np.float64))
        if len(edges) < 2:
            return np.array([-np.inf, np.inf], dtype=np.float64)
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    def _bin_codes(self, X, c):
        values = self._raw_numeric_values(X, c)
        binned = pd.cut(
            values,
            bins=self.bin_edges_[c],
            labels=False,
            include_lowest=True,
        )
        return binned.astype("Float64")

    def _binned_strings(self, X, c):
        codes = self._bin_codes(X, c)
        return codes.map(lambda v: MISSING_TOKEN if pd.isna(v) else str(int(v)))

    def _category_values(self, X, c):
        if c in self.categorical_numeric_cols:
            return X[c].astype("Float64").astype("string")
        return X[c].astype("string")

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def feature_names(self):
        return list(self.feature_names_)

    @property
    def numeric_indices(self):
        return list(self.numeric_indices_)

    @property
    def categorical_indices(self):
        return list(self.categorical_indices_)

    @property
    def n_categories(self):
        return list(self.n_categories_)

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "standardize": self.standardize,
            "add_missing_indicators": self.add_missing_indicators,
            "categorical_numeric_cols": list(self.categorical_numeric_cols),
            "log1p_cols": list(self.log1p_cols),
            "bin_cols": list(self.bin_cols_),
            "n_bins": self.n_bins,
            "bin_strategy": self.bin_strategy,
            "bin_edges": {
                k: [float(v) for v in edges] for k, edges in self.bin_edges_.items()
            },
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "indicator_cols": self.indicator_cols,
            "medians": self.medians,
            "means": self.means,
            "stds": self.stds,
            "categories": self.categories,
            "numeric_indices": self.numeric_indices_,
            "categorical_indices": self.categorical_indices_,
            "n_categories": self.n_categories_,
            "feature_names": self.feature_names_,
            "missing_token": MISSING_TOKEN,
            "unknown_token": UNKNOWN_TOKEN,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NaiveBayesPreprocessor":
        obj = cls(
            standardize=bool(d.get("standardize", True)),
            add_missing_indicators=bool(d.get("add_missing_indicators", True)),
            categorical_numeric_cols=tuple(
                d.get("categorical_numeric_cols", DEFAULT_CATEGORICAL_NUMERIC)
            ),
            log1p_cols=tuple(d.get("log1p_cols", DEFAULT_LOG1P_COLS)),
            bin_cols=tuple(d.get("bin_cols", ())),
            n_bins=int(d.get("n_bins", 20)),
            bin_strategy=d.get("bin_strategy", "quantile"),
        )
        obj.num_cols = list(d["num_cols"])
        obj.cat_cols = list(d["cat_cols"])
        obj.indicator_cols = list(d.get("indicator_cols", []))
        obj.medians = {k: float(v) for k, v in d["medians"].items()}
        obj.means = {k: float(v) for k, v in d.get("means", {}).items()}
        obj.stds = {k: float(v) for k, v in d.get("stds", {}).items()}
        obj.categories = {k: list(v) for k, v in d["categories"].items()}
        obj.bin_cols_ = list(d.get("bin_cols", []))
        obj.bin_edges_ = {
            k: np.asarray(v, dtype=np.float64)
            for k, v in d.get("bin_edges", {}).items()
        }
        obj.numeric_indices_ = list(map(int, d["numeric_indices"]))
        obj.categorical_indices_ = list(map(int, d["categorical_indices"]))
        obj.n_categories_ = list(map(int, d["n_categories"]))
        obj.feature_names_ = list(d["feature_names"])
        return obj

    def __repr__(self):
        return (
            f"NaiveBayesPreprocessor(n_fitur={len(self.feature_names_)}, "
            f"numerik={len(self.num_cols)}, kategorik={len(self.cat_cols)}, "
            f"indikator_missing={len(self.indicator_cols)}, "
            f"binned={len(getattr(self, 'bin_cols_', []))})"
        )
