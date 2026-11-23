from __future__ import annotations
import numpy as np
import pandas as pd

MISSING_TOKEN = "__MISSING__"


class KNNPreprocessor:

    def __init__(self, binarize_cols=("prior_cancellations", "prior_completed_bookings"), cat_weight=1.0):
        if cat_weight <= 0: raise ValueError("cat_weight harus > 0")
        self.cat_weight = float(cat_weight)
        self.binarize_cols = tuple(binarize_cols)
        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.medians: dict[str, float] = {}
        self.thresholds: dict[str, float] = {}
        self.scale_cols: list[str] = []
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.categories: dict[str, list[str]] = {}
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "KNNPreprocessor":
        X = X.copy()
        self.num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.cat_cols = [c for c in X.columns if c not in self.num_cols]
        self.medians = {c: float(X[c].median()) for c in self.num_cols}
        self.thresholds = {}
        for c in self.num_cols:
            if c in self.binarize_cols:
                self.thresholds[c] = float(X[c].median())
        self.scale_cols = [c for c in self.num_cols if c not in self.thresholds]
        self.means, self.stds = {}, {}
        for c in self.scale_cols:
            col = X[c].fillna(self.medians[c])
            mean = float(col.mean())
            std = float(col.std(ddof=0))
            self.means[c] = mean
            self.stds[c] = std if std > 0 and np.isfinite(std) else 1.0
        self.categories = {
            c: sorted(X[c].dropna().astype(str).unique().tolist())
            for c in self.cat_cols
        }
        names = list(self.scale_cols) + list(self.thresholds)
        for c in self.cat_cols:
            names += [f"{c}={v}" for v in self.categories[c]]
            names += [f"{c}={MISSING_TOKEN}"]
        self.feature_names_ = names
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.feature_names_:
            raise RuntimeError("panggil fit() sebelum transform()")
        X = X.copy()
        parts: list[np.ndarray] = []
        for c in self.scale_cols:
            col = X[c].fillna(self.medians[c]).to_numpy(dtype=np.float64)
            parts.append(((col - self.means[c]) / self.stds[c])[:, None])
        for c in self.thresholds:
            col = X[c].fillna(self.medians[c]).to_numpy(dtype=np.float64)
            parts.append((col > self.thresholds[c]).astype(np.float64)[:, None])
        for c in self.cat_cols:
            col = X[c].astype("string").fillna(MISSING_TOKEN).to_numpy()
            cats = self.categories[c]
            index = {v: i for i, v in enumerate(cats)}
            block = np.zeros((len(col), len(cats) + 1), dtype=np.float64)
            for i, v in enumerate(col):
                j = index.get(v)
                block[i, j if j is not None else len(cats)] = 1.0
            if self.cat_weight != 1.0:
                block *= self.cat_weight
            parts.append(block)
        out = np.hstack(parts)
        if not np.isfinite(out).all():
            raise ValueError("hasil transformasi mengandung NaN/inf")
        return out

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "cat_weight": self.cat_weight,
            "binarize_cols": list(self.binarize_cols),
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "medians": self.medians,
            "thresholds": self.thresholds,
            "scale_cols": self.scale_cols,
            "means": self.means,
            "stds": self.stds,
            "categories": self.categories,
            "feature_names": self.feature_names_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KNNPreprocessor":
        obj = cls(binarize_cols=tuple(d.get("binarize_cols", ())),
                  cat_weight=float(d.get("cat_weight", 1.0)))
        obj.num_cols = list(d["num_cols"])
        obj.cat_cols = list(d["cat_cols"])
        obj.medians = {k: float(v) for k, v in d["medians"].items()}
        obj.thresholds = {k: float(v) for k, v in d["thresholds"].items()}
        obj.scale_cols = list(d["scale_cols"])
        obj.means = {k: float(v) for k, v in d["means"].items()}
        obj.stds = {k: float(v) for k, v in d["stds"].items()}
        obj.categories = {k: list(v) for k, v in d["categories"].items()}
        obj.feature_names_ = list(d["feature_names"])
        return obj

    def __repr__(self) -> str:
        return (
            f"KNNPreprocessor(n_fitur={len(self.feature_names_)}, "
            f"numerik={len(self.num_cols)}, kategorik={len(self.cat_cols)})"
        )
