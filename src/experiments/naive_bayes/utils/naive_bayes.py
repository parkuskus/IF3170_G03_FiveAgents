from __future__ import annotations

import numpy as np

EPS = 1e-12
VALID_MODES = ("baseline", "bonus")
VALID_VARIANTS = ("categorical", "hybrid")


class NaiveBayesClassifier:
    """Naive Bayes *from scratch* (Bayesian Network).

    - ``mode="baseline"`` (wajib, default): murni kategorikal (seluruh fitur
      numerik wajib didiskritisasi terlebih dahulu), Laplace smoothing ``+1``
      (``alpha=1.0``), tanpa bobot fitur, dan prediksi MAP (``threshold=0.5``).
      Menyimpan tabel frekuensi ``class_count_``, ``cat_counts_``, prior
      ``class_prior_`` ($P(v_j)$), dan likelihood ``cat_probs_`` ($P(a_i | v_j)$).
    - ``mode="bonus"``: mengizinkan ekstensi di luar slide referensi seperti
      ``variant="hybrid"`` (Gaussian untuk numerik), ``alpha != 1.0``,
      ``feature_weights``, dan ``threshold != 0.5``.
    """

    def __init__(
        self,
        mode="baseline",
        variant=None,
        alpha=1.0,
        var_smoothing=1e-9,
        threshold=0.5,
        numeric_indices=None,
        categorical_indices=None,
        n_categories=None,
        feature_weights=None,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode harus salah satu dari {VALID_MODES}")
        if variant is None:
            variant = "categorical" if mode == "baseline" else "hybrid"
        if variant not in VALID_VARIANTS:
            raise ValueError(f"variant harus salah satu dari {VALID_VARIANTS}")
        if alpha <= 0:
            raise ValueError("alpha harus > 0")
        if var_smoothing < 0:
            raise ValueError("var_smoothing harus >= 0")
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold harus di (0, 1)")

        if mode == "baseline":
            if variant != "categorical":
                raise ValueError(
                    "mode='baseline' tidak mengizinkan Gaussian/hybrid; "
                    "gunakan mode='bonus' untuk variant='hybrid'"
                )
            if not np.isclose(float(alpha), 1.0):
                raise ValueError(
                    "mode='baseline' wajib Laplace smoothing +1 (alpha=1.0); "
                    "gunakan mode='bonus' untuk variasi smoothing"
                )
            if not np.isclose(float(threshold), 0.5):
                raise ValueError(
                    "mode='baseline' wajib MAP murni (threshold=0.5); "
                    "gunakan mode='bonus' untuk threshold tuning"
                )
            if feature_weights is not None:
                raise ValueError(
                    "mode='baseline' tidak mengizinkan feature_weights; "
                    "gunakan mode='bonus'"
                )

        self.mode = mode
        self.variant = variant
        self.alpha = float(alpha)
        self.var_smoothing = float(var_smoothing)
        self.threshold = float(threshold)
        self.feature_weights = (
            None
            if feature_weights is None
            else tuple(float(w) for w in feature_weights)
        )
        self.numeric_indices = (
            None if numeric_indices is None else tuple(int(i) for i in numeric_indices)
        )
        self.categorical_indices = (
            None
            if categorical_indices is None
            else tuple(int(i) for i in categorical_indices)
        )
        self.n_categories = (
            None if n_categories is None else tuple(int(n) for n in n_categories)
        )

        self.classes_ = None
        self.class_count_ = None
        self.class_prior_ = None
        self.class_log_prior_ = None
        self.theta_ = None
        self.var_ = None
        self.cat_counts_ = None
        self.cat_probs_ = None
        self.numeric_indices_ = None
        self.categorical_indices_ = None
        self.n_categories_ = None
        self.n_features_ = None

    def _resolve_indices(self, n_features):
        cat = (
            np.array([], dtype=int)
            if self.categorical_indices is None
            else np.asarray(self.categorical_indices, dtype=int)
        )
        if self.numeric_indices is None:
            num = np.setdiff1d(np.arange(n_features), cat)
        else:
            num = np.asarray(self.numeric_indices, dtype=int)

        if len(np.intersect1d(num, cat)) > 0:
            raise ValueError(
                "numeric_indices dan categorical_indices tidak boleh beririsan"
            )
        combined = np.sort(np.concatenate([num, cat]))
        if not np.array_equal(combined, np.arange(n_features)):
            raise ValueError(
                "numeric_indices + categorical_indices harus mencakup semua fitur"
            )

        if len(cat) > 0:
            if self.n_categories is None:
                raise ValueError(
                    "n_categories wajib diisi jika ada categorical_indices"
                )
            if len(self.n_categories) != len(cat):
                raise ValueError(
                    "panjang n_categories harus sama dengan categorical_indices"
                )
            n_categories = np.asarray(self.n_categories, dtype=int)
            if np.any(n_categories < 1):
                raise ValueError("setiap nilai n_categories harus >= 1")
        else:
            n_categories = np.array([], dtype=int)

        return num, cat, n_categories

    @staticmethod
    def _check_X(X, n_features):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X harus 2 dimensi")
        if X.shape[1] != n_features:
            raise ValueError("jumlah fitur X tidak sesuai dengan model")
        if not np.isfinite(X).all():
            raise ValueError("X mengandung NaN/inf")
        return X

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(np.int64).ravel()
        if X.ndim != 2:
            raise ValueError("X harus 2 dimensi")
        if len(X) != len(y):
            raise ValueError("panjang X dan y harus sama")
        if not np.isfinite(X).all():
            raise ValueError("X mengandung NaN/inf")
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("label harus 0 atau 1")

        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.n_features_ = X.shape[1]
        if self.feature_weights is not None:
            if len(self.feature_weights) != self.n_features_:
                raise ValueError(
                    "panjang feature_weights harus sama dengan jumlah fitur X"
                )
            if any((not np.isfinite(w)) or w < 0.0 for w in self.feature_weights):
                raise ValueError("feature_weights harus berisi nilai finite dan >= 0")
        num, cat, n_categories = self._resolve_indices(self.n_features_)
        if self.mode == "baseline" and len(num) > 0:
            raise ValueError(
                "mode='baseline' hanya menerima fitur kategorikal hasil diskritisasi; "
                "tidak boleh ada fitur numerik (Gaussian)"
            )
        if self.variant == "categorical" and len(num) > 0:
            raise ValueError(
                "variant='categorical' tidak menerima fitur numerik; diskritisasi terlebih dahulu"
            )
        self.numeric_indices_ = num
        self.categorical_indices_ = cat
        self.n_categories_ = n_categories

        self.class_count_ = np.array(
            [int(np.sum(y == c)) for c in self.classes_], dtype=np.float64
        )
        if np.any(self.class_count_ == 0):
            raise ValueError("kedua kelas 0 dan 1 harus ada pada data latih")
        self.class_prior_ = self.class_count_ / self.class_count_.sum()
        self.class_log_prior_ = np.log(self.class_prior_)

        if len(num) > 0:
            theta = np.zeros((2, len(num)), dtype=np.float64)
            var = np.ones((2, len(num)), dtype=np.float64)
            for k, c in enumerate(self.classes_):
                Xk = X[y == c][:, num]
                theta[k] = Xk.mean(axis=0)
                var[k] = Xk.var(axis=0)
            max_var = float(var.max()) if var.size else 0.0
            epsilon = self.var_smoothing * (max_var if max_var > 0 else 1.0)
            self.theta_ = theta
            self.var_ = np.maximum(var + epsilon, EPS)
        else:
            self.theta_ = np.zeros((2, 0), dtype=np.float64)
            self.var_ = np.ones((2, 0), dtype=np.float64)

        self.cat_counts_ = []
        self.cat_probs_ = []
        for j, n_cat in zip(cat, n_categories):
            n_cat = int(n_cat)
            counts = np.zeros((2, n_cat), dtype=np.float64)
            for k, c in enumerate(self.classes_):
                values = X[y == c, j].astype(np.int64)
                values = np.clip(values, 0, n_cat - 1)
                np.add.at(counts[k], values, 1.0)
            probs = (counts + self.alpha) / (
                counts.sum(axis=1, keepdims=True) + self.alpha * n_cat
            )
            self.cat_counts_.append(counts)
            self.cat_probs_.append(probs)
        return self

    def _check_fitted(self):
        if self.classes_ is None:
            raise RuntimeError("panggil fit() sebelum predict()")

    def _resolved_weights(self):
        if self.feature_weights is None:
            return np.ones(self.n_features_, dtype=np.float64)
        weights = np.asarray(self.feature_weights, dtype=np.float64).ravel()
        if len(weights) != self.n_features_:
            raise ValueError(
                "panjang feature_weights harus sama dengan jumlah fitur model"
            )
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("feature_weights harus finite dan >= 0")
        return weights

    def feature_log_likelihood(self, X):
        """Return per-feature, per-class log-likelihood, shape (n, 2, n_features)."""
        self._check_fitted()
        X = self._check_X(X, self.n_features_)
        log_lik = np.zeros((len(X), 2, self.n_features_), dtype=np.float64)

        for pos, j in enumerate(self.numeric_indices_):
            var = self.var_[:, pos]
            diff = X[:, j][:, None] - self.theta_[:, pos][None, :]
            log_lik[:, :, j] = -0.5 * (
                np.log(2.0 * np.pi * var)[None, :] + (diff * diff) / var[None, :]
            )

        for pos, (j, probs) in enumerate(
            zip(self.categorical_indices_, self.cat_probs_)
        ):
            n_cat = probs.shape[1]
            values = X[:, j].astype(np.int64)
            values = np.clip(values, 0, n_cat - 1)
            log_lik[:, 0, j] = np.log(probs[0, values])
            log_lik[:, 1, j] = np.log(probs[1, values])

        return log_lik

    def _log_joint(self, X):
        log_lik = self.feature_log_likelihood(X)
        weights = self._resolved_weights()
        log_joint = np.tile(self.class_log_prior_, (len(log_lik), 1))
        log_joint = log_joint + np.einsum("nkf,f->nk", log_lik, weights)
        return log_joint

    def decision_scores(self, X):
        """Return P(class = 1) for every row."""
        log_joint = self._log_joint(X)
        log_joint = log_joint - log_joint.max(axis=1, keepdims=True)
        prob = np.exp(log_joint)
        prob = prob / np.maximum(prob.sum(axis=1, keepdims=True), EPS)
        class_one_index = int(np.flatnonzero(self.classes_ == 1)[0])
        return prob[:, class_one_index]

    def predict(self, X):
        return (self.decision_scores(X) > self.threshold).astype(np.int64)

    def get_params(self):
        numeric = self.numeric_indices_
        categorical = self.categorical_indices_
        n_categories = self.n_categories_
        if numeric is None:
            numeric = self.numeric_indices
        if categorical is None:
            categorical = self.categorical_indices
        if n_categories is None:
            n_categories = self.n_categories
        return {
            "mode": self.mode,
            "variant": self.variant,
            "alpha": self.alpha,
            "var_smoothing": self.var_smoothing,
            "threshold": self.threshold,
            "numeric_indices": None if numeric is None else list(map(int, numeric)),
            "categorical_indices": None
            if categorical is None
            else list(map(int, categorical)),
            "n_categories": None
            if n_categories is None
            else list(map(int, n_categories)),
            "feature_weights": None
            if self.feature_weights is None
            else list(map(float, self.feature_weights)),
        }

    def export_state(self):
        self._check_fitted()
        state = {
            "classes": self.classes_,
            "class_count": self.class_count_,
            "class_prior": self.class_prior_,
            "class_log_prior": self.class_log_prior_,
            "theta": self.theta_,
            "var": self.var_,
        }
        for i, counts in enumerate(self.cat_counts_):
            state[f"cat_counts_{i}"] = counts
        return state

    @classmethod
    def from_state(cls, params, state):
        params = dict(params)
        if "mode" not in params:
            params["mode"] = (
                "bonus"
                if (
                    params.get("variant") == "hybrid"
                    or params.get("numeric_indices")
                    or not np.isclose(float(params.get("alpha", 1.0)), 1.0)
                    or not np.isclose(float(params.get("threshold", 0.5)), 0.5)
                    or params.get("feature_weights") is not None
                )
                else "baseline"
            )
        obj = cls(**params)
        obj.classes_ = np.asarray(state["classes"], dtype=np.int64)
        obj.class_count_ = np.asarray(state["class_count"], dtype=np.float64)
        obj.class_prior_ = (
            np.asarray(state["class_prior"], dtype=np.float64)
            if "class_prior" in state
            else obj.class_count_ / obj.class_count_.sum()
        )
        obj.class_log_prior_ = np.asarray(state["class_log_prior"], dtype=np.float64)
        obj.theta_ = np.asarray(state["theta"], dtype=np.float64)
        obj.var_ = np.asarray(state["var"], dtype=np.float64)
        obj.n_features_ = obj.theta_.shape[1] + len(
            params.get("categorical_indices") or []
        )
        obj.numeric_indices_ = np.asarray(
            params.get("numeric_indices") or [], dtype=int
        )
        obj.categorical_indices_ = np.asarray(
            params.get("categorical_indices") or [], dtype=int
        )
        obj.n_categories_ = np.asarray(params.get("n_categories") or [], dtype=int)
        obj.cat_counts_ = []
        obj.cat_probs_ = []
        for i in range(len(obj.categorical_indices_)):
            counts = np.asarray(state[f"cat_counts_{i}"], dtype=np.float64)
            n_cat = counts.shape[1]
            probs = (counts + obj.alpha) / (
                counts.sum(axis=1, keepdims=True) + obj.alpha * n_cat
            )
            obj.cat_counts_.append(counts)
            obj.cat_probs_.append(probs)
        return obj

    def to_dict(self):
        state = self.export_state()
        return {
            "params": self.get_params(),
            "state": {k: np.asarray(v).tolist() for k, v in state.items()},
        }

    @classmethod
    def from_dict(cls, d):
        state = {k: np.asarray(v, dtype=np.float64) for k, v in d["state"].items()}
        state["classes"] = state["classes"].astype(np.int64)
        return cls.from_state(d["params"], state)

    def __repr__(self):
        return (
            f"NaiveBayesClassifier(alpha={self.alpha}, "
            f"var_smoothing={self.var_smoothing}, threshold={self.threshold})"
        )
