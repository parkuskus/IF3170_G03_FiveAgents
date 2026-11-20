from __future__ import annotations

import numpy as np

EPS = 1e-12


def _entropy_from_counts(counts):
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(p)))


def mutual_information_with_class(codes, y, n_cat):
    """I(X; C) dari kode kategorikal ``codes`` dan label biner ``y``."""
    joint = np.bincount(y * n_cat + codes, minlength=2 * n_cat).reshape(2, n_cat)
    joint = joint.astype(np.float64)
    h_x = _entropy_from_counts(joint.sum(axis=0))
    h_c = _entropy_from_counts(joint.sum(axis=1))
    h_xc = _entropy_from_counts(joint.ravel())
    return h_x + h_c - h_xc


def conditional_mutual_information(a, b, y, n_a, n_b):
    """I(A; B | C) = sum_c P(c) * I(A; B | C = c)."""
    n = len(y)
    total = 0.0
    for c in (0, 1):
        sel = y == c
        n_c = int(sel.sum())
        if n_c == 0:
            continue
        joint = np.bincount(a[sel] * n_b + b[sel], minlength=n_a * n_b)
        joint = joint.reshape(n_a, n_b).astype(np.float64)
        mi = (
            _entropy_from_counts(joint.sum(axis=1))
            + _entropy_from_counts(joint.sum(axis=0))
            - _entropy_from_counts(joint.ravel())
        )
        total += (n_c / n) * mi
    return total


class KDBClassifier:
    """K-Dependence Bayesian Classifier untuk fitur kategorikal berkode integer.

    Parameter
    ---------
    k : int
        Jumlah maksimum induk fitur untuk setiap fitur (0 = Naive Bayes).
    alpha : float
        Laplace smoothing untuk P(x_i | C).
    m_estimate : float
        Kekuatan m-estimate saat menyusutkan P(x_i | C, induk) ke tingkat
        induk yang lebih sedikit.
    threshold : float
        Ambang P(C = 1 | x) untuk memprediksi kelas 1.
    n_categories : list[int]
        Jumlah kategori tiap kolom. Kode harus berada di ``[0, n_cat)``.
    feature_weights : list[float] | None
        Bobot non-negatif untuk log-likelihood tiap fitur.
    bias : float
        Tambahan pada log-odds kelas 1, dipelajari bersama bobot.
    """

    def __init__(
        self,
        k=2,
        alpha=1.0,
        m_estimate=20.0,
        threshold=0.5,
        n_categories=None,
        feature_weights=None,
        bias=0.0,
    ):
        if int(k) < 0:
            raise ValueError("k harus >= 0")
        if alpha <= 0:
            raise ValueError("alpha harus > 0")
        if m_estimate <= 0:
            raise ValueError("m_estimate harus > 0")
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold harus di (0, 1)")
        if n_categories is None:
            raise ValueError("n_categories wajib diisi")
        self.k = int(k)
        self.alpha = float(alpha)
        self.m_estimate = float(m_estimate)
        self.threshold = float(threshold)
        self.n_categories = tuple(int(n) for n in n_categories)
        self.feature_weights = (
            None
            if feature_weights is None
            else tuple(float(w) for w in feature_weights)
        )
        self.bias = float(bias)
        self.classes_ = None

    # ------------------------------------------------------------------ utils
    def _check_X(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X harus 2 dimensi")
        if X.shape[1] != len(self.n_categories):
            raise ValueError("jumlah kolom X tidak sesuai n_categories")
        if not np.isfinite(X.astype(np.float64)).all():
            raise ValueError("X mengandung NaN/inf")
        codes = X.astype(np.int64)
        upper = np.asarray(self.n_categories, dtype=np.int64) - 1
        return np.clip(codes, 0, upper[None, :])

    def _parent_key(self, codes, parents):
        key = np.zeros(len(codes), dtype=np.int64)
        for q in parents:
            key = key * self.n_categories[q] + codes[:, q]
        return key

    # -------------------------------------------------------------- learning
    def _learn_structure(self, codes, y):
        d = codes.shape[1]
        ncat = self.n_categories
        self.mi_class_ = np.array(
            [mutual_information_with_class(codes[:, j], y, ncat[j]) for j in range(d)]
        )
        cmi = np.zeros((d, d), dtype=np.float64)
        if self.k > 0:
            for i in range(d):
                for j in range(i + 1, d):
                    value = conditional_mutual_information(
                        codes[:, i], codes[:, j], y, ncat[i], ncat[j]
                    )
                    cmi[i, j] = cmi[j, i] = value
        self.cmi_ = cmi
        # Urutan stabil: MI terbesar dulu, seri diputus dengan indeks kolom.
        self.order_ = np.lexsort((np.arange(d), -self.mi_class_))
        self.parents_ = [[] for _ in range(d)]
        for pos, j in enumerate(self.order_):
            candidates = self.order_[:pos]
            if self.k == 0 or len(candidates) == 0:
                continue
            ranked = candidates[
                np.lexsort((candidates, -cmi[j, candidates]))
            ]
            self.parents_[j] = [int(q) for q in ranked[: self.k]]

    def fit(self, X, y):
        y = np.asarray(y).astype(np.int64).ravel()
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("label harus 0 atau 1")
        codes = self._check_X(X)
        if len(codes) != len(y):
            raise ValueError("panjang X dan y harus sama")
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.class_count_ = np.bincount(y, minlength=2).astype(np.float64)
        if np.any(self.class_count_ == 0):
            raise ValueError("kedua kelas 0 dan 1 harus ada pada data latih")
        self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())
        self.n_features_ = codes.shape[1]
        self._learn_structure(codes, y)

        self.base_counts_ = []
        self.tables_ = []
        for j in range(self.n_features_):
            m = self.n_categories[j]
            base = np.bincount(y * m + codes[:, j], minlength=2 * m).reshape(2, m)
            self.base_counts_.append(base.astype(np.float64))
            levels = []
            for level in range(1, len(self.parents_[j]) + 1):
                key = self._parent_key(codes, self.parents_[j][:level])
                flat = key * m + codes[:, j]
                levels.append(
                    {
                        "flat_keys": None,
                        "flat_counts": None,
                        "parent_keys": None,
                        "parent_counts": None,
                    }
                )
                for name, values in (("flat", flat), ("parent", key)):
                    uniq, inv = np.unique(values, return_inverse=True)
                    counts = np.zeros((len(uniq), 2), dtype=np.float64)
                    np.add.at(counts, (inv.ravel(), y), 1.0)
                    levels[-1][f"{name}_keys"] = uniq
                    levels[-1][f"{name}_counts"] = counts
            self.tables_.append(levels)
        return self

    # ------------------------------------------------------------ inference
    @staticmethod
    def _lookup(keys, counts, query):
        out = np.zeros((len(query), 2), dtype=np.float64)
        if len(keys) == 0:
            return out
        pos = np.searchsorted(keys, query)
        pos_clipped = np.minimum(pos, len(keys) - 1)
        found = (pos < len(keys)) & (keys[pos_clipped] == query)
        out[found] = counts[pos_clipped[found]]
        return out

    def _check_fitted(self):
        if self.classes_ is None:
            raise RuntimeError("panggil fit() sebelum predict()")

    def feature_log_likelihood(self, X):
        """Log P(x_i | C, induk) per baris, kelas, dan fitur: (n, 2, d)."""
        self._check_fitted()
        codes = self._check_X(X)
        n = len(codes)
        out = np.zeros((n, 2, self.n_features_), dtype=np.float64)
        for j in range(self.n_features_):
            m = self.n_categories[j]
            base = self.base_counts_[j]
            prob = (base + self.alpha) / (
                base.sum(axis=1, keepdims=True) + self.alpha * m
            )
            p = prob[:, codes[:, j]].T  # (n, 2)
            for level, table in enumerate(self.tables_[j], start=1):
                key = self._parent_key(codes, self.parents_[j][:level])
                flat = key * m + codes[:, j]
                n_xv = self._lookup(table["flat_keys"], table["flat_counts"], flat)
                n_v = self._lookup(table["parent_keys"], table["parent_counts"], key)
                p = (n_xv + self.m_estimate * p) / (n_v + self.m_estimate)
            out[:, :, j] = np.log(np.maximum(p, EPS))
        return out

    def _resolved_weights(self):
        if self.feature_weights is None:
            return np.ones(self.n_features_, dtype=np.float64)
        w = np.asarray(self.feature_weights, dtype=np.float64)
        if len(w) != self.n_features_:
            raise ValueError("panjang feature_weights harus sama dengan jumlah fitur")
        if not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("feature_weights harus finite dan >= 0")
        return w

    def log_odds(self, X):
        lik = self.feature_log_likelihood(X)
        w = self._resolved_weights()
        diff = (lik[:, 1, :] - lik[:, 0, :]) @ w
        prior = self.class_log_prior_[1] - self.class_log_prior_[0]
        return prior + self.bias + diff

    def decision_scores(self, X):
        """P(C = 1 | x) untuk setiap baris."""
        z = np.clip(self.log_odds(X), -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X):
        return (self.decision_scores(X) > self.threshold).astype(np.int64)

    # ------------------------------------------------------------ save/load
    def get_params(self):
        return {
            "k": self.k,
            "alpha": self.alpha,
            "m_estimate": self.m_estimate,
            "threshold": self.threshold,
            "n_categories": list(self.n_categories),
            "feature_weights": None
            if self.feature_weights is None
            else list(self.feature_weights),
            "bias": self.bias,
        }

    def export_state(self):
        """State sebagai dict array NumPy (untuk ``np.savez_compressed``)."""
        self._check_fitted()
        parents = -np.ones((self.n_features_, max(self.k, 1)), dtype=np.int64)
        for j, pa in enumerate(self.parents_):
            parents[j, : len(pa)] = pa
        state = {
            "class_count": self.class_count_,
            "mi_class": self.mi_class_,
            "cmi": self.cmi_,
            "order": self.order_,
            "parents": parents,
        }
        for j in range(self.n_features_):
            state[f"f{j}_base"] = self.base_counts_[j]
            for level, table in enumerate(self.tables_[j], start=1):
                for name, values in table.items():
                    state[f"f{j}_l{level}_{name}"] = values
        return state

    @classmethod
    def from_state(cls, params, state):
        obj = cls(**params)
        obj.classes_ = np.array([0, 1], dtype=np.int64)
        obj.class_count_ = np.asarray(state["class_count"], dtype=np.float64)
        obj.class_log_prior_ = np.log(obj.class_count_ / obj.class_count_.sum())
        obj.mi_class_ = np.asarray(state["mi_class"], dtype=np.float64)
        obj.cmi_ = np.asarray(state["cmi"], dtype=np.float64)
        obj.order_ = np.asarray(state["order"], dtype=np.int64)
        parents = np.asarray(state["parents"], dtype=np.int64)
        obj.n_features_ = len(obj.n_categories)
        obj.parents_ = [[int(q) for q in row if q >= 0] for row in parents]
        obj.base_counts_ = []
        obj.tables_ = []
        for j in range(obj.n_features_):
            obj.base_counts_.append(np.asarray(state[f"f{j}_base"], dtype=np.float64))
            levels = []
            for level in range(1, len(obj.parents_[j]) + 1):
                prefix = f"f{j}_l{level}_"
                levels.append(
                    {
                        "flat_keys": np.asarray(state[prefix + "flat_keys"], dtype=np.int64),
                        "flat_counts": np.asarray(state[prefix + "flat_counts"], dtype=np.float64),
                        "parent_keys": np.asarray(state[prefix + "parent_keys"], dtype=np.int64),
                        "parent_counts": np.asarray(state[prefix + "parent_counts"], dtype=np.float64),
                    }
                )
            obj.tables_.append(levels)
        return obj

    def structure(self, feature_names=None):
        """Daftar edge (induk -> anak) untuk visualisasi graf Bayesian Network."""
        self._check_fitted()
        names = feature_names or [f"x{j}" for j in range(self.n_features_)]
        edges = [("C", names[j]) for j in range(self.n_features_)]
        for j, pa in enumerate(self.parents_):
            edges.extend((names[q], names[j]) for q in pa)
        return edges

    def __repr__(self):
        return (
            f"KDBClassifier(k={self.k}, alpha={self.alpha}, "
            f"m_estimate={self.m_estimate}, threshold={self.threshold})"
        )


def fit_feature_weights(oof_lik, oof_prior, y, lam=1e-3, iters=400, lr=0.5):
    """Pelajari bobot fitur non-negatif & bias pada log-likelihood OOF.

    Mengoptimasi log-loss terregulasi L2 terhadap bobot awal = 1:
        L(w, b) = BCE(sigmoid(d_prior + b + D @ w), y) + lam * ||w - 1||^2
    menggunakan Adam berbasis NumPy murni (deterministik, Zaidi dkk. 2013).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    D = oof_lik[:, 1, :] - oof_lik[:, 0, :]
    dp = oof_prior[:, 1] - oof_prior[:, 0]
    n, d = D.shape
    w = np.ones(d, dtype=np.float64)
    b = 0.0
    mw = np.zeros(d, dtype=np.float64)
    vw = np.zeros(d, dtype=np.float64)
    mb = 0.0
    vb = 0.0
    step = lr * 0.1
    for t in range(1, int(iters) + 1):
        z = np.clip(dp + b + D @ w, -500.0, 500.0)
        s = 1.0 / (1.0 + np.exp(-z))
        g = s - y
        gw = (D.T @ g) / n + 2.0 * lam * (w - 1.0)
        gb = float(g.mean())
        mw = 0.9 * mw + 0.1 * gw
        vw = 0.999 * vw + 0.001 * (gw * gw)
        mb = 0.9 * mb + 0.1 * gb
        vb = 0.999 * vb + 0.001 * (gb * gb)
        mw_hat = mw / (1.0 - 0.9 ** t)
        vw_hat = vw / (1.0 - 0.999 ** t)
        mb_hat = mb / (1.0 - 0.9 ** t)
        vb_hat = vb / (1.0 - 0.999 ** t)
        w = np.maximum(w - step * mw_hat / (np.sqrt(vw_hat) + 1e-8), 0.0)
        b = b - step * mb_hat / (np.sqrt(vb_hat) + 1e-8)
    return w.tolist(), float(b)
