from __future__ import annotations

import numpy as np


class StratifiedKFoldNP:
    """Stratified k-fold berbasis NumPy (itf mirip scikit-learn).

    Setiap kelas diacak dengan ``np.random.default_rng(seed)`` lalu dibagi
    bergiliran (round-robin) ke ``n_splits`` fold sehingga proporsi kelas
    tiap fold hampir sama.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        if int(n_splits) < 2:
            raise ValueError("n_splits harus >= 2")
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def fold_ids(self, y):
        y = np.asarray(y).ravel()
        rng = np.random.default_rng(self.random_state)
        fold = np.empty(len(y), dtype=np.int64)
        offset = 0
        for label in np.unique(y):
            idx = np.flatnonzero(y == label)
            if self.shuffle:
                idx = rng.permutation(idx)
            fold[idx] = (np.arange(len(idx)) + offset) % self.n_splits
            offset += len(idx)
        return fold

    def split(self, X, y):
        fold = self.fold_ids(y)
        for k in range(self.n_splits):
            yield np.flatnonzero(fold != k), np.flatnonzero(fold == k)


class StandardScalerNP:
    """z-score: (x - mean) / std, std nol diganti 1."""

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_


class PCANP:
    """Principal Component Analysis via SVD dari matriks data terpusat.

    Referensi: I. T. Jolliffe, *Principal Component Analysis*, 2nd ed.,
    Springer, 2002 (bab 1-3). Tanda komponen dibuat deterministik dengan
    memaksa elemen beresolusi terbesar tiap komponen bernilai positif.
    """

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        signs = np.sign(vt[np.arange(len(vt)), np.argmax(np.abs(vt), axis=1)])
        signs[signs == 0] = 1.0
        vt = vt * signs[:, None]
        explained = (s**2) / max(n - 1, 1)
        total = explained.sum() if explained.sum() > 0 else 1.0
        k = self.n_components
        if isinstance(k, float) and 0 < k < 1:
            k = int(np.searchsorted(np.cumsum(explained / total), k) + 1)
        k = int(min(k, d))
        self.n_components_ = k
        self.components_ = vt[:k]
        self.explained_variance_ = explained[:k]
        self.explained_variance_ratio_ = explained[:k] / total
        self.singular_values_ = s[:k]
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T


def _codes(col):
    _, inv = np.unique(np.asarray(col), return_inverse=True)
    return inv.ravel()


def mutual_information_discrete(x, y):
    """I(X;Y) dalam nat untuk dua variabel diskret."""
    xi, yi = _codes(x), _codes(y)
    joint = np.zeros((xi.max() + 1, yi.max() + 1), dtype=np.float64)
    np.add.at(joint, (xi, yi), 1.0)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log(joint[nz] / (px @ py)[nz])))


def mutual_info_scores(Z, y, discrete_mask, n_bins=10):
    """Mutual information tiap kolom terhadap label.

    Kolom kontinu didiskretisasi dengan kuantil (``n_bins``) sebelum dihitung.
    """
    scores = np.zeros(Z.shape[1], dtype=np.float64)
    for j in range(Z.shape[1]):
        col = Z[:, j]
        if not discrete_mask[j]:
            edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
            col = np.searchsorted(edges[1:-1], col, side="right")
        scores[j] = mutual_information_discrete(col, y)
    return scores


def chi2_scores(Z, y):
    """Statistik chi-square (sama dengan definisi scikit-learn ``chi2``).

    Fitur diperlakukan sebagai frekuensi non-negatif; observed = jumlah nilai
    fitur per kelas, expected = proporsi kelas x total nilai fitur.
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y).ravel()
    classes = np.unique(y)
    Y = (y[:, None] == classes[None, :]).astype(np.float64)
    observed = Y.T @ Z
    expected = Y.mean(axis=0)[:, None] * Z.sum(axis=0)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = (observed - expected) ** 2 / expected
    terms[~np.isfinite(terms)] = 0.0
    return terms.sum(axis=0)


def macro_f1_np(y_true, y_pred):
    """Macro F1 biner (label 0 dan 1), zero_division=0."""
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_pred = np.asarray(y_pred).astype(np.int64).ravel()
    out = []
    for label in (0, 1):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        denom = 2 * tp + fp + fn
        out.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return float(np.mean(out))
