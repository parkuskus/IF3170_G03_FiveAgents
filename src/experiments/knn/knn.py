from __future__ import annotations
import numpy as np

EPS = 1e-10

class KNNClassifier:
    def __init__(self, n_neighbors=5, weights="uniform"):
        if n_neighbors < 1: raise ValueError("n_neighbors harus >= 1")
        if weights not in ("uniform", "distance"): raise ValueError("weights harus 'uniform' atau 'distance'")
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(np.int64).ravel()
        if X.ndim != 2: raise ValueError("X harus 2 dimensi")
        if len(X) != len(y): raise ValueError("panjang X dan y harus sama")
        if not np.isfinite(X).all(): raise ValueError("X mengandung NaN/inf")
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X, batch_size=2000):
        if self.X_train is None: raise RuntimeError("panggil fit() sebelum predict()")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]: raise ValueError("bentuk X salah")
        n = X.shape[0]
        out = np.empty(n, dtype=np.int64)
        k = min(self.n_neighbors, len(self.X_train))
        t2 = np.einsum("ij,ij->i", self.X_train, self.X_train)[None, :]
        for s in range(0, n, batch_size):
            xb = X[s:s+batch_size]
            d2 = np.einsum("ij,ij->i", xb, xb)[:, None] + t2 - 2.0 * (xb @ self.X_train.T)
            np.maximum(d2, 0.0, out=d2)
            near = np.argpartition(d2, kth=k-1, axis=1)[:, :k]
            dist = np.sqrt(np.take_along_axis(d2, near, axis=1))
            lab = self.y_train[near]
            if self.weights == "uniform":
                s0 = (lab == 0).sum(axis=1)
                s1 = (lab == 1).sum(axis=1)
            else:
                w = 1.0 / (dist + EPS)
                s0 = np.where(lab == 0, w, 0.0).sum(axis=1)
                s1 = np.where(lab == 1, w, 0.0).sum(axis=1)
            out[s:s+batch_size] = (s1 > s0).astype(np.int64)
        return out

    def get_params(self): return {"n_neighbors": self.n_neighbors, "weights": self.weights}