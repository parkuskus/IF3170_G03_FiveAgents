from __future__ import annotations
import numpy as np

EPS = 1e-10

VALID_METRICS = ("euclidean", "manhattan")


class KNNClassifier:
    METRICS = {"euclidean", "manhattan", "chebyshev", "minkowski", "cosine"}

    def __init__(self, n_neighbors=5, weights="uniform", metric="euclidean", p=3):
        if n_neighbors < 1: raise ValueError("n_neighbors harus >= 1")
        if weights not in ("uniform", "distance"): raise ValueError("weights harus 'uniform' atau 'distance'")
        if metric not in self.METRICS: raise ValueError(f"metric harus salah satu dari {sorted(self.METRICS)}")
        if metric == "minkowski" and (not np.isfinite(p) or p < 1): raise ValueError("p Minkowski harus finite dan >= 1")
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.metric = metric
        self.p = p
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

    def kneighbors(self, X, n_neighbors=None, batch_size=2000):
        if self.X_train is None: raise RuntimeError("panggil fit() sebelum predict()")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]: raise ValueError("bentuk X salah")
        if not np.isfinite(X).all(): raise ValueError("X mengandung NaN/inf")
        if batch_size < 1: raise ValueError("batch_size harus >= 1")
        k = min(self.n_neighbors if n_neighbors is None else int(n_neighbors), len(self.X_train))
        if k < 1: raise ValueError("n_neighbors harus >= 1")
        batch_size = min(batch_size, max(1, 4_000_000 // len(self.X_train)))
        n = len(X)
        all_indices = np.empty((n, k), dtype=np.int64)
        all_distances = np.empty((n, k), dtype=np.float64)
        train_norm = np.einsum("ij,ij->i", self.X_train, self.X_train)[None, :]
        train_length = np.sqrt(train_norm)
        for s in range(0, n, batch_size):
            xb = X[s:s + batch_size]
            if self.metric in {"euclidean", "cosine"}:
                dot = xb @ self.X_train.T
            if self.metric == "euclidean":
                d2 = np.einsum("ij,ij->i", xb, xb)[:, None] + train_norm - 2.0 * dot
                np.maximum(d2, 0.0, out=d2)
                distances = np.sqrt(d2)
            elif self.metric == "cosine":
                denominator = np.sqrt(np.einsum("ij,ij->i", xb, xb))[:, None] * train_length
                similarity = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > EPS)
                distances = np.clip(1.0 - similarity, 0.0, 2.0)
            else:
                distances = np.zeros((len(xb), len(self.X_train)), dtype=np.float64)
                for j in range(self.X_train.shape[1]):
                    delta = np.abs(xb[:, j, None] - self.X_train[None, :, j])
                    if self.metric == "manhattan":
                        distances += delta
                    elif self.metric == "chebyshev":
                        np.maximum(distances, delta, out=distances)
                    else:
                        distances += delta ** self.p
                if self.metric == "minkowski":
                    distances **= 1.0 / self.p
            nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
            nearest_distances = np.take_along_axis(distances, nearest, axis=1)
            order = np.argsort(nearest_distances, axis=1)
            nearest = np.take_along_axis(nearest, order, axis=1)
            all_indices[s:s + len(xb)] = nearest
            all_distances[s:s + len(xb)] = np.take_along_axis(distances, nearest, axis=1)
        return all_distances, all_indices

    def predict_from_neighbors(self, distances, indices, n_neighbors=None, weights=None):
        k = min(self.n_neighbors if n_neighbors is None else int(n_neighbors), indices.shape[1])
        weights = self.weights if weights is None else weights
        if weights not in ("uniform", "distance"): raise ValueError("weights harus 'uniform' atau 'distance'")
        labels = self.y_train[indices[:, :k]]
        if weights == "uniform":
            vote_1 = (labels == 1).sum(axis=1)
            vote_0 = (labels == 0).sum(axis=1)
        else:
            neighbor_distances = distances[:, :k]
            vote_weights = 1.0 / (neighbor_distances + EPS)
            vote_0 = np.where(labels == 0, vote_weights, 0.0).sum(axis=1)
            vote_1 = np.where(labels == 1, vote_weights, 0.0).sum(axis=1)
        return (vote_1 > vote_0).astype(np.int64)

    def predict(self, X, batch_size=2000):
        distances, indices = self.kneighbors(X, batch_size=batch_size)
        return self.predict_from_neighbors(distances, indices)

    def get_params(self): return {"n_neighbors": self.n_neighbors, "weights": self.weights, "metric": self.metric, "p": self.p}
