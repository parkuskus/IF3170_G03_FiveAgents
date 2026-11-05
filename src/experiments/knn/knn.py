from __future__ import annotations
import numpy as np

EPS = 1e-10

VALID_METRICS = ("euclidean", "manhattan")


class KNNClassifier:
    def __init__(self, n_neighbors=5, weights="uniform", metric="euclidean", threshold=0.5):
        if n_neighbors < 1: raise ValueError("n_neighbors harus >= 1")
        if weights not in ("uniform", "distance"): raise ValueError("weights harus 'uniform' atau 'distance'")
        if metric not in VALID_METRICS: raise ValueError(f"metric harus salah satu dari {VALID_METRICS}")
        if not (0.0 < threshold < 1.0): raise ValueError("threshold harus di (0, 1)")
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.metric = metric
        self.threshold = float(threshold)
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

    def _knn_scores_and_labels(self, X, batch_size=2000):
        """Return (scores kelas 1, label tetangga, bobot tetangga) per sampel.
        skor = fraksi vote (uniform) atau proporsi bobot (distance) untuk kelas 1."""
        n = X.shape[0]
        scores = np.empty(n, dtype=np.float64)
        k = min(self.n_neighbors, len(self.X_train))
        use_euc = self.metric == "euclidean"
        t2 = np.einsum("ij,ij->i", self.X_train, self.X_train)[None, :] if use_euc else None
        for s in range(0, n, batch_size):
            xb = X[s:s+batch_size]
            if use_euc:
                d2 = np.einsum("ij,ij->i", xb, xb)[:, None] + t2 - 2.0 * (xb @ self.X_train.T)
                np.maximum(d2, 0.0, out=d2)
                near = np.argpartition(d2, kth=k-1, axis=1)[:, :k]
                dist = np.sqrt(np.take_along_axis(d2, near, axis=1))
            else:
                # Manhattan langsung, tetapi train diproses dalam chunk agar
                # tidak membuat array (batch, seluruh_train, fitur) sekaligus.
                best_dist = np.full((len(xb), k), np.inf, dtype=np.float64)
                best_idx = np.full((len(xb), k), -1, dtype=np.int64)
                train_chunk_size = 256
                for t_start in range(0, len(self.X_train), train_chunk_size):
                    t_end = min(t_start + train_chunk_size, len(self.X_train))
                    train_chunk = self.X_train[t_start:t_end]
                    d_chunk = np.abs(xb[:, None, :] - train_chunk[None, :, :]).sum(axis=2)
                    local_k = min(k, d_chunk.shape[1])
                    local_idx = np.argpartition(d_chunk, kth=local_k - 1, axis=1)[:, :local_k]
                    local_dist = np.take_along_axis(d_chunk, local_idx, axis=1)
                    local_idx += t_start
                    candidate_dist = np.concatenate((best_dist, local_dist), axis=1)
                    candidate_idx = np.concatenate((best_idx, local_idx), axis=1)
                    keep = np.argpartition(candidate_dist, kth=k - 1, axis=1)[:, :k]
                    best_dist = np.take_along_axis(candidate_dist, keep, axis=1)
                    best_idx = np.take_along_axis(candidate_idx, keep, axis=1)
                near = best_idx
                dist = best_dist
            lab = self.y_train[near]
            if self.weights == "uniform":
                w = np.ones_like(dist)
            else:
                w = 1.0 / (dist + EPS)
            wsum = w.sum(axis=1)
            s1 = np.where(lab == 1, w, 0.0).sum(axis=1)
            scores[s:s+batch_size] = s1 / np.maximum(wsum, EPS)
        return scores

    def decision_scores(self, X, batch_size=2000):
        """Skor kelas 1 (0..1) per sampel, untuk threshold tuning."""
        if self.X_train is None: raise RuntimeError("panggil fit() sebelum decision_scores()")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]: raise ValueError("bentuk X salah")
        return self._knn_scores_and_labels(X, batch_size)

    def predict(self, X, batch_size=2000):
        scores = self.decision_scores(X, batch_size)
        return (scores > self.threshold).astype(np.int64)

    def get_params(self):
        return {"n_neighbors": self.n_neighbors, "weights": self.weights,
                "metric": self.metric, "threshold": self.threshold}
