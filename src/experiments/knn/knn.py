from __future__ import annotations

from numbers import Integral

import numpy as np

EPS = 1e-10
VALID_METRICS = ("euclidean", "manhattan")


class KNNClassifier:
    def __init__(self, n_neighbors=5, weights="uniform", metric="euclidean", threshold=0.5):
        if not isinstance(n_neighbors, Integral) or n_neighbors < 1:
            raise ValueError("n_neighbors harus >= 1")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights harus 'uniform' atau 'distance'")
        if metric not in VALID_METRICS:
            raise ValueError(f"metric harus salah satu dari {VALID_METRICS}")
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold harus di (0, 1)")
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.metric = metric
        self.threshold = float(threshold)
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError("X harus 2 dimensi")
        if len(X) == 0:
            raise ValueError("data train tidak boleh kosong")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("label y harus 0 atau 1")
        if len(X) != len(y):
            raise ValueError("panjang X dan y harus sama")
        if not np.isfinite(X).all():
            raise ValueError("X mengandung NaN/inf")
        self.X_train = X
        self.y_train = y.astype(np.int64)
        return self

    @staticmethod
    def _nearest(distances, indices, k):
        """Urutkan jarak yang seri berdasarkan indeks baris train."""
        selected = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        cutoff = np.take_along_axis(distances, selected, axis=1).max(axis=1)
        at_cutoff = distances == cutoff[:, None]
        selected_at_cutoff = np.take_along_axis(at_cutoff, selected, axis=1)
        ties = at_cutoff.sum(axis=1) > selected_at_cutoff.sum(axis=1)
        for row in np.flatnonzero(ties):
            order = np.lexsort((indices[row], distances[row]))
            selected[row] = order[:k]
        return selected

    def _knn_scores_and_labels(self, X, batch_size=2000):
        """Menghasilkan skor kelas 1 dari k tetangga terdekat per baris."""
        n = X.shape[0]
        scores = np.empty(n, dtype=np.float64)
        k = min(self.n_neighbors, len(self.X_train))
        use_euclidean = self.metric == "euclidean"
        train_norm = (
            np.einsum("ij,ij->i", self.X_train, self.X_train)[None, :]
            if use_euclidean else None
        )

        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            xb = X[start:stop]

            if use_euclidean:
                d2 = (
                    np.einsum("ij,ij->i", xb, xb)[:, None]
                    + train_norm
                    - 2.0 * (xb @ self.X_train.T)
                )
                np.maximum(d2, 0.0, out=d2)
                indices = np.broadcast_to(np.arange(d2.shape[1]), d2.shape)
                nearest = self._nearest(d2, indices, k)
                distances = np.sqrt(np.take_along_axis(d2, nearest, axis=1))
            else:
                # Hitung Manhattan langsung dengan chunk train agar tidak
                # mengalokasikan array sebesar (batch, seluruh_train, fitur).
                best_dist = np.full((len(xb), k), np.inf, dtype=np.float64)
                best_idx = np.full((len(xb), k), -1, dtype=np.int64)
                train_chunk_size = 256
                for train_start in range(0, len(self.X_train), train_chunk_size):
                    train_stop = min(train_start + train_chunk_size, len(self.X_train))
                    train_chunk = self.X_train[train_start:train_stop]
                    d_chunk = np.abs(
                        xb[:, None, :] - train_chunk[None, :, :]
                    ).sum(axis=2)
                    local_k = min(k, d_chunk.shape[1])
                    indices = np.broadcast_to(np.arange(train_start, train_stop), d_chunk.shape)
                    local_idx = self._nearest(d_chunk, indices, local_k)
                    local_dist = np.take_along_axis(d_chunk, local_idx, axis=1)
                    local_idx += train_start

                    candidate_dist = np.concatenate((best_dist, local_dist), axis=1)
                    candidate_idx = np.concatenate((best_idx, local_idx), axis=1)
                    keep = self._nearest(candidate_dist, candidate_idx, k)
                    best_dist = np.take_along_axis(candidate_dist, keep, axis=1)
                    best_idx = np.take_along_axis(candidate_idx, keep, axis=1)

                nearest = best_idx
                distances = best_dist

            labels = self.y_train[nearest]
            if self.weights == "uniform":
                weights = np.ones_like(distances)
            else:
                weights = 1.0 / (distances + EPS)
            total_weight = weights.sum(axis=1)
            class_1_weight = np.where(labels == 1, weights, 0.0).sum(axis=1)
            scores[start:stop] = class_1_weight / np.maximum(total_weight, EPS)

        return scores

    def decision_scores(self, X, batch_size=2000):
        """Menghasilkan skor kelas 1 pada rentang [0, 1]."""
        if not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError("batch_size harus bilangan bulat positif")
        if self.X_train is None:
            raise RuntimeError("panggil fit() sebelum decision_scores()")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]:
            raise ValueError("bentuk X salah")
        if not np.isfinite(X).all():
            raise ValueError("X mengandung NaN/inf")
        return self._knn_scores_and_labels(X, batch_size)

    def predict(self, X, batch_size=2000):
        scores = self.decision_scores(X, batch_size)
        return (scores > self.threshold).astype(np.int64)

    def get_params(self):
        return {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "metric": self.metric,
            "threshold": self.threshold,
        }
