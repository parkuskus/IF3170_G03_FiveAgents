from __future__ import annotations
import time
from typing import Literal
import numpy as np


KernelType = Literal["linear", "poly", "rbf"]


class SVMFromScratch:
    def __init__(
        self,
        C: float = 1.0,
        kernel: KernelType = "rbf",
        degree: int = 3,
        gamma: str | float = "scale",
        coef0: float = 1.0,
        class_weight: str | dict | None = None,
        tol: float = 1e-3,
        max_iter: int = 500,
        solver: str = "smo",
        random_state: int = 42,
    ):
        self.C = float(C)
        self.kernel = kernel
        self.degree = int(degree)
        self.gamma = gamma
        self.coef0 = float(coef0)
        self.class_weight = class_weight
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.solver = solver
        self.random_state = int(random_state)

        self.gamma_val_ = 0.0
        self.alphas_ = None
        self.b_ = 0.0
        self.w_ = None
        self.support_indices_ = None
        self.support_vectors_ = None
        self.support_y_ = None
        self.support_alphas_ = None
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.fit_time_sec_ = 0.0
        self.predict_time_sec_ = 0.0
        self.slack_total = 0.0
        self.is_fitted_ = False

    def _resolve_gamma(self, x: np.ndarray) -> float:
        if isinstance(self.gamma, (int, float)):
            return float(self.gamma)
        n_features = x.shape[1]
        if self.gamma == "scale":
            var = float(np.var(x))
            return 1.0 / (n_features * var) if var > 1e-12 else 1.0 / n_features
        if self.gamma == "auto":
            return 1.0 / n_features
        raise ValueError(f"gamma tidak valid: {self.gamma}")

    def _compute_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            return np.dot(x1, x2.T)

        if self.kernel == "poly":
            # batasi basis agar tidak overflow
            base = self.gamma_val_ * np.dot(x1, x2.T) + self.coef0
            base = np.clip(base, -1e4, 1e4)
            return np.power(base, self.degree)

        if self.kernel == "rbf":
            x1_sq = np.sum(x1 ** 2, axis=1, keepdims=True)
            x2_sq = np.sum(x2 ** 2, axis=1, keepdims=True)
            sq_dist = x1_sq + x2_sq.T - 2.0 * np.dot(x1, x2.T)
            sq_dist = np.maximum(sq_dist, 0.0)
            return np.exp(-self.gamma_val_ * sq_dist)

        raise ValueError(f"kernel tidak dikenal: {self.kernel}")

    def _calculate_sample_weights(self, y_raw: np.ndarray) -> np.ndarray:
        n = len(y_raw)
        c_vec = np.full(n, self.C, dtype=np.float64)
        if self.class_weight is None:
            return c_vec

        if self.class_weight == "balanced":
            counts = np.bincount(y_raw, minlength=2)
            w0 = n / (2.0 * max(counts[0], 1))
            w1 = n / (2.0 * max(counts[1], 1))
            c_vec[y_raw == 0] *= w0
            c_vec[y_raw == 1] *= w1
            return c_vec

        if isinstance(self.class_weight, dict):
            w0 = float(self.class_weight.get(0, 1.0))
            w1 = float(self.class_weight.get(1, 1.0))
            c_vec[y_raw == 0] *= w0
            c_vec[y_raw == 1] *= w1
            return c_vec

        return c_vec

    def fit(self, x: np.ndarray, y: np.ndarray):
        start_time = time.perf_counter()
        x = np.asarray(x, dtype=np.float64)
        y_raw = np.asarray(y, dtype=np.int64)

        assert set(np.unique(y_raw)).issubset({0, 1})
        y_pm = np.where(y_raw == 1, 1.0, -1.0)
        n_samples, n_features = x.shape

        self.gamma_val_ = self._resolve_gamma(x)
        c_bounds = self._calculate_sample_weights(y_raw)

        # primal solver cepat khusus linear skala besar
        if self.kernel == "linear" and (self.solver == "primal" or n_samples > 10000 and self.solver != "smo_exact"):
            self._fit_linear_primal(x, y_pm, c_bounds)
        else:
            self._fit_dual_smo(x, y_pm, c_bounds)

        self.is_fitted_ = True
        self.slack_total = float(self.calculate_slack(x, y_raw))
        self.fit_time_sec_ = time.perf_counter() - start_time
        return self

    def _fit_linear_primal(self, x: np.ndarray, y_pm: np.ndarray, c_bounds: np.ndarray):
        # subgradient descent dengan momentum dan decay
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = x.shape

        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0

        batch_size = min(256, n_samples)
        iters = self.max_iter * 20
        lr0 = 0.05
        indices = np.arange(n_samples)

        for step in range(1, iters + 1):
            lr = lr0 / (1.0 + 0.001 * step)
            batch_idx = rng.choice(indices, size=batch_size, replace=False)
            xb = x[batch_idx]
            yb = y_pm[batch_idx]
            cb = c_bounds[batch_idx]

            scores = np.dot(xb, w) + b
            margins = yb * scores
            mis_mask = margins < 1.0

            grad_w = (1.0 / n_samples) * w
            grad_b = 0.0

            if np.any(mis_mask):
                loss_weights = cb[mis_mask] * yb[mis_mask]
                grad_w -= np.dot(loss_weights, xb[mis_mask]) / batch_size
                grad_b -= np.sum(loss_weights) / batch_size

            w -= lr * grad_w
            b -= lr * grad_b

        self.w_ = w
        self.b_ = float(b)

        # simpan support vector semu untuk representasi dual
        scores = np.dot(x, w) + b
        margins = y_pm * scores
        sv_mask = margins <= 1.0 + self.tol
        sv_idx = np.flatnonzero(sv_mask)
        if len(sv_idx) == 0:
            sv_idx = np.argsort(margins)[:100]

        self.support_indices_ = sv_idx
        self.support_vectors_ = x[sv_idx]
        self.support_y_ = y_pm[sv_idx]
        self.support_alphas_ = np.ones(len(sv_idx), dtype=np.float64)

    def _fit_dual_smo(self, x: np.ndarray, y_pm: np.ndarray, c_bounds: np.ndarray):
        # koordinat descent dual smo dengan row caching on-the-fly
        rng = np.random.default_rng(self.random_state)
        n_samples = x.shape[0]

        alphas = np.zeros(n_samples, dtype=np.float64)
        b = 0.0
        k_diag = np.zeros(n_samples, dtype=np.float64)

        if self.kernel == "linear":
            k_diag = np.sum(x ** 2, axis=1)
        elif self.kernel == "poly":
            base = self.gamma_val_ * np.sum(x ** 2, axis=1) + self.coef0
            k_diag = np.power(np.clip(base, -1e4, 1e4), self.degree)
        elif self.kernel == "rbf":
            k_diag = np.ones(n_samples, dtype=np.float64)

        # buffer f_i = sum(alpha_j * y_j * K_ij) + b
        f_cache = np.zeros(n_samples, dtype=np.float64)

        for iteration in range(self.max_iter):
            num_changed = 0

            # iterasi seluruh sampel atau support vectors aktif
            active_idx = np.flatnonzero((alphas > 1e-7) & (alphas < c_bounds - 1e-7))
            target_idx = active_idx if len(active_idx) > 0 and (iteration % 2 == 1) else np.arange(n_samples)
            if len(target_idx) > 2000:
                target_idx = rng.choice(target_idx, size=2000, replace=False)

            for i in target_idx:
                e_i = (f_cache[i] + b) - y_pm[i]
                r_i = e_i * y_pm[i]

                # cek pelanggaran kkt
                if (r_i < -self.tol and alphas[i] < c_bounds[i]) or (r_i > self.tol and alphas[i] > 0.0):
                    # pilih pasangan j
                    if len(active_idx) > 1:
                        j = int(rng.choice(active_idx))
                        if j == i:
                            j = int(rng.choice(target_idx))
                    else:
                        j = int(rng.choice(n_samples))
                    if i == j:
                        continue

                    e_j = (f_cache[j] + b) - y_pm[j]

                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]

                    if y_pm[i] != y_pm[j]:
                        low = max(0.0, alpha_j_old - alpha_i_old)
                        high = min(c_bounds[j], c_bounds[j] + alpha_j_old - alpha_i_old)
                    else:
                        low = max(0.0, alpha_i_old + alpha_j_old - c_bounds[i])
                        high = min(c_bounds[j], alpha_i_old + alpha_j_old)

                    if low >= high:
                        continue

                    # komputasi kernel parsial k_ij
                    k_ij = float(self._compute_kernel(x[i:i+1], x[j:j+1])[0, 0])
                    eta = k_diag[i] + k_diag[j] - 2.0 * k_ij

                    # periksa stabilitas kurvatur
                    if eta <= 1e-12:
                        continue

                    alpha_j_new = alpha_j_old + (y_pm[j] * (e_i - e_j)) / eta
                    alpha_j_new = np.clip(alpha_j_new, low, high)

                    if abs(alpha_j_new - alpha_j_old) < 1e-5:
                        continue

                    alpha_i_new = alpha_i_old + y_pm[i] * y_pm[j] * (alpha_j_old - alpha_j_new)

                    # update b
                    b1 = b - e_i - y_pm[i] * (alpha_i_new - alpha_i_old) * k_diag[i] - y_pm[j] * (alpha_j_new - alpha_j_old) * k_ij
                    b2 = b - e_j - y_pm[i] * (alpha_i_new - alpha_i_old) * k_ij - y_pm[j] * (alpha_j_new - alpha_j_old) * k_diag[j]

                    if 0.0 < alpha_i_new < c_bounds[i]:
                        b_new = b1
                    elif 0.0 < alpha_j_new < c_bounds[j]:
                        b_new = b2
                    else:
                        b_new = 0.5 * (b1 + b2)

                    diff_i = (alpha_i_new - alpha_i_old) * y_pm[i]
                    diff_j = (alpha_j_new - alpha_j_old) * y_pm[j]

                    # update cache f lokal pada subset
                    if diff_i != 0.0:
                        f_cache += diff_i * self._compute_kernel(x[i:i+1], x)[0]
                    if diff_j != 0.0:
                        f_cache += diff_j * self._compute_kernel(x[j:j+1], x)[0]

                    alphas[i] = alpha_i_new
                    alphas[j] = alpha_j_new
                    b = b_new
                    num_changed += 1

            if num_changed == 0 and len(target_idx) == n_samples:
                break

        self.alphas_ = alphas
        self.b_ = float(b)

        sv_mask = alphas > 1e-6
        sv_idx = np.flatnonzero(sv_mask)
        if len(sv_idx) == 0:
            sv_idx = np.argsort(alphas)[-100:]

        self.support_indices_ = sv_idx
        self.support_vectors_ = x[sv_idx]
        self.support_y_ = y_pm[sv_idx]
        self.support_alphas_ = alphas[sv_idx]

        if self.kernel == "linear":
            self.w_ = np.sum((self.support_alphas_ * self.support_y_)[:, None] * self.support_vectors_, axis=0)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("model belum di-fit")
        x = np.asarray(x, dtype=np.float64)

        if self.kernel == "linear" and self.w_ is not None:
            return np.dot(x, self.w_) + self.b_

        # inferensi kernel hanya terhadap support vectors
        k_sv = self._compute_kernel(x, self.support_vectors_)
        weights = self.support_alphas_ * self.support_y_
        return np.dot(k_sv, weights) + self.b_

    def predict(self, x: np.ndarray) -> np.ndarray:
        start_time = time.perf_counter()
        scores = self.decision_function(x)
        preds = np.where(scores >= 0.0, 1, 0).astype(np.int64)
        self.predict_time_sec_ = time.perf_counter() - start_time
        return preds

    def calculate_slack(self, x: np.ndarray, y_raw: np.ndarray) -> float:
        y_pm = np.where(y_raw == 1, 1.0, -1.0)
        scores = self.decision_function(x)
        margins = y_pm * scores
        slacks = np.maximum(0.0, 1.0 - margins)
        return float(np.sum(slacks))


if __name__ == "__main__":
    # pengujian unit cepat
    rng = np.random.default_rng(42)
    x_toy = rng.normal(size=(200, 10))
    y_toy = ((x_toy[:, 0] + x_toy[:, 1] > 0)).astype(np.int64)

    for kern in ["linear", "poly", "rbf"]:
        model = SVMFromScratch(kernel=kern, max_iter=20, random_state=42)
        model.fit(x_toy, y_toy)
        preds = model.predict(x_toy)
        assert set(np.unique(preds)).issubset({0, 1})
        assert len(preds) == len(y_toy)
        assert model.slack_total >= 0.0
        print(f"kernel {kern:<6} | acc: {np.mean(preds == y_toy):.3f} | slack: {model.slack_total:.2f} | sv: {len(model.support_indices_)}")
