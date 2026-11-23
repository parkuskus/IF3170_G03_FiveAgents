from __future__ import annotations
import time
from typing import Literal
import numpy as np

# referensi metode:
# - cortes & vapnik (1995), support-vector networks
# - platt (1998), sequential minimal optimization (smo)
# - williams & seeger (2001), using the nystrom method to speed up kernel machines
# - keerthi & decoste (2005), a modified finite newton method for fast solution of large scale linear svms
# - hsieh et al. (2008), dual coordinate descent method for large-scale svm

KernelType = Literal["linear", "poly", "rbf"]


class SVMFromScratch:
    def __init__(
        self,
        C: float = 1.0,
        kernel: KernelType = "rbf",
        degree: int = 3,
        gamma: str | float = "scale",
        coef0: float = 0.0,
        class_weight: str | dict | None = None,
        tol: float = 1e-3,
        max_iter: int = 35,
        n_components: int = 1800,
        threshold: float = 0.0,
        solver: str = "auto",
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
        self.n_components = int(n_components)
        self.threshold = float(threshold)
        self.solver = solver
        self.random_state = int(random_state)

        self.gamma_val_ = 0.0
        self.alphas_ = None
        self.b_ = 0.0
        self.w_ = None
        self.dual_coef_ = None
        self.support_indices_ = None
        self.support_vectors_ = None
        self.support_y_ = None
        self.support_alphas_ = None
        self.n_support_ = 0
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
        if isinstance(self.gamma, str) and self.gamma.startswith("scale*"):
            mult = float(self.gamma.split("*", 1)[1])
            var = float(np.var(x))
            base = 1.0 / (n_features * var) if var > 1e-12 else 1.0 / n_features
            return base * mult
        raise ValueError(f"gamma tidak valid: {self.gamma}")

    def _compute_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            return np.dot(x1, x2.T)

        if self.kernel == "poly":
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

        self.gamma_val_ = self._resolve_gamma(x)
        c_bounds = self._calculate_sample_weights(y_raw)

        if self.solver == "smo" or (self.solver == "auto" and len(x) <= 1500 and self.kernel != "linear"):
            self._fit_exact_smo(x, y_pm, c_bounds)
        elif self.solver == "dcd":
            self._fit_rkhs_dcd(x, y_raw, y_pm, c_bounds)
        else:
            self._fit_rkhs_newton(x, y_raw, y_pm, c_bounds)

        self.is_fitted_ = True
        self.slack_total = float(self.calculate_slack(x, y_raw))
        self.fit_time_sec_ = time.perf_counter() - start_time
        return self

    def _fit_rkhs_newton(self, x: np.ndarray, y_raw: np.ndarray, y_pm: np.ndarray, c_bounds: np.ndarray):
        # active-set modified finite newton pada l2-hinge svm di ruang rkhs (keerthi & decoste 2005)
        rng = np.random.default_rng(self.random_state)
        n_samples = len(x)

        if self.kernel == "linear":
            z = x
            sv_idx = None
            l_inv_t = None
        else:
            m = min(self.n_components, n_samples)
            counts = np.bincount(y_raw, minlength=2)
            m0 = int(round(m * counts[0] / n_samples))
            m1 = m - m0
            idx0 = rng.choice(np.flatnonzero(y_raw == 0), size=min(m0, counts[0]), replace=False)
            idx1 = rng.choice(np.flatnonzero(y_raw == 1), size=min(m1, counts[1]), replace=False)
            sv_idx = np.sort(np.concatenate([idx0, idx1]))
            sv_x = x[sv_idx]

            k_mm = self._compute_kernel(sv_x, sv_x)
            k_nm = self._compute_kernel(x, sv_x)
            reg_mat = k_mm + 1e-5 * np.eye(len(sv_idx))

            try:
                chol_l = np.linalg.cholesky(reg_mat)
                l_inv_t = np.linalg.solve(chol_l, np.eye(len(sv_idx))).T
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(reg_mat)
                keep = eigvals > 1e-7
                l_inv_t = eigvecs[:, keep] * (1.0 / np.sqrt(eigvals[keep]))

            z = np.dot(k_nm, l_inv_t)

        d = z.shape[1]
        z_aug = np.hstack([z, np.ones((n_samples, 1), dtype=np.float64)])
        w_aug = np.zeros(d + 1, dtype=np.float64)
        prev_sv = None
        max_steps = min(max(self.max_iter, 10), 15)

        for it in range(max_steps):
            margins = y_pm * np.dot(z_aug, w_aug)
            sv_mask = margins < 1.0
            if not np.any(sv_mask):
                break
            if prev_sv is not None and np.sum(sv_mask ^ prev_sv) <= 2 and it >= 3:
                break
            prev_sv = sv_mask

            z_sv = z_aug[sv_mask]
            wc = c_bounds[sv_mask]
            zw = z_sv * (2.0 * wc)[:, None]
            hess = np.dot(zw.T, z_sv)
            idx_d = np.arange(d)
            hess[idx_d, idx_d] += 1.0
            hess[d, d] += 1e-6
            rhs = np.dot(zw.T, y_pm[sv_mask])
            w_aug = np.linalg.solve(hess, rhs)

        scores = np.dot(z_aug, w_aug)
        margins = y_pm * scores
        self.alphas_ = 2.0 * c_bounds * np.maximum(0.0, 1.0 - margins)
        active_mask = self.alphas_ > 1e-6
        self.n_support_ = int(np.sum(active_mask))

        if self.kernel == "linear":
            self.w_ = w_aug[:-1]
            self.b_ = float(w_aug[-1])
            self.support_indices_ = np.flatnonzero(active_mask)
            self.support_vectors_ = x[self.support_indices_]
            self.support_y_ = y_pm[self.support_indices_]
            self.support_alphas_ = self.alphas_[self.support_indices_]
            self.dual_coef_ = self.support_alphas_ * self.support_y_
        else:
            self.w_ = None
            self.b_ = float(w_aug[-1])
            self.dual_coef_ = np.dot(l_inv_t, w_aug[:-1])
            self.support_indices_ = sv_idx
            self.support_vectors_ = x[sv_idx]
            self.support_y_ = y_pm[sv_idx]
            self.support_alphas_ = np.abs(self.dual_coef_)

    def _fit_rkhs_dcd(self, x: np.ndarray, y_raw: np.ndarray, y_pm: np.ndarray, c_bounds: np.ndarray):
        # dual coordinate descent di ruang rkhs untuk seluruh sampel
        rng = np.random.default_rng(self.random_state)
        n_samples = len(x)

        if self.kernel == "linear":
            z = np.hstack([x, np.ones((n_samples, 1), dtype=np.float64)])
            sv_idx = None
            proj = None
        else:
            m = min(self.n_components, n_samples)
            counts = np.bincount(y_raw, minlength=2)
            m0 = int(round(m * counts[0] / n_samples))
            m1 = m - m0
            idx0 = rng.choice(np.flatnonzero(y_raw == 0), size=min(m0, counts[0]), replace=False)
            idx1 = rng.choice(np.flatnonzero(y_raw == 1), size=min(m1, counts[1]), replace=False)
            sv_idx = np.sort(np.concatenate([idx0, idx1]))
            sv_x = x[sv_idx]

            k_mm = self._compute_kernel(sv_x, sv_x)
            k_nm = self._compute_kernel(x, sv_x)

            eigvals, eigvecs = np.linalg.eigh(k_mm + 1e-6 * np.eye(len(sv_idx)))
            keep = eigvals > 1e-7
            proj = eigvecs[:, keep] * (1.0 / np.sqrt(eigvals[keep]))
            z = np.hstack([np.dot(k_nm, proj), np.ones((n_samples, 1), dtype=np.float64)])

        q_diag = np.maximum(np.sum(z ** 2, axis=1), 1e-12)
        alphas = np.zeros(n_samples, dtype=np.float64)
        w_z = np.zeros(z.shape[1], dtype=np.float64)

        batch_size = min(64, n_samples)
        for ep in range(self.max_iter):
            perm = rng.permutation(n_samples)
            if ep < 5:
                step_scale = 0.20
            elif ep < 15:
                step_scale = 0.10
            else:
                step_scale = 0.05

            max_violation = 0.0
            for s in range(0, n_samples, batch_size):
                blk = perm[s : s + batch_size]
                zb = z[blk]
                yb = y_pm[blk]

                margins = yb * np.dot(zb, w_z)
                grad = 1.0 - margins

                a_old = alphas[blk]
                a_new = np.clip(a_old + step_scale * grad / q_diag[blk], 0.0, c_bounds[blk])
                diff = a_new - a_old

                if np.any(np.abs(diff) > 1e-9):
                    alphas[blk] = a_new
                    w_z += np.dot(diff * yb, zb)
                    max_violation = max(max_violation, float(np.max(np.abs(diff))))

            if max_violation < self.tol * 1e-2 and ep >= 5:
                break

        self.alphas_ = alphas
        active_mask = alphas > 1e-5
        self.n_support_ = int(np.sum(active_mask))

        if self.kernel == "linear":
            self.w_ = w_z[:-1]
            self.b_ = float(w_z[-1])
            self.support_indices_ = np.flatnonzero(active_mask)
            self.support_vectors_ = x[self.support_indices_]
            self.support_y_ = y_pm[self.support_indices_]
            self.support_alphas_ = alphas[self.support_indices_]
            self.dual_coef_ = self.support_alphas_ * self.support_y_
        else:
            self.w_ = None
            self.b_ = float(w_z[-1])
            self.dual_coef_ = np.dot(proj, w_z[:-1])
            self.support_indices_ = sv_idx
            self.support_vectors_ = x[sv_idx]
            self.support_y_ = y_pm[sv_idx]
            self.support_alphas_ = np.abs(self.dual_coef_)

    def _fit_exact_smo(self, x: np.ndarray, y_pm: np.ndarray, c_bounds: np.ndarray):
        # smo penuh pada matriks gram untuk data kecil
        rng = np.random.default_rng(self.random_state)
        n_samples = len(x)
        k_mat = self._compute_kernel(x, x)
        k_diag = np.maximum(np.diag(k_mat), 1e-12)

        alphas = np.zeros(n_samples, dtype=np.float64)
        u = np.zeros(n_samples, dtype=np.float64)
        b = 0.0

        for _ in range(self.max_iter):
            num_changed = 0
            for i in range(n_samples):
                e_i = (u[i] + b) - y_pm[i]
                r_i = e_i * y_pm[i]
                if (r_i < -self.tol and alphas[i] < c_bounds[i]) or (r_i > self.tol and alphas[i] > 0.0):
                    j = int(rng.integers(0, n_samples - 1))
                    if j >= i:
                        j += 1
                    e_j = (u[j] + b) - y_pm[j]
                    ai_old, aj_old = alphas[i], alphas[j]

                    if y_pm[i] != y_pm[j]:
                        low = max(0.0, aj_old - ai_old)
                        high = min(c_bounds[j], c_bounds[i] + aj_old - ai_old)
                    else:
                        low = max(0.0, ai_old + aj_old - c_bounds[i])
                        high = min(c_bounds[j], ai_old + aj_old)
                    if low >= high:
                        continue

                    eta = k_diag[i] + k_diag[j] - 2.0 * k_mat[i, j]
                    if eta <= 1e-12:
                        continue

                    aj_new = float(np.clip(aj_old + y_pm[j] * (e_i - e_j) / eta, low, high))
                    if abs(aj_new - aj_old) < 1e-6:
                        continue
                    ai_new = ai_old + y_pm[i] * y_pm[j] * (aj_old - aj_new)

                    di = (ai_new - ai_old) * y_pm[i]
                    dj = (aj_new - aj_old) * y_pm[j]
                    u += di * k_mat[:, i] + dj * k_mat[:, j]
                    alphas[i], alphas[j] = ai_new, aj_new

                    free = (alphas > 1e-6) & (alphas < c_bounds - 1e-6)
                    if np.any(free):
                        b = float(np.mean(y_pm[free] - u[free]))
                    num_changed += 1
            if num_changed == 0:
                break

        self.alphas_ = alphas
        self.b_ = float(b)
        sv_mask = alphas > 1e-6
        if not np.any(sv_mask):
            sv_mask[0] = True
        self.support_indices_ = np.flatnonzero(sv_mask)
        self.support_vectors_ = x[self.support_indices_]
        self.support_y_ = y_pm[self.support_indices_]
        self.support_alphas_ = alphas[self.support_indices_]
        self.dual_coef_ = self.support_alphas_ * self.support_y_
        self.n_support_ = len(self.support_indices_)
        if self.kernel == "linear":
            self.w_ = np.dot(self.dual_coef_, self.support_vectors_)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("model belum di-fit")
        x = np.asarray(x, dtype=np.float64)

        if self.kernel == "linear" and self.w_ is not None:
            return np.dot(x, self.w_) + self.b_

        # evaluasi kernel dalam batch agar hemat memori
        batch_size = 2048
        scores = np.empty(len(x), dtype=np.float64)
        for s in range(0, len(x), batch_size):
            xb = x[s : s + batch_size]
            k_sv = self._compute_kernel(xb, self.support_vectors_)
            scores[s : s + batch_size] = np.dot(k_sv, self.dual_coef_) + self.b_
        return scores

    def predict(self, x: np.ndarray, threshold: float | None = None) -> np.ndarray:
        start_time = time.perf_counter()
        th = self.threshold if threshold is None else float(threshold)
        scores = self.decision_function(x)
        preds = np.where(scores >= th, 1, 0).astype(np.int64)
        self.predict_time_sec_ = time.perf_counter() - start_time
        return preds

    @staticmethod
    def _best_macro_f1_threshold(scores: np.ndarray, y_true: np.ndarray, n_points: int = 201) -> tuple[float, float]:
        best_f1 = -1.0
        best_th = 0.0
        th_candidates = np.linspace(np.percentile(scores, 5), np.percentile(scores, 95), n_points)
        for th in th_candidates:
            p = np.where(scores >= th, 1, 0)
            tp = np.sum((p == 1) & (y_true == 1))
            fp = np.sum((p == 1) & (y_true == 0))
            fn = np.sum((p == 0) & (y_true == 1))
            tn = np.sum((p == 0) & (y_true == 0))

            f1_1 = 2.0 * tp / max(2 * tp + fp + fn, 1)
            f1_0 = 2.0 * tn / max(2 * tn + fp + fn, 1)
            macro_f1 = 0.5 * (f1_0 + f1_1)
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_th = float(th)
        return best_f1, best_th

    def tune_threshold(self, x_val: np.ndarray, y_val: np.ndarray, n_points: int = 201) -> float:
        # cari ambang batas yang memaksimalkan macro f1
        scores = self.decision_function(x_val)
        y_true = np.asarray(y_val, dtype=np.int64)
        _, best_th = self._best_macro_f1_threshold(scores, y_true, n_points=n_points)
        self.threshold = best_th
        return best_th

    def __getstate__(self):
        import zlib
        import pickle
        state = self.__dict__.copy()
        heavy = {
            "support_vectors_": state.pop("support_vectors_", None),
            "alphas_": state.pop("alphas_", None),
            "dual_coef_": state.pop("dual_coef_", None),
            "support_y_": state.pop("support_y_", None),
            "support_alphas_": state.pop("support_alphas_", None),
            "support_indices_": state.pop("support_indices_", None),
        }
        state["_heavy_arrays_zlib_"] = zlib.compress(pickle.dumps(heavy), level=6)
        return state

    def __setstate__(self, state):
        import zlib
        import pickle
        heavy_bytes = state.pop("_heavy_arrays_zlib_", None)
        self.__dict__.update(state)
        if heavy_bytes is not None:
            heavy = pickle.loads(zlib.decompress(heavy_bytes))
            self.support_vectors_ = heavy["support_vectors_"]
            self.alphas_ = heavy["alphas_"]
            self.dual_coef_ = heavy["dual_coef_"]
            self.support_y_ = heavy["support_y_"]
            self.support_alphas_ = heavy["support_alphas_"]
            self.support_indices_ = heavy["support_indices_"]

    def tune_threshold_oof(self, x: np.ndarray, y: np.ndarray, n_splits: int = 5, n_points: int = 201) -> float:
        # kalibrasi ambang batas berbasis out-of-fold secara efisien
        x = np.asarray(x, dtype=np.float64)
        y_true = np.asarray(y, dtype=np.int64)
        rng = np.random.default_rng(self.random_state)
        folds = np.zeros(len(y_true), dtype=np.int64)
        for lbl in (0, 1):
            idx = np.flatnonzero(y_true == lbl)
            rng.shuffle(idx)
            for pos, sample_i in enumerate(idx):
                folds[sample_i] = pos % n_splits

        # gunakan subsample basis nystrom yang representatif agar kalibrasi threshold cepat
        sub_comp = min(self.n_components, 900)
        oof_scores = np.zeros(len(y_true), dtype=np.float64)
        for f in range(n_splits):
            va_m = folds == f
            tr_m = ~va_m
            clone = SVMFromScratch(
                C=self.C,
                kernel=self.kernel,
                degree=self.degree,
                gamma=self.gamma,
                coef0=self.coef0,
                class_weight=self.class_weight,
                tol=self.tol,
                max_iter=min(self.max_iter, 20),
                n_components=sub_comp,
                threshold=0.0,
                solver=self.solver,
                random_state=self.random_state + f,
            )
            clone.fit(x[tr_m], y_true[tr_m])
            oof_scores[va_m] = clone.decision_function(x[va_m])

        _, best_th = self._best_macro_f1_threshold(oof_scores, y_true, n_points=n_points)
        self.threshold = best_th
        return best_th

    def calculate_slack(self, x: np.ndarray, y_raw: np.ndarray) -> float:
        y_pm = np.where(y_raw == 1, 1.0, -1.0)
        scores = self.decision_function(x)
        margins = y_pm * scores
        slacks = np.maximum(0.0, 1.0 - margins)
        return float(np.sum(slacks))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    x_toy = rng.normal(size=(200, 10))
    y_toy = (x_toy[:, 0] + x_toy[:, 1] > 0).astype(np.int64)

    for kern in ["linear", "poly", "rbf"]:
        model = SVMFromScratch(kernel=kern, max_iter=25, random_state=42)
        model.fit(x_toy, y_toy)
        preds = model.predict(x_toy)
        assert set(np.unique(preds)).issubset({0, 1})
        assert len(preds) == len(y_toy)
        assert model.slack_total >= 0.0
        print(f"kernel {kern:<6} | acc: {np.mean(preds == y_toy):.3f} | slack: {model.slack_total:.2f} | sv: {model.n_support_}")
