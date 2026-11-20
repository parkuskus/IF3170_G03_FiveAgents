from math import exp

import numpy as np


class LogisticRegressionScratch:

    def __init__(
        self,
        learning_rate=0.001,
        max_iter=100,
        reg_strength=0.001,
        l1_ratio=0.0,
        threshold=0.5,
        random_state=42,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.reg_strength = reg_strength
        self.l1_ratio = l1_ratio
        self.threshold = threshold
        self.random_state = random_state

    @staticmethod
    def sigmoid(z):
        z = np.asarray(z, dtype=np.float64)
        q = np.exp(-np.abs(z))
        return np.where(z >= 0, 1.0 / (1.0 + q), q / (1.0 + q))

    @staticmethod
    def _matrix(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError('X must be a 2D numeric matrix with at least one feature')
        if not np.isfinite(X).all():
            raise ValueError('X contains NaN/inf; apply preprocessing first')
        return X

    def _validate_parameters(self):
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError('max_iter must be a positive integer (number of epochs)')
        if not np.isfinite(self.reg_strength) or self.reg_strength < 0:
            raise ValueError('reg_strength must be finite and nonnegative')
        if not np.isfinite(self.l1_ratio) or not 0 <= self.l1_ratio <= 1:
            raise ValueError('l1_ratio must be between 0 and 1')
        if not np.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError('threshold must be strictly between 0 and 1')
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, (int, np.integer)) or self.random_state < 0:
            raise ValueError('random_state must be a nonnegative integer')

    def _objective(self, X, y, w, b):
        z = X @ w + b
        log_loss = np.mean(np.logaddexp(0.0, z) - y * z)
        penalty = self.l1_ratio * np.abs(w).sum()
        penalty += 0.5 * (1.0 - self.l1_ratio) * np.dot(w, w)
        return float(log_loss + self.reg_strength * penalty)

    def fit(self, X, y):
        self._validate_parameters()
        X = self._matrix(X)
        y = np.asarray(y, dtype=np.float64)
        if len(X) == 0 or y.shape != (len(X),) or not np.isin(y, [0, 1]).all():
            raise ValueError('y must contain one binary label per nonempty row')
        if set(np.unique(y)) != {0.0, 1.0}:
            raise ValueError('Training requires both classes 0 and 1')

        n, d = X.shape
        w, b = np.zeros(d, dtype=np.float64), 0.0
        rng = np.random.default_rng(self.random_state)
        eta = self.learning_rate
        l1 = self.reg_strength * self.l1_ratio
        l2 = self.reg_strength * (1.0 - self.l1_ratio)
        self.loss_history_ = [self._objective(X, y, w, b)]

        for _ in range(self.max_iter):
            for i in rng.permutation(n):
                xi = X[i]
                z = float(xi @ w + b)
                q = exp(-abs(z))
                probability = 1.0 / (1.0 + q) if z >= 0 else q / (1.0 + q)
                error = y[i] - probability
                direction = error * xi
                if l2:
                    direction -= l2 * w
                if l1:
                    direction -= l1 * np.sign(w)
                w += eta * direction
                b += eta * error

            objective = self._objective(X, y, w, b)
            if not np.isfinite(objective):
                raise FloatingPointError('Training diverged; check scaling or reduce learning_rate')
            self.loss_history_.append(objective)

        self.coef_ = w
        self.intercept_ = float(b)
        self.n_features_in_ = d
        self.n_iter_ = self.max_iter
        return self

    def _prediction_matrix(self, X):
        if not hasattr(self, 'coef_'):
            raise RuntimeError('Call fit before prediction')
        X = self._matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}')
        return X

    def objective(self, X, y):
        X = self._prediction_matrix(X)
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (len(X),) or len(y) == 0 or not np.isin(y, [0, 1]).all():
            raise ValueError('y must contain one binary label per nonempty row')
        return self._objective(X, y, self.coef_, self.intercept_)

    def predict_proba(self, X):
        X = self._prediction_matrix(X)
        return self.sigmoid(X @ self.coef_ + self.intercept_)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(np.int64)
