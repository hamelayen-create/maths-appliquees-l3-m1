"""Implémentations pédagogiques : régression, PCA, SVM soft-margin."""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erreur quadratique moyenne."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient de détermination R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


class LinearRegression:
    def __init__(self, ridge: float = 0.0):
        self.ridge = ridge
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.loss_history_: list[float] = []
        self.path_: list[np.ndarray] = []

    def fit_normal(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_des = np.column_stack([np.ones(len(X)), X])
        n_features = X_des.shape[1]
        reg = self.ridge * np.eye(n_features)
        reg[0, 0] = 0.0  # ne pas régulariser l'intercept
        beta = np.linalg.solve(X_des.T @ X_des + reg, X_des.T @ y)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        self.loss_history_ = []
        self.path_ = []
        return self

    def fit_gd(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.1,
        n_iter: int = 2000,
        store_path: bool = False,
        path_every: int = 1,
    ) -> "LinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        self.loss_history_ = []
        self.path_ = []
        for t in range(n_iter):
            pred = X @ w + b
            err = pred - y
            loss = 0.5 * float(np.mean(err**2)) + 0.5 * self.ridge * float(np.dot(w, w))
            self.loss_history_.append(loss)
            if store_path and (t % path_every == 0):
                self.path_.append(np.concatenate([[b], w.copy()]))
            w -= lr * ((X.T @ err) / n + self.ridge * w)
            b -= lr * float(err.mean())
        self.coef_ = w
        self.intercept_ = b
        if store_path:
            self.path_.append(np.concatenate([[b], w.copy()]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) - self.predict(X)


class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # SVD économique
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        var = (s**2) / (len(X) - 1)
        self.explained_variance_ratio_ = var / var.sum()
        self.components_ = vt[: self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=float) - self.mean_
        return Xc @ self.components_.T


class LinearSVM:
    """SVM soft-margin primal par sous-gradient (hinge)."""

    def __init__(self, C: float = 1.0, lr: float = 0.01, n_iter: int = 3000):
        self.C = C
        self.lr = lr
        self.n_iter = n_iter
        self.w_: np.ndarray | None = None
        self.b_: float = 0.0
        self.loss_history_: list[float] = []

    def _hinge_objective(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
        margins = y * (X @ w + b)
        hinge = np.maximum(0.0, 1.0 - margins)
        return 0.5 * float(np.dot(w, w)) / len(X) + self.C * float(np.mean(hinge))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVM":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # y in {-1, +1}
        y = np.where(y <= 0, -1.0, 1.0)
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        rng = np.random.default_rng(0)
        self.loss_history_ = []
        for t in range(1, self.n_iter + 1):
            i = int(rng.integers(0, n))
            xi, yi = X[i], y[i]
            margin = yi * (xi @ w + b)
            lr_t = self.lr / (1 + 0.001 * t)
            if margin < 1:
                w = w - lr_t * (w / n - self.C * yi * xi)
                b = b + lr_t * self.C * yi
            else:
                w = w - lr_t * (w / n)
            if t % 50 == 0 or t == 1:
                self.loss_history_.append(self._hinge_objective(X, y, w, b))
        self.w_ = w
        self.b_ = b
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.w_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1)


def make_regression(n: int = 200, d: int = 3, noise: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = X @ w_true + 0.5 + noise * rng.normal(size=n)
    return X, y, w_true


def make_blobs(n: int = 200, seed: int = 1):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(loc=[-1.0, -1.0], scale=0.6, size=(n // 2, 2))
    X2 = rng.normal(loc=[1.0, 1.0], scale=0.6, size=(n - n // 2, 2))
    X = np.vstack([X1, X2])
    y = np.array([-1] * (n // 2) + [1] * (n - n // 2))
    return X, y
