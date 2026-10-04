#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml import LinearRegression, LinearSVM, PCA, make_blobs, make_regression


def main() -> None:
    print("=== ML from scratch ===")
    X, y, w_true = make_regression()
    reg = LinearRegression(ridge=0.1).fit_normal(X, y)
    reg_gd = LinearRegression(ridge=0.1).fit_gd(X, y, lr=0.05, n_iter=5000)
    print("w_true     ", np.round(w_true, 3))
    print("w_normal   ", np.round(reg.coef_, 3))
    print("w_gd       ", np.round(reg_gd.coef_, 3))
    print("MSE normal ", float(np.mean((reg.predict(X) - y) ** 2)))

    # PCA sur données corrélées
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 2))
    A = np.array([[1.0, 0.8], [0.8, 1.0]])
    Xc = Z @ np.linalg.cholesky(A).T
    pca = PCA(n_components=2).fit(Xc)
    print("PCA variance ratios:", np.round(pca.explained_variance_ratio_, 3))

    Xb, yb = make_blobs()
    svm = LinearSVM(C=1.0, lr=0.05, n_iter=4000).fit(Xb, yb)
    acc = float(np.mean(svm.predict(Xb) == yb))
    print(f"SVM train accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
