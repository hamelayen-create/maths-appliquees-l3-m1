import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml import LinearRegression, LinearSVM, PCA, make_blobs, make_regression


def test_regression_recovers_signal():
    X, y, w_true = make_regression(n=400, noise=0.05, seed=2)
    model = LinearRegression(ridge=0.01).fit_normal(X, y)
    assert np.linalg.norm(model.coef_ - w_true) < 0.2


def test_gd_close_to_normal():
    X, y, _ = make_regression(n=250, seed=3)
    m1 = LinearRegression(ridge=0.1).fit_normal(X, y)
    m2 = LinearRegression(ridge=0.1).fit_gd(X, y, lr=0.05, n_iter=12000)
    assert np.linalg.norm(m1.coef_ - m2.coef_) < 0.25


def test_pca_variance_sums_to_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    pca = PCA(n_components=3).fit(X)
    assert abs(pca.explained_variance_ratio_.sum() - 1.0) < 1e-10
    Z = pca.transform(X)
    assert Z.shape == (100, 3)


def test_svm_separates_blobs():
    X, y = make_blobs(n=300, seed=4)
    svm = LinearSVM(C=1.0, lr=0.05, n_iter=5000).fit(X, y)
    acc = np.mean(svm.predict(X) == y)
    assert acc > 0.9
