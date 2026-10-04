"""Tests liés aux résultats numériques commentés dans le rapport."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml import LinearRegression, LinearSVM, PCA, make_blobs, make_regression, mse, r2_score


def test_mse_and_r2_helpers():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 3.0])
    assert mse(y, pred) == 0.0
    assert abs(r2_score(y, pred) - 1.0) < 1e-12


def test_rapport_regression_mse_bounded():
    """Résultat du rapport §7 : MSE d'entraînement raisonnable sur make_regression."""
    X, y, _ = make_regression(n=200, d=3, noise=0.1, seed=0)
    model = LinearRegression(ridge=0.1).fit_normal(X, y)
    err = mse(y, model.predict(X))
    assert err < 0.05
    assert r2_score(y, model.predict(X)) > 0.9


def test_rapport_gd_loss_decreases():
    X, y, _ = make_regression(n=150, d=2, seed=11)
    model = LinearRegression(ridge=0.05).fit_gd(X, y, lr=0.1, n_iter=500)
    assert len(model.loss_history_) == 500
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_rapport_pca_first_component_dominates_correlated():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 2))
    A = np.array([[1.0, 0.8], [0.8, 1.0]])
    X = Z @ np.linalg.cholesky(A).T
    pca = PCA(n_components=2).fit(X)
    assert pca.explained_variance_ratio_[0] > 0.7
    assert abs(pca.explained_variance_ratio_.sum() - 1.0) < 1e-10


def test_rapport_svm_accuracy_threshold():
    X, y = make_blobs(n=200, seed=1)
    svm = LinearSVM(C=1.0, lr=0.05, n_iter=4000).fit(X, y)
    acc = float(np.mean(svm.predict(X) == y))
    assert acc > 0.85
    assert len(svm.loss_history_) > 0
