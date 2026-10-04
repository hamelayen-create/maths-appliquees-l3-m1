# 02 — Optimisation de portefeuille Markowitz

**Niveau :** L3  
**Maths :** optimisation quadratique, algèbre linéaire, covariance, Lagrange

## Problème

Minimiser le risque pour un rendement cible \(\mu_p\) :
\[
\min_w\; w^\top \Sigma w
\quad\text{s.c.}\quad
w^\top\mu = \mu_p,\quad
\mathbf{1}^\top w = 1,\quad
w \ge 0
\]

## Méthodes

- Frontière efficiente analytique (sans contrainte \(w\ge 0\))
- QP avec CVXPY (long-only)
- Shrinkage Ledoit–Wolf de la covariance
- Backtest synthétique (rendements simulés)

## Usage

```bash
python src/demo.py
pytest tests/
```
