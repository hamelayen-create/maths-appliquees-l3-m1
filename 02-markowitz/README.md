# 02 — Optimisation de portefeuille Markowitz

**Niveau :** L3  
**Domaine :** finance quantitative  
**Maths :** optimisation quadratique, multiplicateurs de Lagrange, matrices de covariance, shrinkage, ratio de Sharpe

Minimiser le risque pour un rendement cible $\mu_p$ :

$$
\min_w\; w^\top \Sigma w
\quad\text{s.c.}\quad
w^\top\mu = \mu_p,\quad
\mathbf{1}^\top w = 1,\quad
w \ge 0
$$

## Rapport de cours (polycopié)

Le cours-projet complet (≥ 15 pages équivalentes) est ici :

- **[report/RAPPORT.md](report/RAPPORT.md)** — théorie, algorithmes, expériences, exercices corrigés
- **[report/BIBLIOGRAPHY.bib](report/BIBLIOGRAPHY.bib)** — références BibTeX
- **[report/figures/](report/figures/)** — figures PNG (nuage risque–rendement, frontières, shrinkage, backtest, …)
- Génération des figures : `python3 report/make_figures.py`

## Méthodes implémentées

| Méthode | Fonction | Fichier |
|--------|----------|---------|
| Frontière analytique (Lagrange) | `efficient_frontier_analytic` | `src/portfolio.py` |
| QP long-only (CVXPY) | `optimize_long_only` | idem |
| Frontière long-only | `efficient_frontier_long_only` | idem |
| Shrinkage Ledoit–Wolf (pédagogique) | `ledoit_wolf_cov` | idem |
| GMV / max Sharpe | `gmv_weights`, `max_sharpe_weights` | idem |
| Backtest equity | `backtest_equity` | idem |
| Données synthétiques | `simulate_returns` | idem |

## Usage

Depuis ce dossier (dépendances : `pip install -r ../requirements.txt`) :

```bash
python3 src/demo.py
python3 -m pytest tests/ -q
python3 report/make_figures.py
```

## Structure

```text
02-markowitz/
  README.md
  src/portfolio.py   # cœur mathématique
  src/demo.py
  tests/
  report/
    RAPPORT.md
    BIBLIOGRAPHY.bib
    make_figures.py
    figures/
```
