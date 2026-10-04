# 07 — PageRank, chaînes de Markov et algèbre linéaire sparse

**Niveau :** L3  
**Maths :** graphes orientés, matrices stochastiques, méthode de la puissance, Laplacien / Fiedler

Sur un graphe orienté, le score PageRank \(r\) est la mesure stationnaire du *surfeur aléatoire* amorti :
\[
r = \alpha \tilde{P}^\top r + \frac{1-\alpha}{N}\mathbf{1}.
\]

Ce dossier propose un **cours-projet autonome** (théorie, preuves, expériences, exercices corrigés) branché sur une implémentation sparse NumPy/SciPy.

## Rapport

- Cours complet : [`report/RAPPORT.md`](report/RAPPORT.md) (≥ 15 pages équivalentes)
- Figures : [`report/figures/`](report/figures/) (générées par [`report/make_figures.py`](report/make_figures.py))
- Bibliographie : [`report/BIBLIOGRAPHY.bib`](report/BIBLIOGRAPHY.bib)

## Méthodes implémentées

| Fonction | Rôle |
|----------|------|
| `random_graph` | digraphe Erdős–Rényi sparse |
| `pagerank_power` | méthode de la puissance (+ historique des résidus) |
| `pagerank_linear` | système \((I-\alpha P^\top)r=(1-\alpha)v\) via GMRES |
| `fiedler_vector` | 2ᵉ vecteur propre du Laplacien (partition) |

## Usage

```bash
# Démonstration
python src/demo.py

# Tests
pytest tests/

# Régénérer les figures du rapport
python report/make_figures.py
```

## Objectifs d’apprentissage

1. Modéliser le surfeur aléatoire et le damping \(\alpha\)
2. Implémenter power method et résolution linéaire
3. Relier à la théorie spectrale des graphes (Fiedler)
