# 07 — PageRank, graphes et Markov

**Niveau :** L3  
**Maths :** chaînes de Markov, valeurs propres, méthode de la puissance, matrices sparse

## Problème

Sur un graphe orienté, le score PageRank \(r\) vérifie
\[
r = \alpha P^\top r + \frac{1-\alpha}{N}\mathbf{1}
\]
où \(P\) est la matrice de transition (avec gestion des dangling nodes).

## Méthodes

- Construction sparse de \(P\)
- Itération de la puissance
- Comparaison avec un solveur linéaire sparse
- Bonus : vecteur de Fiedler (Laplacien)

## Usage

```bash
python src/demo.py
pytest tests/
```
