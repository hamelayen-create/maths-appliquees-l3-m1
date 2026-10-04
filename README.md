# Maths appliquées L3 / M1

Neuf projets de mathématiques appliquées (finance, physique, informatique), pensés pour un portfolio GitHub : théorie courte, code exécutable, tests et figures.

## Projets

| # | Dossier | Domaine | Niveau | Thème |
|---|---------|---------|--------|-------|
| 1 | [`01-black-scholes`](01-black-scholes) | Finance | L3/M1 | Black–Scholes : formule fermée, EDP, Monte Carlo |
| 2 | [`02-markowitz`](02-markowitz) | Finance | L3 | Optimisation de portefeuille (Markowitz / QP) |
| 3 | [`03-vol-smile`](03-vol-smile) | Finance | M1 | Volatilité implicite, smile SVI, surface |
| 4 | [`04-oscillateurs`](04-oscillateurs) | Physique | L3 | Oscillateurs, résonance, Duffing, Fourier |
| 5 | [`05-edp-chaleur-ondes`](05-edp-chaleur-ondes) | Physique | L3/M1 | EDP chaleur / ondes, différences finies |
| 6 | [`06-schrodinger`](06-schrodinger) | Physique | M1 | Schrödinger 1D : spectre et paquets d'ondes |
| 7 | [`07-pagerank`](07-pagerank) | Info | L3 | PageRank, Markov, algèbre linéaire sparse |
| 8 | [`08-ml-from-scratch`](08-ml-from-scratch) | Info | L3/M1 | Régression, PCA, SVM soft-margin |
| 9 | [`09-crypto-info`](09-crypto-info) | Info | L3 | RSA pédagogique, entropie, code de Hamming |

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer tous les tests

```bash
pytest -q
```

## Lancer une démo par projet

```bash
python 01-black-scholes/src/demo.py
python 02-markowitz/src/demo.py
# ... idem pour les autres
```

## Structure d'un projet

```text
0X-nom/
  README.md      # énoncé, maths, usages
  src/           # bibliothèque + demo.py
  tests/         # pytest
  notebooks/     # explorations (optionnel)
```

## Publier sur GitHub

```bash
cd maths-appliquees-l3-m1
# créer un repo vide sur GitHub, puis :
git remote add origin git@github.com:<ton-user>/<ton-repo>.git
git branch -M main
git push -u origin main
```

Ou conserve la branche actuelle `cursor/maths-appliquees-l3-m1-f007` et pousse-la telle quelle.

## Licence

MIT — usage pédagogique libre.
