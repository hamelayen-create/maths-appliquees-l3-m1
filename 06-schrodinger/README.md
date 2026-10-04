# 06 — Schrödinger 1D : spectre, paquets d'ondes, tunnellisation

**Niveau :** M1  
**Domaine :** physique quantique numérique  
**Maths :** opérateurs auto-adjoints, matrices hermitiennes, FFT, EDP de Schrödinger

## Accroche

Comment calculer le spectre d’un puits quantique, propager un paquet d’ondes sans casser la norme $L^2$, et mesurer l’effet tunnel à travers une barrière ? Ce module répond par un mini-solveur 1D (différences finies + split-operator FFT) et un polycopié complet.

## Rapport de cours

Polycopié autonome (≥ 15 pages équivalentes) :

- [`report/RAPPORT.md`](report/RAPPORT.md) — cours-projet complet (théorie, algorithmes, expériences, exercices corrigés)
- [`report/BIBLIOGRAPHY.bib`](report/BIBLIOGRAPHY.bib) — références BibTeX
- [`report/figures/`](report/figures/) — figures PNG (≥ 8)
- Génération : `python report/make_figures.py`

## Problème

Résoudre numériquement
$$
i\hbar\partial_t\psi = -\frac{\hbar^2}{2m}\partial_{xx}\psi + V(x)\psi
$$
en 1D : spectre d’un puits, propagation d’un paquet, tunnellisation.

## Méthodes

- Discrétisation du Hamiltonien (différences finies)
- Diagonalisation dense pour les états stationnaires (`eigenpairs`)
- Split-operator FFT (Strang) pour la dynamique
- Conservation de $\|\psi\|_2$ et estimateurs transmission / réflexion

## Usage

Depuis ce dossier :

```bash
python3 src/demo.py
python3 -m pytest tests/
python3 report/make_figures.py
```

## Structure

```text
06-schrodinger/
  src/quantum.py      # API numérique
  src/demo.py         # démo console
  tests/              # pytest
  report/RAPPORT.md   # polycopié
  report/figures/     # PNG
```
