# 05 — EDP chaleur et ondes (différences finies)

**Niveau :** L3 / M1  
**Maths :** EDP linéaires, séparation de variables, stabilité de von Neumann, CFL, ordre de convergence

Cours-projet autonome : de la physique 1D (barre thermique, corde vibrante) jusqu’aux schémas FTCS / BTCS / Crank–Nicolson / leapfrog, avec expériences reproductibles et mesure d’ordres.

## Rapport

- **[report/RAPPORT.md](report/RAPPORT.md)** — polycopié complet (≥ 15 pages équivalentes)
- **[report/BIBLIOGRAPHY.bib](report/BIBLIOGRAPHY.bib)** — références BibTeX
- **Figures :** [report/figures/](report/figures/) (générées par `report/make_figures.py`)

## Problèmes

- Chaleur 1D : $u_t = \kappa u_{xx}$
- Onde 1D : $u_{tt} = c^2 u_{xx}$

## Méthodes (code dans `src/pde.py`)

| Schéma | Fonction | Stabilité |
|--------|----------|-----------|
| FTCS | `heat_ftcs` | $r=\kappa\Delta t/\Delta x^2\le 1/2$ |
| BTCS | `heat_btcs` | inconditionnelle |
| Crank–Nicolson | `heat_crank_nicolson` | inconditionnelle |
| Leapfrog | `wave_leapfrog` | CFL $\lambda=c\Delta t/\Delta x\le 1$ |

## Usage

Depuis ce dossier :

```bash
python3 src/demo.py              # erreurs L2 + ordre observé
python3 report/make_figures.py   # régénère les PNG du rapport
python3 -m pytest tests/ -q      # tests de convergence / stabilité
```

Dépendances : `numpy`, `scipy`, `matplotlib`, `pytest` (voir `../requirements.txt`).
