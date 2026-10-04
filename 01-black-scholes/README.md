# 01 — Black–Scholes : formule fermée, EDP et Monte Carlo

**Niveau :** L3 fin / M1 début  
**Maths :** EDP, GBM, intégration gaussienne, différences finies, Monte Carlo

## Accroche

Trois façons indépendantes de pricer un call européen sous Black–Scholes — formule fermée, EDP Crank–Nicolson, Monte Carlo — et de mesurer leurs erreurs. Le dossier est un **cours-projet autonome** : théorie, code, figures et exercices corrigés.

## Rapport

Polycopié complet (≥ 15 pages / 6500+ mots) :

- **[report/RAPPORT.md](report/RAPPORT.md)** — cours détaillé en français
- **[report/BIBLIOGRAPHY.bib](report/BIBLIOGRAPHY.bib)** — références BibTeX
- **[report/figures/](report/figures/)** — figures PNG (GBM, prix, MC, EDP, surface, Grecs…)
- Génération des figures : `python3 report/make_figures.py`

## Problème

Prix d'un call européen sous l'hypothèse Black–Scholes :
\[
dS_t = r S_t\,dt + \sigma S_t\,dW_t
\]
Le prix \(V(t,S)\) vérifie
\[
\partial_t V + r S \partial_S V + \tfrac12\sigma^2 S^2\partial_{SS}V - rV = 0
\]
avec \(V(T,S)=(S-K)^+\).

## Méthodes

1. **Formule fermée** (N(d1), N(d2)) — `src/bs_closed_form.py`
2. **EDP** : Crank–Nicolson en variables log-spot — `src/bs_pde.py`
3. **Monte Carlo** : simulation de \(S_T\), estimateur \(\mathrm{e}^{-rT}(S_T-K)^+\) — `src/bs_mc.py`

## Usage

```bash
# Démo comparative (formule fermée / EDP / MC)
python3 src/demo.py

# Régénérer les figures du rapport
python3 report/make_figures.py

# Tests
pytest tests/
```

## Structure

```text
01-black-scholes/
  README.md
  src/           # closed form, PDE, MC, demo
  tests/         # parité, convergence MC/PDE, Grecs
  report/        # RAPPORT.md, figures, make_figures.py
```

## Extensions

Grecs pathwise, barrière, contrôle de variance (antithetic / control variate) — voir section 8 du rapport.
