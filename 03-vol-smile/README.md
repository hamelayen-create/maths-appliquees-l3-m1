# 03 — Smile de volatilité implicite et calibration SVI

**Niveau :** M1 — Finance quantitative  
**Maths :** inversion numérique Black–Scholes, optimisation non linéaire, modèle SVI raw, no-arbitrage butterfly/calendar

Les prix d’options ne se résument pas à une volatilité constante. Ce projet montre comment extraire le **smile** \(\sigma_{\mathrm{imp}}(K)\), le paramétriser par **SVI**, diagnostiquer les arbitrages, puis construire une **surface** simple.

## Rapport de cours

Cours-projet complet (≥ 15 pages équivalentes) :

- [report/RAPPORT.md](report/RAPPORT.md) — polycopié pédagogique
- [report/BIBLIOGRAPHY.bib](report/BIBLIOGRAPHY.bib) — références BibTeX
- [report/figures/](report/figures/) — figures PNG (≥ 8)
- Génération : `python3 report/make_figures.py`

## Problème

À partir de prix de calls, extraire \(\sigma_{\mathrm{imp}}(K)\) puis calibrer un smile SVI :

\[
w(k) = a + b\bigl(\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}\bigr)
\]

où \(k=\log(K/F)\) et \(w=\sigma_{\mathrm{imp}}^2 T\).

## Méthodes

1. Implied vol par Brent (et Newton en comparaison)
2. Fit SVI par moindres carrés bornés (TRF)
3. Diagnostic butterfly \(g(k)\ge 0\) et projection calendaire
4. Construction d’une surface \(w(k,T)\) / \(\sigma(K,T)\)

## Structure du code

| Fichier | Rôle |
|---------|------|
| `src/implied_vol.py` | `bs_call`, `implied_vol_call`, `implied_vol_newton` |
| `src/svi.py` | SVI, calibration, smile synthétique, surface, butterfly |
| `src/demo.py` | expérience reproductible du rapport |
| `tests/test_vol.py` | tests unitaires + borne RMSE du rapport |

## Usage

Depuis ce dossier :

```bash
python3 src/demo.py
python3 -m pytest tests/ -q
python3 report/make_figures.py
```

Dépendances : `numpy`, `scipy`, `matplotlib`, `pytest` (voir `../requirements.txt`).

## Objectifs d’apprentissage

- Extraire la volatilité implicite par inversion BS
- Calibrer un smile SVI et discuter le no-arbitrage
- Construire une surface simple et l’utiliser pour pricer des vanilles
