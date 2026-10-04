# 04 — Oscillateurs, résonance et systèmes dynamiques

**Niveau :** L3  
**Maths :** EDO linéaires, espace de phase, résonance, Fourier, Duffing

Étudier l’oscillateur harmonique amorti / forcé, la résonance et le facteur de qualité, puis l’oscillateur de Duffing (portrait de phase, spectre FFT, hystérésis qualitative).

## Rapport de cours

Polycopié complet (≥ 15 pages équivalentes) :

- [report/RAPPORT.md](report/RAPPORT.md)
- Figures : [report/figures/](report/figures/)
- Bibliographie : [report/BIBLIOGRAPHY.bib](report/BIBLIOGRAPHY.bib)
- Génération des figures : `python3 report/make_figures.py`

## Méthodes

- Solutions analytiques (libre : 3 régimes ; forcé : régime permanent)
- Intégration RK4 (pédagogique) et RK45 (SciPy)
- Courbe de résonance $A(\omega)$, facteur de qualité $Q$
- Spectre FFT et section de Poincaré (Duffing)

## Usage

```bash
# depuis ce dossier
python3 src/demo.py
python3 -m pytest tests/
python3 report/make_figures.py
```

## Structure

```text
src/oscillators.py   # API mathématique
src/demo.py          # démonstration console
tests/               # tests de non-régression liés au rapport
report/              # cours, figures, bib
```
