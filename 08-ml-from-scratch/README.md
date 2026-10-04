# 08 — ML from scratch : régression, PCA, SVM

**Niveau :** L3 / M1  
**Maths :** moindres carrés, ridge, descente de gradient, SVD / ACP, hinge loss, biais–variance

Implémentations NumPy pédagogiques (sans scikit-learn) pour comprendre *from scratch* trois piliers de l’apprentissage statistique linéaire : régression, PCA et SVM soft-margin.

## Rapport

Cours-projet complet (≥ 15 pages équivalentes) :

- [report/RAPPORT.md](report/RAPPORT.md) — polycopié (théorie, preuves, expériences, exercices corrigés)
- [report/BIBLIOGRAPHY.bib](report/BIBLIOGRAPHY.bib) — références BibTeX
- [report/figures/](report/figures/) — figures PNG générées
- [report/make_figures.py](report/make_figures.py) — script de génération

## Contenu du code

- Régression linéaire et ridge (`LinearRegression.fit_normal`, `fit_gd`)
- PCA via SVD (`PCA.fit` / `transform`)
- SVM linéaire soft-margin par sous-gradient (`LinearSVM`)
- Helpers `mse`, `r2_score`, générateurs `make_regression` / `make_blobs`

## Usage

```bash
# depuis ce dossier
python src/demo.py
pytest tests/
python report/make_figures.py
```

## Structure

```text
08-ml-from-scratch/
  README.md
  src/ml.py src/demo.py
  tests/
  report/RAPPORT.md
  report/BIBLIOGRAPHY.bib
  report/make_figures.py
  report/figures/*.png
```
