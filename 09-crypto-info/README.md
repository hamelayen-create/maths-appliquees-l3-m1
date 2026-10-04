# 09 — RSA pédagogique, entropie et code de Hamming

**Niveau :** L3  
**Maths :** arithmétique modulaire, entropie de Shannon, codes correcteurs  
**Rapport :** [report/RAPPORT.md](report/RAPPORT.md) (≥ 15 pages équivalentes)

## Accroche

Comment chiffrer un message avec de l’arithmétique modulaire, mesurer l’incertitude d’une source, et corriger une erreur de bit sur un canal bruité ? Ce dossier relie **RSA pédagogique**, **entropie / capacité du BSC**, et **Hamming(7,4)** dans un cours-projet reproductible.

> **Avertissement :** l’implémentation RSA est **strictement pédagogique** (petites clés). Elle n’offre **aucune sécurité réelle** et ne doit pas protéger de vraies données.

## Contenu

- RSA pédagogique (Euclide étendu, \(\varphi(n)\), petites clés)
- Entropie de Shannon, entropie binaire, capacité du canal BSC
- Code de Hamming (7,4) : encodage, syndrome, correction 1 bit
- Expériences BER avant/après correction + figures

## Lancer la démo et les tests

```bash
cd 09-crypto-info
python src/demo.py
python -m pytest tests/ -q
```

## Générer les figures du rapport

```bash
python report/make_figures.py
```

Les PNG sont écrits dans `report/figures/`. Bibliographie : `report/BIBLIOGRAPHY.bib`.

## Structure

```text
09-crypto-info/
  src/crypto_info.py   # fonctions principales
  src/demo.py
  tests/test_crypto.py
  report/RAPPORT.md
  report/make_figures.py
  report/figures/
```
