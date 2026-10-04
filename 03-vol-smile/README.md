# 03 — Smile de volatilité implicite et calibration SVI

**Niveau :** M1  
**Maths :** inversion numérique, optimisation non linéaire, modèle SVI

## Problème

À partir de prix de calls, extraire \(\sigma_{\mathrm{imp}}(K)\) puis calibrer un smile SVI :
\[
w(k) = a + b\bigl(\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}\bigr)
\]
où \(k=\log(K/F)\) et \(w=\sigma_{\mathrm{imp}}^2 T\).

## Méthodes

1. Implied vol par Brent (inversion BS)
2. Fit SVI par moindres carrés
3. Construction d'une surface simple (plusieurs maturités)
4. Pricing d'un Asian arithmétique MC sous BS local (vol ATM)

## Usage

```bash
python src/demo.py
pytest tests/
```
