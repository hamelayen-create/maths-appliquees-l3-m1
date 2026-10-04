# 01 — Black–Scholes : formule fermée, EDP et Monte Carlo

**Niveau :** L3 fin / M1 début  
**Maths :** EDP, GBM, intégration gaussienne, différences finies, Monte Carlo

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

1. **Formule fermée** (N(d1), N(d2))
2. **EDP** : Crank–Nicolson en variables log-spot
3. **Monte Carlo** : simulation de \(S_T\), estimateur \(\mathrm{e}^{-rT}(S_T-K)^+\)

## Usage

```bash
python src/demo.py
pytest tests/
```

## Extensions

Grecs pathwise, barrière, contrôle de variance (antithetic / control variate).
