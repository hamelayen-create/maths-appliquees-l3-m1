# 05 — EDP chaleur et ondes (différences finies)

**Niveau :** L3 / M1  
**Maths :** EDP linéaires, stabilité de von Neumann, ordre de convergence

## Problèmes

- Chaleur 1D : \(u_t = \kappa u_{xx}\)
- Onde 1D : \(u_{tt} = c^2 u_{xx}\)

## Méthodes

- FTCS / BTCS / Crank–Nicolson (chaleur)
- Schéma saute-mouton (onde) avec CFL
- Mesure d'ordre observé vs solution exacte

## Usage

```bash
python src/demo.py
pytest tests/
```
