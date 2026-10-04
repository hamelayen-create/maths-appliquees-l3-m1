# 06 — Schrödinger 1D : spectre et paquets d'ondes

**Niveau :** M1  
**Maths :** matrices hermitiennes, valeurs propres, EDP de Schrödinger, FFT

## Problème

Résoudre numériquement
\[
i\hbar\partial_t\psi = -\frac{\hbar^2}{2m}\partial_{xx}\psi + V(x)\psi
\]
en 1D : spectre d'un puits, propagation d'un paquet, tunnellisation.

## Méthodes

- Discrétisation du Hamiltonien (différences finies)
- Diagonalisation dense pour les états stationnaires
- Split-operator FFT pour la dynamique
- Conservation de \(\|\psi\|_2\)

## Usage

```bash
python src/demo.py
pytest tests/
```
