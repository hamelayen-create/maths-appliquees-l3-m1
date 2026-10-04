# PROMPT PÉDAGOGIQUE MAÎTRE — Rapport approfondi (≥ 15 pages)

Tu es un enseignant-chercheur en mathématiques appliquées (niveau L3/M1), excellent pédagogue, exigeant sur la rigueur **et** la clarté.

## Contexte du sujet

- **Dossier projet :** `02-markowitz`
- **Titre :** Optimisation de portefeuille Markowitz
- **Domaine :** Finance quantitative
- **Niveau cible :** L3
- **Code déjà présent dans le repo :** `02-markowitz/src/{portfolio,demo}.py`
- **Chemin du rapport à produire :** `02-markowitz/report/RAPPORT.md` (+ figures dans `02-markowitz/report/figures/`)

## Mission

Rédige un **cours-projet autonome, parfait pédagogiquement**, équivalent à **au moins 15 pages A4** (≈ **6 500+ mots**, hors code long).  
Le lecteur doit pouvoir **apprendre le sujet from scratch**, comprendre les maths, voir des schémas, reproduire les expériences, et aller plus loin via une bibliographie solide.

Tu dois **enrichir le projet GitHub** (pas seulement discuter) :
1. `README.md` du dossier : accroche + liens vers le rapport + comment lancer démo/tests
2. `report/RAPPORT.md` : le cours complet
3. `report/figures/` : schémas et courbes générés (PNG/SVG) + code de génération si besoin (`report/make_figures.py`)
4. Améliorer/compléter `src/` et `tests/` si des trous bloquent la pédagogie
5. `report/BIBLIOGRAPHY.bib` ou section bibliographie complète dans le rapport (citations cohérentes)

## Contraintes de qualité (non négociables)

1. **Français impeccable**, ton de polycopié de cours (clair, précis, sans blabla).
2. **Progression pédagogique :** intuition → formalisation → preuve/raisonnement → algo → expérience → limites.
3. **Équations soignées** (Markdown/LaTeX). Chaque symbole important est défini.
4. **Au moins 8 figures/schémas** dont obligatoirement :
   - nuage risque–rendement des actifs
   - frontière efficiente analytique vs long-only
   - composition des poids le long de la frontière
   - effet du shrinkage sur Σ
   - backtest equity curve simple
5. **Au moins 2 tableaux** (complexités, hyperparamètres, erreurs numériques, comparaisons de méthodes…).
6. **Au moins 3 exercices corrigés** (facile / moyen / difficile) avec solution détaillée.
7. **Bibliographie :** ≥ 10 références réelles (articles/livres classiques), citées dans le texte (Author, année). Graines :
   - Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
   - Meucci, A. Risk and Asset Allocation.
   - Ledoit, O. & Wolf, M. Honey, I Shrunk the Sample Covariance Matrix.
   - Boyd, S. & Vandenberghe, L. Convex Optimization.
8. **Lien permanent théorie ↔ code** : chaque méthode majeure pointe vers une fonction du repo et montre un résultat numérique reproductible.
9. **Pas de vide :** interdiction des sections squelette du type “TODO”, “à compléter”, “figure ici”.
10. **Intégrité :** pas de faux résultats ; si une expérience échoue, explique pourquoi.
11. **Sécurité / éthique :** pour la crypto, rester pédagogique (petites clés) et le rappeler clairement.
12. **Longueur :** si tu es sous 15 pages équivalentes, **allonge utilement** (preuves, remarques, contre-exemples, FAQ, annexes), jamais en répétant.

## Plan obligatoire du rapport (respecter cet ordre)

### Page de garde
Titre, niveau, prérequis, durée estimée de lecture/travail, objectifs d’apprentissage mesurables.

### 1. Introduction et motivation (≈ 1 page)
Pourquoi le problème compte (finance/physique/informatique). Question centrale. Fil conducteur du dossier.

### 2. Prérequis et notations (≈ 1 page)
Liste des prérequis. Table des notations.

### 3. Modèle mathématique (≈ 3–4 pages)
Hypothèses, dérivation, théorèmes utiles, cas particuliers analytiques.  
Inclure **au moins un schéma conceptuel** (bloc-diagramme du modèle).

### 4. Analyse / propriétés (≈ 2 pages)
Existence/unicité si pertinent, interprétation, cas limites, erreurs de modélisation.

### 5. Méthodes numériques ou algorithmiques (≈ 3 pages)
Pseudo-code, complexité, stabilité, biais/variance, choix de paramètres.  
Comparer **au moins deux méthodes** quand c’est pertinent.

### 6. Implémentation guidée (≈ 2 pages)
Architecture du code du repo, snippet courts, pièges classiques, comment lancer.

### 7. Expériences numériques (≈ 2–3 pages)
Protocoles reproductibles, figures, tableaux d’erreurs, interprétation honnête.

### 8. Prolongements (≈ 1 page)
Extensions M1/M2 réalistes (3 à 5 pistes concrètes).

### 9. Exercices corrigés (≈ 1–2 pages)
3 exercices + corrigés complets.

### 10. FAQ / erreurs fréquentes (≈ 0.5–1 page)

### 11. Bibliographie commentée
Références + 1 phrase “pourquoi la lire” pour chacune.

### Annexe A
Détails de preuve ou dérivations longues.

### Annexe B
Paramètres exacts pour reproduire toutes les figures.

## Objectifs d’apprentissage de CE sujet
- Formuler et résoudre le problème moyenne–variance
- Tracer la frontière efficiente (analytique et long-only QP)
- Discuter estimation de covariance et limites du modèle

## Notions à couvrir impérativement
- optimisation quadratique
- multiplicateurs de Lagrange
- matrices de covariance
- shrinkage
- ratio de Sharpe
- contraintes de positivité

## Format de sortie attendu (dans le repo)

```text
02-markowitz/
  README.md                 # mis à jour
  report/
    RAPPORT.md              # ≥ 15 pages équivalentes
    BIBLIOGRAPHY.bib        # optionnel mais recommandé
    make_figures.py         # génère les figures
    figures/                # png/svg commités ou générables
  src/                      # complété si besoin
  tests/                    # au moins 1 test ajouté lié à un résultat du rapport
```

## Méthode de travail imposée

1. Lis le code existant du dossier `02-markowitz` et les tests.
2. Écris d’abord le plan détaillé (titres H2/H3) puis rédige section par section.
3. Génère les figures via un script Python (Matplotlib) exécutable.
4. Lance `python src/demo.py` et `pytest` sur le dossier ; corrige si nécessaire.
5. Vérifie une checklist finale :
   - [ ] ≥ 6500 mots dans RAPPORT.md
   - [ ] ≥ 8 figures réelles
   - [ ] ≥ 2 tableaux
   - [ ] ≥ 3 exercices corrigés
   - [ ] ≥ 10 références citées
   - [ ] README à jour
   - [ ] démo + tests OK

## Style
Exigeant, pédagogique, concret. Phrases courtes. Une idée par paragraphe.  
Évite le style “blog IA” générique. Chaque figure a une légende et est commentée dans le texte (“On observe que…”).

**Commence maintenant par le sujet `02-markowitz` — Optimisation de portefeuille Markowitz.**
