#!/usr/bin/env python3
"""
Génère, pour chaque projet du monorepo, un prompt pédagogique prêt à coller
dans Cursor / ChatGPT / Claude, afin de produire un rapport ≥ 15 pages.

Usage:
  python3 scripts/generer_prompts_pedagogiques.py
  python3 scripts/generer_prompts_pedagogiques.py --projet 01-black-scholes
  python3 scripts/generer_prompts_pedagogiques.py --stdout 01-black-scholes

Sortie:
  prompts/<dossier>/PROMPT.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "prompts"

PROJETS = [
    {
        "id": "01-black-scholes",
        "titre": "Black–Scholes : formule fermée, EDP et Monte Carlo",
        "domaine": "Finance quantitative",
        "niveau": "L3 fin / M1 début",
        "objectifs": [
            "Dériver l'EDP de Black–Scholes et la formule fermée call/put",
            "Implémenter et comparer pricing EDP (Crank–Nicolson) vs Monte Carlo",
            "Analyser erreurs, convergence, variance et Grecs",
        ],
        "notions": [
            "mouvement brownien géométrique",
            "EDP paraboliques",
            "différences finies",
            "Monte Carlo et réduction de variance",
            "put-call parity",
            "Grecs (Δ, Γ, Vega)",
        ],
        "figures_min": [
            "trajectoires GBM",
            "prix call vs K et vs σ",
            "convergence MC (estimateur ± IC)",
            "erreur EDP vs pas d'espace/temps",
            "surface de prix V(S,t)",
        ],
        "biblio_seed": [
            "Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. JPE.",
            "Hull, J. Options, Futures, and Other Derivatives.",
            "Wilmott, P., Howison, S., Dewynne, J. The Mathematics of Financial Derivatives.",
            "Glasserman, P. Monte Carlo Methods in Financial Engineering.",
        ],
        "code_existant": "01-black-scholes/src/{bs_closed_form,bs_pde,bs_mc,demo}.py",
    },
    {
        "id": "02-markowitz",
        "titre": "Optimisation de portefeuille Markowitz",
        "domaine": "Finance quantitative",
        "niveau": "L3",
        "objectifs": [
            "Formuler et résoudre le problème moyenne–variance",
            "Tracer la frontière efficiente (analytique et long-only QP)",
            "Discuter estimation de covariance et limites du modèle",
        ],
        "notions": [
            "optimisation quadratique",
            "multiplicateurs de Lagrange",
            "matrices de covariance",
            "shrinkage",
            "ratio de Sharpe",
            "contraintes de positivité",
        ],
        "figures_min": [
            "nuage risque–rendement des actifs",
            "frontière efficiente analytique vs long-only",
            "composition des poids le long de la frontière",
            "effet du shrinkage sur Σ",
            "backtest equity curve simple",
        ],
        "biblio_seed": [
            "Markowitz, H. (1952). Portfolio Selection. Journal of Finance.",
            "Meucci, A. Risk and Asset Allocation.",
            "Ledoit, O. & Wolf, M. Honey, I Shrunk the Sample Covariance Matrix.",
            "Boyd, S. & Vandenberghe, L. Convex Optimization.",
        ],
        "code_existant": "02-markowitz/src/{portfolio,demo}.py",
    },
    {
        "id": "03-vol-smile",
        "titre": "Smile de volatilité implicite et calibration SVI",
        "domaine": "Finance quantitative",
        "niveau": "M1",
        "objectifs": [
            "Extraire la volatilité implicite par inversion BS",
            "Calibrer un smile SVI et discuter no-arbitrage",
            "Construire une surface simple et l'utiliser pour pricer",
        ],
        "notions": [
            "volatilité implicite",
            "smile / skew",
            "modèle SVI raw",
            "optimisation non linéaire",
            "conditions d'arbitrage butterfly/calendar",
            "forward et total variance",
        ],
        "figures_min": [
            "smile σ_imp(K) pour une maturité",
            "fit SVI vs marché (synthétique)",
            "résidus de calibration",
            "surface w(k,T) ou σ(K,T)",
            "sensibilité des paramètres SVI",
        ],
        "biblio_seed": [
            "Gatheral, J. The Volatility Surface.",
            "Gatheral, J. & Jacquier, A. Arbitrage-free SVI volatility surfaces.",
            "Hagan et al. Managing Smile Risk (SABR).",
            "Rouah, F. The Heston Model and its Extensions in Matlab and C#.",
        ],
        "code_existant": "03-vol-smile/src/{implied_vol,svi,demo}.py",
    },
    {
        "id": "04-oscillateurs",
        "titre": "Oscillateurs, résonance et systèmes dynamiques",
        "domaine": "Physique",
        "niveau": "L3",
        "objectifs": [
            "Résoudre analytiquement l'oscillateur amorti/forcé",
            "Expliquer la résonance et le facteur de qualité",
            "Étudier Duffing : phase space, spectre FFT",
        ],
        "notions": [
            "EDO linéaires à coeff. constants",
            "régimes sous/critique/sur-amorti",
            "réponse fréquentielle",
            "portrait de phase",
            "non-linéarité (Duffing)",
            "analyse de Fourier discrète",
        ],
        "figures_min": [
            "x(t) libre amorti (analytique vs numérique)",
            "courbe de résonance A(ω)",
            "portrait de phase Duffing",
            "spectre FFT régime permanent",
            "diagramme qualitatif de bifurcation (si pertinent)",
        ],
        "biblio_seed": [
            "Arnold, V. I. Ordinary Differential Equations.",
            "Strogatz, S. Nonlinear Dynamics and Chaos.",
            "Feynman, Lectures on Physics, Vol. I (résonance).",
            "Jordan, D. & Smith, P. Nonlinear ODEs.",
        ],
        "code_existant": "04-oscillateurs/src/{oscillators,demo}.py",
    },
    {
        "id": "05-edp-chaleur-ondes",
        "titre": "EDP chaleur et ondes : analyse et schémas numériques",
        "domaine": "Physique / Analyse numérique",
        "niveau": "L3 / M1",
        "objectifs": [
            "Présenter chaleur et ondes (1D) avec solutions exactes",
            "Construire FTCS/BTCS/CN et leapfrog, analyser CFL/stabilité",
            "Mesurer ordres de convergence numériques",
        ],
        "notions": [
            "EDP linéaires",
            "séparation de variables",
            "différences finies",
            "stabilité de von Neumann",
            "condition CFL",
            "ordre de consistance / convergence",
        ],
        "figures_min": [
            "évolution u(x,t) chaleur",
            "corde vibrante u(x,t)",
            "erreur L2 vs Δx (pente = ordre)",
            "domaine de stabilité (schéma explicite)",
            "comparaison schémas (dispersion/diffusion numérique)",
        ],
        "biblio_seed": [
            "Evans, L. C. Partial Differential Equations.",
            "LeVeque, R. Finite Difference Methods for ODEs and PDEs.",
            "Allaire, G. Analyse numérique et optimisation.",
            "Strikwerda, J. Finite Difference Schemes and Partial Differential Equations.",
        ],
        "code_existant": "05-edp-chaleur-ondes/src/{pde,demo}.py",
    },
    {
        "id": "06-schrodinger",
        "titre": "Schrödinger 1D : spectre, paquets d'ondes, tunnellisation",
        "domaine": "Physique quantique numérique",
        "niveau": "M1",
        "objectifs": [
            "Discrétiser le Hamiltonien et calculer un spectre",
            "Propager un paquet d'ondes (split-operator)",
            "Illustrer la tunnellisation et la conservation de la norme",
        ],
        "notions": [
            "équation de Schrödinger",
            "opérateurs auto-adjoints",
            "valeurs / vecteurs propres",
            "FFT et split-operator",
            "effet tunnel",
            "conservation de probabilité",
        ],
        "figures_min": [
            "potentiel V(x) et densités |ψ_n|²",
            "spectre numérique vs analytique (puits)",
            "animation/snapshots |ψ(x,t)|²",
            "norme ||ψ|| au cours du temps",
            "transmission vs énergie / largeur de barrière",
        ],
        "biblio_seed": [
            "Griffiths, D. Introduction to Quantum Mechanics.",
            "Thijssen, J. Computational Physics.",
            "Tannor, D. Introduction to Quantum Mechanics: A Time-Dependent Perspective.",
            "Press et al. Numerical Recipes (FFT / PDE).",
        ],
        "code_existant": "06-schrodinger/src/{quantum,demo}.py",
    },
    {
        "id": "07-pagerank",
        "titre": "PageRank, chaînes de Markov et algèbre linéaire sparse",
        "domaine": "Informatique / maths discrètes",
        "niveau": "L3",
        "objectifs": [
            "Modéliser le surfeur aléatoire et le damping",
            "Implémenter power method et résolution linéaire",
            "Relier à la théorie spectrale des graphes (Fiedler)",
        ],
        "notions": [
            "graphes orientés",
            "matrices stochastiques",
            "chaînes de Markov",
            "méthode de la puissance",
            "dangling nodes",
            "Laplacien et vecteur de Fiedler",
        ],
        "figures_min": [
            "petit graphe annoté avec scores",
            "convergence ||r_{k+1}-r_k||",
            "effet de α (damping)",
            "top-k nœuds sur graphe aléatoire",
            "partition induite par Fiedler",
        ],
        "biblio_seed": [
            "Page, L. et al. The PageRank Citation Ranking (1999).",
            "Langville, A. & Meyer, C. Google's PageRank and Beyond.",
            "Newman, M. Networks: An Introduction.",
            "Golub, G. & Van Loan, C. Matrix Computations.",
        ],
        "code_existant": "07-pagerank/src/{pagerank,demo}.py",
    },
    {
        "id": "08-ml-from-scratch",
        "titre": "Apprentissage statistique from scratch : régression, PCA, SVM",
        "domaine": "Informatique / statistiques",
        "niveau": "L3 / M1",
        "objectifs": [
            "Dériver et coder régression linéaire/ridge",
            "Expliquer PCA via SVD",
            "Formuler SVM soft-margin et l'entraîner (sous-gradient)",
        ],
        "notions": [
            "moindres carrés",
            "régularisation ridge",
            "descente de gradient",
            "SVD / ACP",
            "hinge loss",
            "biais–variance",
        ],
        "figures_min": [
            "fit régression + résidus",
            "chemin de convergence du gradient",
            "variance expliquée PCA",
            "projection 2D PCA",
            "frontière de décision SVM",
        ],
        "biblio_seed": [
            "Hastie, Tibshirani, Friedman. The Elements of Statistical Learning.",
            "Bishop, C. Pattern Recognition and Machine Learning.",
            "Boyd & Vandenberghe. Convex Optimization.",
            "Cortes, C. & Vapnik, V. Support-Vector Networks (1995).",
        ],
        "code_existant": "08-ml-from-scratch/src/{ml,demo}.py",
    },
    {
        "id": "09-crypto-info",
        "titre": "RSA pédagogique, entropie de Shannon et code de Hamming",
        "domaine": "Informatique / maths discrètes",
        "niveau": "L3",
        "objectifs": [
            "Expliquer RSA via arithmétique modulaire (usage pédagogique)",
            "Calculer entropie et capacité d'un BSC",
            "Encoder/décoder Hamming(7,4) avec correction 1 bit",
        ],
        "notions": [
            "arithétique modulaire",
            "Euclide étendu",
            "fonction indicatrice d'Euler",
            "entropie de Shannon",
            "canal binaire symétrique",
            "codes linéaires / matrice de contrôle",
        ],
        "figures_min": [
            "schéma de principe RSA (clés, chiffrement)",
            "courbe H(p) et capacité BSC",
            "structure Hamming(7,4) (bits data/parité)",
            "syndrome → position d'erreur",
            "taux d'erreur avant/après correction",
        ],
        "biblio_seed": [
            "Shannon, C. (1948). A Mathematical Theory of Communication.",
            "Cover, T. & Thomas, J. Elements of Information Theory.",
            "Trappe, W. & Washington, L. Introduction to Cryptography with Coding Theory.",
            "MacKay, D. Information Theory, Inference, and Learning Algorithms.",
        ],
        "code_existant": "09-crypto-info/src/{crypto_info,demo}.py",
    },
]


MASTER_PROMPT = """# PROMPT PÉDAGOGIQUE MAÎTRE — Rapport approfondi (≥ 15 pages)

Tu es un enseignant-chercheur en mathématiques appliquées (niveau L3/M1), excellent pédagogue, exigeant sur la rigueur **et** la clarté.

## Contexte du sujet

- **Dossier projet :** `{id}`
- **Titre :** {titre}
- **Domaine :** {domaine}
- **Niveau cible :** {niveau}
- **Code déjà présent dans le repo :** `{code_existant}`
- **Chemin du rapport à produire :** `{id}/report/RAPPORT.md` (+ figures dans `{id}/report/figures/`)

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
3. **Équations soignées** (Markdown/LaTeX **compatible GitHub**).
   Utiliser uniquement `$...$` (inline) et `$$...$$` (display), **jamais** `\\(...\\)` ni `\\[...\\]`.
   Chaque symbole important est défini. Échapper `<`/`>` hors maths (`&lt;` / `&gt;`) pour éviter un rendu HTML cassé.
4. **Au moins 8 figures/schémas** dont obligatoirement :
{figures}
5. **Au moins 2 tableaux** (complexités, hyperparamètres, erreurs numériques, comparaisons de méthodes…).
6. **Au moins 3 exercices corrigés** (facile / moyen / difficile) avec solution détaillée.
7. **Bibliographie :** ≥ 10 références réelles (articles/livres classiques), citées dans le texte (Author, année). Graines :
{biblio}
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
{objectifs}

## Notions à couvrir impérativement
{notions}

## Format de sortie attendu (dans le repo)

```text
{id}/
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

1. Lis le code existant du dossier `{id}` et les tests.
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

**Commence maintenant par le sujet `{id}` — {titre}.**
"""


def render_prompt(p: dict) -> str:
    figures = "\n".join(f"   - {f}" for f in p["figures_min"])
    biblio = "\n".join(f"   - {b}" for b in p["biblio_seed"])
    objectifs = "\n".join(f"- {o}" for o in p["objectifs"])
    notions = "\n".join(f"- {n}" for n in p["notions"])
    return MASTER_PROMPT.format(
        id=p["id"],
        titre=p["titre"],
        domaine=p["domaine"],
        niveau=p["niveau"],
        code_existant=p["code_existant"],
        figures=figures,
        biblio=biblio,
        objectifs=objectifs,
        notions=notions,
    )


def write_prompts(selected: list[dict], to_stdout: bool) -> None:
    if to_stdout:
        print(render_prompt(selected[0]))
        return

    OUT_DIR.mkdir(exist_ok=True)
    index_lines = [
        "# Prompts pédagogiques (rapports ≥ 15 pages)",
        "",
        "Générés par `scripts/generer_prompts_pedagogiques.py`.",
        "",
        "Usage typique dans Cursor :",
        "1. Ouvre `prompts/<projet>/PROMPT.md`",
        "2. Lance un agent en mode Agent avec ce prompt",
        "3. Vérifie `report/RAPPORT.md`, figures, tests",
        "",
        "| Projet | Fichier |",
        "|--------|---------|",
    ]

    for p in selected:
        dest_dir = OUT_DIR / p["id"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "PROMPT.md"
        dest.write_text(render_prompt(p), encoding="utf-8")
        index_lines.append(f"| `{p['id']}` | [`{p['id']}/PROMPT.md`]({p['id']}/PROMPT.md) |")
        print(f"OK  {dest.relative_to(ROOT)}")

    (OUT_DIR / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"OK  { (OUT_DIR / 'README.md').relative_to(ROOT) }")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projet",
        help="Identifiant du projet (ex: 01-black-scholes). Défaut: tous.",
    )
    parser.add_argument(
        "--stdout",
        metavar="PROJET",
        help="Affiche le prompt du projet sur stdout (sans écrire de fichier).",
    )
    args = parser.parse_args()

    if args.stdout:
        match = [p for p in PROJETS if p["id"] == args.stdout]
        if not match:
            raise SystemExit(f"Projet inconnu: {args.stdout}")
        write_prompts(match, to_stdout=True)
        return

    selected = PROJETS
    if args.projet:
        selected = [p for p in PROJETS if p["id"] == args.projet]
        if not selected:
            ids = ", ".join(p["id"] for p in PROJETS)
            raise SystemExit(f"Projet inconnu: {args.projet}. Choisir parmi: {ids}")

    write_prompts(selected, to_stdout=False)


if __name__ == "__main__":
    main()
