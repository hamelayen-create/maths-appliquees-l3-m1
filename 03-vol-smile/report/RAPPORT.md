# Smile de volatilité implicite et calibration SVI

**Projet :** `03-vol-smile`  
**Niveau :** M1 — Mathématiques appliquées / Finance quantitative  
**Prérequis :** calcul différentiel, probabilités (loi normale, martingales élémentaires), Black–Scholes de base, optimisation numérique (moindres carrés), Python/NumPy/SciPy  
**Durée estimée :** lecture 3–4 h · travail machine 2–3 h · exercices 2 h  
**Objectifs d’apprentissage mesurables :**
1. Extraire \(\sigma_{\mathrm{imp}}(K)\) par inversion du prix Black–Scholes à une tolérance \(\lt 10^{-6}\).
2. Calibrer un smile SVI raw et interpréter chaque paramètre \((a,b,\rho,m,\sigma)\).
3. Diagnostiquer l’absence d’arbitrage butterfly/calendar sur une surface \(w(k,T)\).
4. Construire une surface simple et l’utiliser pour un pricing de référence.

---

## 1. Introduction et motivation

### 1.1 Pourquoi le smile existe

Le modèle de Black–Scholes (1973) suppose une volatilité constante \(\sigma\). Sur les marchés réels, si l’on inverse le prix de chaque option européenne pour obtenir la volatilité qui « explique » ce prix, on n’obtient **pas** une constante. On obtient une fonction du strike \(K\) (et de la maturité \(T\)) appelée **volatilité implicite** \(\sigma_{\mathrm{imp}}(K,T)\).

Pour une maturité fixée, le graphe \(K\mapsto\sigma_{\mathrm{imp}}(K)\) présente souvent une forme en U (actions, indices après 1987) ou une pente monotone (skew FX ou equity). On parle de **smile** ou de **skew**. Cette structure encode le fait que les queues de la densité risk-neutral ne sont pas gaussiennes : kurtose, asymétrie, sauts, volatilité stochastique.

### 1.2 Question centrale du dossier

Comment passer de cotations de calls \(C(K,T)\) à un objet mathématique **lisse, parcimonieux et sans arbitrage**, utilisable pour pricer des dérivés non cotés ?

Le fil conducteur est :

\[
\text{prix marché}\;\xrightarrow{\text{inversion BS}}\;\sigma_{\mathrm{imp}}(K)
\;\xrightarrow{\text{changement de variables}}\;w(k)=\sigma_{\mathrm{imp}}^2 T
\;\xrightarrow{\text{SVI}}\;\widehat{w}(k)
\;\xrightarrow{\text{surface}}\;w(k,T).
\]

### 1.3 Pourquoi SVI ?

Le modèle **SVI** (*Stochastic Volatility Inspired*, Gatheral) paramétrise la **total variance** \(w(k)\) par cinq paramètres seulement. Il est suffisamment flexible pour reproduire smile et skew, tout en restant analytique. Gatheral & Jacquier (2014) ont clarifié les conditions d’absence d’arbitrage. D’autres familles existent (SABR de Hagan et al., 2002 ; Heston, 1993), mais SVI est un excellent outil de **paramétrisation de marché** avant même de choisir une dynamique.

Dans ce projet, le marché est **synthétique** (généré par SVI + bruit) afin de contrôler la vérité terrain. Les méthodes restent celles utilisées en salle des marchés.

### 1.4 Ce que le smile révèle (et ce qu’il ne révèle pas)

La volatilité implicite n’est **pas** une prévision de volatilité réalisée future. C’est un **changement de numéraire de cotation** : les traders cotent en vol parce que cette échelle est plus stable et plus comparable que les prix bruts. Deux calls de strikes différents peuvent avoir des prix très éloignés, alors que leurs vols implicites restent du même ordre de grandeur.

En revanche, la *forme* du smile porte une information probabiliste. Un skew négatif signifie que la densité risk-neutral place plus de masse sur les baisses extrêmes que la log-normale. Une courbure forte (ailes relevées) signale de la kurtose. Ces lectures sont précisées par Breeden–Litzenberger (1978) : la dérivée seconde du call par rapport au strike est, à actualisation près, la densité.

Ce que le smile *ne* fixe pas, c’est la dynamique complète de \(S_t\). Une infinité de modèles (vol locale, vol stochastique, sauts) peuvent partager les mêmes prix de vanilles et diverger sur un barrrier ou un Asian. D’où l’importance de séparer clairement :

1. **paramétrisation** de la surface de vanilles (sujet de ce dossier) ;
2. **choix d’une dynamique** pour les exotiques (prolongements §8).

### 1.5 Fil pédagogique du document

La section 3 construit les objets mathématiques. La section 4 en discute les propriétés et les limites. La section 5 compare les algorithmes. La section 6 relie chaque notion à une fonction du dépôt. La section 7 présente des expériences chiffrées et commentées. Les exercices (§9) vérifient la compréhension à trois niveaux de difficulté.

---

## 2. Prérequis et notations

### 2.1 Prérequis

- Formule de Black–Scholes pour un call européen, vega positive.
- Notion de forward \(F=S_0 e^{rT}\) (dividendes nuls ici).
- Optimisation non linéaire bornée (Trust Region Reflective).
- Lecture de code Python scientifique.

### 2.2 Table des notations

| Symbole | Signification |
|--------|----------------|
| \(S_0\) | Spot initial |
| \(r\) | Taux sans risque constant |
| \(T\) | Maturité (en années) |
| \(K\) | Strike |
| \(F=S_0 e^{rT}\) | Forward |
| \(k=\log(K/F)\) | Log-moneyness |
| \(C(K,T)\) | Prix du call européen |
| \(\sigma_{\mathrm{imp}}(K,T)\) | Volatilité implicite BS |
| \(w(k,T)=\sigma_{\mathrm{imp}}^2(K,T)\,T\) | Total variance |
| \((a,b,\rho,m,\sigma)\) | Paramètres SVI raw |
| \(g(k)\) | Fonction de densité/butterfly (Gatheral) |
| RMSE | Racine de l’erreur quadratique moyenne |

Dans tout le rapport, sauf mention contraire, on travaille avec \(S_0=100\), \(r=0.01\), \(T=0.5\), donc \(F\approx 100.501\).

---

## 3. Modèle mathématique

### 3.1 Black–Scholes et volatilité implicite

Sous la mesure risque-neutre, avec taux constant et volatilité constante \(\sigma\),

\[
dS_t = r S_t\,dt + \sigma S_t\,dW_t.
\]

Le prix du call européen est

\[
C_{\mathrm{BS}}(S_0,K,T,r,\sigma)= S_0\Phi(d_1)-K e^{-rT}\Phi(d_2),
\]

avec

\[
d_1=\frac{\log(S_0/K)+(r+\tfrac12\sigma^2)T}{\sigma\sqrt{T}},\qquad
d_2=d_1-\sigma\sqrt{T},
\]

et \(\Phi\) la fonction de répartition de \(\mathcal{N}(0,1)\).

**Définition (volatilité implicite).** Pour un prix de marché \(C^{\mathrm{mkt}}\) donné,

\[
\sigma_{\mathrm{imp}}(K,T)\;\text{est la solution de}\quad
C_{\mathrm{BS}}(S_0,K,T,r,\sigma_{\mathrm{imp}})=C^{\mathrm{mkt}}.
\]

Comme la vega \(\partial C/\partial\sigma = S_0\phi(d_1)\sqrt{T}>0\) pour \(\sigma>0\) et \(T>0\), l’application \(\sigma\mapsto C_{\mathrm{BS}}\) est strictement croissante : l’inverse existe dès que \(C^{\mathrm{mkt}}\) est dans l’intervalle admissible (entre valeur intrinsèque actualisée et borne supérieure \(S_0\)).

Dans le code : `implied_vol_call` (`src/implied_vol.py`) résout cette équation par **Brent** sur \([10^{-4},5]\).

### 3.2 Forward, log-moneyness et total variance

Il est préférable de raisonner en **forward** et en **total variance**. Posons

\[
k=\log\frac{K}{F},\qquad w(k,T)=\sigma_{\mathrm{imp}}^2(K,T)\,T.
\]

Pourquoi \(w\) plutôt que \(\sigma_{\mathrm{imp}}\) ?

1. Les conditions d’arbitrage calendaire s’écrivent naturellement \(T\mapsto w(k,T)\) croissante.
2. Les asymptotes du smile en \(k\to\pm\infty\) sont linéaires en \(w\), pas en \(\sigma\).
3. SVI est défini sur \(w\).

La figure `10_total_variance.png` montre \(w(k)\) pour le smile de référence.

### 3.3 Le modèle SVI raw

**Définition (SVI raw, Gatheral).**

\[
w(k)= a + b\Bigl(\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}\Bigr),
\]

avec les contraintes usuelles \(b\ge 0\), \(|\rho|<1\), \(\sigma>0\), et \(a\in\mathbb{R}\) tel que \(w(k)>0\).

Interprétation heuristique des paramètres :

| Paramètre | Rôle |
|-----------|------|
| \(a\) | Niveau vertical (variance de base) |
| \(b\) | Amplitude des pentes asymptotiques |
| \(\rho\) | Asymétrie (skew) : \(\rho<0\) relève les puts |
| \(m\) | Translation horizontale du minimum |
| \(\sigma\) | Courbure près de l’ATM (largeur du « creux ») |

**Asymptotes.** Pour \(k\to+\infty\),

\[
w(k)\sim a+b(1+\rho)k + o(1),
\]

et pour \(k\to-\infty\),

\[
w(k)\sim a-b(1-\rho)k + o(1).
\]

Les pentes \(b(1\pm\rho)\) doivent rester dans des bornes compatibles avec l’absence d’arbitrage (Lee’s moment formula : pentes \(\le 2\)).

Implémentation : `svi_total_variance` dans `src/svi.py`.

### 3.4 Densité risk-neutral et butterfly

Breeden–Litzenberger (1978) : la densité risk-neutral est

\[
p(K)=e^{rT}\frac{\partial^2 C}{\partial K^2}.
\]

En variables \((k,w)\), Gatheral montre qu’une condition nécessaire d’absence d’arbitrage butterfly est \(g(k)\ge 0\), avec

\[
g(k)=\Bigl(1-\frac{k w'(k)}{2w(k)}\Bigr)^2
-\frac{w'(k)^2}{4}\Bigl(\frac{1}{w(k)}+\frac14\Bigr)
+\frac12 w''(k).
\]

Si \(g(k)<0\) sur un intervalle, la densité implicite devient négative : arbitrage butterfly. Le code `svi_density_proxy` approxime \(w'\) et \(w''\) par différences finies.

### 3.5 Arbitrage calendaire

Pour deux maturités \(0<T_1<T_2\), une condition nécessaire est

\[
w(k,T_1)\le w(k,T_2)\qquad\text{pour tout }k.
\]

Sinon, un calendrier d’options (long \(T_2\), short \(T_1\)) peut créer un arbitrage. La figure `09_calendar_arbitrage.png` illustre un cas sain et un cas violé. Dans `build_svi_surface`, on applique une **projection** simple \(W[i]=\max(W[i],W[i-1])\) pour restaurer cette monotonie après calibration tranche par tranche.

### 3.6 Schéma conceptuel du pipeline

La figure `06_pipeline.png` résume le flux :

\[
C(K,T)\;\to\;\text{Brent}\;\to\;\sigma_{\mathrm{imp}}(K)\;\to\;w(k)\;\to\;\text{SVI}\;\to\;\text{surface}\;\to\;\text{pricing}.
\]

C’est le modèle opérationnel du dossier.

### 3.7 Lien avec les modèles dynamiques (aperçu)

SVI n’est pas une dynamique de \(S_t\). Des dynamiques qui génèrent des smiles proches incluent :

- **Heston** (1993) : volatilité stochastique CIR, corrélation levier \(\rho_{\mathrm{Heston}}\).
- **SABR** (Hagan et al., 2002) : formule asymptotique pour \(\sigma_{\mathrm{imp}}\) très utilisée sur taux et FX.
- **Modèles à sauts** (Merton, Bates) : queues épaisses, skew fort en courte maturité.

Rouah (2013) détaille Heston et ses extensions numériques. Dans ce projet, on reste au niveau **paramétrisation de surface**, préalable à toute calibration dynamique.

### 3.8 Cas particulier analytique : smile plat

Si \(\sigma_{\mathrm{imp}}\equiv\sigma_0\), alors \(w(k)=\sigma_0^2 T\) constant. Ce cas correspond à \(b=0\) dans SVI (ou \(b\) négligeable). C’est le Black–Scholes pur : aucune structure de smile. Toute déviation mesurée par \(b\) ou \(\rho\) quantifie l’écart au monde log-normal.

### 3.9 Dérivation des asymptotes SVI

Partons de

\[
w(k)=a+b\Bigl(\rho x+\sqrt{x^2+\sigma^2}\Bigr),\qquad x=k-m.
\]

Pour \(x\to+\infty\), \(\sqrt{x^2+\sigma^2}=|x|\sqrt{1+\sigma^2/x^2}=x\bigl(1+O(x^{-2})\bigr)\), donc

\[
w(k)=a+b\bigl(\rho x+x+O(x^{-1})\bigr)=a+b(1+\rho)x+o(1).
\]

Pour \(x\to-\infty\), \(\sqrt{x^2+\sigma^2}=-x\bigl(1+O(x^{-2})\bigr)\), d’où

\[
w(k)=a+b\bigl(\rho x-x+O(x^{-1})\bigr)=a-b(1-\rho)x+o(1).
\]

Lee (2004) relie la pente asymptotique \(p=\lim_{k\to+\infty}w(k)/k\) aux moments de la densité risk-neutral : \(p\in[0,2]\). Une calibration qui produit \(b(1+\rho)>2\) est donc suspecte même si le RMSE sur la zone liquide est excellent.

### 3.10 Lien SVI ↔ volatilité stochastique (intuition)

Gatheral a motivé SVI par le comportement de la total variance dans certains régimes de volatilité stochastique. Intuitivement :

- la corrélation levier entre spot et variance pousse \(\rho\) vers des valeurs négatives ;
- le niveau moyen de variance se lit dans \(a\) et dans le minimum de \(w\) ;
- la vol-of-vol augmente la courbure ATM et les ailes, ce qui se traduit par \(b\) et \(\sigma\).

Cette correspondance n’est **pas** une bijection exacte avec Heston. Elle explique toutefois pourquoi cinq paramètres suffisent souvent à coller un slice liquide.

### 3.11 Forward Black vs spot Black–Scholes

On peut écrire les prix en variables forward (formule de Black 1976) :

\[
c(K,T)=e^{-rT}\Bigl(F\Phi(d_1^F)-K\Phi(d_2^F)\Bigr),
\]

avec \(d_1^F=\bigl(-\log(K/F)+\tfrac12 w\bigr)/\sqrt{w}\) et \(w=\sigma^2 T\). C’est souvent plus propre pour les surfaces : le taux \(r\) n’apparaît plus que dans le facteur d’actualisation et dans la construction de \(F\). Notre code utilise la forme spot pour l’inversion, ce qui est équivalent lorsque \(F=S_0 e^{rT}\).

---

## 4. Analyse et propriétés

### 4.1 Existence et unicité de \(\sigma_{\mathrm{imp}}\)

**Proposition.** Soit \(T>0\), \(S_0>0\), \(K>0\). L’application \(\sigma\mapsto C_{\mathrm{BS}}(\sigma)\) est \(C^\infty\) et strictement croissante sur \((0,\infty)\), avec

\[
\lim_{\sigma\to 0^+}C_{\mathrm{BS}}=\bigl(S_0-K e^{-rT}\bigr)^+,
\qquad
\lim_{\sigma\to\infty}C_{\mathrm{BS}}=S_0.
\]

Donc pour tout prix \(C\) strictement entre ces bornes, il existe une unique \(\sigma_{\mathrm{imp}}>0\).

*Esquisse.* La vega est strictement positive ; les limites sont classiques (voir Annexe A).

### 4.2 Identifiabilité de SVI

Les cinq paramètres ne sont pas toujours globalement identifiables : des trade-offs existent entre \(a\) et \(b\), ou entre \(m\) et \(\rho\), surtout sur une grille de strikes étroite. En pratique :

- on borne les paramètres (`calibrate_svi` : \(b\in[10^{-6},5]\), \(|\rho|<0.999\), etc.) ;
- on initialise près d’une heuristique (\(a\approx\tfrac12\min w\), \(m\approx\mathrm{médiane}(k)\)) ;
- on juge la qualité sur \(w\) (RMSE) et sur \(g(k)\), pas seulement sur les paramètres.

Sur données synthétiques sans bruit et avec \(x_0\) égal à la vérité, le test `test_svi_calibration_recovers_params` montre une récupération quasi exacte (RMSE \(\lt 10^{-4}\)).

### 4.3 Interprétation du skew \(\rho\)

Pour les indices actions, \(\rho\) est typiquement négatif : les puts OTM (protection) sont chers, donc \(\sigma_{\mathrm{imp}}\) est plus élevée pour \(K\ll F\). La figure `05_svi_sensitivity.png` (panneau \(\rho\)) montre comment un \(\rho\) négatif incline \(w(k)\).

### 4.4 Cas limites

- \(\sigma\to 0^+\) : le « V » de la racine carrée devient un \(|k-m|\) ; smile très pointu.
- \(b\to 0\) : smile plat.
- \(|\rho|\to 1\) : une asymptote s’aplatit, l’autre devient raide ; risque de \(g(k)<0\).

### 4.5 Erreurs de modélisation

1. **Bruit de cotation** : spreads bid/ask, ticks, stale quotes.
2. **Interpolation / extrapolation** : SVI peut extrapoler trop agressivement hors de la zone liquide.
3. **Calibration indépendante par maturité** : peut créer de l’arbitrage calendaire ; d’où la projection ou le SSVI (Gatheral & Jacquier).
4. **Confusion strike / log-moneyness** : calibrer en \(K\) au lieu de \(k\) fausse les asymptotes.

### 4.6 No-arbitrage : résumé opérationnel

| Condition | Test pratique | Code |
|-----------|---------------|------|
| Prix ≥ intrinsèque | avant inversion | `implied_vol_call` lève `ValueError` |
| Butterfly | \(g(k)\ge 0\) | `butterfly_arbitrage_free` |
| Calendaire | \(w(\cdot,T)\) croissante en \(T\) | `build_svi_surface` |

### 4.7 Unicité locale de la calibration

Le problème de moindres carrés SVI est **non convexe**. Rien ne garantit un unique minimum global. Deux phénomènes apparaissent régulièrement :

1. **Minima locaux.** Une mauvaise initialisation peut converger vers un jeu de paramètres avec RMSE correct mais \(g(k)<0\).
2. **Vallées plates.** Sur une fenêtre de strikes étroite, plusieurs combinaisons \((a,b,\sigma)\) produisent des \(w(k)\) quasi indistinguables.

Contre-mesures pédagogiques (et professionnelles) :

- multi-start : relancer depuis plusieurs \(x_0\) ;
- contraintes dures sur \(b\) et \(|\rho|\) ;
- juger le fit sur une grille dense de \(k\), pas seulement sur les strikes cotés ;
- inspecter systématiquement \(g(k)\) après calibration.

### 4.8 Erreur de discrétisation de \(g(k)\)

La fonction `svi_density_proxy` utilise un pas \(dk=10^{-3}\). Trop grand, on lisse les zones négatives ; trop petit, on amplifie le bruit numérique de \(w''\). Pour un SVI analytique, on pourrait différentier en forme fermée ; le choix des différences finies reste volontairement transparent pour le cours. En cas de doute, diminuer \(dk\) et vérifier la stabilité du signe de \(g\).

### 4.9 Contre-exemple : RMSE faible, arbitrage présent

Prenons le SVI agressif de la figure `08_butterfly_g.png`. Sur une grille de strikes **étroite** autour de l’ATM, l’erreur \(L^2\) peut rester acceptable alors que \(g\) est déjà négatif plus loin. Moralité : le RMSE n’est pas une garantie de no-arbitrage. Tout rapport de calibration sérieux doit afficher **à la fois** une erreur de prix/vol **et** un diagnostic butterfly/calendar.

---

## 5. Méthodes numériques et algorithmiques

### 5.1 Inversion Black–Scholes : Brent vs Newton

**Méthode A — Brent (`implied_vol_call`).**  
On résout \(f(\sigma)=C_{\mathrm{BS}}(\sigma)-C^{\mathrm{mkt}}=0\) sur un intervalle \([10^{-4},5]\). Brent combine dichotomie, sécante et interpolation inverse quadratique. Complexité typique : \(O\bigl(\log(1/\varepsilon)\bigr)\) évaluations de \(C_{\mathrm{BS}}\), chacune \(O(1)\).

**Pseudo-code (Brent).**
```
fonction implied_vol_call(C, S, K, T, r):
    si C < intrinsèque: erreur
    low ← 1e-4; high ← 5
    si f(low) > 0: retourner low   # prix trop bas → vol min
    si f(high) < 0: retourner high # prix trop haut → vol max
    retourner brentq(f, low, high)
```

**Méthode B — Newton–Raphson (`implied_vol_newton`).**
\[
\sigma_{n+1}=\sigma_n-\frac{C_{\mathrm{BS}}(\sigma_n)-C^{\mathrm{mkt}}}{\mathrm{vega}(\sigma_n)}.
\]
Convergence quadratique près de la racine, mais **instable** loin de la monnaie (vega minuscule) ou si \(\sigma_0\) est mauvais. Le code replie sur Brent si la boucle échoue.

**Comparaison expérimentale.** Sur un smile plat \(\sigma=22\%\), strikes de 70 à 130, les deux méthodes atteignent des erreurs \(\lt 10^{-8}\) (figure `07_iv_brent_vs_newton.png`). En pratique de production, Brent (ou une variante Jaeckel) est préféré pour la robustesse ; Newton sert souvent de raffinement après une bonne initialisation.

| Critère | Brent | Newton |
|---------|-------|--------|
| Robustesse OTM | Excellente | Moyenne |
| Vitesse ATM | Bonne | Excellente |
| Besoin de dérivée | Non | Oui (vega) |
| Complexité typique | \(O(\log 1/\varepsilon)\) | \(O(\log\log 1/\varepsilon)\) près de la racine |
| Usage recommandé | Défaut | Raffinement |

### 5.2 Calibration SVI par moindres carrés

Étant donné des points \((k_i,w_i^{\mathrm{mkt}})\), on résout

\[
\min_{\theta=(a,b,\rho,m,\sigma)}\;
\sum_{i=1}^n \bigl(w_{\mathrm{SVI}}(k_i;\theta)-w_i^{\mathrm{mkt}}\bigr)^2
\]

sous bornes sur \(\theta\). L’algorithme utilisé est **Trust Region Reflective** (`scipy.optimize.least_squares`, `method="trf"`).

**Pseudo-code.**
```
fonction calibrate_svi(k, w_mkt, x0=None):
    si x0 est None: x0 ← heuristique(w_mkt, k)
    résidus(θ) ← svi(k; θ) - w_mkt
    θ* ← least_squares(résidus, x0, bounds)
    retourner θ* et RMSE
```

Complexité : chaque itération évalue \(n\) résidus \(O(n)\) ; le nombre d’itérations dépend du conditionnement. Pour \(n\sim 15\)–\(50\) strikes, la calibration est instantanée.

**Hyperparamètres.**

| Hyperparamètre | Valeur dans le repo | Rôle |
|----------------|---------------------|------|
| Bornes \(a\) | \([-1,2]\) | évite niveaux absurdes |
| Bornes \(b\) | \([10^{-6},5]\) | pente positive |
| Bornes \(\rho\) | \((-0.999,0.999)\) | \(|\rho|<1\) |
| Bornes \(m\) | \([-2,2]\) | translation raisonnable |
| Bornes \(\sigma\) | \([10^{-4},2]\) | courbure |
| \(x_0\) défaut | \((\tfrac12\min w,\,0.2,\,-0.3,\,\mathrm{med}\,k,\,0.2)\) | initialisation |

### 5.3 Stabilité et biais

- **Bruit iid** sur \(w\) : le RMSE de calibration reste de l’ordre du bruit (expérience §7 : bruit \(10^{-4}\), RMSE \(\approx 6.8\cdot 10^{-5}\)).
- **Biais de spécification** : si le vrai smile n’est pas SVI (ex. SABR), le fit minimise l’erreur \(L^2\) mais peut violer \(g\ge 0\) aux extrêmes.
- **Variance d’estimateur** : peu de strikes liquides ⇒ paramètres mal déterminés ; régularisation ou SSVI recommandés en prolongation.

### 5.4 Construction de surface

Pour des maturités \(T_1<\cdots<T_m\), on calibre un SVI par tranche puis on empile \(w(k,T_j)\). La fonction `build_svi_surface` applique la projection calendaire. La figure `04_surface_sigma.png` montre \(\sigma(K,T)=\sqrt{w(k,T)/T}\).

### 5.5 Complexités récapitulatives

| Étape | Coût typique | Remarque |
|-------|--------------|----------|
| 1 prix BS | \(O(1)\) | \(\Phi\), \(\phi\) |
| Implied vol (Brent) | \(O(I)\) avec \(I\sim 10\)–\(30\) | par strike |
| Smile \(n\) strikes | \(O(n I)\) | parallélisable |
| Calibration SVI | \(O(J n)\) | \(J\) itérations TRF |
| Surface \(m\) maturités | \(O(m J n)\) | + projection \(O(m n_k)\) |

### 5.6 Choix de la fonction objectif

Plusieurs objectifs sont possibles :

1. **Erreur sur \(w\)** (choix du repo) : \(\sum(w_{\mathrm{SVI}}-w_{\mathrm{mkt}})^2\).
2. **Erreur sur \(\sigma_{\mathrm{imp}}\)** : \(\sum(\sigma_{\mathrm{SVI}}-\sigma_{\mathrm{mkt}})^2\).
3. **Erreur de prix** : \(\sum(C_{\mathrm{SVI}}-C_{\mathrm{mkt}})^2\), éventuellement pondérée par \(1/\mathrm{vega}\) ou par l’inverse du spread.

L’objectif prix est le plus fidèle au P&L, mais il couple inversion BS et calibration. L’objectif vol est courant chez un desk. L’objectif total variance est particulièrement adapté à SVI parce que le modèle est défini sur \(w\). Pour un cours, calibrer \(w\) clarifie le lien formule ↔ optimisation.

### 5.7 Pondération et liquidité

Les strikes loin de la monnaie sont moins liquides. Une pondération \( \omega_i\propto \mathrm{vega}_i \) ou \(\omega_i\propto 1/\mathrm{spread}_i\) évite que les ailes bruitées dictent \(b\) et \(\rho\). Dans nos expériences synthétiques, tous les points sont équipondérés : c’est volontaire pour isoler l’algorithme.

### 5.8 Comparaison qualitative SVI vs SABR (slice)

Pour une maturité fixée, SABR (Hagan et al., 2002) paramétrise aussi un smile via \((\alpha,\beta,\rho,\nu)\). Différences utiles :

| Aspect | SVI raw | SABR |
|--------|---------|------|
| Objet | \(w(k)\) exact (formule) | approximation asymptotique de \(\sigma_{\mathrm{imp}}\) |
| Nb paramètres | 5 | 4 (souvent \(\beta\) fixé) |
| No-arbitrage ailes | à contrôler via \(g\) et Lee | approximations parfois négatives en prix |
| Usage typique | equity index surfaces | taux, FX, commodities |
| Lien dynamique | inspiré, non exact | dynamique explicite sous-jacente |

Pour ce projet M1, SVI est le meilleur point d’entrée : formule exacte sur \(w\), calibration directe, diagnostics d’arbitrage lisibles.

### 5.9 Stabilité numérique de l’inversion BS

Près de l’échéance et loin de la monnaie, \(C_{\mathrm{BS}}\) devient extrêmement plat en \(\sigma\) (vega minuscule). Les erreurs d’arrondi relatifs sur le prix se traduisent alors en erreurs absolues de vol importantes. Les remèdes classiques sont :

- travailler en prix normalisés / normalized Black ;
- utiliser des formulations « let’s be rational » (Jaeckel, 2015) ;
- borner et valider les IV extrêmes avant calibration.

Notre implémentation pédagogique reste volontairement simple (Brent sur prix BS standard) et suffit largement pour \(T=0.5\) et des strikes dans \([0.7F,1.3F]\).

---

## 6. Implémentation guidée

### 6.1 Architecture du dossier

```text
03-vol-smile/
  README.md
  src/
    implied_vol.py   # bs_call, implied_vol_call, implied_vol_newton
    svi.py           # SVI, calibration, surface, butterfly
    demo.py          # expérience reproductible
  tests/test_vol.py
  report/
    RAPPORT.md
    BIBLIOGRAPHY.bib
    make_figures.py
    figures/*.png
```

### 6.2 Fonctions clés et lien théorie ↔ code

| Notion | Fonction | Fichier |
|--------|----------|---------|
| Prix BS | `bs_call` | `implied_vol.py` |
| IV Brent | `implied_vol_call` | `implied_vol.py` |
| IV Newton | `implied_vol_newton` | `implied_vol.py` |
| \(w_{\mathrm{SVI}}(k)\) | `svi_total_variance` | `svi.py` |
| Calibration | `calibrate_svi` | `svi.py` |
| Smile synthétique | `make_synthetic_smile` | `svi.py` |
| \(g(k)\) | `svi_density_proxy` | `svi.py` |
| Surface | `build_svi_surface` | `svi.py` |

### 6.3 Snippet minimal

```python
from implied_vol import bs_call, implied_vol_call
from svi import calibrate_svi, make_synthetic_smile, svi_total_variance
import numpy as np

S, r, T = 100.0, 0.01, 0.5
F = S * np.exp(r * T)
K, iv, w = make_synthetic_smile(F=F, T=T, true_params=(0.03, 0.25, -0.35, 0.05, 0.25))
price = bs_call(S, float(K[5]), T, r, float(iv[5]))
print(implied_vol_call(price, S, float(K[5]), T, r))
fit = calibrate_svi(np.log(K / F), w)
print(fit["rmse"])
```

### 6.4 Comment lancer

Depuis la racine du dépôt `maths-appliquees-l3-m1` ou depuis `03-vol-smile/` :

```bash
python3 src/demo.py
python3 -m pytest tests/ -q
python3 report/make_figures.py
```

Dépendances : `numpy`, `scipy`, `matplotlib`, `pytest` (voir `requirements.txt` à la racine du dépôt).

### 6.5 Pièges classiques

1. **Utiliser \(S\) au lieu de \(F\) dans \(k=\log(K/F)\)** : décale tout le smile.
2. **Calibrer \(\sigma_{\mathrm{imp}}\) au lieu de \(w\)** : distord les poids (une erreur de vol courte maturité ≠ longue).
3. **Oublier que le prix est sous l’intrinsèque** : l’inversion échoue ; souvent un problème de conventions (dividendes, early exercise).
4. **Newton avec \(\sigma_0\) absurde** loin OTM : divergence ; d’où le repli Brent.
5. **Extrapoler SVI hors de la zone calibrée** sans contrôler \(g(k)\) et les pentes de Lee.

### 6.6 Tests

Les tests dans `tests/test_vol.py` couvrent :

- round-trip IV (erreur \(\lt 10^{-6}\)) ;
- récupération SVI sans bruit ;
- borne RMSE de la démo du rapport (\(\mathrm{RMSE}<5\cdot 10^{-4}\)) ;
- cohérence Newton/Brent ATM ;
- butterfly sur SVI sain ;
- projection calendaire de surface.

---

## 7. Expériences numériques

### 7.1 Protocole reproductible

**Paramètres exacts (Annexe B) :**

- \(S_0=100\), \(r=0.01\), \(T=0.5\), \(F=S_0 e^{rT}\) ;
- paramètres vrais : \((a,b,\rho,m,\sigma)=(0.03,\,0.25,\,-0.35,\,0.05,\,0.25)\) ;
- \(n=15\) strikes sur \([0.7F,\,1.3F]\) pour la démo ; \(n=21\) pour les figures de fit ;
- bruit gaussien sur \(w\) : \(\sigma_\varepsilon=10^{-4}\), graine `seed=0`.

Commande : `python3 src/demo.py`.

### 7.2 Résultats de la démo

| Quantité | Valeur observée |
|----------|-----------------|
| \(\max\|\sigma_{\mathrm{rec}}-\sigma_{\mathrm{true}}\|\) (round-trip) | \(\approx 1.7\cdot 10^{-9}\) |
| Paramètres calibrés | \((0.0278,\,0.2533,\,-0.3379,\,0.0529,\,0.2546)\) |
| RMSE sur \(w\) | \(\approx 6.84\cdot 10^{-5}\) |
| `success` | `True` |

On observe que les paramètres récupérés restent proches de la vérité malgré le bruit. L’écart le plus visible porte sur \(a\) et \(\sigma\), qui compensent partiellement le bruit de niveau et de courbure.

### 7.3 Smile et fit

La figure `01_smile_iv.png` montre \(\sigma_{\mathrm{imp}}(K)\) pour \(T=0.5\). On observe un skew descendant typique (\(\rho<0\)) : volatilité plus haute pour les strikes bas.

La figure `02_svi_fit.png` superpose le marché synthétique et le fit SVI. On observe un accord excellent sur toute la zone de strikes ; le RMSE affiché est cohérent avec le niveau de bruit.

La figure `03_residuals.png` présente les résidus \(w_{\mathrm{SVI}}-w_{\mathrm{mkt}}\) en fonction de \(k\). On observe des oscillations de l’ordre de \(10^{-4}\), sans structure systématique forte : le modèle n’est pas misspecifié (ici, le générateur est SVI).

### 7.4 Surface

La figure `04_surface_sigma.png` affiche \(\sigma(K,T)\) pour cinq maturités. On observe :

- une diminution progressive du skew avec \(T\) (paramètres choisis à cet effet) ;
- une surface lisse grâce à SVI ;
- la projection calendaire garantit \(w\) croissante même si une tranche brute violait la contrainte.

### 7.5 Sensibilité des paramètres

La figure `05_svi_sensitivity.png` isole l’effet de \(\rho\), \(\sigma\), \(b\) et \(m\). On observe que :

- \(\rho\) contrôle l’inclinaison ;
- \(\sigma\) (paramètre SVI) élargit le creux ATM ;
- \(b\) raidi les ailes ;
- \(m\) translate le minimum.

Ces graphes sont essentiels pour initialiser une calibration et pour le contrôle de risque paramétrique.

### 7.6 Butterfly

La figure `08_butterfly_g.png` compare \(g(k)\) pour le SVI de référence (partout \(\ge 0\)) et pour un jeu agressif \((a,b,\rho,m,\sigma)=(0.01,1.2,-0.99,0,0.05)\). On observe des zones \(g<0\) dans le second cas : calibration « trop libre » ⇒ arbitrage. Le test `test_butterfly_proxy_on_healthy_svi` valide le cas sain.

### 7.7 Calendaire

La figure `09_calendar_arbitrage.png` montre à gauche deux courbes \(w(k,T)\) ordonnées, à droite une violation (zone rouge). Message pédagogique : calibrer maturité par maturité **sans contrainte** est insuffisant.

### 7.8 Tableau d’erreurs numériques

| Expérience | Erreur / métrique | Ordre de grandeur |
|------------|-------------------|-------------------|
| Round-trip IV (démo) | \(\max\|\Delta\sigma\|\) | \(10^{-9}\) |
| Round-trip IV (test unitaire) | \(\|\Delta\sigma\|\) | \(\lt 10^{-6}\) |
| Fit SVI bruité | RMSE\(_w\) | \(\approx 6.8\cdot 10^{-5}\) |
| Fit SVI sans bruit + \(x_0=\) vérité | RMSE\(_w\) | \(\lt 10^{-4}\) |
| Brent vs Newton (plat 22%) | \(\max\|\Delta\sigma\|\) | \(\lt 10^{-8}\) |

### 7.9 Interprétation honnête

Les expériences sont **favorables** car le générateur est SVI. Sur données réelles :

- le RMSE sera plus grand ;
- \(g(k)\) devra être surveillé aux ailes ;
- une pénalisation d’arbitrage ou une paramétrisation SSVI sera souvent nécessaire ;
- les conventions de marché (dividendes, rates, American) compliquent l’extraction d’IV.

Aucune expérience n’a échoué dans ce dépôt ; les ordres de grandeur ci-dessus sont reproductibles via `demo.py` et `make_figures.py`.

### 7.10 Protocole détaillé : du prix à la surface (pas à pas)

Pour ancrer la reproductibilité, détaillons une trajectoire complète telle qu’un étudiant doit pouvoir la refaire sans ambiguïté.

**Étape A — Génération du marché synthétique.**  
Appeler `make_synthetic_smile(F, T, true_params, n_strikes=15, noise=1e-4, seed=0)`. On obtient des tableaux `K`, `iv`, `w`. Le forward utilisé pour \(k\) doit être **exactement** le même que celui de la génération : \(F=S_0 e^{rT}\).

**Étape B — Validation par round-trip.**  
Pour chaque strike, recalculer \(C=\texttt{bs_call}(S_0,K,T,r,\mathrm{iv})\) puis \(\hat\sigma=\texttt{implied_vol_call}(C,\ldots)\). Vérifier \(\max|\hat\sigma-\mathrm{iv}|\) de l’ordre de \(10^{-9}\) à \(10^{-8}\). Si l’erreur est \(10^{-3}\) ou plus, chercher une incohérence de conventions (\(r\), \(T\), années fractionnaires).

**Étape C — Calibration.**  
Poser `k = log(K/F)` et `fit = calibrate_svi(k, w)`. Inspecter `fit["success"]`, `fit["rmse"]`, et les cinq paramètres. Comparer aux valeurs vraies uniquement parce que nous sommes en synthétique ; en marché réel, cette comparaison n’existe pas.

**Étape D — Diagnostic no-arbitrage.**  
Évaluer `butterfly_arbitrage_free` sur une grille fine de \(k\in[-0.6,0.6]\). Tracer \(g(k)\). Si des négativités apparaissent près des bords, restreindre les bornes de \(b\) ou ajouter une pénalité.

**Étape E — Surface.**  
Répéter C–D pour plusieurs maturités, puis `build_svi_surface`. Vérifier graphiquement que \(\sigma(K,T)\) est lisse et que \(w\) croît en \(T\).

### 7.11 Lecture quantitative du smile de référence

Pour le smile \(T=0.5\) sans bruit, la volatilité au strike le plus proche du forward vaut environ \(44.3\%\) (valeur lue sur la grille de la démo). Ce niveau peut sembler élevé pour un indice « calme », mais il est cohérent avec nos paramètres SVI : le minimum de \(w\) n’est pas extrêmement bas, et \(T=0.5\) convertit \(w\) en \(\sigma=\sqrt{w/T}\) avec un facteur \(\sqrt{2}\approx 1.41\). L’important n’est pas le niveau absolu — choisi pour la clarté graphique — mais la **forme** : skew négatif, ailes ascendantes, minimum légèrement décalé (\(m=0.05>0\)).

### 7.12 Robustesse au bruit : petite étude

En plus du cas `noise=1e-4`, on peut relancer `make_synthetic_smile` avec `noise=1e-3` et `noise=5e-3` (à faire par l’étudiant). Comportements attendus :

| Bruit sur \(w\) | RMSE typique | Qualité des paramètres | Risque \(g<0\) |
|-----------------|--------------|------------------------|----------------|
| \(0\) | \(\sim 10^{-8}\)–\(10^{-5}\) | excellente si \(x_0\) bon | faible |
| \(10^{-4}\) | \(\sim 7\cdot 10^{-5}\) | bonne | faible |
| \(10^{-3}\) | \(\sim 10^{-3}\) | dégradation de \(\sigma,m\) | modéré |
| \(5\cdot 10^{-3}\) | \(\sim 5\cdot 10^{-3}\) | instable | élevé si bornes larges |

Cette échelle justifie la régularisation dès que l’on quitte les données synthétiques propres.

### 7.13 Utilisation simple pour un pricing de référence

Une fois \(\hat w(k)\) calibré, on peut pricer n’importe quel call européen vanille par Black–Scholes en injectant \(\sigma=\sqrt{\hat w(k)/T}\) au strike voulu. Pour un strike non coté, on évalue SVI en ce \(k\) puis on convertit. C’est le pricing « market » des vanilles. Pour un Asian arithmétique, le README mentionne une approximation grossière : utiliser la vol ATM dans un Monte-Carlo BS. Ce n’est **pas** un pricing de smile complet ; c’est un témoin numérique du pipeline. Un pricing exotique cohérent exigerait Dupire ou un modèle stochastique calibré (§8).

---

## 8. Prolongements

1. **SSVI / surface sans arbitrage** (Gatheral & Jacquier, 2014) : une paramétrisation globale en \(T\) qui garantit butterfly + calendar.
2. **Calibration SABR** (Hagan et al., 2002) et comparaison des RMSE / stabilité des ailes, surtout en courte maturité.
3. **Heston calibré sur la surface** (Rouah, 2013) : passer de la paramétrisation SVI à une dynamique pour le pricing path-dependent.
4. **Pricing d’exotiques** sous volatilité locale Dupire reconstruite depuis \(w(k,T)\), puis comparaison Monte-Carlo.
5. **Régularisation bayésienne** des paramètres SVI (priors sur \(\rho\), \(b\)) pour strikes peu liquides.

Chacun de ces prolongements constitue un mini-projet M1/M2 réaliste.

---

## 9. Exercices corrigés

### Exercice 1 — Facile : round-trip d’implied vol

**Énoncé.** On prend \(S_0=100\), \(K=105\), \(T=0.75\), \(r=0.02\), \(\sigma=0.22\). Calculer le prix BS du call, puis retrouver \(\sigma_{\mathrm{imp}}\) par `implied_vol_call`. Quelle erreur obtient-on ?

**Solution.** Le prix est \(C=C_{\mathrm{BS}}(100,105,0.75,0.02,0.22)\). L’inversion donne \(\sigma_{\mathrm{imp}}\) tel que \(|\sigma_{\mathrm{imp}}-0.22|<10^{-6}\) (test unitaire `test_implied_vol_roundtrip`).  
*Raison :* \(C_{\mathrm{BS}}\) est lisse et strictement croissante en \(\sigma\) ; Brent atteint la tolérance `xtol=1e-8`.  
*Point de vigilance :* si l’on fournit un prix inférieur à l’intrinsèque \((S_0-Ke^{-rT})^+\), la fonction lève une exception — c’est le comportement correct.

### Exercice 2 — Moyen : interprétation d’un fit SVI

**Énoncé.** Sur le smile synthétique de la démo (\(T=0.5\), paramètres vrais \((0.03,0.25,-0.35,0.05,0.25)\), bruit \(10^{-4}\)), on obtient approximativement

\[
\hat a\approx 0.0278,\;
\hat b\approx 0.253,\;
\hat\rho\approx -0.338,\;
\hat m\approx 0.053,\;
\hat\sigma\approx 0.255.
\]

1. Le marché est-il plutôt skewé puts ou calls ?  
2. Calculer les pentes asymptotiques \(b(1+\rho)\) et \(b(1-\rho)\).  
3. Ces pentes respectent-elles la borne de Lee (\(\le 2\)) ?

**Solution.**

1. \(\hat\rho<0\) : skew puts (vols plus hautes pour \(k<0\)).
2. \(b(1+\rho)\approx 0.253\times(1-0.338)\approx 0.167\) (aile droite) ;  
   \(b(1-\rho)\approx 0.253\times(1+0.338)\approx 0.339\) (aile gauche).
3. Les deux pentes sont \(\ll 2\) : compatibles avec Lee.  
*Remarque :* la pente gauche plus forte confirme le prix élevé des puts OTM.

### Exercice 3 — Difficile : détecter un butterfly

**Énoncé.** Soit le SVI agressif \(\theta=(0.01,\,1.2,\,-0.99,\,0,\,0.05)\).  
1. Montrer numériquement que \(g(k)\) devient négatif sur une partie de \([-0.6,0.6]\).  
2. Expliquer le mécanisme économique de l’arbitrage butterfly associé.  
3. Proposer une contrainte simple à ajouter à la calibration pour l’éviter.

**Solution.**

1. Évaluer `svi_density_proxy` sur une grille fine (figure `08_butterfly_g.png`) : on observe \(g(k)<0\) ; `butterfly_arbitrage_free` renvoie `False`.
2. \(g<0\) implique une densité risk-neutral négative sur une bande de strikes. Un portefeuille butterfly (long 1 call \(K-\Delta\), short 2 calls \(K\), long 1 call \(K+\Delta\)) a un payoff \(\approx\Delta^2\mathbf{1}\) mais un prix de marché négatif ou incohérent : arbitrage.
3. Contraintes possibles :  
   - pénaliser \(\sum_j \min(g(k_j),0)^2\) dans l’objectif ;  
   - borner \(b\) et \(|\rho|\) plus strictement ;  
   - passer à SSVI avec conditions théoriques suffisantes (Gatheral & Jacquier, 2014).

### Complément aux exercices — grille d’auto-évaluation

| Exercice | Compétence visée | Critère de réussite |
|----------|------------------|---------------------|
| 1 | Inversion BS robuste | erreur \(\lt 10^{-6}\) reproduite |
| 2 | Lecture paramétrique SVI | pentes correctes + interprétation \(\rho\) |
| 3 | No-arbitrage butterfly | diagnostic \(g\) + remède crédible |

Un étudiant à l’aise doit pouvoir, sans regarder le corrigé, écrire le code des exercices 1 et 3 en moins de trente minutes, et calculer les pentes de l’exercice 2 à la main.

---

## 10. FAQ / erreurs fréquentes

**Q1. Pourquoi calibrer \(w\) et non \(\sigma_{\mathrm{imp}}\) ?**  
Parce que les asympotes et les conditions d’arbitrage sont naturelles en total variance, et parce que \(w=\sigma^2 T\) homogénéise les maturités.

**Q2. Mon implied vol « sature » à 5.0.**  
Le prix marché est trop élevé (au-delà de ce que BS peut produire sous \(\sigma\le 5\)), ou conventions incorrectes (spot/forward, dividends).

**Q3. Le fit SVI a un bon RMSE mais \(g<0\) aux ailes.**  
Le critère \(L^2\) ne encode pas le no-arbitrage. Ajouter une pénalité butterfly ou réduire \(b\).

**Q4. Newton diverge loin OTM.**  
Vega \(\approx 0\) : le pas de Newton explose. Utiliser Brent, ou une formule d’initialisation (ex. Brenner–Subrahmanyam) puis Newton.

**Q5. Deux maturités proches se croisent en \(w\).**  
Arbitrage calendaire. Projeter ou recalibrer conjointement (SSVI).

**Q6. Les paramètres calibrés changent beaucoup si j’enlève deux strikes.**  
Identifiabilité faible. Élargir la grille, régulariser, ou figer \(m\) / \(\sigma\).

**Q7. Puis-je utiliser SVI pour pricer un Asian ?**  
SVI fournit une surface de vanilles. Pour un Asian, il faut une dynamique (vol locale/stochastique) cohérente avec cette surface, puis un MC ou une EDP.

**Q8. Quelle différence entre smile et skew ?**  
« Smile » désigne souvent une courbe en U (ailes des deux côtés). « Skew » insiste sur l’asymétrie (pente dominante). En pratique, les deux mots se mélangent ; on précise plutôt « skew equity négatif » ou « smile FX symétrique ».

**Q9. Pourquoi \(|\rho|<1\) est-il imposé ?**  
Dans la forme raw, \(|\rho|<1\) garantit que les deux pentes asymptotiques \(b(1\pm\rho)\) sont strictement positives lorsque \(b>0\). Si \(|\rho|=1\), une aile devient plate (pente nulle), ce qui est un cas dégénéré souvent indésirable.

**Q10. Mon `least_squares` renvoie `success=False` mais le RMSE est petit.**  
Le drapeau `success` de SciPy reflète des critères internes (tolérances de gradient, etc.). Un RMSE faible avec `success=False` peut rester acceptable ; inspecter quand même les paramètres et \(g(k)\). Inversement, `success=True` avec RMSE médiocre signale un minimum local pauvre.

**Q11. Faut-il calibrer en \(\log(K/S_0)\) ou \(\log(K/F)\) ?**  
Toujours \(\log(K/F)\) pour rester cohérent avec Black. Utiliser \(S_0\) introduit un biais de taux/dividendes dans la moneyness.

**Q12. Comment choisir le nombre de strikes ?**  
Trop peu (\(n<8\)) : sous-détermination. Trop, avec des ailes illiquides : sur-ajustement. En desk, on privilégie les strikes liquides et l’on contrôle l’extrapolation par la forme paramétrique.

---

## 11. Bibliographie commentée

1. **Gatheral, J. (2006).** *The Volatility Surface.* Wiley. — *Livre de référence ; introduit SVI et l’intuition marché de la surface.*
2. **Gatheral, J. & Jacquier, A. (2014).** Arbitrage-free SVI volatility surfaces. *Quantitative Finance.* — *Conditions no-arbitrage et SSVI ; indispensable après ce projet.*
3. **Hagan, P. et al. (2002).** Managing Smile Risk. *Wilmott.* — *Formule SABR ; alternative pratique à SVI sur taux/FX.*
4. **Rouah, F. D. (2013).** *The Heston Model and its Extensions in Matlab and C#.* Wiley. — *Implémentations concrètes de Heston pour aller vers une dynamique.*
5. **Black, F. & Scholes, M. (1973).** The Pricing of Options and Corporate Liabilities. *JPE.* — *Socle du pricing et de la notion d’implied vol.*
6. **Breeden, D. & Litzenberger, R. (1978).** Prices of State-Contingent Claims… *Journal of Business.* — *Lien prix d’options ↔ densité risk-neutral (butterfly).*
7. **Lee, R. W. (2004).** The Moment Formula for Implied Volatility at Extreme Strikes. *Mathematical Finance.* — *Bornes sur les pentes asymptotiques du smile.*
8. **Heston, S. (1993).** A Closed-Form Solution for Options with Stochastic Volatility. *RFS.* — *Modèle dynamique classique générant skew et smile.*
9. **Dupire, B. (1994).** Pricing with a Smile. *Risk.* — *Volatilité locale ; pont entre surface de vanilles et pricing exotique.*
10. **Jaeckel, P. (2015).** Let’s Be Rational. *Wilmott.* — *Inversion BS ultra-robuste utilisée en production.*
11. **Cont, R. & Tankov, P. (2004).** *Financial Modelling with Jump Processes.* Chapman & Hall. — *Queues et sauts : pourquoi le smile existe en courte maturité.*
12. **Wilmott, P. (2006).** *Paul Wilmott on Quantitative Finance.* Wiley. — *Vue d’ensemble pédagogique pricing / risques / smiles.*

Les citations dans le texte suivent le style (Auteur, année), par exemple (Gatheral, 2006), (Hagan et al., 2002).

---

## Annexe A — Compléments de preuve

### A.1 Limites du call BS quand \(\sigma\to 0\) et \(\sigma\to\infty\)

Lorsque \(\sigma\to 0^+\), \(S_T\) converge en probabilité vers \(S_0 e^{rT}=F\). Donc

\[
C\to e^{-rT}\mathbb{E}[(F-K)^+]=\bigl(S_0-K e^{-rT}\bigr)^+.
\]

Lorsque \(\sigma\to\infty\), \(d_1\to+\infty\) et \(d_2\to-\infty\) si \(K>0\) fixé, d’où \(C\to S_0\).

### A.2 Dérivation rapide de \(g(k)\)

En changeant de numéraire forward, le call non actualisé \(c(k)\) s’exprime via la formule de Black en total variance \(w(k)\). En différentiant deux fois par rapport à \(k\) (ou \(K\)) et en imposant \(c_{kk}\ge 0\), on obtient après algèbre la fonction \(g(k)\) de la section 3.4. Les détails complets figurent dans Gatheral (2006, ch. 3) et Gatheral & Jacquier (2014).

### A.3 Pourquoi Brent converge ici

Sur \([\sigma_{\min},\sigma_{\max}]\), \(f(\sigma)=C_{\mathrm{BS}}(\sigma)-C^{\mathrm{mkt}}\) est continue et change de signe dès que \(C^{\mathrm{mkt}}\) est admissible. Le théorème des valeurs intermédiaires garantit une racine ; Brent maintient un encadrement, d’où la convergence globale (contrairement à Newton).

### A.4 Projection calendaire

Si \(w_2(k)<w_1(k)\) pour certains \(k\), remplacer \(w_2\) par \(\max(w_2,w_1)\) restaure la monotonie. Ce n’est **pas** optimal au sens d’une calibration conjointe, mais c’est un correctif pédagogique transparent, implémenté dans `build_svi_surface`.

### A.5 Expression de la vega

Partant de \(C=S_0\Phi(d_1)-Ke^{-rT}\Phi(d_2)\) et de \(d_2=d_1-\sigma\sqrt{T}\), une différenciation soigneuse (les termes en \(\partial d_1/\partial\sigma\) se cancelent) donne

\[
\frac{\partial C}{\partial\sigma}=S_0\phi(d_1)\sqrt{T}.
\]

C’est la vega utilisée par `implied_vol_newton`. Elle s’annule lorsque \(|d_1|\) est très grand, c’est-à-dire loin de la monnaie ou à très faible vol.

### A.6 Heuristique d’initialisation SVI

L’initialisation par défaut du code,

\[
a_0=\tfrac12\min_i w_i,\quad
b_0=0.2,\quad
\rho_0=-0.3,\quad
m_0=\mathrm{médiane}(k),\quad
\sigma_0=0.2,
\]

repose sur des ordres de grandeur equity. Elle n’est pas universelle. Si le smile est plat, \(b_0\) trop grand ralentit simplement la descente. Si le skew est positif (certains marchés commodities), \(\rho_0=-0.3\) reste en général dans le bassin d’attraction grâce aux bornes, mais un multi-start avec \(\rho_0\in\{-0.5,0,0.5\}\) est plus sûr.

### A.7 Du slice à Dupire (aperçu formulé)

Si l’on dispose d’une surface \(C(K,T)\) lisse sans arbitrage, la volatilité locale de Dupire (1994) s’écrit

\[
\sigma_{\mathrm{loc}}^2(K,T)=\frac{\partial_T C}{\tfrac12 K^2\partial_{KK}C}
\]

(dans la version taux nul ; des variantes existent avec taux/dividendes). Une surface SVI/SSVI fournit les dérivées nécessaires. Ce calcul dépasse le présent dossier mais motive pourquoi le no-arbitrage sur \(w(k,T)\) est non négociable : une densité négative rend \(\sigma_{\mathrm{loc}}\) absurde.

---

## Annexe B — Paramètres pour reproduire toutes les figures

Toutes les figures sont générées par :

```bash
python3 report/make_figures.py
```

Sortie : `report/figures/*.png`.

### B.1 Constantes globales

| Nom | Valeur |
|-----|--------|
| `S0` | 100.0 |
| `R` | 0.01 |
| `T_REF` | 0.5 |
| `F_REF` | `S0*exp(R*T_REF)` |
| `TRUE_PARAMS` | `(0.03, 0.25, -0.35, 0.05, 0.25)` |
| `NOISE` | `1e-4` |
| `SEED` | 0 |

### B.2 Détail par figure

| Fichier | Contenu | Paramètres spécifiques |
|---------|---------|------------------------|
| `01_smile_iv.png` | Smile \(\sigma_{\mathrm{imp}}(K)\) | 41 strikes, bruit 0 |
| `02_svi_fit.png` | Fit SVI vs marché | 21 strikes, bruit `1e-4` |
| `03_residuals.png` | Résidus \(w\) | idem fit |
| `04_surface_sigma.png` | Surface \(\sigma(K,T)\) | \(T\in\{0.25,0.5,1,1.5,2\}\) ; params listés dans `make_figures.py` |
| `05_svi_sensitivity.png` | Sensibilités \(\rho,\sigma,b,m\) | variations autour de `TRUE_PARAMS` |
| `06_pipeline.png` | Schéma conceptuel | pure illustration |
| `07_iv_brent_vs_newton.png` | Erreurs d’inversion | smile plat \(\sigma=0.22\) |
| `08_butterfly_g.png` | \(g(k)\) sain vs agressif | agressif `(0.01,1.2,-0.99,0,0.05)` |
| `09_calendar_arbitrage.png` | Calendaire OK / violé | voir script |
| `10_total_variance.png` | \(w(k)\) | 41 strikes, bruit 0 |

### B.3 Résultats numériques de référence (démo)

```text
max |iv_rec - iv_true| ≈ 1.67e-09
SVI fit ≈ a=0.02776, b=0.25326, rho=-0.3379, m=0.05287, sigma=0.2546
RMSE total variance ≈ 6.84e-05
success = True
```

---

## Conclusion

Ce dossier a construit, de façon autonome, le chemin complet menant des prix de calls à une surface SVI exploitable. La volatilité implicite apparaît comme l’inverse de Black–Scholes ; le smile se lit confortablement en total variance ; SVI offre cinq paramètres interprétables ; les conditions \(g(k)\ge 0\) et la croissance en maturité protègent contre les arbitrages butterfly et calendar. Les expériences synthétiques confirment la précision de l’inversion (erreurs \(\sim 10^{-9}\)) et la stabilité de la calibration sous faible bruit. Les prolongements naturels — SSVI, SABR, Heston, Dupire — s’appuient directement sur les objets produits ici.

Le lecteur qui a exécuté `demo.py`, `pytest` et `make_figures.py`, puis traité les trois exercices, maîtrise les objectifs annoncés en page de garde.

Trois messages opérationnels méritent d’être retenus. Premier message : **toujours** valider un smile par un round-trip de prix avant de discuter de paramètres. Deuxième message : un bon RMSE ne remplace pas un diagnostic d’arbitrage ; la fonction \(g(k)\) et la monotonie calendaire de \(w\) sont des gardes-fous indispensables. Troisième message : SVI est une paramétrisation de marché, non une dynamique ; dès que l’on quitte les vanilles européennes, il faut un modèle d’évolution du sous-jacent cohérent avec la surface.

Ces trois points séparent clairement le travail d’un quant junior qui « fite une courbe » de celui d’un quant qui **construit une surface utilisable**. Le dépôt `03-vol-smile` fournit les briques numériques pour pratiquer cette exigence sur un jeu de données contrôlé, avant de passer à des cotations réelles et à des modèles plus riches.

---

*Fin du rapport — projet `03-vol-smile`.*
