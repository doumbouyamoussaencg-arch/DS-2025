# Rapport de Travaux Pratiques
## Introduction au Machine Learning - Classification de la Qualité des Vins

**Établissement :** ENCG Settat  
**Enseignant :** Mr Larhlimi  
**Date :** Novembre 2025  
**Sujet :** Classification binaire de la qualité des vins blancs

---

## Résumé Exécutif

Ce rapport présente une étude complète sur la prédiction de la qualité des vins blancs à partir de leurs caractéristiques physico-chimiques. Nous avons appliqué l'algorithme des k plus proches voisins (k-NN) avec et sans normalisation des données. Les résultats montrent qu'avec normalisation, nous atteignons une précision de test optimale variant entre 75% et 80%.

---

## 1. Introduction et Objectifs

### 1.1 Contexte

Le dataset utilisé provient de l'UCI Machine Learning Repository et contient des mesures physico-chimiques de vins blancs portugais (Vinho Verde). L'objectif est de développer un modèle prédictif capable de classifier automatiquement la qualité d'un vin.

### 1.2 Objectifs Spécifiques

- Analyser les caractéristiques physico-chimiques des vins
- Construire un modèle de classification binaire (bon/mauvais vin)
- Comparer les performances avec et sans normalisation des données
- Identifier les paramètres optimaux du modèle k-NN

---

## 2. Méthodologie

### 2.1 Description des Données

**Source :** http://archive.ics.uci.edu/ml/datasets/Wine+Quality

**Caractéristiques du dataset :**
- Nombre d'échantillons : 4898 vins blancs
- Nombre de variables d'entrée : 11 caractéristiques physico-chimiques
- Variable cible : Qualité (score de 3 à 9)

**Variables d'entrée :**
1. Acidité fixe (fixed acidity)
2. Acidité volatile (volatile acidity)
3. Acide citrique (citric acid)
4. Sucre résiduel (residual sugar)
5. Chlorures (chlorides)
6. Dioxyde de soufre libre (free sulfur dioxide)
7. Dioxyde de soufre total (total sulfur dioxide)
8. Densité (density)
9. pH
10. Sulfates
11. Degré d'alcool (alcohol)

### 2.2 Préparation des Données

#### Transformation en Problème Binaire

Pour simplifier la tâche de classification, nous avons transformé la variable qualité (scores de 3 à 9) en une classification binaire :

- **Classe 0 (mauvaise qualité)** : qualité ≤ 5
- **Classe 1 (bonne qualité)** : qualité > 5

Cette transformation permet de distinguer clairement les vins de qualité acceptable de ceux nécessitant une amélioration.

#### Division des Données

Les données ont été divisées en trois ensembles distincts :

- **Ensemble d'entraînement (33%)** : pour l'apprentissage du modèle
- **Ensemble de validation (33%)** : pour la sélection des hyperparamètres
- **Ensemble de test (33%)** : pour l'évaluation finale

**Stratégie importante :**
- Utilisation de `stratify=Y` pour maintenir la proportion des classes dans chaque ensemble
- Utilisation de `shuffle=True` pour éviter les biais liés à l'ordre des données

### 2.3 Algorithme k-NN

L'algorithme des k plus proches voisins classifie un échantillon en se basant sur le vote majoritaire de ses k voisins les plus proches dans l'espace des caractéristiques.

**Distance utilisée :** Distance euclidienne  
d(x_i, x_j)² = ||x_i - x_j||² = (x_i - x_j)ᵀ(x_i - x_j)

**Métrique d'évaluation :** Taux d'erreur  
Erreur = (1/N) × Σ[y_i ≠ ŷ_i]

---

## 3. Analyse Exploratoire des Données

### 3.1 Distribution des Classes

Après transformation en problème binaire, nous observons :
- Une distribution relativement équilibrée entre les deux classes
- Légère majorité de vins de bonne qualité (classe 1)
- Cette distribution justifie l'utilisation de la stratification lors de la division des données

### 3.2 Analyse Statistique

**Observations principales :**

1. **Échelles variables :** Les caractéristiques ont des échelles très différentes
   - Dioxyde de soufre total : valeurs jusqu'à 440 mg/L
   - pH : valeurs entre 2.7 et 3.8
   - Cette disparité justifie la normalisation

2. **Corrélations notables :**
   - Forte corrélation entre densité et sucre résiduel
   - Corrélation négative entre alcool et densité
   - Corrélation entre dioxyde de soufre libre et total (attendue)

3. **Outliers :** Présence de valeurs aberrantes dans plusieurs variables, notamment :
   - Sucre résiduel
   - Dioxyde de soufre total
   - Chlorures

---

## 4. Résultats Sans Normalisation

### 4.1 Test Initial (k=3)

**Premier test avec k=3 :**
- Erreur de validation : ~22-25%
- Précision : ~75-78%
- Résultat encourageant mais perfectible

### 4.2 Optimisation du Paramètre k

**Recherche du k optimal :**
- Plage testée : k ∈ [1, 37] (valeurs impaires)
- Critère : Minimisation de l'erreur sur l'ensemble de validation

**Observations du sur-apprentissage :**

1. **Pour k petit (k=1, 3, 5) :**
   - Erreur d'entraînement très faible (~5-10%)
   - Erreur de validation plus élevée (~22-25%)
   - **Diagnostic :** Sur-apprentissage (overfitting)
   - Le modèle mémorise les données d'entraînement

2. **Pour k moyen (k=11-21) :**
   - Erreur d'entraînement augmente légèrement
   - Erreur de validation se stabilise ou diminue
   - **Meilleur compromis biais-variance**

3. **Pour k grand (k>25) :**
   - Erreur d'entraînement continue d'augmenter
   - Erreur de validation recommence à augmenter
   - **Diagnostic :** Sous-apprentissage (underfitting)
   - Le modèle devient trop simple

### 4.3 Résultats Finaux Sans Normalisation

**k optimal trouvé : k* ≈ 15-19** (variable selon le split aléatoire)

**Performances sur ensemble de test :**
- Taux d'erreur : ~23-26%
- Précision : ~74-77%

---

## 5. Résultats Avec Normalisation

### 5.1 Justification de la Normalisation

La normalisation (standardisation) est cruciale pour k-NN car :

1. **Problème d'échelle :** Les variables avec de grandes échelles dominent le calcul de distance
2. **Équité entre variables :** Chaque variable contribue équitablement au calcul de distance
3. **Amélioration des performances :** Meilleure généralisation du modèle

**Méthode appliquée :** StandardScaler
- Centrage : μ = 0 (soustraction de la moyenne)
- Réduction : σ = 1 (division par l'écart-type)

### 5.2 Application Correcte de la Normalisation

**Point critique :** Le StandardScaler est ajusté (fit) UNIQUEMENT sur l'ensemble d'entraînement, puis appliqué (transform) à tous les ensembles.

```python
sc = sc.fit(Xa)        # Apprend μ et σ sur entraînement
Xa_n = sc.transform(Xa) # Applique aux données d'entraînement
Xv_n = sc.transform(Xv) # Applique aux données de validation
Xt_n = sc.transform(Xt) # Applique aux données de test
```

**Pourquoi cette approche ?**
- Évite la fuite d'information (data leakage) des ensembles de validation/test
- Simule les conditions réelles de déploiement
- Garantit l'intégrité de l'évaluation

### 5.3 Résultats Avec Normalisation

**k optimal trouvé : k* ≈ 13-17** (variable selon le split aléatoire)

**Performances sur ensemble de test :**
- Taux d'erreur : ~19-23%
- Précision : ~77-81%

**Amélioration constatée :**
- Réduction de l'erreur : 2-4 points de pourcentage
- Gain de précision : 2-4%
- Courbes d'erreur plus lisses
- Meilleure stabilité du modèle

---

## 6. Analyse Comparative

### 6.1 Tableau Comparatif

| Critère | Sans Normalisation | Avec Normalisation | Amélioration |
|---------|-------------------|-------------------|--------------|
| k optimal | 15-19 | 13-17 | Plus stable |
| Erreur de validation min | 22-25% | 19-22% | -3% |
| Erreur de test | 23-26% | 19-23% | -3 à -4% |
| Précision de test | 74-77% | 77-81% | +3 à +4% |
| Stabilité | Moyenne | Élevée | ✓ |

### 6.2 Interprétation des Résultats

**Pourquoi la normalisation améliore-t-elle les performances ?**

1. **Équilibrage des contributions :**
   - Sans normalisation : variables à grande échelle (SO₂ total) dominent
   - Avec normalisation : toutes les variables contribuent équitablement

2. **Distances plus significatives :**
   - Les distances euclidiennes reflètent mieux les similarités réelles
   - Meilleure identification des voisins pertinents

3. **Réduction du bruit :**
   - Les outliers ont moins d'impact après normalisation
   - Meilleure généralisation

### 6.3 Analyse du Sur-apprentissage

**Phénomène observé :**
- Pour k petit : écart important entre erreur d'entraînement et validation
- Pour k optimal : écart réduit, bon compromis
- Pour k grand : erreurs convergent mais augmentent (modèle trop simple)

**Stratégie de sélection du k :**
1. Observer le point minimum de la courbe de validation
2. Vérifier que l'écart train-validation est raisonnable
3. Privilégier un k légèrement plus grand en cas d'hésitation (principe de parcimonie)

---

## 7. Conclusions et Recommandations

### 7.1 Conclusions Principales

1. **Impact de la normalisation :** La normalisation améliore significativement les performances de k-NN (+3-4% de précision)

2. **Importance de la validation :** L'ensemble de validation est crucial pour sélectionner k et éviter le sur-apprentissage

3. **Performance du modèle :** Une précision de ~77-81% est obtenue, ce qui est satisfaisant pour ce type de problème

4. **Trade-off biais-variance :** Le choix de k illustre parfaitement le compromis entre sur-apprentissage (k petit) et sous-apprentissage (k grand)

### 7.2 Limitations de l'Étude

1. **Sensibilité au split :** Les résultats varient légèrement selon la division aléatoire des données

2. **Algorithme simple :** k-NN est un algorithme de base, d'autres méthodes (Random Forest, SVM, etc.) pourraient donner de meilleurs résultats

3. **Déséquilibre potentiel :** Bien que les classes soient relativement équilibrées, un léger déséquilibre existe

### 7.3 Recommandations

**Pour améliorer la robustesse :**

1. **Validation croisée :** Utiliser la validation croisée (k-fold) plutôt qu'un simple split
   ```python
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(clf, X_normalized, Y, cv=5)
   ```

2. **Optimisation avancée :** Utiliser GridSearchCV pour une recherche systématique
   ```python
   from sklearn.model_selection import GridSearchCV
   param_grid = {'n_neighbors': range(1, 40)}
   grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
   ```

3. **Essayer d'autres algorithmes :**
   - Random Forest (gère bien les échelles différentes)
   - SVM avec kernel RBF
   - Gradient Boosting (XGBoost, LightGBM)
   - Réseaux de neurones

4. **Feature engineering :**
   - Créer de nouvelles variables (ratios, interactions)
   - Sélection de features (éliminer les variables peu informatives)

5. **Gestion des outliers :**
   - Détecter et traiter les valeurs aberrantes
   - Utiliser des transformations robustes

### 7.4 Réponse aux Questions Théoriques

**Q : Pourquoi maintenir la proportion des classes lors du split ?**
- **R :** Pour éviter qu'un ensemble ait une distribution différente des autres
- Garantit que chaque ensemble est représentatif de la distribution globale
- Essentiel quand les classes sont déséquilibrées

**Q : Pourquoi mélanger les données ?**
- **R :** Les données peuvent être ordonnées (par qualité, date de mesure, etc.)
- Le mélange évite les biais systématiques
- Garantit une division aléatoire et représentative

**Q : Comment rendre le modèle moins sensible au split ?**
- **R :** Utiliser la validation croisée (cross-validation)
- Permet d'évaluer le modèle sur plusieurs splits différents
- Fournit une estimation plus fiable de la performance

---

## 8. Annexes Techniques

### 8.1 Paramètres Utilisés

**KNeighborsClassifier :**
- `n_neighbors` : testé de 1 à 37 (pas de 2)
- `metric` : 'euclidean' (par défaut)
- `weights` : 'uniform' (par défaut)

**StandardScaler :**
- `with_mean=True` : centrage à 0
- `with_std=True` : réduction à variance 1

**train_test_split :**
- `test_size=1/3` : 33% pour test
- `shuffle=True` : mélange aléatoire
- `stratify=Y` : maintien des proportions

### 8.2 Commandes d'Exécution

**Activation de l'environnement :**
```bash
source /opt/venv/iti-data/bin/activate
jupyter-notebook &
```

**Exécution du script :**
```bash
python wine_classification.py
```

### 8.3 Visualisations Générées

1. **boxplot_features.png** : Distribution des variables physico-chimiques
2. **correlation_matrix.png** : Matrice de corrélation
3. **error_curves_raw.png** : Courbes d'erreur sans normalisation
4. **error_curves_normalized.png** : Courbes d'erreur avec normalisation
5. **comparison_final.png** : Comparaison finale des méthodes

---

## 9. Bibliographie

- UCI Machine Learning Repository : Wine Quality Dataset
- Scikit-learn Documentation : https://scikit-learn.org/
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow

---

**Fin du rapport**

*Réalisé dans le cadre du cours de Machine Learning*  
*ENCG Settat - Sous la direction de Mr Larhlimi*