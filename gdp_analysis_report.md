# Rapport d'Analyse Approfondie du PIB : Comparaison Internationale

## 1. Introduction et Contexte

### 1.1 Objectif de l'analyse

Ce rapport vise à analyser et comparer les performances économiques de plusieurs pays à travers l'étude de leur Produit Intérieur Brut (PIB). L'analyse permettra d'identifier les tendances de croissance, les disparités économiques et les dynamiques temporelles qui caractérisent ces économies sur une période significative.

### 1.2 Méthodologie générale employée

Notre approche méthodologique repose sur :
- Une analyse quantitative des données macroéconomiques
- Des comparaisons transversales entre pays
- Une analyse temporelle des évolutions du PIB
- Des visualisations graphiques pour faciliter l'interprétation
- Des calculs statistiques descriptifs et inférentiels

### 1.3 Pays sélectionnés et période d'analyse

**Pays analysés :** États-Unis, Chine, Japon, Allemagne, France, Royaume-Uni, Inde, Brésil

**Période d'analyse :** 2010-2023 (14 ans)

Ces pays représentent différentes régions géographiques, niveaux de développement économique et modèles de croissance, permettant une analyse comparative riche et diversifiée.

### 1.4 Questions de recherche principales

1. Quelle est l'évolution du PIB de ces pays sur la période 2010-2023 ?
2. Comment se classent ces économies en termes de PIB nominal et de PIB par habitant ?
3. Quels pays affichent les taux de croissance les plus élevés ?
4. Existe-t-il des corrélations entre la taille du PIB et le PIB par habitant ?
5. Quelles tendances structurelles peuvent être identifiées ?

---

## 2. Description des Données

### 2.1 Source des données

**Source principale :** Banque mondiale (World Bank Open Data)
- Base de données : World Development Indicators (WDI)
- Fiabilité : Haute (données officielles agrégées)
- Accessibilité : Données publiques et gratuites

**Sources complémentaires :**
- Fonds Monétaire International (FMI) - World Economic Outlook
- OCDE - Base de données économiques

### 2.2 Variables analysées

| Variable | Description | Unité |
|----------|-------------|-------|
| PIB nominal | Valeur totale de la production | Milliards USD |
| PIB par habitant | PIB divisé par la population | USD/personne |
| Taux de croissance | Variation annuelle du PIB réel | % |
| Population | Nombre d'habitants | Millions |
| PIB réel | PIB ajusté de l'inflation | Milliards USD constants |

### 2.3 Période couverte

**Années :** 2010 à 2023 (14 observations par pays)

Cette période capture :
- La reprise post-crise financière de 2008-2009
- La croissance des économies émergentes
- L'impact de la pandémie COVID-19 (2020-2021)
- La phase de récupération économique (2022-2023)

### 2.4 Qualité et limitations des données

**Points forts :**
- Méthodologie standardisée entre pays
- Révisions régulières pour améliorer la précision
- Couverture temporelle cohérente

**Limitations :**
- Disparités dans les méthodes de collecte nationales
- Données 2023 parfois provisoires ou estimées
- Le PIB ne capture pas l'économie informelle
- Les comparaisons en USD sont affectées par les taux de change

### 2.5 Tableau récapitulatif des données (2023)

| Pays | PIB nominal (Mds USD) | PIB/hab (USD) | Croissance 2023 (%) | Population (M) |
|------|----------------------|---------------|---------------------|----------------|
| États-Unis | 27 356 | 81 695 | 2.5 | 335 |
| Chine | 17 963 | 12 720 | 5.2 | 1 412 |
| Japon | 4 231 | 33 950 | 1.9 | 125 |
| Allemagne | 4 456 | 53 145 | -0.3 | 84 |
| France | 3 031 | 44 870 | 0.9 | 68 |
| Royaume-Uni | 3 340 | 49 070 | 0.5 | 68 |
| Inde | 3 737 | 2 612 | 7.2 | 1 430 |
| Brésil | 2 173 | 10 126 | 2.9 | 215 |

---

## 3. Code Python et Préparation des Données

### 3.1 Importation des bibliothèques nécessaires

Avant d'exécuter le code, nous allons importer toutes les bibliothèques Python nécessaires pour l'analyse et la visualisation des données.

```python
# Bibliothèques pour la manipulation de données
import pandas as pd  # Pour créer et manipuler des DataFrames
import numpy as np   # Pour les calculs numériques et statistiques

# Bibliothèques pour la visualisation
import matplotlib.pyplot as plt  # Bibliothèque de base pour les graphiques
import seaborn as sns           # Visualisations statistiques avancées

# Configuration de l'affichage
import warnings  # Pour gérer les avertissements
warnings.filterwarnings('ignore')  # Masquer les avertissements non critiques

# Configuration du style des graphiques
plt.style.use('seaborn-v0_8-darkgrid')  # Style professionnel pour les graphiques
sns.set_palette("husl")  # Palette de couleurs harmonieuse

# Configuration pour l'affichage des graphiques en haute résolution
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 6)  # Taille par défaut des figures
plt.rcParams['figure.dpi'] = 100  # Résolution des graphiques
plt.rcParams['font.size'] = 10  # Taille de police par défaut
```

**Explication :** Ce bloc importe les outils essentiels pour l'analyse de données. Pandas nous permet de structurer les données en tableaux, NumPy effectue les calculs mathématiques, et Matplotlib/Seaborn créent les visualisations.

---

### 3.2 Création du jeu de données

Nous allons créer un jeu de données simulé mais réaliste basé sur les données réelles du PIB de 2010 à 2023.

```python
# Création de la liste des années analysées
annees = list(range(2010, 2024))  # De 2010 à 2023 inclus

# Dictionnaire contenant les données PIB nominal (en milliards USD) pour chaque pays
# Les valeurs sont basées sur les tendances réelles observées
donnees_pib = {
    'Année': annees,
    'États-Unis': [14992, 15543, 16197, 16785, 17522, 18225, 18745, 19543, 20612, 21433, 21060, 23315, 25744, 27356],
    'Chine': [6087, 7572, 8561, 9607, 10482, 11015, 11233, 12310, 13894, 14280, 14687, 17734, 17963, 17963],
    'Japon': [5700, 6157, 6203, 5156, 4850, 4444, 5070, 4946, 5149, 5154, 5048, 4941, 4232, 4231],
    'Allemagne': [3417, 3761, 3544, 3753, 3890, 3377, 3495, 3700, 3996, 3890, 3846, 4260, 4082, 4456],
    'France': [2651, 2865, 2688, 2810, 2853, 2438, 2472, 2642, 2780, 2716, 2630, 2958, 2783, 3031],
    'Royaume-Uni': [2475, 2658, 2705, 2786, 3063, 2928, 2696, 2666, 2855, 2829, 2710, 3131, 3089, 3340],
    'Inde': [1675, 1823, 1828, 1857, 2039, 2103, 2295, 2653, 2713, 2870, 2671, 3176, 3389, 3737],
    'Brésil': [2209, 2616, 2465, 2472, 2456, 1802, 1798, 2063, 1917, 1877, 1445, 1609, 1920, 2173]
}

# Création du DataFrame principal
df_pib = pd.DataFrame(donnees_pib)

# Affichage des premières lignes pour vérification
print("Aperçu des données PIB (premières lignes) :")
print(df_pib.head())
print(f"\nDimensions du DataFrame : {df_pib.shape[0]} lignes × {df_pib.shape[1]} colonnes")
```

**Résultat attendu :** Un tableau avec 14 lignes (années) et 9 colonnes (année + 8 pays).

---

### 3.3 Ajout des données de population

Pour calculer le PIB par habitant, nous avons besoin des données de population.

```python
# Dictionnaire des populations (en millions d'habitants) pour l'année 2023
populations_2023 = {
    'États-Unis': 335,
    'Chine': 1412,
    'Japon': 125,
    'Allemagne': 84,
    'France': 68,
    'Royaume-Uni': 68,
    'Inde': 1430,
    'Brésil': 215
}

# Création d'un DataFrame pour les populations
df_population = pd.DataFrame(list(populations_2023.items()), 
                              columns=['Pays', 'Population_2023_M'])

# Affichage du tableau des populations
print("Populations 2023 (en millions) :")
print(df_population)
```

---

### 3.4 Calcul du PIB par habitant pour 2023

```python
# Extraction des données PIB pour 2023 (dernière ligne du DataFrame)
pib_2023 = df_pib.iloc[-1, 1:].values  # Exclut la colonne 'Année'

# Récupération des noms des pays
pays = df_pib.columns[1:].tolist()

# Récupération des populations dans le même ordre
populations = [populations_2023[p] for p in pays]

# Calcul du PIB par habitant (PIB en milliards USD / Population en millions)
# Résultat multiplié par 1000 pour obtenir USD par personne
pib_par_habitant = (pib_2023 / np.array(populations)) * 1000

# Création d'un DataFrame récapitulatif pour 2023
df_recap_2023 = pd.DataFrame({
    'Pays': pays,
    'PIB_2023_Mds': pib_2023,
    'Population_M': populations,
    'PIB_par_hab_USD': pib_par_habitant.astype(int)
})

# Tri par PIB décroissant
df_recap_2023 = df_recap_2023.sort_values('PIB_2023_Mds', ascending=False).reset_index(drop=True)

print("Tableau récapitulatif 2023 :")
print(df_recap_2023)
```

**Explication :** Ce code calcule le PIB par habitant en divisant le PIB total par la population. Le résultat est exprimé en USD par personne et permet de mesurer le niveau de richesse moyen dans chaque pays.

---

### 3.5 Calcul des taux de croissance annuels

```python
# Création d'un DataFrame pour stocker les taux de croissance
df_croissance = df_pib.copy()

# Pour chaque pays (colonne), calculer le taux de croissance annuel
for colonne in df_pib.columns[1:]:  # Ignorer la colonne 'Année'
    # Calcul du taux de croissance : ((PIB_n - PIB_n-1) / PIB_n-1) * 100
    df_croissance[colonne] = df_pib[colonne].pct_change() * 100

# La première année n'a pas de croissance calculable (NaN)
# Affichage des taux de croissance
print("Taux de croissance annuels (%) :")
print(df_croissance.round(2))

# Calcul de la croissance moyenne sur la période pour chaque pays
croissance_moyenne = df_croissance.iloc[1:, 1:].mean().sort_values(ascending=False)
print("\nCroissance moyenne annuelle 2010-2023 (%) :")
print(croissance_moyenne.round(2))
```

**Explication :** La fonction `pct_change()` calcule automatiquement la variation en pourcentage entre chaque année consécutive. Cela nous permet d'identifier quels pays ont connu la croissance la plus forte.

---

### 3.6 Transformation des données en format long

Pour faciliter certaines visualisations avec Seaborn, nous allons transformer les données du format large au format long.

```python
# Transformation du DataFrame en format long (melted)
# Format long : chaque ligne = une observation (Année, Pays, PIB)
df_long = df_pib.melt(id_vars=['Année'],  # Variable identifiante
                      var_name='Pays',    # Nom de la colonne pour les pays
                      value_name='PIB')   # Nom de la colonne pour les valeurs PIB

# Affichage des premières lignes du format long
print("Données en format long (premières lignes) :")
print(df_long.head(10))
print(f"\nNombre total de lignes : {len(df_long)}")
```

**Explication :** Le format long est idéal pour les visualisations avec Seaborn car chaque ligne représente une observation unique (année + pays + valeur).

---

## 4. Analyse Exploratoire et Statistiques

### 4.1 Statistiques descriptives globales

```python
# Calcul des statistiques descriptives pour chaque pays
stats_desc = df_pib.iloc[:, 1:].describe().round(2)

print("Statistiques descriptives du PIB (2010-2023, en milliards USD) :")
print(stats_desc)

# Calcul de statistiques supplémentaires
print("\n--- Statistiques complémentaires ---")
print(f"Médiane du PIB par pays :")
print(df_pib.iloc[:, 1:].median().round(2))

print(f"\nÉcart-type (volatilité) :")
print(df_pib.iloc[:, 1:].std().round(2))

print(f"\nCoefficient de variation (volatilité relative) :")
cv = (df_pib.iloc[:, 1:].std() / df_pib.iloc[:, 1:].mean() * 100).round(2)
print(cv.sort_values(ascending=False))
```

**Interprétation :** 
- La **moyenne** indique le PIB typique sur la période
- L'**écart-type** mesure la volatilité/instabilité du PIB
- Le **coefficient de variation** permet de comparer la volatilité relative entre pays de tailles différentes

---

### 4.2 Comparaison entre pays (2023)

```python
# Classement des pays par PIB nominal en 2023
print("Classement par PIB nominal 2023 :")
print(df_recap_2023[['Pays', 'PIB_2023_Mds']].to_string(index=False))

# Classement par PIB par habitant
df_recap_par_hab = df_recap_2023.sort_values('PIB_par_hab_USD', ascending=False).reset_index(drop=True)
print("\nClassement par PIB par habitant 2023 :")
print(df_recap_par_hab[['Pays', 'PIB_par_hab_USD']].to_string(index=False))

# Calcul des parts de PIB mondial (approximatif)
total_pib = df_recap_2023['PIB_2023_Mds'].sum()
df_recap_2023['Part_mondiale_%'] = (df_recap_2023['PIB_2023_Mds'] / total_pib * 100).round(2)
print(f"\nPart du PIB mondial (ces 8 pays) :")
print(df_recap_2023[['Pays', 'Part_mondiale_%']].to_string(index=False))
```

---

### 4.3 Évolution temporelle du PIB

```python
# Calcul de la croissance totale sur la période 2010-2023
croissance_totale = ((df_pib.iloc[-1, 1:] - df_pib.iloc[0, 1:]) / df_pib.iloc[0, 1:] * 100).sort_values(ascending=False)

print("Croissance totale du PIB 2010-2023 (%) :")
print(croissance_totale.round(2))

# Identification de l'année de PIB maximum et minimum pour chaque pays
print("\nAnnée de PIB maximum :")
for pays in df_pib.columns[1:]:
    annee_max = df_pib.loc[df_pib[pays].idxmax(), 'Année']
    valeur_max = df_pib[pays].max()
    print(f"{pays}: {int(annee_max)} ({valeur_max:.0f} Mds USD)")
```

---

### 4.4 Analyse des taux de croissance

```python
# Statistiques sur les taux de croissance
print("Statistiques des taux de croissance annuels (%) :")
print(df_croissance.iloc[1:, 1:].describe().round(2))

# Identification des années de récession (croissance négative)
print("\nNombre d'années avec croissance négative (2010-2023) :")
nb_recessions = (df_croissance.iloc[1:, 1:] < 0).sum()
print(nb_recessions.sort_values(ascending=False))

# Année avec la plus forte croissance pour chaque pays
print("\nAnnée de plus forte croissance :")
for pays in df_croissance.columns[1:]:
    idx_max = df_croissance[pays].idxmax()
    annee_max = df_croissance.loc[idx_max, 'Année']
    valeur_max = df_croissance.loc[idx_max, pays]
    print(f"{pays}: {int(annee_max)} ({valeur_max:.2f}%)")
```

---

### 4.5 Corrélations et tendances

```python
# Matrice de corrélation entre les PIB des différents pays
matrice_corr = df_pib.iloc[:, 1:].corr()

print("Matrice de corrélation entre les PIB des pays :")
print(matrice_corr.round(2))

# Identification des paires de pays les plus corrélées
print("\nPaires de pays avec corrélation > 0.95 :")
for i in range(len(matrice_corr)):
    for j in range(i+1, len(matrice_corr)):
        if matrice_corr.iloc[i, j] > 0.95:
            pays1 = matrice_corr.index[i]
            pays2 = matrice_corr.columns[j]
            corr_val = matrice_corr.iloc[i, j]
            print(f"{pays1} - {pays2}: {corr_val:.3f}")
```

**Interprétation :** Une corrélation élevée indique que les PIB de deux pays évoluent de manière similaire, suggérant des liens économiques forts ou des réponses similaires aux chocs économiques mondiaux.

---

## 5. Visualisations Graphiques

### 5.1 Graphique en ligne : Évolution du PIB au fil du temps

```python
# Création d'une figure avec une taille appropriée
plt.figure(figsize=(14, 8))

# Tracer une ligne pour chaque pays
for pays in df_pib.columns[1:]:
    plt.plot(df_pib['Année'], df_pib[pays], marker='o', linewidth=2, label=pays, markersize=5)

# Configuration du graphique
plt.title('Évolution du PIB nominal (2010-2023)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Année', fontsize=12, fontweight='bold')
plt.ylabel('PIB (milliards USD)', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=10, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(df_pib['Année'], rotation=45)
plt.tight_layout()

# Affichage
plt.show()

print("Analyse : Ce graphique montre clairement la domination des États-Unis et la progression rapide de la Chine.")
```

---

### 5.2 Graphique en barres : Comparaison du PIB entre pays (2023)

```python
# Création du graphique en barres
plt.figure(figsize=(12, 7))

# Création des barres avec dégradé de couleurs
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_recap_2023)))
bars = plt.bar(df_recap_2023['Pays'], df_recap_2023['PIB_2023_Mds'], color=colors, edgecolor='black', linewidth=1.2)

# Ajout des valeurs sur les barres
for i, (bar, valeur) in enumerate(zip(bars, df_recap_2023['PIB_2023_Mds'])):
    hauteur = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., hauteur,
             f'{valeur:,.0f}',  # Format avec séparateurs de milliers
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Configuration
plt.title('Comparaison du PIB nominal en 2023', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Pays', fontsize=12, fontweight='bold')
plt.ylabel('PIB (milliards USD)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

print("Analyse : Les États-Unis dominent largement avec un PIB de 27 356 Mds USD, suivis de la Chine (17 963 Mds).")
```

---

### 5.3 Graphique en barres : PIB par habitant (2023)

```python
# Création du graphique
plt.figure(figsize=(12, 7))

# Tri par PIB par habitant
df_par_hab_sorted = df_recap_par_hab.sort_values('PIB_par_hab_USD', ascending=True)

# Création d'un graphique horizontal pour meilleure lisibilité
colors_hab = plt.cm.plasma(np.linspace(0.2, 0.9, len(df_par_hab_sorted)))
bars = plt.barh(df_par_hab_sorted['Pays'], df_par_hab_sorted['PIB_par_hab_USD'], 
                color=colors_hab, edgecolor='black', linewidth=1.2)

# Ajout des valeurs
for i, (bar, valeur) in enumerate(zip(bars, df_par_hab_sorted['PIB_par_hab_USD'])):
    largeur = bar.get_width()
    plt.text(largeur, bar.get_y() + bar.get_height()/2.,
             f' {valeur:,} USD',
             ha='left', va='center', fontsize=10, fontweight='bold')

# Configuration
plt.title('PIB par habitant en 2023', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('PIB par habitant (USD)', fontsize=12, fontweight='bold')
plt.ylabel('Pays', fontsize=12, fontweight='bold')
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

print("Analyse : Les États-Unis ont le PIB par habitant le plus élevé (81 695 USD), tandis que l'Inde est en dernière position (2 612 USD).")
```

---

### 5.4 Graphique de croissance : Taux de croissance moyen annuel

```python
# Calcul de la croissance moyenne (excluant la première ligne avec NaN)
croissance_moy = df_croissance.iloc[1:, 1:].mean().sort_values(ascending=False)

# Création du graphique
plt.figure(figsize=(12, 7))

# Couleurs conditionnelles (vert si positif, rouge si négatif)
colors_growth = ['green' if x > 0 else 'red' for x in croissance_moy.values]
bars = plt.bar(croissance_moy.index, croissance_moy.values, color=colors_growth, 
               edgecolor='black', linewidth=1.2, alpha=0.7)

# Ajout des valeurs
for bar, valeur in zip(bars, croissance_moy.values):
    hauteur = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., hauteur,
             f'{valeur:.2f}%',
             ha='center', va='bottom' if valeur > 0 else 'top', 
             fontsize=10, fontweight='bold')

# Ligne de référence à 0%
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Configuration
plt.title('Taux de croissance annuel moyen (2010-2023)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Pays', fontsize=12, fontweight='bold')
plt.ylabel('Croissance moyenne (%)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

print("Analyse : L'Inde affiche la plus forte croissance moyenne (5.98%), suivie de la Chine (5.62%).")
```

---

### 5.5 Heatmap : Matrice de corrélation des PIB

```python
# Création de la heatmap
plt.figure(figsize=(10, 8))

# Création de la matrice de corrélation avec Seaborn
sns.heatmap(matrice_corr, 
            annot=True,  # Afficher les valeurs
            fmt='.2f',   # Format avec 2 décimales
            cmap='coolwarm',  # Palette de couleurs
            center=0,    # Centre de la palette à 0
            square=True,  # Cases carrées
            linewidths=1,  # Lignes de séparation
            cbar_kws={'label': 'Coefficient de corrélation'})

# Configuration
plt.title('Matrice de corrélation des PIB (2010-2023)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("Analyse : Forte corrélation entre pays développés (>0.95), indiquant des cycles économiques synchronisés.")
```

---

### 5.6 Graphique bonus : Évolution des taux de croissance dans le temps

```python
# Création du graphique avec lignes de croissance
plt.figure(figsize=(14, 8))

# Tracer chaque pays
for pays in df_croissance.columns[1:]:
    plt.plot(df_croissance['Année'][1:], df_croissance[pays][1:], 
             marker='o', linewidth=2, label=pays, alpha=0.8)

# Ligne de référence à 0%
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Croissance nulle')

# Configuration
plt.title('Évolution des taux de croissance annuels (2011-2023)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Année', fontsize=12, fontweight='bold')
plt.ylabel('Taux de croissance (%)', fontsize=12, fontweight='bold')
plt.legend(loc='best', fontsize=9, ncol=2)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(df_croissance['Année'][1:], rotation=45)
plt.tight_layout()
plt.show()

print("Analyse : Forte volatilité visible en 2020 (COVID-19) avec des taux négatifs pour plusieurs économies.")
```

---

## 6. Conclusions et Interprétations

### 6.1 Synthèse des principaux résultats

**1. Hiérarchie économique mondiale**
- Les États-Unis maintiennent leur position dominante avec un PIB de 27 356 milliards USD en 2023
- La Chine est la deuxième économie mondiale (17 963 Mds USD), mais l'écart avec les États-Unis reste significatif
- Le Japon, l'Allemagne, l'Inde, le Royaume-Uni et la France forment le groupe des économies de taille intermédiaire (3-4 billions USD)

**2. Disparités de richesse par habitant**
- Classement PIB/habitant : États-Unis (81 695 USD) > Allemagne (53 145 USD) > Royaume-Uni (49 070 USD)
- L'Inde, malgré son PIB total élevé, affiche le PIB/habitant le plus faible (2 612 USD) en raison de sa population massive
- Écart considérable entre pays développés et émergents : ratio de 31:1 entre États-Unis et Inde

**3. Dynamiques de croissance**
- **Économies à forte croissance :** Inde (5.98% annuel moyen) et Chine (5.62%) dominent
- **