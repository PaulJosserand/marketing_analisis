import pandas as pd
from pandasql import sqldf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression


def analyse_df (df) :
    #Nombre de ligne du dataframe
    print('DATAFRAME') 
    print('     Nombre de ligne:', df.shape[0])
    #Nombre d'attributs du dataframe
    print('     Nombre attributs:', df.shape[1])

    list_attributs = []
    for i in range (df.shape[1]) : list_attributs.append(df.columns[i])

    for attribut in list_attributs:
        print()
        print(f'Attribut :{attribut}')
        print('     Nombre de données :', df[attribut].shape[0])
        print('     Nombre de données uniques:', df[attribut].nunique())
        print('     Nombre de NaN:', df[attribut].isna().sum())
        print('     Type de données :', df[attribut].dtype)
        print('     Exemple de données :', df[attribut].unique()[:5])
        try : print('     Minimum :', df[attribut].min())
        except : pass
        try : print('     Maximum :', df[attribut].max())
        except : pass
        print()
        
    
    
    
def check_NaN_in_df (df):
    list_attributs = []
    for i in range (df.shape[1]) : list_attributs.append(df.columns[i])

    for attribut in list_attributs:
        print(attribut)
        print('     Nombre de NaN:', df[attribut].isna().sum())
        
        
def distribution_curve(df, attributStr):
    # Configuration du style des graphiques seaborn
    sns.set(style="darkgrid")

    # Calcul des statistiques
    mean = df[attributStr].mean()
    median = df[attributStr].median()
    q1 = df[attributStr].quantile(0.25)
    q3 = df[attributStr].quantile(0.75)

    # Tracer la courbe de distribution avec un histogramme plus détaillé
    plt.figure(figsize=(10, 6))
    
    sns.histplot(df[attributStr], kde=True, bins=20, color='blue', stat='density', kde_kws={'bw_adjust': 3}) #kde_kws permet de lisser plus ou moins la courbe

    # Ajouter des lignes verticales pour les statistiques
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='-', linewidth=2, label=f'Median: {median:.2f}')
    plt.axvline(q1, color='purple', linestyle='-.', linewidth=2, label=f'Q1: {q1:.2f}')
    plt.axvline(q3, color='orange', linestyle='-.', linewidth=2, label=f'Q3: {q3:.2f}')

    # Ajouter des labels et un titre
    plt.xlabel(attributStr)
    plt.ylabel('Density')
    plt.title(f'Distribution Curve of {attributStr}')

    # Ajouter une légende
    plt.legend()

    # Afficher le graphique
    plt.show()
    
    
    
def filter_aberrant_values (df, attributStr):
    #On utilise la loi des 3 sigma pour filtrer les données : on ne retient pas les 0,15% des valeurs les plus basses et les 0,15% des valeurs les plus hautes
    seuil_bas = df[attributStr].quantile(0.0015)
    seuil_haut = df[attributStr].quantile(0.9985)
    return df[(df[attributStr] > seuil_bas) & (df[attributStr] < seuil_haut)]



def histogramme_baton(df, abscisse_attribut, ordonnee_attribut, title, chiffres_signiInt):
    df_graphique = df

    # Listes des données
    abscisse = df_graphique[abscisse_attribut].tolist()
    ordonnee = df_graphique[ordonnee_attribut].tolist()

    # Configuration du style des graphiques seaborn
    sns.set(style="darkgrid")
    
    # Créer un graphique à barres juxtaposées avec la moyenne totale par jour
    plt.figure(figsize=(18, 6))
    bar_width = 0.35
    colors = plt.cm.viridis(np.linspace(0, 1, len(abscisse)))  # Couleurs pour les barres

    bars = plt.bar(abscisse, ordonnee, bar_width, label=ordonnee_attribut, color=colors)

    # Ajouter les valeurs au-dessus de chaque barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.{chiffres_signiInt}f}', ha='center', va='bottom')
 
    # Appliquer l'échelle logarithmique à l'axe y
    plt.yscale('log')

    plt.xlabel(abscisse_attribut)
    plt.ylabel(ordonnee_attribut)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    
    
def plot_linear_regression (df, abscisseStr, ordonneeStr) :
    # Calculer la moyenne mobile sur une semaine
    rolling_average = df[ordonneeStr].rolling(window=14).mean()

    # Convertir les dates en nombres de jours depuis le début
    days_since_start = (df[abscisseStr] - df[abscisseStr].min()).dt.days

    # Créer un modèle de régression linéaire
    model = LinearRegression()

    # Adapter le modèle aux données existantes
    model.fit(np.array(days_since_start).reshape(-1, 1), df[ordonneeStr])

    # Générer les prédictions de la régression linéaire sur l'intervalle des données existantes
    predictions = model.predict(np.array(days_since_start).reshape(-1, 1))

    # Tracer la courbe des données originales
    plt.figure(figsize=(10, 6))
    plt.plot(df[abscisseStr], df[ordonneeStr], marker='o', label=abscisseStr)

    # Tracer la moyenne mobile sur une semaine
    plt.plot(df[abscisseStr], rolling_average, color='red', label='Moyenne mobile (2 semaines)')

    # Tracer la projection de l'évolution de la courbe
    plt.plot(df[abscisseStr], predictions, linestyle='--', color='magenta', label='Projection (régression linéaire)')

    # Ajouter des titres et des labels
    plt.title(f"Évolution {ordonneeStr} dans le temps")
    plt.xlabel(abscisseStr)
    plt.ylabel(ordonneeStr)
    plt.grid(True)
    plt.legend()

    # Afficher le graphique
    plt.show()
    
    
def check_NaN_in_df (df):
    list_attributs = []
    for i in range (df.shape[1]) : list_attributs.append(df.columns[i])

    for attribut in list_attributs:
        print(attribut)
        print('     Nombre de NaN:', df[attribut].isna().sum())