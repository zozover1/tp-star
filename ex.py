import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Charger le jeu de données
data = pd.read_csv('total_stars.csv')

# Nettoyer les colonnes décalées
df_cleaned = data.copy()
for col in df_cleaned.columns:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

# Sélectionner les caractéristiques nécessaires
features = ['Mass', 'Distance', 'Radius']
df_selected = df_cleaned[features]

# Imputer les valeurs manquantes
imputer = SimpleImputer(strategy='mean')  # Vous pouvez changer la stratégie si nécessaire
df_imputed = pd.DataFrame(imputer.fit_transform(df_selected), columns=features)

# Normaliser les données avec MinMaxScaler
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_imputed), columns=features)

# Utiliser l'algorithme K-means pour le regroupement
kmeans = KMeans(n_clusters=4, random_state=42)  # Vous pouvez ajuster le nombre de clusters selon votre besoin
df_normalized['Cluster'] = kmeans.fit_predict(df_normalized)

# Afficher le nombre d'étoiles par cluster
print(df_normalized['Cluster'].value_counts())

# Afficher un graphique avec matplotlib
plt.scatter(df_normalized['Mass'], df_normalized['Distance'], c=df_normalized['Cluster'], cmap='viridis')
plt.xlabel('Mass')
plt.ylabel('Distance')
plt.title('Clustering of Stars')
plt.show()
