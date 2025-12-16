import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

# --- Carga de Archivo ---
ruta_entrada = os.path.join(
    "data",
    "raw",
    "detenidosaprehendidos",
    "aprehendidos_detenidos_raw.csv"
)

df = pd.read_csv(ruta_entrada)

# Conversion de tipo de dato
df['fecha_dt'] = pd.to_datetime(df['fecha_dt'])

# featuring
df['hora'] = df['fecha_dt'].dt.hour
df['dia_semana'] = df['fecha_dt'].dt.dayofweek

# --- Latitud y Longitud ---
# Limites aproximados de Ecuador Continental
lat_min, lat_max = -6, 2
lon_min, lon_max = -82, -74

# Filtrar el DataFrame
df_clean = df[
    (df['latitud'] >= lat_min) & (df['latitud'] <= lat_max) &
    (df['longitud'] >= lon_min) & (df['longitud'] <= lon_max)
].copy()

print(f"Registros originales: {len(df)}")
print(f"Registros limpios: {len(df_clean)}")

# --- Clustering ---
coords = df_clean[['latitud', 'longitud']]
coords_rad = np.radians(coords)

kms_per_radian = 6371.0088
epsilon = 0.5 / kms_per_radian  # Radio de 500 metros (0.5 km)

db = DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric='haversine')

df_clean['cluster'] = db.fit_predict(coords_rad)


# Verificar cuántos clusters salieron (excluyendo el -1 que es ruido)
n_clusters = len(set(df_clean['cluster'])) - (1 if -1 in df_clean['cluster'] else 0)
print(f"Número de clusters encontrados: {n_clusters}")

# Top 5 Clusters más grandes
top_clusters = df_clean['cluster'].value_counts().head(5)
print("Top 5 Clusters más grandes (ID y cantidad de infracciones):")
print(top_clusters)

# En detalle del cluster más grande 
id_cluster_mas_grande = top_clusters.index[0]
data_cluster = df_clean[df_clean['cluster'] == id_cluster_mas_grande]

print(f"\n--- Análisis del Cluster {id_cluster_mas_grande} ---")
print(data_cluster['presunta_infraccion'].value_counts().head(3))
print(f"Cantón predominante: {data_cluster['nombre_canton'].mode()[0]}")
print(f"Hora promedio de incidentes: {data_cluster['hora'].mean():.1f}")


# --- Visualizacion ---

# Gráfico de Dispersión - Todos los clusters
# Puntos de Ruido
ruido = df_clean[df_clean['cluster'] == -1]
plt.scatter(ruido['longitud'], ruido['latitud'], c='#E0E0E0', s=2, label='Ruido', alpha=0.5)

# Clusters
clusters = df_clean[df_clean['cluster'] != -1]
plt.scatter(clusters['longitud'], clusters['latitud'], c=clusters['cluster'], cmap='Spectral', s=10)

plt.title(f'Mapa de Calor de Infracciones (Total Clusters: {n_clusters})')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.axis('equal') 
plt.show()

# Gráfico de Boxplot - Cluster más grandes - Horas de las infracciones
top_clusters_ids = [1, 27, 2, 40]
df_top = df_clean[df_clean['cluster'].isin(top_clusters_ids)]

plt.figure(figsize=(12, 6))
sns.boxplot(x='cluster', y='hora', data=df_top)
plt.title('¿A qué hora ocurren las infracciones en cada zona caliente?')
plt.show()