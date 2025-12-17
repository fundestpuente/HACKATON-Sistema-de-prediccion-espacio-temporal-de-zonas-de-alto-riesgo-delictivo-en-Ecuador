import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import joblib

# Carga de Archivo 
ruta_entrada = os.path.join(
    "data",
    "raw",
    "ECU911",
    "ecu911_unificado.csv"
)

df = pd.read_csv(ruta_entrada, sep=';', encoding='latin1')

# Conversión de tipo de dato
df['fecha_dt'] = pd.to_datetime(df['fecha_dt'])

# Featuring temporal
df['hora'] = df['fecha_dt'].dt.hour
df['dia_semana'] = df['fecha_dt'].dt.dayofweek


# --- Latitud y Longitud ---
# Límites aproximados de Ecuador Continental
lat_min, lat_max = -6, 2
lon_min, lon_max = -82, -74

df_clean = df[
    (df['lat_grid'] >= lat_min) & (df['lat_grid'] <= lat_max) &
    (df['lon_grid'] >= lon_min) & (df['lon_grid'] <= lon_max)
].copy()

print(f"Registros originales: {len(df)}")
print(f"Registros limpios: {len(df_clean)}")


# Clustering Espacial (DBSCAN)
coords = df_clean[['lat_grid', 'lon_grid']]
coords_rad = np.radians(coords)


kms_per_radian = 6371.0088
epsilon = 1.0 / kms_per_radian   # 1 km

db = DBSCAN(
    eps=epsilon,
    min_samples=30,
    metric='haversine',
    algorithm='ball_tree'
)

df_clean['cluster'] = db.fit_predict(coords_rad)

n_clusters = len(set(df_clean['cluster'])) - (1 if -1 in df_clean['cluster'] else 0)
print(f"Clusters encontrados: {n_clusters}")


# Resumen Estratégico por Cluster
resumen_estrategico = (
    df_clean[df_clean['cluster'] != -1]
    .groupby('cluster')
    .agg({
        'tipo_evento': lambda x: x.mode()[0],   # Evento dominante
        'canton': lambda x: x.mode()[0],        # Cantón principal
        'hora': 'mean',                         # Hora promedio
        'latitud': 'mean',                      # Centroide Y
        'longitud': 'mean'                      # Centroide X
    })
    .rename(columns={
        'tipo_evento': 'evento_top',
        'canton': 'ubicacion_top'
    })
)

# Conteo de eventos por cluster
resumen_estrategico['total_eventos'] = df_clean['cluster'].value_counts()


# --- Visualización ---
# Top 10 clusters más grandes
top_10_clusters_ids = (
    df_clean[df_clean['cluster'] != -1]['cluster']
    .value_counts()
    .nlargest(10)
    .index
)

# Cálculo de centroides
centros_top = (
    df_clean[df_clean['cluster'].isin(top_10_clusters_ids)]
    .groupby('cluster')[['latitud', 'longitud']]
    .mean()
)

plt.figure(figsize=(14, 10))

# Ruido
ruido = df_clean[df_clean['cluster'] == -1]
plt.scatter(
    ruido['longitud'], ruido['latitud'],
    c="#D6D6D6", s=2, alpha=0.3, label='Ruido'
)

# Clusters
clusters_all = df_clean[df_clean['cluster'] != -1]
plt.scatter(
    clusters_all['longitud'], clusters_all['latitud'],
    c=clusters_all['cluster'],
    cmap='Spectral', s=10, alpha=0.5
)

# Centroides
plt.scatter(
    centros_top['longitud'], centros_top['latitud'],
    c='black', marker='X', s=100,
    edgecolors='white', linewidths=1.5,
    label='Top 10 Epicentros'
)

# Etiquetas
for cid in top_10_clusters_ids:
    plt.annotate(
        f"ZONA CRÍTICA {cid}",
        (centros_top.loc[cid, 'longitud'], centros_top.loc[cid, 'latitud']),
        textcoords="offset points",
        xytext=(0, 12),
        ha='center',
        fontsize=7,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, edgecolor='black')
    )

plt.title(
    'Zonas Prioritarias ECU911 según Concentración de Eventos\nTop 10 Clústeres',
    fontsize=15,
    pad=20
)
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.legend(loc='upper right')
plt.axis('equal')
plt.grid(alpha=0.3)

# Guardar gráfico
ruta_grafico_1 = os.path.join(
    "data",
    "graphics",
    "ecu911",
    "zonas-prioritarias-ecu911-top-10-clusteres.png"
)
os.makedirs(os.path.dirname(ruta_grafico_1), exist_ok=True)
plt.savefig(ruta_grafico_1)
print("Gráfico de Zonas Prioritarias ECU911 guardado!")


# --- Mapa de Calor Temporal ---
def categorizar_hora(h):
    if 0 <= h < 6:
        return '1. Madrugada'
    elif 6 <= h < 12:
        return '2. Mañana'
    elif 12 <= h < 18:
        return '3. Tarde'
    else:
        return '4. Noche'

df_clean['bloque_horario'] = df_clean['hora'].apply(categorizar_hora)

# Mapeo de nombres
mapeo_nombres = {}
for cid in top_10_clusters_ids:
    nombre = df_clean[df_clean['cluster'] == cid]['canton'].mode()[0]
    mapeo_nombres[cid] = f"{nombre} (C-{cid})"

df_top = df_clean[df_clean['cluster'].isin(top_10_clusters_ids)].copy()
df_top['canton_ordenado'] = df_top['cluster'].map(mapeo_nombres)

resumen_temporal = pd.crosstab(
    df_top['canton_ordenado'],
    df_top['bloque_horario'],
    normalize='index'
) * 100

orden_final = [mapeo_nombres[cid] for cid in top_10_clusters_ids]
resumen_temporal = resumen_temporal.reindex(orden_final)

plt.figure(figsize=(12, 9))
sns.heatmap(
    resumen_temporal,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    linewidths=.5,
    cbar_kws={'label': '% de Eventos'}
)

plt.title(
    'Perfil Temporal de las 10 Zonas con Más Eventos ECU911',
    fontsize=15,
    pad=20
)
plt.xlabel('Bloque Horario')
plt.ylabel('Cantón e ID de Cluster')
plt.tight_layout()

ruta_grafico_2 = os.path.join(
    "data",
    "graphics",
    "ecu911",
    "perfil-temporal-zonas-criticas-ecu911.png"
)
os.makedirs(os.path.dirname(ruta_grafico_2), exist_ok=True)
plt.savefig(ruta_grafico_2)
print("Gráfico de Perfil Temporal ECU911 guardado!")


# --- Artefactos ---
ruta_modelo = os.path.join("model", "modelo_dbscan_ecu911.joblib")
ruta_perfiles = os.path.join("model", "perfiles_clusters_ecu911.joblib")

os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)

joblib.dump(db, ruta_modelo)
print("Modelo DBSCAN ECU911 guardado exitosamente.")

perfiles_clusters = resumen_estrategico.to_dict(orient='index')
joblib.dump(perfiles_clusters, ruta_perfiles)
print("Perfiles de clusters ECU911 guardados exitosamente.")

