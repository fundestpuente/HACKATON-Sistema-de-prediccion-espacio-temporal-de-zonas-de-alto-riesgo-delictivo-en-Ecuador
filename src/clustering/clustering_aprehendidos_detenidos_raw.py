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

# Mapeo de cantones por cada cluster
mapeo_cantones = df_clean[df_clean['cluster'] != -1].groupby('cluster')['nombre_canton'].agg(lambda x: x.mode()[0]).to_dict()


# Calcular la proporción del delito más frecuente por cada cluster
dominancia = df_clean.groupby('cluster')['presunta_infraccion'].value_counts(normalize=True).unstack(fill_value=0)
dominancia['max_dominancia'] = dominancia.max(axis=1)
dominancia['delito_principal'] = dominancia.idxmax(axis=1)

dominancia['nombre_canton'] = dominancia.index.map(mapeo_cantones)
# Filtrar solo clusters reales y ver los que tienen más del 60% de especialización
especializados = dominancia[
    (dominancia.index != -1) & 
    (dominancia['max_dominancia'] > 0.6)
][['nombre_canton', 'delito_principal', 'max_dominancia']]

# Orden por nivel de dominancia
especializados = especializados.sort_values(by='max_dominancia', ascending=False)
ruta_especializados = os.path.join(
    "data",
    "processed",
    "clusters_con_delitos_especializados_por_Canton.csv"
)
os.makedirs(os.path.dirname(ruta_especializados), exist_ok=True)
especializados.to_csv(ruta_especializados)

print("Archivo 'clusters_con_delitos_especializados_por_Canton.csv' con existo!")

# --- Visualizacion ---

# Obtención de los ID de los 10 cluster más grandes
top_10_clusters_ids = df_clean[df_clean['cluster'] != -1]['cluster'].value_counts().nlargest(10).index

# Gráfico de Dispersión - Todos los clusters

# Calculo de los centroides para los 10 cluster
centros_top = df_clean[df_clean['cluster'].isin(top_10_clusters_ids)].groupby('cluster')[['latitud', 'longitud']].mean()

plt.figure(figsize=(14, 10))

# Puntos de Ruido
ruido = df_clean[df_clean['cluster'] == -1]
plt.scatter(ruido['longitud'], ruido['latitud'], c="#D6D6D6", s=2, label='Ruido', alpha=0.3)

# Puntos de los clusters
clusters_all = df_clean[df_clean['cluster'] != -1]
plt.scatter(clusters_all['longitud'], clusters_all['latitud'], c=clusters_all['cluster'], cmap='Spectral', s=10, alpha=0.5)

# Puntos de los centroides
plt.scatter(centros_top['longitud'], centros_top['latitud'], 
            c='black', marker='X', s=100, edgecolors='white', linewidths=1.5, label='Top 10 Epicentros')

# Etiqueta para los centroides
for i in top_10_clusters_ids:
    plt.annotate(f"ZONA CRÍTICA {i}", 
                    (centros_top.loc[i, 'longitud'], centros_top.loc[i, 'latitud']),
                    textcoords="offset points", 
                    xytext=(0,12), 
                    ha='center', 
                    fontsize=7, 
                    fontweight='bold',
                    color='black',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, edgecolor='black')
                )

plt.title('Zonas Prioritarias según Concentración de Detenciones\nTop 10 Clústeres', fontsize=15, pad=20)
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.legend(loc='upper right')
plt.axis('equal')
plt.grid(True, alpha=0.3)

#Guardar grafico
ruta_grafico_1 = os.path.join(
    "data",
    "graphics",
    "detenidos",
    "zonas-prioritarias-segun-concentracion-de-detenciones-top-10-clusteres.png"
)
os.makedirs(os.path.dirname(ruta_grafico_1), exist_ok=True)
plt.savefig(ruta_grafico_1)
print("Gráfico de Zonas Prioritarias guardado!")

# Mapa de Calor - 10 Clusters más grandes por Bloque Horario

# Categorias de tiempo
def categorizar_hora(h):
    if 0 <= h < 6: return '1. Madrugada'
    elif 6 <= h < 12: return '2. Mañana'
    elif 12 <= h < 18: return '3. Tarde'
    else: return '4. Noche'

df_clean['bloque_horario'] = df_clean['hora'].apply(categorizar_hora)

# Mapeo de ID del cluster con el nombre del cantón mediante la moda
mapeo_nombres = {}
for cid in top_10_clusters_ids:
    nombre = df_clean[df_clean['cluster'] == cid]['nombre_canton'].mode()[0]
    mapeo_nombres[cid] = f"{nombre} (C-{cid})"

# Preparacion de los datos
df_top = df_clean[df_clean['cluster'].isin(top_10_clusters_ids)].copy()
df_top['canton_ordenado'] = df_top['cluster'].map(mapeo_nombres)

# Generación de la tabla cruzada
resumen_temporal = pd.crosstab(
    df_top['canton_ordenado'], 
    df_top['bloque_horario'], 
    normalize='index'
) * 100

# Reordenamiento según el tamaño del cluster
orden_final = [mapeo_nombres[cid] for cid in top_10_clusters_ids]
resumen_temporal = resumen_temporal.reindex(orden_final)

plt.figure(figsize=(12, 9))
sns.heatmap(resumen_temporal, annot=True, fmt=".1f", cmap="YlOrRd", 
            linewidths=.5, cbar_kws={'label': '% de Detenciones en el Cantón'})

plt.title('Perfil Temporal de las 10 Zonas con Más Detenciones\n(Ordenadas de mayor a menor volumen)', fontsize=15, pad=20)
plt.xlabel('Bloque Horario', fontsize=12)
plt.ylabel('Cantón e ID de Cluster', fontsize=12)
plt.tight_layout()

#Guardar frafico
ruta_grafico_2 = os.path.join(
    "data",
    "graphics",
    "detenidos",
    "perfil-temporal-de-las-10-zonas-con-mas-detenciones.png"
)
os.makedirs(os.path.dirname(ruta_grafico_2), exist_ok=True)
plt.savefig(ruta_grafico_2)
print("Gráfico de Perfil Temporal guardado!")