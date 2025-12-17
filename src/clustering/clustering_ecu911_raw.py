import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import contextily as ctx
from matplotlib.colors import LogNorm

# 1. CONFIGURACIÓN DE RUTAS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUTA_ENTRADA = os.path.join(BASE_DIR, "data", "raw", "ecu911", "ecu911_limpio_final.csv")
RUTA_GRAFICOS = os.path.join(BASE_DIR, "data", "graphics", "ecu911")
os.makedirs(RUTA_GRAFICOS, exist_ok=True)

# 2. CARGA Y FILTRADO
print("Cargando 20 millones de registros... esto puede tardar un poco.")
df = pd.read_csv(
    RUTA_ENTRADA, 
    usecols=["lat_grid", "lon_grid", "conteo_llamadas_riesgo"]
)

# Filtro para Ecuador
df = df[
    (df["lat_grid"].between(-5.0, 1.5)) & 
    (df["lon_grid"].between(-81.0, -75.0))
]
print(f"Registros filtrados listos: {len(df):,}")

# 3. CREACIÓN DEL GRID Y CLUSTERING VISUAL

gridsize = 250  # Mayor resolución para ver calles/barrios
lon_min, lon_max = df["lon_grid"].min(), df["lon_grid"].max()
lat_min, lat_max = df["lat_grid"].min(), df["lat_grid"].max()


grid, yedges, xedges = np.histogram2d(
    df["lat_grid"], 
    df["lon_grid"], 
    bins=gridsize, 
    weights=df["conteo_llamadas_riesgo"],
    range=[[lat_min, lat_max], [lon_min, lon_max]]
)

# Aplicamos el Suavizado (Clustering)
# Sigma 1.5 a 2.0 es ideal para ver zonas urbanas
grid_smooth = gaussian_filter(grid, sigma=1.8)

# Reemplazamos ceros por NaN para que las zonas sin datos sean transparentes
grid_smooth[grid_smooth < 0.1] = np.nan

# 4. VISUALIZACIÓN FINAL
fig, ax = plt.subplots(figsize=(15, 12))

# Límites del mapa
extent = [lon_min, lon_max, lat_min, lat_max]

# Dibujamos el Heatmap con escala Logarítmica
im = ax.imshow(
    grid_smooth,
    extent=extent,
    origin="lower",
    cmap="YlOrRd",  # Amarillo -> Naranja -> Rojo
    alpha=0.6,      # Transparencia para ver el mapa base
    norm=LogNorm(vmin=1, vmax=np.nanmax(grid_smooth)), # Escala logarítmica
    zorder=2
)

# Añadimos el mapa base de OpenStreetMap
try:
    print("Descargando mapa base...")
    ctx.add_basemap(
        ax, 
        crs="EPSG:4326", 
        source=ctx.providers.CartoDB.Positron, 
        zorder=1
    )
except Exception as e:
    print(f"Nota: No se pudo cargar el mapa base (requiere internet). Error: {e}")

# Personalización estética
plt.colorbar(im, fraction=0.03, pad=0.04, label="Intensidad de Riesgo (Escala Log)")
ax.set_title("Análisis de Densidad Geoespacial", fontsize=16)
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")

# Guardar
ruta_salida = os.path.join(RUTA_GRAFICOS, "clustering_final_ecu911.png")
plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
print(f"Proceso finalizado. Imagen guardada en: {ruta_salida}")





