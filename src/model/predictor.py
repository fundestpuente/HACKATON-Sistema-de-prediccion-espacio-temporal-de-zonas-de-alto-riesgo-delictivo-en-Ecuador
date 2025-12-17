#prediccion del modelo entrenado
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import haversine_distances


#MÓDULO DE CARGA DE RECURSOS

# Funciones utilitarias para la carga serializada de modelos y dataFrames
def cargar_modelo(ruta):
    return joblib.load(ruta)

def cargar_dataset(ruta):
    return pd.read_csv(ruta)

#PREPARACIÓN DEL GRID Y PREDICCIÓN

def preparar_grid(df, fecha_dt):
    columnas_modelo = [
        "lat_grid", "lon_grid", "mes", "dia", "dia_semana",
        "conteo_delitos_graves", "conteo_llamadas_riesgo"
    ]

    #Obtiene las coordenadas únicas de la cuadrícula 
    df_grid = df[["lat_grid", "lon_grid"]].drop_duplicates().copy()

    df_grid["mes"] = fecha_dt.month
    df_grid["dia"] = fecha_dt.day
    df_grid["dia_semana"] = fecha_dt.weekday() # 'weekday' retorna 0=Lunes a 6=Domingo

    # Se asume conteo cero para features de eventos pasados en la fecha futura de predicción
    df_grid["conteo_delitos_graves"] = 0
    df_grid["conteo_llamadas_riesgo"] = 0

    return df_grid[columnas_modelo]


def predecir_riesgo(modelo, df_grid):
    df_grid = df_grid.copy()
    # Ejecuta la inferencia (predicción) del modelo sobre los datos del grid
    df_grid["prediccion_riesgo"] = modelo.predict(df_grid)
    return df_grid

#FILTRADO GEOGRÁFICO

def filtrar_por_zona(df, limites):

    return df[
        (df["lon_grid"] >= limites["lon_min"]) &
        (df["lon_grid"] <= limites["lon_max"]) &
        (df["lat_grid"] >= limites["lat_min"]) &
        (df["lat_grid"] <= limites["lat_max"])
    ].copy()

# INFORMACIÓN ADICIAONAL

def diagnosticar_prediccion(model, profile,lat, lon):
    """
    Determina si una coordenada de riesgo predicha coincide con uncluster hitorico.
    
    :param model: modelo DBSCAN
    :param profile: perfil del cLuster
    :param lat: Latitud
    :param lon: Longitud
    :return: perfil(dict) del cluster cercano a la coordenada geografica
    """
    punto_nuevo = np.radians([[lat, lon]])
    
    puntos_core = model.components_
    etiquetas_core = model.labels_[model.core_sample_indices_]
    
    distancias = haversine_distances(punto_nuevo, puntos_core)

    indice_min = np.argmin(distancias)
    distancia_min = distancias[0, indice_min]
    
    if distancia_min <= model.eps:
        cluster_id = etiquetas_core[indice_min]

        if cluster_id != -1 and cluster_id in profile:
            perfil = profile[cluster_id].copy()
            # Convertir posibles valores numpy a tipos nativos de Python para JSON
            for k, v in perfil.items():
                if hasattr(v, 'item'): # Detecta tipos de numpy
                    perfil[k] = v.item()
            return perfil
    return dict()


# CODIGO DE PRUEBA
if __name__ == "__main__":
    try:
        modelo_geo = joblib.load('model/modelo_dbscan_detenciones.joblib')
        info_clusters = joblib.load('model/perfiles_clusters_detenciones.joblib')
    except Exception as e:
        print(e)
    

    perfil = diagnosticar_prediccion(modelo_geo,info_clusters,-2.10,-79.9)

    print("cantidad: ",len(perfil))

    for clave, valor in perfil.items():
        print(f"{clave}: {valor}")