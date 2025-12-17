import pandas as pd
import numpy as np
import os

#carga datasets
ruta_padre = os.path.join(os.pardir, os.pardir, "data")
ruta_procesados = os.path.join(ruta_padre, "processed")
datos_aprehendidos = os.path.join(ruta_padre,"raw","detenidosaprehendidos", "aprehendidos_detenidos_raw.csv")
datos_911 = os.path.join(ruta_padre,"raw", "ecu911" ,"ecu911_unificado.csv")
nombre_datos_procesados = os.path.join(ruta_procesados, "dataset_entrenamiento_final.csv")

df_apre = pd.read_csv(datos_aprehendidos)
df_911 = pd.read_csv(datos_911)

# Normalizacion de fechas
df_911["fecha"] = pd.to_datetime(df_911["fecha_dt"], errors="coerce").dt.date
df_apre["fecha"] = pd.to_datetime(df_apre["fecha_dt"], errors="coerce").dt.date

df_apre = df_apre.rename(columns= {
    "latitud": "lat_grid",
    "longitud": "lon_grid"
})

#union de datasets por celdas 
df_union = df_apre.merge(
    df_911,
    on=["lat_grid", "lon_grid", "fecha"],
    how="outer",
    suffixes=("_apre", "_911")
)

#Imputación de valores faltantes (NaN) en variables de conteo (Targets)
for col in ["conteo_delitos", "conteo_delitos_graves", "conteo_llamadas_riesgo"]:
    if col in df_union.columns:
        df_union[col].fillna(0, inplace=True) #Imputación con cero
    else:
        df_union[col] = 0  # si no existe, la creamos

#Ingeniería de Características Temporales

if "mes" not in df_union.columns:
    df_union["mes"] = pd.to_datetime(df_union["fecha"]).dt.month  #Extracción del mes

if "dia" not in df_union.columns:
    df_union["dia"] = pd.to_datetime(df_union["fecha"]).dt.day  #Extracción del dia del mes

if "dia_semana" not in df_union.columns:
    df_union["dia_semana"] = pd.to_datetime(df_union["fecha"]).dt.dayofweek   #Extracción del día de la semana
    

# Eliminar franja_horaria porque no contiene datos importantes
if "franja_horaria" in df_union.columns:
    df_union.drop(columns=["franja_horaria"], inplace=True)

# features
features = [
    "lat_grid", "lon_grid",
    "mes", "dia", "dia_semana",
    "franja_horaria",
    "conteo_delitos",
    "conteo_delitos_graves",
    "conteo_llamadas_riesgo"
]

# asegurarse de que las columnas existan
features = [c for c in features if c in df_union.columns]

df_final = df_union[features].copy()

# guardar dataset final
df_final.to_csv(nombre_datos_procesados, index=False)
print("dataset de entramiento guardado: dataset_entranamiento_final.csv")
print("Registros totales:", len(df_final))
print("Columnas:", df_final.columns.tolist())