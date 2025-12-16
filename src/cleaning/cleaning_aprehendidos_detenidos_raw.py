import pandas as pd
import numpy as np
import os


# CARGA DEL DATASET

ruta_excel = os.path.join(
    "data",
    "raw",
    "detenidosaprehendidos",
    "dataset",
    "mdi_detenidosaprehendidos_pm_2025_enero_octubre.xlsx"
)

ruta_salida = os.path.join(
    "data",
    "raw",
    "detenidosaprehendidos",
    "aprehendidos_detenidos_raw.csv"
)

# Cargar datos

df = pd.read_excel(
    ruta_excel,
    sheet_name=1,
    engine="openpyxl"
)

print(f"Registros originales: {len(df)}")


# Normalizacion latitud y longitud
for col in ["latitud", "longitud"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Eliminar coordenadas inválidas
df = df.dropna(subset=["latitud", "longitud"])
df = df[(df["latitud"] != 0) & (df["longitud"] != 0)]

print(f"Registros con coordenadas válidas: {len(df)}")

# Union de fecha y hora en un solo campo datetime
df["fecha_dt"] = pd.to_datetime(
    df["fecha_detencion_aprehension"].astype(str) + " " +
    df["hora_detencion_aprehension"].astype(str),
    errors="coerce"
)

# eliminar registros con fecha inválida
df = df.dropna(subset=["fecha_dt"])

print(f"Registros con fecha válida: {len(df)}")



# Normalizacion de campos geograficos
cols_geo = [
    "nombre_provincia",
    "nombre_canton",
    "nombre_parroquia"
]

for col in cols_geo:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.upper()
    )
#selección de columnas finales
cols_finales = [
    "fecha_dt",
    "latitud",
    "longitud",
    "nombre_provincia",
    "nombre_canton",
    "nombre_parroquia",
    "presunta_infraccion"
]

df_clean = df[cols_finales].copy()

print("Columnas finales:", df_clean.columns.tolist())
print(f"Registros finales limpios: {len(df_clean)}")

#finalmente, guardar el dataset limpio
os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

df_clean.to_csv(
    ruta_salida,
    index=False,
    encoding="utf-8"
)

print(f"Archivo guardado en: {ruta_salida}")
