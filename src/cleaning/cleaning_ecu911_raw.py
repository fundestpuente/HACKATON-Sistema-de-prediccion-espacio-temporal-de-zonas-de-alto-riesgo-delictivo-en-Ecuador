import glob
import pandas as pd
import numpy as np
import os


RUTA_ENTRADA = os.path.join("data", "raw", "ECU911", "dataset")
RUTA_SALIDA = os.path.join("data", "raw", "ECU911", "ecu911_unificado.csv")
RUTA_CATALOGO = os.path.join(
    "data", "processed", "catalogo_parroquias_ecuador.csv"
)


patron_busqueda = os.path.join(RUTA_ENTRADA, "*.csv")
archivos_csv = glob.glob(patron_busqueda)

lista_dfs = []

if not archivos_csv:
    print(f"ERROR: No se encontraron archivos CSV en: {RUTA_ENTRADA}")
    print("Verifica que la carpeta exista y tenga archivos dentro.")
    exit()

print(f"Se encontraron {len(archivos_csv)} archivos para procesar.")

for archivo in archivos_csv:
    nombre_archivo = os.path.basename(archivo)
    print(f"-> Leyendo: {nombre_archivo}...")
    
    try:
        df_temp = pd.read_csv(
            archivo, 
            sep=';', 
            encoding='utf-8', 
            dtype={'Cod_Parroquia': str},
            on_bad_lines='skip' 
        )

        if "servicio" in df_temp.columns:
            df_temp["servicio"] = (
                df_temp["servicio"]
                .astype(str)
                .str.strip()
                .str.upper()
            )
            df_temp = df_temp[df_temp["servicio"] == "SEGURIDAD CIUDADANA"]

        lista_dfs.append(df_temp)

    except Exception as e:
        print(f"   [!] Error leyendo {nombre_archivo}: {e}")


if lista_dfs:
    df = pd.concat(lista_dfs, ignore_index=True)
    print(f"\nCarga completada. Total de registros brutos: {len(df)}")
else:
    print("No se pudo cargar ningún archivo. Terminando.")
    exit()


print("\nProcesando columna de fechas...")


df["fecha_dt"] = pd.to_datetime(
    df["Fecha"], 
    format="%d/%m/%Y", 
    errors="coerce"  
)


nulos_fecha = df["fecha_dt"].isna().sum()
if nulos_fecha > 0:
    print(f"-> Se eliminaron {nulos_fecha} registros con fechas inválidas o vacías.")
    df = df.dropna(subset=["fecha_dt"])

print(f"Registros válidos tras limpieza de fechas: {len(df)}")


cols_texto = [
    "provincia", 
    "Canton", 
    "Parroquia", 
    "Servicio", 
    "Subtipo"
]

print("\nNormalizando textos (Mayúsculas y espacios)...")

for col in cols_texto:
    if col in df.columns:
        
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        print(f"   [!] Advertencia: La columna '{col}' no se encontró en el archivo.")

df = df.rename(columns={
    "Cod_Parroquia": "cod_parroquia"
})

if "cod_parroquia" in df.columns:
    df["cod_parroquia"] = (
        df["cod_parroquia"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(6)
    )

#Cargar catalogo de parroquias
catalogo = pd.read_csv(
    RUTA_CATALOGO,
    dtype={"cod_parroquia": str}
)

# Unir con catálogo para obtener lat/lon
df = df.merge(catalogo, on="cod_parroquia", how="left")


#Limpieza de latitud y longitud
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

df = df.dropna(subset=["lat", "lon"])
df = df[(df["lat"] != 0) & (df["lon"] != 0)]

print(f"Registros con coordenadas válidas: {len(df)}")

#Grid Espacial
df["lat_grid"] = df["lat"].round(3)
df["lon_grid"] = df["lon"].round(3)

#Features Temporales
df["mes"] = df["fecha_dt"].dt.month
df["dia"] = df["fecha_dt"].dt.day
df["dia_semana"] = df["fecha_dt"].dt.dayofweek


# TARGET: Conteo de llamadas
df_group = (
    df.groupby(["lat_grid", "lon_grid", "fecha_dt"])
    .size()
    .reset_index(name="conteo_llamadas_riesgo")
)

df = df.merge(
    df_group,
    on=["lat_grid", "lon_grid", "fecha_dt"],
    how="left"
)


df.columns = [c.lower() for c in df.columns]

os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)

print(f"\nGuardando archivo unificado en: {RUTA_SALIDA}")

df.to_csv(RUTA_SALIDA, index=False, sep=';', encoding='utf-8')

print("--- ¡PROCESO TERMINADO EXITOSAMENTE! ---")
