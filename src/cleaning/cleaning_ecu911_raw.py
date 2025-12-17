import pandas as pd
import glob
import os

#carga de dataset ecu911
ruta_padre = os.path.join(os.pardir, os.pardir, "data")
ruta_datos_originales = os.path.join(ruta_padre, "raw", "ecu911", "dataset")
archivos_csv = glob.glob(os.path.join(ruta_datos_originales, "*.csv"))
catalogo_parroquias = os.path.join(ruta_padre, "processed","catalogo_parroquias_ecuador.csv")

nombre_datos_procesados = os.path.join(ruta_padre,"processed","ecu911_limpio_final.csv")

lista_dfs = []

for archivo in archivos_csv:
    try:
        df_temp = pd.read_csv(
            archivo,
            sep=";",
            encoding="UTF-8",
            dtype={"Cod_Parroquia": str},
            on_bad_lines="skip"
        )
        
        # Normalizacion de columnas
        df_temp.columns = (
            df_temp.columns.astype(str)
            .str.replace("ï»¿", "")
            .str.strip()
            .str.lower()
        )
        
        # Filtrar seguridad ciudadana
        if "servicio" in df_temp.columns:
            df_temp = df_temp[df_temp["servicio"] == "Seguridad Ciudadana"]
        
        lista_dfs.append(df_temp)

    except Exception as e:
        print(f"Error procesando {archivo}: {e}")

df_911 = pd.concat(lista_dfs, ignore_index=True)

# Fechas
df_911["fecha_dt"] = pd.to_datetime(df_911["fecha"], errors="coerce", dayfirst=True)
df_911 = df_911.dropna(subset=["fecha_dt"])

# Códigos parroquia normalizados
df_911["cod_parroquia"] = (
    df_911["cod_parroquia"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.zfill(6)
)
#cargar catalogo de parroquias
catalogo = pd.read_csv(catalogo_parroquias, dtype={"cod_parroquia": str})
# Unir con catálogo para obtener lat/lon
df_911 = df_911.merge(catalogo, on="cod_parroquia", how="left")

# Eliminar registros sin coordenadas
df_911 = df_911.dropna(subset=["lat", "lon"])
#grid espacial
df_911["lat_grid"] = df_911["lat"].round(3)
df_911["lon_grid"] = df_911["lon"].round(3)
#featrures temporales
df_911["mes"] = df_911["fecha_dt"].dt.month
df_911["dia"] = df_911["fecha_dt"].dt.day
df_911["dia_semana"] = df_911["fecha_dt"].dt.dayofweek

#targe ecu911 - conteo de llamadas por dia y zona
df_group = (
    df_911.groupby(["lat_grid", "lon_grid", "fecha_dt"])
    .size()
    .reset_index(name="conteo_llamadas_riesgo")
)

df_911 = df_911.merge(
    df_group, 
    on=["lat_grid", "lon_grid", "fecha_dt"], 
    how="left"
)

# Guardar dataset limpio
df_911.to_csv(nombre_datos_procesados, index=False)

print("ECU911 procesado correctamente")
print(f"Registros finales: {len(df_911)}")
