# api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from src.model.predictor import (
    cargar_modelo, cargar_dataset,
    preparar_grid, predecir_riesgo,
    filtrar_por_zona, diagnosticar_prediccion
)
from src.model.zonas import ZONAS

app = Flask(__name__)
CORS(app) 

# rutas de archivos
ruta_dataset = os.path.join("data","processed","dataset_entrenamiento_final.csv")
ruta_modelo = os.path.join("model","modelo_riesgo_delictivo.pkl")
ruta_dbscan = os.path.join("model", "modelo_dbscan_detenciones.joblib")
ruta_perfiles = os.path.join("model", "perfiles_clusters_detenciones.joblib")

# Cargar modelo y dataset al iniciar
print("🔄 Cargando modelo y dataset...")
modelo = cargar_modelo(ruta_modelo)
df = cargar_dataset(ruta_dataset)

# Cargar junto con el modelo de riesgo
print("🔄 Cargando recursos de diagnóstico...")
modelo_dbscan = cargar_modelo(ruta_dbscan)
perfiles_clusters = cargar_modelo(ruta_perfiles)

print(" Sistema listo")

@app.route('/api/predecir', methods=['POST'])
def predecir():
    try:
        # Recibir datos del frontend
        data = request.json
        fecha_str = data.get('fecha')
        zona = data.get('zona')

        # Validaciones
        if not fecha_str or not zona:
            return jsonify({'error': 'Fecha y zona son requeridas'}), 400

        if zona not in ZONAS:
            return jsonify({'error': 'Zona no válida'}), 400

        # Convertir fecha
        fecha_dt = pd.to_datetime(fecha_str)

        # Preparar grid y predecir
        df_grid = preparar_grid(df, fecha_dt)
        df_pred = predecir_riesgo(modelo, df_grid)

        # Filtrar por zona
        limites = ZONAS[zona]
        df_zona = filtrar_por_zona(df_pred, limites)

        if df_zona.empty:
            return jsonify({'error': 'No hay datos para esta zona'}), 404

        # Normalizar valores de riesgo a rango 0-1
        riesgo_min = df_zona["prediccion_riesgo"].min()
        riesgo_max = df_zona["prediccion_riesgo"].max()

        if riesgo_max - riesgo_min < 0.1:
            riesgo_min = df_zona["prediccion_riesgo"].quantile(0.1)
            riesgo_max = df_zona["prediccion_riesgo"].quantile(0.9)

        denominador = riesgo_max - riesgo_min
        if denominador == 0:
            df_zona_norm = df_zona.copy()
            df_zona_norm["riesgo_norm"] = 0.5  # Valor neutro si no hay variación
        else:
            df_zona_norm = df_zona.copy()
            df_zona_norm["riesgo_norm"] = (
                (df_zona["prediccion_riesgo"] - riesgo_min) / denominador
            ).clip(0, 1)

        df_zona_norm["riesgo_norm"] = df_zona_norm["riesgo_norm"].fillna(0)

        # api.py - Alrededor de la línea 65
        riesgo_min = df_zona["prediccion_riesgo"].min()
        riesgo_max = df_zona["prediccion_riesgo"].max()

        # Si los valores están muy juntos, forzamos una escala para que haya contraste
        if (riesgo_max - riesgo_min) < 1e-7:
            df_zona_norm = df_zona.copy()
            df_zona_norm["riesgo_norm"] = 0.1 # Fondo azul si no hay riesgo real
        else:
            df_zona_norm = df_zona.copy()
            # Normalización Min-Max: Estira los valores para usar todo el espectro (0 a 1)
            df_zona_norm["riesgo_norm"] = (
                (df_zona["prediccion_riesgo"] - riesgo_min) / (riesgo_max - riesgo_min)
            ).clip(0, 1)

        # Limpieza de valores nulos
        df_zona_norm["riesgo_norm"] = df_zona_norm["riesgo_norm"].fillna(0)

        # Formatear datos para el heatmap
        # Formato: [[lat, lon, intensidad], [lat, lon, intensidad], ...]
        heat_data = df_zona_norm[["lat_grid", "lon_grid", "riesgo_norm"]].values.tolist()

        return jsonify({
            'datos': heat_data,
            'puntos': len(df_zona),
            'estadisticas': {
                'riesgo_min': float(riesgo_min),
                'riesgo_max': float(riesgo_max),
                'riesgo_promedio': float(df_zona["prediccion_riesgo"].mean())
            }
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/zonas', methods=['GET'])
def obtener_zonas():
    """Obtiene la lista de zonas y sus límites para evitar duplicar configuración en el frontend."""
    zonas_ordenadas = sorted(ZONAS.items(), key=lambda z: z[0])
    return jsonify({
        'zonas': [nombre for nombre, _ in zonas_ordenadas],
        'detalles': {nombre: limites for nombre, limites in zonas_ordenadas}
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Verificar que el servidor está funcionando"""
    return jsonify({'status': 'OK', 'message': 'API funcionando correctamente'})


@app.route('/api/diagnosticar', methods=['POST'])
def diagnosticar():
    try:
        data = request.json
        lat, lon = data.get('lat'), data.get('lon')
        # Llamada a la función del predictor.py
        perfil = diagnosticar_prediccion(modelo_dbscan, perfiles_clusters, lat, lon) 
        
        if not perfil:
            return jsonify({'encontrado': False, 'mensaje': 'Sin antecedentes cercanos.'})
            
        return jsonify({'encontrado': True, 'perfil': perfil})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)