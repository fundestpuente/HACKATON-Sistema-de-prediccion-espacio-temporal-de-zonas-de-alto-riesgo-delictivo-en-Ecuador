# HACKATON-Sistema-de-prediccion-espacio-temporal-de-zonas-de-alto-riesgo-delictivo-en-Ecuador


**Índice:**
- [Descripción del proyecto](#descripción-del-proyecto)
- [Integrantes del grupo](#integrantes-del-grupo)
- [Instalación](#instalacion)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Ejecución](#ejecución)
- [Uso](#uso)
- [Herramientas Implementadas](#herramientas-implementadas)

---

# Descripción del proyecto
Este proyecto tiene como objetivo desarrollar un sistema de inteligencia artificial capaz de identificar y predecir zonas de alto riesgo delictivo en Ecuador, integrando registros de detenidos con datos de llamadas de emergencia. Para ello, se realiza un procesamiento y análisis geoespacial de la información, se entrenan modelos de predicción (XGBoost) y se generan visualizaciones mediante mapas interactivos.

---

# Integrantes del Grupo
- Angie Alfonso
- Christian Zavala
- Fernando Alvarez 
- Fiorella Quijana
- Joseph Carrera
---

# Instalación

## Requisitos
- Python 3.11+ (Recomendado)
- Git
- Git LFS

## Pasos
1. **Clonar el repositorio**
    ```bash
    git clone https://github.com/fundestpuente/SIC-Sistema-de-prediccion-espacio-temporal-de-zonas-de-alto-riesgo-delictivo-en-Ecuador.git
    cd SIC-Sistema-de-prediccion-espacio-temporal-de-zonas-de-alto-riesgo-delictivo-en-Ecuador
    ```

2. **Crear y activar el entorno virtual (Recomendado)**
    ```bash
    python -m venv venv
    ```
    Activar entorno en Linux/MacOS
    ```bash
    source venv/bin/activate
    ```
    Activar entorno en Windows
    ```bash
    .\venv\Scripts\activate
    ```

3. **Instalar dependencias**
    ```bash
    pip install -r requirements.txt
    ```

---

# Estructura del proyecto
    SIC-Sistema-de-prediccion-espacio-temporal-de-zonas-de-alto-riesgo-delictivo-en-Ecuador/
    |
    ├── data/
    |   ├── graphics/            <- Graficos estadisticos
    │   ├── raw/                 <- Datos originales
    │   └── processed/           <- Datasets procesados
    |
    ├── models/                  <- modelos guardados
    |
    ├── src/
    │   ├── cleaning/            <- Scripts para el preprocesamiento
    |   ├── clustering/          <- Scripts de entrenamiento para cluster              
    │   └── models/              <- Scripts de entrenamiento y predicción
    │
    ├── api.py                   <- app (backend)
    ├── index.html               <- app (frontend)
    ├── requirements.txt         <- librerias necesarias
    |
    └── README.md

---

# Ejecución

En la carpeta del proyecto ejecutar el siguiente comando:
```bash
python api.py
```
El comando inicia el servidor de manera local y la aplicación se accede mediante el archivo `index.html`

---

# Uso
El funcionamiento de la aplicación es el mismo sin importar su versión.

1. Seleccione el día mediante el calendario
2. Seleccione la provincia
3. Presione el botón **Predecir** o **Generar Predicción**
4. Ya puede visualizar el mapa con la predicción delictiva de ese día y provincia

---

# Herramientas Implementadas
- **Lenguaje:** Python 3.1x
- **Librerías principales:** pandas, joblib, xgboost, scikit-learn