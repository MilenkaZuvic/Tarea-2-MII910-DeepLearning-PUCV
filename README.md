# Tarea-2-MII910-DeepLearning-PUCV

__Pontificia Universidad Católica de Valparaíso — Escuela de Ingeniería Informática__ 
__Semestre 1-2026 · Profesor: Carlos Valle__

## Integrantes

- Nicolás Zárate
- David Cáceres
- Milenka Zuvic
- Brayan López

# Parte 1 de la Tarea - Clasificación de Niveles de Obesidad

## Descripción

El proyecto aborda un problema de **pronóstico de series temporales multivariadas**: dado un histórico de mediciones horarias de contaminantes y variables meteorológicas, se predice la concentración de PM2.5 de las **próximas 24 horas** a partir de una ventana de entrada de **48 horas**.

Se entrenan y comparan dos arquitecturas recurrentes (LSTM y GRU), cada una con dos capas recurrentes y una capa densa intermedia, ajustando hiperparámetros mediante búsqueda aleatoria.

### Dataset

[Beijing Multi-Site Air-Quality Data — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)

Mediciones horarias (2013–2017) de la estación **Guanyuan**:

- **Contaminantes:** PM2.5, PM10, SO2, NO2, CO, O3
- **Meteorológicas:** temperatura (TEMP), presión (PRES), punto de rocío (DEWP), lluvia (RAIN), velocidad del viento (WSPM), dirección del viento (wd)

## Estructura del notebook (`Parte_1.ipynb`)

| Sección | Contenido |
|---------|-----------|
| **1.a** Carga y preprocesamiento | Carga del CSV, análisis exploratorio (series, distribuciones, correlaciones, ACF), manejo de faltantes, división temporal 70/20/10, codificación cíclica de la hora/día, escalado y generación de secuencias deslizantes (48h → 24h) |
| **1.b** Función de evaluación | Cálculo del MSE en la **escala original** (invirtiendo el scaler) |
| **1.c** Búsqueda LSTM | Función de entrenamiento, espacios de búsqueda y *random search* (15 trials) |
| **1.d** Búsqueda GRU | Misma estrategia aplicada a GRU, midiendo además el tiempo de entrenamiento |
| **1.e** Comparación | LSTM vs GRU en MSE, número de parámetros, tiempo y curvas de aprendizaje |

### Decisiones metodológicas clave

- **División cronológica** (no aleatoria) para preservar la dependencia temporal y evitar *data leakage*.
- **Imputación por interpolación temporal** de los faltantes, en lugar de eliminar filas, para no romper la continuidad de las ventanas.
- **Codificación cíclica** (seno/coseno) de hora y día de la semana para que horas vecinas (23 y 0) queden próximas en el espacio de entrada.
- **Scaler ajustado solo con el conjunto de entrenamiento** y aplicado a validación y test.

## Requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels tensorflow
```

> Se recomienda ejecutar en **Google Colab** con acelerador (GPU/TPU).

## Uso

1. Descargar el dataset desde UCI y ubicarlo de modo que la ruta coincida con la del notebook:

   ```
   beijing+multi+site+air+quality+data/PRSA_Data_20130301-20170228/PRSA_Data_Guanyuan_20130301-20170228.csv
   ```

2. Abrir y ejecutar `Parte_1.ipynb` de principio a fin.

## Resultados

| Métrica | Mejor LSTM | Mejor GRU |
|---------|-----------:|----------:|
| MSE en test (escala original) | 6.253,03 | **6.108,22** |
| Número de parámetros | 493.480 | **104.408** |
| Tiempo de entrenamiento | — | menor |

**Conclusión:** la arquitectura **GRU** resulta más adecuada para este problema. Alcanza un menor error en test con casi **5× menos parámetros**, menor tiempo de entrenamiento y una curva de aprendizaje más estable, siguiendo mejor la tendencia del PM2.5 real. El desempeño global queda limitado principalmente por el *distribution shift* entre el periodo de entrenamiento (2013–2015) y el de validación/test (2016–2017), y por la baja autocorrelación de la serie a 24 horas, más que por la elección de hiperparámetros.
