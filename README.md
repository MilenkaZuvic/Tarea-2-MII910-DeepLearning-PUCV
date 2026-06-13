# Tarea-2-MII910-DeepLearning-PUCV

__Pontificia Universidad Católica de Valparaíso — Escuela de Ingeniería Informática__ 
__Semestre 1-2026 · Profesor: Carlos Valle__

## Integrantes

- Nicolás Zárate
- David Cáceres
- Milenka Zuvic
- Brayan López

# Parte 1 de la Tarea - Redes Neuronales Recurrentes

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

#    Parte 2: Clasificación Jerárquica de Comidas del Mundo

Solución para el desafío Kaggle de **clasificación jerárquica de imágenes de comida** en dos niveles simultáneos: el **origen geográfico** de la cocina y el **plato específico**. La métrica oficial es **Macro F1** sobre las 107 clases (6 de nivel 1 + 101 de nivel 2).

🔗 [Enlace al desafío en Kaggle](https://www.kaggle.com/competitions/clasificacion-jerarquica-de-comidas-del-mundo)

> **Resultado final (v10): Public Score `0.93258`** — una mejora de **+0.05155** sobre el baseline (`0.88103`).

---

## 📋 Tabla de contenidos

- [Descripción del problema](#-descripción-del-problema)
- [Enfoque de la solución](#-enfoque-de-la-solución)
- [Estructura del notebook](#-estructura-del-notebook)
- [Resultados](#-resultados)
- [Evolución de versiones](#-evolución-de-versiones-v1--v10)
- [Requisitos](#-requisitos)

---

## 🎯 Descripción del problema

Cada imagen activa **exactamente una clase de cada nivel**:

| Nivel | Clases | IDs | Descripción |
|-------|:------:|:----:|-------------|
| **Nivel 1** (categoría) | 6 | `0–5` | 5 orígenes geográficos + 1 categoría temática (Postres & Dulces) |
| **Nivel 2** (plato) | 101 | `6–106` | Plato específico |

**Formato de submission** (`label` = nivel 1 + nivel 2 separados por espacio):

```
filename,label
042137.jpg,5 6
081432.jpg,1 101
```

---

## 🧠 Enfoque de la solución

El diseño central combina un **encoder compartido** con una **doble cabeza jerárquica** y mascarado de coherencia en inferencia.

### Decisiones de diseño clave
- **Backbone:** `convnextv2_base.fcmae_ft_in22k_in1k` (transfer learning con `timm`). Modelos adicionales en el ensemble final: ConvNeXt V2-Large @384 y EVA-02-Base @448.
- **Doble cabeza** sobre un encoder único: 6 logits (nivel 1) + 101 logits (nivel 2), con dropout.
- **Pérdida combinada** `w1·L1 + w2·L2` con `w1 < w2` (nivel 2 domina la métrica).
- **Focal Loss** (γ=2) para enfocar el entrenamiento en ejemplos difíciles.
- **Label smoothing** + **augmentation fuerte** (RandAugment, RandomResizedCrop, RandomErasing, ColorJitter, MixUp/CutMix) contra el ruido de etiquetas.
- **Mascarado jerárquico en inferencia:** dado que cada plato pertenece a un único origen (relación muchos-a-uno), se enmascaran los logits incoherentes → elimina toda una clase de errores triviales.
- **Fine-tuning en 2 fases:** primero solo las cabezas (backbone congelado), luego todo con `lr` menor.
- **Ensemble multi-seed**, **model soup**, **pseudo-labeling** y **blend de múltiples familias** de modelos.

### Técnicas de optimización
- Entrenamiento con **AMP** (mixed precision), `channels_last`, TF32 y gradient accumulation (batch efectivo = 32).
- **Sistema de caché** reutilizable para pasos costosos (pre-decode/resize de imágenes, HP search, entrenamiento, predicción) → permite reanudar tras reinicios.
- **Autodetección de GPU/CUDA** y de la ruta del dataset.

---

## 📓 Estructura del notebook

El notebook [`Parte_2.ipynb`](Parte_2.ipynb) sigue estas secciones:

| Sección | Contenido |
|---------|-----------|
| **0. Config y requerimientos** | Instalación de dependencias, configuración central (`CFG`), reproducibilidad, GPU, caché |
| **1. Análisis exploratorio (EDA)** | Distribución por nivel, desbalance, calidad de imágenes, relación plato→origen |
| **2. Preprocesamiento** | Split estratificado, transformaciones, data augmentation, `Dataset` |
| **3. Modelos explorados** | Arquitectura jerárquica, Focal Loss, MixUp, mascarado en inferencia |
| **4. Búsqueda de hiperparámetros** | Grid pequeño sobre subset estratificado con early-stop agresivo |
| **5. Resultados y discusión** | Curvas, matriz de confusión, peores clases, errores |
| **6. Generación de submission** | Predicción sobre test + verificación de formato |
| **7. Evolución de versiones** | Historial v1 → v10 con scores |
| **8. Investigación de mejoras** | Análisis de las 9 palancas exploradas y hallazgos transversales |

### Hallazgos del EDA
- Nivel 1: desbalance moderado (ratio máx/mín ≈ 4.1).
- Nivel 2: **perfectamente balanceado** (800 imágenes/clase, ratio = 1.00).
- Cada plato → un único origen (relación muchos-a-uno verificada) → valida el mascarado jerárquico.

---

## 📊 Resultados

- La Macro F1 global está dominada por las 101 clases de nivel 2 (más numerosas y confusas); el nivel 1 alcanza F1 alto con facilidad gracias al encoder pre-entrenado.
- Las clases de nivel 2 con peor F1 son visualmente similares entre sí o sufren más el ruido de etiquetas.
- El mascarado jerárquico garantiza coherencia plato→origen y elimina errores triviales.

---

## 📈 Evolución de versiones (v1 → v10)

| Versión | Cambio principal | Public Score | Δ |
|:-------:|------------------|:------------:|:----:|
| **v1** | Baseline: ConvNeXt V2 **Tiny**, doble cabeza, label smoothing, mascarado | 0.88103 | — |
| **v2** | Focal Loss + CutMix/MixUp + augmentation fuerte | 0.88146 | +0.00043 |
| v3 | Infraestructura de ensemble multi-seed | *(no subido)* | — |
| **v4** | Backbone **Base** (89M) + TTA multi-vista + ensemble 2 seeds | 0.91282 | **+0.03136** |
| v5/v6 | TTA multi-escala, decisión desacoplada, FixRes 288 | 0.91218–0.91265 | *(negativos)* |
| **v7a** | **Model soup** (promedio de pesos s42+s7) | 0.91352 | +0.00070 |
| **v7b** | **Pseudo-labeling** (conf≥0.8) + fine-tune full-data | 0.91457 | +0.00105 |
| **v8a** | **ConvNeXt V2-Large @384** (Kaggle T4, 9.2h) | 0.91658 | +0.00201 |
| **v8** | **Blend 2 familias** (local + Large-ft3) | 0.92329 | +0.00671 |
| **v9a** | **EVA-02-Base @448** (ViT/MIM, Kaggle T4×2, 5.7h) | 0.92685 | +0.00356 |
| **v9** | **Blend 3 familias** (1/3 local + 1/3 Large + 1/3 EVA) | 0.93030 | +0.00345 |
| **v10 (final)** | **EVA-02 v2** + blend EVA-pesado (25/25/50) | **0.93258** | **+0.00228** |

**Lecciones transversales:**
1. La capacidad del backbone es la palanca dominante (v1→v4: +0.032).
2. La validación local **no es proxy fiable** para diferencias <0.002.
3. Lo que transfiere al leaderboard son los **mecanismos monótonos**: capacidad, datos, votantes fuertes, promediado de pesos.
4. La **diversidad de familias** de modelos fue la palanca compuesta más rentable.
5. Ponderar el miembro más fuerte del blend supera al blend uniforme.

---

## ⚙️ Requisitos

- **Python 3.9+**
- **PyTorch** (con CUDA recomendado; funciona en CPU pero lento)
- Dependencias principales (se instalan automáticamente en la sección 0 del notebook):

```
torch
timm
albumentations
scikit-learn
pandas
numpy<2
matplotlib
seaborn
pillow
tqdm
```


### Dataset
Estructura esperada (carpeta con):
```
train.csv
test_public.csv
sample_submission.csv
dict.npy          # id -> nombre legible
train/            # imágenes de entrenamiento
test/             # imágenes de test
```

---
