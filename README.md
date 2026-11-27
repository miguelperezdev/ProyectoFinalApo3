# Clasificación automática de la calidad de frutas con visión por computador

Proyecto final del curso **Algoritmos y Programación III** (semestre 2026-2), Ingeniería de Sistemas, Facultad de Ingeniería, Diseño y Ciencias Aplicadas, **Universidad Icesi**, Cali, Colombia.

**Estado del proyecto:** terminado.

- **Informe (IEEE):** [`reports/informe_final.pdf`](reports/informe_final.pdf)
- **Video de presentación:** https://canva.link/gkvnycpm3mnc2wd
- **Repositorio:** https://github.com/miguelperezdev/ProyectoFinalApo3

## Integrantes

| Nombre | Código |
|---|---|
| Miguel Pérez Ojeda | A00407054 |
| Danna Isabel Zambrano Sánchez | A00404131 |
| Mateo Berrío Villa | A00399867 |
| Sebastián Romero León | A00404670 |

**Profesor:** Milton Sarria Paja

## Objetivo

En plazas de mercado y en la agroindustria, las frutas se clasifican a mano según su tamaño, madurez y defectos. Ese proceso es lento y subjetivo, genera pérdidas económicas y desperdicio de alimentos, y no sigue un estándar común. Este proyecto desarrolla un **sistema automático de clasificación de calidad** que, a partir de la foto de una fruta individual sobre un fondo uniforme:

1. reconoce la **fruta** (manzana, banano, guayaba, limón, naranja o granada),
2. asigna un **grado de calidad** (Primera, Segunda o Descarte), y
3. estima su **tamaño relativo** (pequeño, mediano o grande).

El sistema se entrega con una interfaz gráfica que permite cargar una imagen o capturarla con la cámara, y con un modo en tiempo real que simula una línea de empaque.

## Resultados

Conjunto de prueba (1.614 imágenes, usado una sola vez), con partición por ráfagas y sin copias aumentadas. El F1-macro es de las 12 clases fruta × calidad, con intervalo de confianza *bootstrap* del 95 %.

| Modelo | F1-macro | Exactitud de calidad | Exactitud de fruta | Brecha train−test |
|---|---|---|---|---|
| Regresión logística | 0,847 [0,828; 0,862] | 0,942 | 0,872 | 0,139 |
| SVM-RBF | 0,891 [0,876; 0,905] | 0,952 | 0,924 | 0,109 |
| Random Forest | 0,905 [0,890; 0,919] | 0,953 | 0,919 | 0,095 |
| XGBoost | **0,912** [0,898; 0,925] | **0,965** | **0,936** | 0,088 |
| CNN (modelo final) | 0,909 [0,895; 0,922] | 0,958 | 0,932 | **0,037** |

- **XGBoost y la CNN están empatados** (McNemar: p = 0,22 en calidad y p = 0,63 en la tarea conjunta). Se eligió la CNN porque sobreajusta menos, generaliza mejor y responde en 6,2 ms por imagen en CPU, frente a 36,6 ms de XGBoost.
- **Grados con la CNN:** 95,8 % de precisión en Primera, 99,5 % en Descarte y 5,9 % de frutas enviadas a revisión (Segunda).
- **Limitación principal:** en las fotos de *Mixed*, que vienen de sesiones distintas, la exactitud de fruta baja a 59 %. Además, un modelo que solo mira el borde de la foto acierta la calidad el 90 % de las veces, así que el fondo delata la etiqueta. Las cifras de prueba valen dentro de las condiciones del dataset.
- **Fotos propias:** con 48 manzanas propias, la CNN acierta la fruta en el 83 % y el grado en el 38 %. Si se reentrena sumándolas, el grado sube al 92 % sin bajar el F1 de prueba (0,908), pero 122 fotos propias de frutas desconocidas muestran que el modelo ajustado aprende el contexto de la casa: el 83 % de ellas pasa por manzana. Por eso la app mantiene la CNN original.

El análisis completo está en [`docs/crisp-dm/05_evaluacion.md`](docs/crisp-dm/05_evaluacion.md) y en el [informe](reports/informe_final.pdf).

## Metodología: CRISP-DM

Cada fase se documenta en [`docs/crisp-dm/`](docs/crisp-dm/):

| Fase | Documento | Código / evidencia |
|---|---|---|
| 1. Comprensión del negocio | [01_comprension_negocio.md](docs/crisp-dm/01_comprension_negocio.md) | — |
| 2. Comprensión de los datos | [02_comprension_datos.md](docs/crisp-dm/02_comprension_datos.md) | `notebooks/01_eda.ipynb` |
| 3. Preparación de los datos | [03_preparacion_datos.md](docs/crisp-dm/03_preparacion_datos.md) | `src/data/`, `src/features/` |
| 4. Modelado | [04_modelado.md](docs/crisp-dm/04_modelado.md) | `src/models/`, `src/training/` |
| 5. Evaluación | [05_evaluacion.md](docs/crisp-dm/05_evaluacion.md) | `src/evaluation/`, `experiments/results/` |
| 6. Despliegue | [06_despliegue.md](docs/crisp-dm/06_despliegue.md) | `src/app/` |

### Modelos

- **Machine learning clásico:** regresión logística (línea base), SVM-RBF, Random Forest y XGBoost, sobre 418 características de color, textura y forma. Hiperparámetros con `GridSearchCV` y validación cruzada de 5 pliegues agrupada por ráfagas.
- **Deep learning:** CNN entrenada desde cero en CPU, con 3 bloques convolucionales, *global average pooling* y una capa densa (113.068 parámetros). La tasa de aprendizaje y el *dropout* se eligieron con una rejilla 2×2 en validación.
- **Grados:** todos los modelos estiman P(fruta, calidad). De ahí se obtiene P(buena), y una regla de rechazo envía a Segunda los casos dudosos. Los umbrales se ajustan en validación.

## Datos

| Fuente | Descripción | Ubicación |
|---|---|---|
| [Fruit Quality Classification (Kaggle)](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification) | 19.526 imágenes de 6 frutas (manzana, banano, guayaba, limón, naranja y granada) en calidades *Good*, *Bad* y *Mixed*. Licencia CC0. Proviene de FruitNet [1]. | `data/raw/` (no se versiona) |
| Fotos propias del grupo | 48 manzanas anotadas, 122 fotos de frutas que el modelo no conoce y 22 manzanas excluidas, todas tomadas por el equipo. | [`data/own/`](data/own/) |

## Estructura del repositorio

```
├── data/
│   ├── raw/            # Dataset de Kaggle (se descarga, no se sube a git)
│   ├── own/            # Fotos propias: images/ + labels.csv, otras_frutas/, excluidas/
│   ├── interim/        # Datos intermedios (recortes, segmentaciones)
│   └── processed/      # Datos listos para entrenar
├── docs/
│   ├── enunciado/      # Enunciado, competencias, rúbrica y plantilla IEEE
│   └── crisp-dm/       # Documentación por fase de CRISP-DM
├── notebooks/          # Exploración y experimentos (numerados por fase)
├── src/
│   ├── main.py         # Ejecuta todas las etapas del proyecto en orden
│   ├── data/           # Descarga, carga y partición de datos
│   ├── features/       # Extracción de características (color, textura, tamaño)
│   ├── models/         # Definición de modelos (ML clásico y CNN)
│   ├── training/       # Entrenamiento y ajuste de hiperparámetros
│   ├── evaluation/     # Métricas, matrices de confusión, análisis de errores
│   ├── app/            # Interfaz (Streamlit), anotación y modo en tiempo real
│   └── utils/          # Funciones auxiliares
├── experiments/        # Resultados, logs y modelos entrenados
├── reports/            # Informe final (IEEE) y figuras
├── tests/              # Pruebas unitarias
├── requirements.txt
└── LICENSE
```

## Cómo empezar

```bash
# 1. Clonar el repositorio
git clone https://github.com/miguelperezdev/ProyectoFinalApo3.git
cd ProyectoFinalApo3

# 2. Crear el entorno virtual e instalar dependencias (Python 3.11 o superior)
python3 -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Descargar el dataset (~3 GB, no necesita token de Kaggle)
python -m src.data.download
```

## Cómo reproducir los resultados

Con todo el proyecto en un solo comando, o por etapas:

```bash
python -m src.main                 # todas las etapas, de la descarga a la evaluación
python -m src.main datos           # solo preparación
python -m src.main ml cnn          # solo entrenamiento
python -m src.main evaluacion      # solo evaluación
python -m src.main --listar        # muestra los comandos de cada etapa
python -m src.main desconocidas    # prueba con frutas que el modelo no conoce
python -m src.main ajuste          # ajuste fino con las fotos propias
```

Tiempos aproximados en un portátil sin GPU (CPU de 16 hilos):

| Etapa | Qué hace | Salida | Tiempo |
|---|---|---|---|
| `descarga` | Descarga el dataset | `data/raw/` | según la conexión |
| `datos` | Índice, partición, preprocesamiento, características y recortes de *Mixed* | `data/interim/`, `data/processed/` | 3 min |
| `ml` | Modelos clásicos y control de solo fondo | `experiments/results/metrics_*.json` | 30 min |
| `cnn` | CNN final | `experiments/checkpoints/cnn.pt` | 40 min |
| `evaluacion` | Comparación, grados, Grad-CAM y datos externos | `experiments/results/`, `reports/figures/` | 2 min |
| `desconocidas` | Prueba de fruta desconocida con `data/own/otras_frutas/` (no corre por defecto) | `experiments/results/otras_frutas_cnn.json` | 1 min |
| `ajuste` | Ajuste fino de la CNN con las fotos propias (no corre por defecto) | `experiments/results/finetune_own.json`, `cnn_own.pt` | 20 min |

Las pruebas unitarias se corren con `pytest -q`, y el informe se regenera con `python reports/build_informe.py` (ver [`reports/README.md`](reports/README.md)).

### Aplicación

```bash
streamlit run src/app/app.py          # clasificador: cargar imagen o usar la cámara
streamlit run src/app/annotate.py     # anotación manual de los recortes de Mixed

python -m src.app.realtime --fuente 0                            # tiempo real con la cámara
python -m src.app.realtime --fuente data/interim/mixed_crops \
    --max-imagenes 25 --pausa 0.8 --salida linea.mp4              # línea de empaque simulada
```

El modo con cámara abre una ventana, así que necesita `opencv-python`, que es el que trae `requirements.txt`. Con `opencv-python-headless` solo funciona la salida a video.

El modelo CNN final (`experiments/checkpoints/cnn.pt`, ~450 KB) se incluye en el repositorio, así que la aplicación funciona sin reentrenar.

### Fotos propias

1. Tomar las fotos siguiendo [`data/own/README.md`](data/own/README.md) y guardarlas en `data/own/images/`.
2. Borrar los metadatos (GPS y fecha): `python -m src.data.strip_exif`.
3. Anotar cada foto en `data/own/labels.csv`.
4. Volver a correr `python -m src.evaluation.evaluate`.

El set final tiene **48 manzanas** tomadas en casa y anotadas en `data/own/labels.csv` (46 Primera, 2 Segunda; 16 pequeñas, 16 medianas y 16 grandes). La CNN acierta la fruta en el **83 %**, pero el grado solo coincide en el **38 %** y el tamaño en el **33 %**; el detalle está en `docs/crisp-dm/05_evaluacion.md` (§5.6) y en `experiments/results/own_photos_pred_cnn.csv`. Además, el equipo tomó **122 fotos de frutas que el modelo no conoce** (fresa, lulo, tomate y tomate de árbol) en `data/own/otras_frutas/`, y las 22 manzanas que no cumplieron el protocolo quedaron en `data/own/excluidas/`.

Con estas fotos se hicieron dos pruebas (detalle en `docs/crisp-dm/05_evaluacion.md` §5.6):

```bash
python -m src.main desconocidas    # ¿detecta el modelo una fruta que no conoce? (~1 min)
python -m src.main ajuste          # ajuste fino de la CNN con Kaggle + fotos propias (~20 min)
```

| CNN | Fruta (propias) | Grado (propias) | F1 prueba Kaggle | Frutas desconocidas rechazadas | Frutas desconocidas tomadas por manzana |
|---|---|---|---|---|---|
| Original (la de la app) | 83 % | 38 % | 0,909 | 21 % | 57 % |
| Ajustada con fotos propias | 100 % | 92 % | 0,908 | 9 % | 83 % |

## Entregables

- [x] Código fuente documentado (`src/`)
- [x] Notebooks de exploración y experimentos (`notebooks/`)
- [x] Diagrama de flujo personalizado de CRISP-DM (`reports/figures/diagrama_crisp_dm.svg`)
- [x] Interfaz gráfica funcionando (`src/app/`)
- [x] Informe final en formato IEEE, máximo 7 páginas (`reports/informe_final.pdf`)
- [x] Fotos propias del grupo, 30 a 50 (`data/own/`)
- [x] Video de presentación de máximo 10 minutos ([enlace](https://canva.link/gkvnycpm3mnc2wd))

## Referencias

[1] V. Meshram and K. Patil, "FruitNet: Indian fruits image dataset with quality for machine learning applications," *Data in Brief*, vol. 40, p. 107686, 2022, doi: 10.1016/j.dib.2021.107686.

[2] R. Park, "Fruit Quality Classification," Kaggle, 2022. [Online]. Available: https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification

## Licencia

Este proyecto se distribuye bajo la licencia GNU GPL v2. Ver [LICENSE](LICENSE).
