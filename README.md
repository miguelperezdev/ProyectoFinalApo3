# Clasificación automática de la calidad de frutas con visión por computador

Proyecto final del curso **Algoritmos y Programación III** (semestre 2026-2), Ingeniería de Sistemas, Facultad de Ingeniería, Diseño y Ciencias Aplicadas, **Universidad Icesi**, Cali, Colombia.

**Estado del proyecto:** 🟡 En desarrollo

## Integrantes

| Nombre | Código |
|---|---|
| Miguel Pérez Ojeda | A00407054 |
| Danna Isabel Zambrano Sánchez | A00404131 |
| Mateo Berrío Villa | A00399867 |

**Profesor:** Milton Sarria Paja

## Objetivo

En plazas de mercado y en la agroindustria, las frutas se clasifican a mano según su tamaño, madurez y defectos. Ese proceso es lento y subjetivo, genera pérdidas económicas y desperdicio de alimentos, y no sigue un estándar común. Este proyecto desarrolla un **sistema automático de clasificación de calidad** que, a partir de la foto de una fruta individual sobre un fondo uniforme:

1. asigna una **clase de calidad** (3 categorías), y
2. estima su **tamaño relativo** (pequeño, mediano o grande).

El sistema se entrega con una interfaz gráfica que permite cargar una imagen o capturarla con la cámara.

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

- **Machine learning clásico** (al menos dos, con búsqueda de hiperparámetros por validación cruzada): SVM, Random Forest y XGBoost, sobre características de color, textura y forma.
- **Deep learning:** CNN pequeña entrenada desde cero (2 o 3 bloques convolucionales + pooling + capas densas).

## Datos

| Fuente | Descripción | Ubicación |
|---|---|---|
| [Fruit Quality Classification (Kaggle)](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification) | 19.526 imágenes de 6 frutas (manzana, banano, guayaba, limón, naranja y granada) en calidades *Good*, *Bad* y *Mixed*. Licencia CC0. Proviene de FruitNet [1]. | `data/raw/` (no se versiona) |
| Fotos propias del grupo | 30 a 50 fotos tomadas y anotadas por el equipo. | [`data/own/`](data/own/) |

## Estructura del repositorio

```
├── data/
│   ├── raw/            # Dataset de Kaggle (se descarga, no se sube a git)
│   ├── own/            # Fotos propias + labels.csv
│   ├── interim/        # Datos intermedios (recortes, segmentaciones)
│   └── processed/      # Datos listos para entrenar
├── docs/
│   ├── enunciado/      # Enunciado, competencias, rúbrica y plantilla IEEE
│   └── crisp-dm/       # Documentación por fase de CRISP-DM
├── notebooks/          # Exploración y experimentos (numerados por fase)
├── src/
│   ├── data/           # Descarga, carga y partición de datos
│   ├── features/       # Extracción de características (color, textura, tamaño)
│   ├── models/         # Definición de modelos (ML clásico y CNN)
│   ├── training/       # Entrenamiento y ajuste de hiperparámetros
│   ├── evaluation/     # Métricas, matrices de confusión, análisis de errores
│   ├── app/            # Interfaz gráfica (Streamlit)
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
git clone <url-del-repo>
cd ProyectoFinalApo3

# 2. Crear el entorno virtual e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Descargar el dataset (~3 GB, no necesita token de Kaggle)
python -m src.data.download
```

## Entregables

- [ ] Código fuente documentado (`src/`)
- [ ] Notebooks de exploración y experimentos (`notebooks/`)
- [ ] Diagrama de flujo personalizado de CRISP-DM
- [ ] Interfaz gráfica funcionando (`src/app/`)
- [ ] Informe final en formato IEEE, máximo 7 páginas (`reports/`)
- [ ] Video de presentación de máximo 10 minutos (_[enlace]_)

## Referencias

[1] V. Meshram and K. Patil, "FruitNet: Indian fruits dataset with quality for machine learning applications," *Data in Brief*, vol. 40, p. 107686, 2022, doi: 10.1016/j.dib.2021.107686.

[2] R. Park, "Fruit Quality Classification," Kaggle, 2022. [Online]. Available: https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification

## Licencia

Este proyecto se distribuye bajo la licencia GNU GPL v2. Ver [LICENSE](LICENSE).
