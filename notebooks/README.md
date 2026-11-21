# Notebooks

Análisis y experimentos, numerados según las fases de CRISP-DM. Todos están **ejecutados** (con sus salidas y figuras) para poder leerlos en GitHub sin correr nada.

| Notebook | Fase | Contenido |
|---|---|---|
| [`01_eda.ipynb`](01_eda.ipynb) | 2 · Comprensión de los datos | Distribución de clases, copias aumentadas, ráfagas, resolución, color por calidad y sesgo de fondo |
| [`02_preprocesamiento.ipynb`](02_preprocesamiento.ipynb) | 3 · Preparación | Verificación de la partición sin fuga, recorte, características manuales, tamaño y segmentación de *Mixed* |
| [`03_modelos_ml.ipynb`](03_modelos_ml.ipynb) | 4 · Modelado (ML clásico) | Búsqueda de hiperparámetros, sobreajuste, importancia de características y experimento de solo fondo |
| [`04_cnn.ipynb`](04_cnn.ipynb) | 4 · Modelado (CNN) | Arquitectura, búsqueda de hiperparámetros, curvas de aprendizaje y Grad-CAM |
| [`05_evaluacion.ipynb`](05_evaluacion.ipynb) | 5 · Evaluación | Comparación con intervalos de confianza, McNemar, grados, criterios de éxito, errores y datos externos |

Los notebooks **no reentrenan** los modelos: cargan los resultados de `experiments/` (ver «Cómo reproducir los resultados» en el README principal). Para ejecutarlos de nuevo se necesitan el dataset descargado y los pasos de preparación:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```
