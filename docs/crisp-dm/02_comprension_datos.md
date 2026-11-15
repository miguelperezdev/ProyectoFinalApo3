# Fase 2 · Comprensión de los datos

> Evidencia: [`notebooks/01_eda.ipynb`](../../notebooks/01_eda.ipynb) · Código: `src/data/manifest.py`

## 2.1 Fuentes

| Fuente | Contenido | Uso |
|---|---|---|
| *Fruit Quality Classification* (Kaggle, CC0), derivado de **FruitNet** [Meshram & Patil, 2022] | 19.526 imágenes · 6 frutas (manzana, banano, guayaba, limón, naranja, granada) · calidad *Good*, *Bad* y *Mixed* | Entrenamiento, validación y prueba |
| Carpeta *Mixed quality* | 873 fotos con varias frutas de calidades mezcladas | Segmentadas en 3.640 recortes individuales para la evaluación externa y la anotación manual |
| Fotos propias del grupo (`data/own/`) | 30 a 50 fotos tomadas en casa o en el mercado, una fruta sobre fondo uniforme | Evaluación de la generalización en condiciones reales de despliegue |

## 2.2 Hallazgos

| Aspecto | Hallazgo | Implicación |
|---|---|---|
| **Copias aumentadas** | El 39 % de las imágenes etiquetadas son copias de otra foto (mismo timestamp con sufijo). En *granada buena*, 5.940 archivos corresponden a solo 1.154 fotos originales (81 % de copias) | Inflan una clase y causan **fuga de información** si una copia cae en entrenamiento y otra en prueba |
| **Desbalance** | 5,5 : 1 con los archivos originales y 1,5 : 1 tras deduplicar | Leve: se corrige con pesos de clase |
| **Ráfagas** | La misma fruta fotografiada desde 4 o 5 ángulos con segundos de diferencia | Las imágenes no son independientes: hay que partir por grupos |
| **Resolución** | Principalmente 256×256 y 256×192 (horizontal y vertical); unas pocas de 3000×4000 o más | Se homogeneiza con recorte central y 128×128 |
| **Variabilidad de escena** | Una o varias frutas por foto, manos y guantes, fondos de pasto, baldosa, tela o pared | El dataset no cumple el supuesto "una fruta sobre fondo uniforme" del enunciado |
| **Señal de calidad** | Las frutas malas tienen más píxeles marrones en banano, granada, guayaba y limón, pero no en manzana ni naranja. El brillo de las malas es *mayor* en varias frutas, por efecto del fondo | Ningún indicador de color separa las clases por sí solo: se combinan color, textura y forma |
| **Sesgo de fondo** | En 5 de 6 frutas, las fotos de frutas malas se tomaron sobre fondos más claros. Un Random Forest que **solo ve la franja de 16 px del borde** acierta la calidad en el **85,4 %** de validación (el azar sería 50 %) | Las métricas en la prueba de Kaggle son una cota **optimista**; se necesita evaluación externa y análisis de dónde mira el modelo (Grad-CAM) |

## 2.3 Figuras

- `reports/figures/eda_distribucion_clases.svg`: archivos frente a fotos originales por clase.
- `reports/figures/eda_muestras.svg`: ejemplos por clase.
- `reports/figures/eda_imagen_promedio.svg`: imagen promedio por clase (muestra las diferencias de fondo).
- `reports/figures/eda_color_calidad.svg`: fracción marrón y brillo por fruta y calidad.
- `reports/figures/eda_sesgo_fondo.svg`: luminosidad del borde por clase.
