# Fotos propias del grupo

El enunciado pide de **30 a 50 imágenes propias** de frutas reales, tomadas en plazas de mercado, supermercados o en casa. Sirven para medir algo que el dataset de Kaggle no puede medir: **cómo funciona el modelo con fotos nuevas**, otros fondos, otra luz y frutas colombianas.

## Cómo tomar las fotos

- **Una sola fruta por foto**, centrada, sobre un **fondo liso y uniforme** (cartulina blanca, gris o una mesa lisa).
- Cámara del celular **a unos 30 cm** de la fruta, mirando desde arriba o de frente. La distancia fija es necesaria para que el tamaño relativo sea comparable entre fotos.
- Luz natural o lo más pareja posible, sin sombras fuertes ni flash directo.
- Solo las 6 frutas que conoce el modelo: **manzana, banano, guayaba, limón, naranja y granada**.
- Incluir variedad dentro de cada fruta: sanas, con golpes o manchas leves y con podredumbre; pequeñas, medianas y grandes.
- Sugerencia de reparto (≈ 42 fotos): por cada fruta, 3 de Primera, 2 de Segunda y 2 de Descarte.
- **Privacidad:** no deben aparecer rostros, personas, manos con anillos o tatuajes visibles, ni nombres o logos de negocios. En un mercado, pedir permiso al vendedor y explicar que es para un trabajo de la universidad.

## Organización

1. Guardar las imágenes **planas** en `data/own/images/` con nombres como `manzana_001.jpg`. No usar subcarpetas: `python -m src.data.strip_exif` y `src/evaluation/evaluate.py` solo leen el primer nivel.
2. Borrar los metadatos (ubicación GPS, fecha, modelo del celular): `python -m src.data.strip_exif`.
3. Anotar cada foto en `labels.csv`, **antes** de ver la predicción del modelo (para no sesgar la etiqueta). Ninguna fila puede quedar con campos vacíos: la evaluación no valida los vacíos y los contaría como errores.

> **Frutas desconocidas.** El modelo solo reconoce manzana, banano, guayaba, limón, naranja y granada. Las fotos de otras frutas (fresa, lulo, tomate, tomate de árbol) no van en `labels.csv` ni en `images/`: van en `otras_frutas/` y se registran en `otras_frutas.csv`. Sirven para la prueba de fruta desconocida.

| columna | valores |
|---|---|
| `archivo` | nombre del archivo, p. ej. `manzana_001.jpg` |
| `fruta` | `apple`, `banana`, `guava`, `lime`, `orange`, `pomegranate` |
| `calidad` | `Primera`, `Segunda`, `Descarte` (definiciones abajo) |
| `tamano` | `pequeno`, `mediano`, `grande` (comparado con otras frutas de la misma especie) |
| `lugar` | `casa`, `mercado`, `supermercado` |
| `autor` | `miguel`, `danna`, `mateo`, `sebastian` |

Ejemplo:

```csv
archivo,fruta,calidad,tamano,lugar,autor
manzana_001.jpg,apple,Primera,mediano,casa,miguel
banano_003.jpg,banana,Descarte,grande,mercado,danna
```

### Set actual de manzanas

El set final tiene **48 fotos de manzana** en `data/own/images/` (`manzana_001.jpeg` … `manzana_059.jpeg`, con huecos) tomadas en casa y anotadas en `labels.csv`: 46 Primera y 2 Segunda (no hay Descarte), 16 pequeñas, 16 medianas y 16 grandes.

De las 70 fotos iniciales se descartaron 22 que no cumplían el protocolo: 6 con varias frutas por foto, 11 con sticker de producto y 5 en las que la mano ocupaba buena parte de la imagen.

Resultado de la evaluación con la CNN: fruta 83 % (40/48), grado 38 % (18/48) y tamaño 33 % (16/48). El detalle está en `experiments/results/own_photos_pred_cnn.csv` y en `docs/crisp-dm/05_evaluacion.md` §5.6.

## Contenido de la carpeta

| Ruta | Qué es | Fotos | Uso |
|---|---|---|---|
| `images/` + `labels.csv` | Manzanas que cumplen el protocolo, anotadas | 48 | Evaluación (y ajuste fino, ver abajo) |
| `otras_frutas/` + `otras_frutas.csv` | Fresa (52), lulo (14), tomate (9) y tomate de árbol (47) | 122 | Prueba de fruta desconocida |

Todas las fotos son del equipo y están sin metadatos EXIF (`python -m src.data.strip_exif` revisa las dos carpetas). Los 40 duplicados exactos que había en la colección original se borraron tras comprobar con MD5 que eran copias idénticas; `otras_frutas.csv` guarda el nombre original de cada foto.

## Pruebas hechas con estas fotos

| Prueba | Comando | Tiempo | Salida |
|---|---|---|---|
| Fruta desconocida: ¿el modelo puede decir «no sé qué fruta es»? | `python -m src.main desconocidas` | ~1 min | `experiments/results/otras_frutas_cnn.json`, `reports/figures/res_otras_frutas.svg` |
| Ajuste fino: entrenar con Kaggle **y** las fotos propias | `python -m src.main ajuste` | ~20 min | `experiments/results/finetune_own.json`, `experiments/checkpoints/cnn_own.pt` |

El ajuste fino usa validación cruzada de 4 pliegues sobre las 48 manzanas: cada foto se evalúa con un modelo que no la vio, y cada modelo se vuelve a medir en el conjunto de prueba de Kaggle para comprobar que no empeora. No sobrescribe la CNN final (`cnn.pt`). El detalle está en `src/training/finetune_own.py`.

**Resultados.** El ajuste fino sube el acierto en las manzanas propias (fruta 83 % → 100 %, grado 38 % → 92 %) sin bajar el F1 de prueba (0,909 → 0,908). Pero la prueba de fruta desconocida muestra el costo: después del ajuste, el 83 % de las fresas, lulos y tomates pasa por manzana (antes, el 57 %), porque todas las fotos de entrenamiento eran manzanas sanas de la misma casa. Por eso la app sigue usando la CNN original. Para que el ajuste sirva, las próximas fotos deben incluir las seis frutas, los tres grados y varios lugares. Análisis completo en `docs/crisp-dm/05_evaluacion.md` §5.6.

## Definición de los grados

| Grado | Cuándo usarlo |
|---|---|
| **Primera** | Sana, sin defectos visibles o con marcas mínimas en la cáscara. Se vendería a precio pleno. |
| **Segunda** | Defectos leves: pequeños golpes, manchas superficiales, maduración irregular. Se vendería más barata o para jugos. |
| **Descarte** | Podredumbre, moho, partes blandas o hundidas, cáscara rota. No se vendería. |

Si dos integrantes no están de acuerdo con el grado de una foto, se discute y se anota el consenso.

## Evaluación

Con las fotos y `labels.csv` listos, `python -m src.evaluation.evaluate` compara la predicción con la anotación (fruta, grado y tamaño) y guarda el detalle en `experiments/results/own_photos_pred_cnn.csv`. Con las fotos anotadas también se pueden recalibrar los umbrales de tamaño de cada fruta (`calibrate_thresholds` en `src/features/size.py`, que usa los terciles de d_norm).
