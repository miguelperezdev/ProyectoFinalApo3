# Fase 3 · Preparación de los datos

> Código: `src/data/` y `src/features/`

## 3.1 Pipeline

```
data/raw (19.526 JPG)
  │  manifest.py      → etiquetas, captura original, ráfaga (ventana de 60 s), md5
  ▼
data/interim/manifest.csv
  │  split.py         → 1 imagen por captura + StratifiedGroupKFold
  ▼
data/processed/splits.csv   (train 8.078 · val 1.620 · test 1.614 · mixed 873)
  │  preprocess.py    → EXIF, recorte central cuadrado, 128×128
  ▼
data/processed/images_<split>.npz
  │  handcrafted.py   → 418 características (ML clásico)
  ▼
data/processed/features_full_<split>.npz
```

## 3.2 Decisiones y justificación

**Deduplicación.** Se conserva una imagen por captura: la de nombre más corto, que es la original. Las copias aumentadas no aportan información nueva y causarían fuga entre conjuntos.

**Partición agrupada y estratificada.** Con `StratifiedGroupKFold` se separa 1/7 para prueba y 1/6 del resto para validación (≈ 72/14/14). Ninguna ráfaga se reparte entre conjuntos (se verifica: 0 grupos compartidos), y cada conjunto conserva la proporción de las 12 clases. La carpeta *Mixed* nunca se usa para entrenar.

**Recorte central y redimensionamiento.** Se recorta un cuadrado de lado $s = \min(H, W)$ centrado y se redimensiona a 128×128 con interpolación por área (promedio de los píxeles fuente, que evita el *aliasing* al reducir). El recorte conserva la proporción de la fruta (no la deforma) y descarta parte del fondo.

**Normalización.**
- ML clásico: estandarización z-score por característica, $x'_j = (x_j - \mu_j)/\sigma_j$, con $\mu_j$ y $\sigma_j$ calculados **solo con el conjunto de entrenamiento** dentro del `Pipeline`, de modo que en cada pliegue de validación cruzada no se filtra información de validación. Es indispensable para la SVM-RBF y la regresión logística, porque ambas dependen de distancias o productos punto y, sin escalar, las características de mayor rango dominarían.
- CNN: normalización por canal $x' = (x/255 - \mu_c)/\sigma_c$ con la media y la desviación por canal del conjunto de entrenamiento. Centra los datos y acelera la convergencia del descenso por gradiente.

**Características manuales (418 dimensiones).**

| Bloque | Dim. | Fundamento |
|---|---|---|
| Histogramas H (18), S (8) y V (8) | 34 | HSV separa el tono (madurez, podredumbre) de la intensidad (iluminación). Los histogramas normalizados son invariantes a la posición |
| Momentos de color en HSV y CIELab | 18 | Media $\mu$, desviación $\sigma$ y asimetría $\gamma = E[(X-\mu)^3]/\sigma^3$. En CIELab, la distancia euclidiana se aproxima a la diferencia de color percibida |
| Fracción oscura y marrón | 2 | Indicadores directos de manchas y podredumbre |
| LBP uniforme (P=8, R=1; P=16, R=2) | 28 | $LBP = \sum_{p} s(g_p - g_c)\,2^p$: textura local invariante a cambios monótonos de brillo |
| GLCM (6 propiedades × 2 distancias) | 12 | Matriz de co-ocurrencia de grises: contraste $\sum (i-j)^2 P_{ij}$, homogeneidad $\sum P_{ij}/(1+|i-j|)$, energía, correlación |
| HOG (64×64, celdas de 16, 9 orientaciones) | 324 | Histogramas de gradientes orientados: silueta y bordes (grietas, arrugas) |

**Desbalance.** Pesos de clase $w_k = N/(K\,N_k)$ en la función de pérdida (`class_weight="balanced"`), de modo que cada clase aporta lo mismo a la pérdida total. Se prefirió esto a sobremuestrear, porque con un desbalance de 1,5 : 1 no hace falta duplicar imágenes.

**Aumento de datos (solo CNN, solo entrenamiento).** Recorte aleatorio redimensionado (escala 0,6–1), volteos horizontales y verticales, rotación de ±25° y variación leve de brillo, contraste y saturación. El tono casi no se altera (±0,02) porque el color es la señal de calidad. El aumento obliga a la red a reconocer la fruta en distintas posiciones y escalas, lo que reduce la dependencia del fondo.

**Segmentación de la carpeta *Mixed*.** (1) El color del fondo es el centro del grupo más grande de un k-means (k = 4) en Lab; la primera versión usaba la mediana del borde, pero el borde a veces contenía ventanas, bordes de mesa u otras frutas y se detectaban como frutas. (2) Se umbraliza con Otsu la distancia al fondo, con la luminosidad a peso 0,3 para ignorar sombras. (3) Las frutas que se tocan se separan con *watershed* sobre la transformada de distancia (cada fruta, casi redonda, tiene un máximo en su centro). (4) Se descartan regiones que tocan el borde de la imagen, de área fuera de 0,4–25 %, con solidez < 0,85, aspecto > 3, ocupación de su caja < 0,45 o luminosidad media L* < 40/255 (franjas negras del fondo). Resultado: 873 fotos → 3.640 recortes (la primera versión daba 1.066, muchos con varias frutas o con fondo). En dos inspecciones de 60 recortes al azar, ~85 % contienen una sola fruta completa; los inválidos (pasto, franjas oscuras, primeros planos) se marcan como *Inválido* al anotar con `streamlit run src/app/annotate.py`.

**Privacidad.** `python -m src.data.strip_exif` elimina el GPS y la fecha de las fotos propias.
