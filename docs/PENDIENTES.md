# Pendientes del proyecto

> Estado verificado el **14 de noviembre de 2026** sobre `main`. Este documento resume lo que ya quedó resuelto y lo que todavía depende del equipo. Los commits van en español, a nombre de Miguel (`Miguel <perezojedamiguel99@gmail.com>`) y **sin línea de co-autor**; el agente no hace `push`.

## Resumen

| Punto | Estado | Quién |
|---|---|---|
| 1. CNN reentrenada | **Resuelto: se volvió a la versión anterior (opción A).** | Agente |
| 2. Fotos propias | **Resuelto: 48 manzanas anotadas, evaluadas y documentadas.** | Agente/Equipo |
| 3. Anotación de *Mixed* | Opcional; no se hizo. | Equipo |
| 4. Video | Guion listo; falta grabar y pegar el enlace. | Equipo |
| 5. Git | Commits locales; **no se hizo push**. | Equipo |
| 6. Nuevo integrante | **Resuelto: Sebastián Romero León (A00404670) añadido al README, al informe (autores en dos filas, 6 páginas) y al guion del video (bloques 4 y 8).** | Agente |
| 7. Pruebas con las fotos propias | **Resuelto: ejecutadas y documentadas (14-nov).** Ver §7. | Agente |

---

## 1. CNN reentrenada (resuelto)

Se eligió la **opción A**: conservar la versión que ya estaba documentada en el informe. Se revirtieron los 37 archivos modificados de `experiments/` y `reports/figures/` a `HEAD`.

Además se regeneraron `experiments/results/proba_cnn_{val,test,mixed}.npy` desde el checkpoint antiguo (`experiments/checkpoints/cnn.pt`). La verificación dio **delta 0,00** frente a `metrics_cnn.json` en validación y prueba:

| Métrica | Valor restaurado |
|---|---|
| F1-macro test | 0,9094 |
| Exactitud de calidad test | 0,9579 |
| Exactitud de fruta test | 0,9318 |
| Mejor época | 34 |
| Brecha train−test | 3,7 puntos |

La copia de la CNN nueva (checkpoint, probabilidades y parche de los 37 archivos) está en `experiments/checkpoints/respaldo_cnn_nueva/` (local, git la ignora), por si el equipo decidiera reconsiderarlo. Antes estaba en `/tmp`, que se borra al reiniciar.

> **Si algún día se elige la opción B**, no basta con cambiar el F1: también cambian la exactitud por fruta (manzana 0,957 → 0,910; granada 0,947 → 0,989; naranja 0,891 → 0,861), la cobertura y precisión de grados, Grad-CAM, los recortes de *Mixed* y McNemar (p = 0,22 → 0,054). Habría que actualizar `reports/informe.md`, `docs/crisp-dm/04_modelado.md`, `docs/crisp-dm/05_evaluacion.md`, `README.md`, `docs/guion_video.md` y los notebooks 04 y 05, y regenerar el PDF y el DOCX.

---

## 2. Fotos propias (resuelto)

### Estado final

- **48 fotos de manzana** en `data/own/images/`, planas y sin EXIF. El equipo confirmó que son propias.
- `data/own/labels.csv` tiene **48 filas y cero celdas vacías**: 46 Primera, 2 Segunda y 0 Descarte; 16 pequeñas, 16 medianas y 16 grandes; todas de `casa`, autor `miguel`.
- Se excluyeron **22 fotos** que no cumplían el protocolo; están en `data/own/excluidas/` (versionadas, sin EXIF): 6 con varias frutas por foto (026–031), 11 con sticker de producto (060–061 y 063–070) y 5 con mano dominante (006, 007, 017, 024 y 037).
- `data/own/renombrado_manzanas.csv` conserva la correspondencia con los nombres originales y ahora incluye `estado` (`usada`/`excluida`) y `motivo`.
- Se eliminó `data/own/labels_manzanas_pendiente.csv`: era la plantilla de 70 filas y ya no aplica.
- Las **122 fotos de frutas que el modelo no conoce** están en `data/own/otras_frutas/` (fresa 52, lulo 14, tomate 9 y tomate de árbol 47), renombradas como `fresa_001.jpeg`… y registradas en `data/own/otras_frutas.csv` con su nombre original. Los 40 duplicados exactos se borraron tras comprobar por MD5 que eran copias idénticas. La carpeta `data/own_ood/` ya no existe.

> El conteo anterior de “233 fotos” incluía `.gitkeep` y contaba duplicados. En realidad eran 232 imágenes, de las cuales solo 192 son únicas.

### Evaluación

`python -m src.evaluation.evaluate` generó `experiments/results/own_photos_pred_cnn.csv` y añadió `own_photos_cnn` a `experiments/results/resumen_evaluacion.json`:

| Métrica | Resultado |
|---|---|
| Fruta | **83 %** (40/48) |
| Grado | **38 %** (18/48) |
| Tamaño | **33 %** (16/48) |

El detalle está en `docs/crisp-dm/05_evaluacion.md` §5.6. Las cifras ya se escribieron en `reports/informe.md`, `docs/guion_video.md` (bloque 6) y `README.md`, y el PDF y el DOCX se regeneraron con `python reports/build_informe.py`.

### Por qué el grado y el tamaño coinciden poco

- **Grado.** La probabilidad media de «buena» en las fotos propias es 0,28 (mediana 0,17), muy por debajo de la del conjunto de prueba. Los umbrales de la CNN (0,030 y 0,301) se calibraron en el dataset, así que el 54 % de las manzanas sanas cae en Segunda. No es un error de etiqueta: es el mismo cambio de dominio que se ve en *Mixed*.
- **Tamaño.** La anotación es relativa a las 48 manzanas (terciles), pero la app usa umbrales absolutos por fruta (`d_norm` 0,40 y 0,55 para manzana). Las fotos propias se tomaron más cerca, así que 45 de 48 se predicen como grandes. Al recalibrar con `calibrate_thresholds` sobre estas fotos, los umbrales para manzana quedan en (0,604, 0,652).
- **Cobertura del set.** Solo hay manzanas y ninguna en estado de Descarte, así que esa clase no se puede evaluar con fotos propias. Ampliar el set con banano, guayaba, limón, naranja y granada —incluyendo frutas podridas— es la mejora más útil antes de un despliegue real.

### Si se amplía el set

1. Guardar las fotos planas en `data/own/images/` con nombres como `banano_001.jpeg`.
2. Borrar metadatos: `python -m src.data.strip_exif` (solo recorre el primer nivel).
3. Añadir una fila por foto a `data/own/labels.csv` sin dejar campos vacíos.
4. Volver a correr `python -m src.evaluation.evaluate`.
5. Actualizar `docs/crisp-dm/05_evaluacion.md`, `reports/informe.md`, `docs/guion_video.md` y `README.md` si cambian las cifras.


---

## 3. Anotación de recortes de *Mixed* (opcional)

No es obligatoria, pero estaba en el plan. Hay **3.640 recortes** en `data/interim/mixed_crops/` y ninguno tiene grado anotado.

- La herramienta es `src/app/annotate.py` y guarda en **`data/own/mixed_annotations.csv`**, no en la columna `grade` de `data/interim/mixed_crops/annotation.csv`. Ese `annotation.csv` es intermedio, lo genera `src/data/segment_mixed.py` y no se versiona; no hay que llenarlo a mano.
- Para anotar:

  ```bash
  streamlit run src/app/annotate.py
  ```

- Al terminar, volver a correr `python -m src.evaluation.evaluate` y actualizar `docs/crisp-dm/05_evaluacion.md` (§5.6) y el informe si cambian las cifras de *Mixed*. Es trabajo manual de una persona; no lo puede hacer el agente.

---

## 4. Video

- El guion está en `docs/guion_video.md` y dura 9:30 (límite 10:00).
- El bloque 6 ya incluye las cifras de las fotos propias y el bloque 7 usa una manzana sana, una con mancha y una manzana mala del dataset para mostrar los tres grados.
- Grabar, subir el video y poner el enlace en `README.md`, línea del entregable:

  ```markdown
  - [x] Video de presentación de máximo 10 minutos ([enlace](https://...))
  ```

---

## 5. Git

- `main` tiene commits locales por delante de `origin/main`. El equipo los sube de a poco; **el agente no debe hacer `push`**.
- Los commits van en español, a nombre del usuario y **sin línea de co-autor**.
- Para revisar antes de subir:

  ```bash
  git status
  git log --oneline -5
  git diff --stat
  ```

---

---

## 7. Pruebas con las fotos propias (resuelto)

Se ejecutaron el 14 de noviembre de 2026. Ninguna cambia el modelo de la app.

```bash
python -m src.main ajuste          # 21 min; genera cnn_own.pt (local, git lo ignora)
python -m src.main desconocidas    # 1-2 min; evalúa cnn y, si existe, cnn_own
```

Hay que correrlas en ese orden para que la figura compare los dos modelos.

| CNN | Fruta (propias) | Grado (propias) | F1 prueba | Calidad prueba | Desconocidas rechazadas | Desconocidas → manzana |
|---|---|---|---|---|---|---|
| Original (`cnn.pt`) | 83 % | 38 % | 0,909 | 0,958 | 21 % | 57 % |
| Ajustada (validación cruzada, 4 pliegues) | 100 % | 92 % | 0,908 | 0,951 | 9 % (`cnn_own`) | 83 % (`cnn_own`) |

- **Decisión: la app y el informe siguen con la CNN original.** La mejora en las manzanas propias es en parte un atajo: como todas las fotos de entrenamiento eran manzanas sanas de la misma casa, la red aprendió «ese contexto → manzana buena», y ahora fresas y tomates de árbol pasan por manzana.
- Con la CNN original, la confianza no separa frutas desconocidas de manzanas propias (las dos se rechazan en el 21 %).
- Cifras en `experiments/results/finetune_own.json`, `finetune_own_pred.csv`, `otras_frutas_cnn.json` y `otras_frutas_cnn_own.json`; figura `reports/figures/res_otras_frutas.svg`.
- Ya se actualizaron `docs/crisp-dm/05_evaluacion.md` §5.6 y §5.8, `reports/informe.md` (párrafo, Fig. 6, resumen y conclusiones; el PDF sigue en 6 páginas), `README.md`, `data/own/README.md` y el bloque 6 del guion.
- **Para mejorar de verdad:** tomar fotos de las seis frutas, en los tres grados (sobre todo Descarte) y en varios lugares, y volver a correr las dos pruebas.

---

## Comandos de verificación rápida

```bash
# El informe no debe pasar de 7 páginas y el PDF debe abrir
pdfinfo reports/informe_final.pdf | grep Pages

# Las métricas restauradas deben coincidir con el informe
python - <<'PY'
import json
m = json.load(open("experiments/results/metrics_cnn.json"))
print("F1 test:", round(m["test"]["joint_f1_macro"], 3))
print("mejor época:", m["best_epoch"])
PY

# Fotos propias: 48 manzanas, 122 otras frutas y 22 excluidas
for d in images otras_frutas excluidas; do echo "$d: $(find data/own/$d -type f -name '*.jpeg' | wc -l)"; done

# labels.csv debe tener 48 filas y ninguna celda vacía
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/own/labels.csv")
print("filas:", len(df), "nulos:", int(df.isna().sum().sum()))
print(df["calidad"].value_counts().to_dict())
PY

# Resultados de las fotos propias
python - <<'PY'
import json
r = json.load(open("experiments/results/resumen_evaluacion.json"))
print(r["own_photos_cnn"])
PY
```
