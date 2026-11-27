# Fase 6 · Despliegue

> Código: `src/app/app.py` (interfaz web), `src/app/realtime.py` (tiempo real y línea de empaque), `src/models/predictor.py` (lógica de predicción compartida)

## 6.1 Aplicación

```bash
streamlit run src/app/app.py
```

La interfaz, hecha en Streamlit, permite:

- **cargar una foto** (JPG o PNG) o **capturarla con la cámara** del computador o del celular;
- elegir el modelo (la CNN por defecto, o los modelos clásicos si se reentrenaron);
- ver la **fruta reconocida** con su confianza, el **grado de calidad** (Primera, Segunda, Descarte) con P(buena), el **tamaño** (pequeño, mediano o grande) con el diámetro normalizado y el contorno de segmentación, y el tiempo de inferencia;
- recibir **advertencias** cuando la confianza en la fruta es baja (< 50 %) o cuando la segmentación no es confiable (fondo no uniforme).

### Tiempo real y simulación de la línea de empaque

El enunciado pide ver **en tiempo real** la calidad de una fruta presentada ante la cámara. `st.camera_input` de Streamlit toma fotos sueltas, así que se agregó un modo de video con OpenCV:

```bash
python -m src.app.realtime                                    # cámara del computador
python -m src.app.realtime --fuente data/interim/mixed_crops  # simula una línea de empaque
python -m src.app.realtime --fuente data/interim/mixed_crops --salida linea.mp4   # graba el resultado
```

Cada cuadro se clasifica y se dibuja sobre el video la fruta, el grado, P(buena), el tamaño y los cuadros por segundo. Para que la etiqueta no parpadee entre cuadros, las probabilidades se suavizan con una media móvil exponencial, $\bar P_t = \alpha P_t + (1-\alpha)\bar P_{t-1}$ con $\alpha = 0{,}4$, y el grado se calcula sobre $\bar P_t$. En el modo de **línea de empaque** (extensión opcional del enunciado), las imágenes de una carpeta pasan una a una frente a la "cámara" y se lleva el conteo de frutas por grado, como en una banda transportadora.

## 6.2 Arquitectura

```
 Cámara / archivo
       │
       ▼
 Preprocesamiento ── EXIF → recorte central → 128×128
       │
       ├──► CNN (113 K parámetros) ──► P(fruta, calidad) ∈ Δ¹¹
       │                                   │
       │                    ┌──────────────┴──────────────┐
       │                    ▼                             ▼
       │            P(buena) = Σ_f P(f, buena)     fruta = argmax_f P(f)
       │                    │
       │                    ▼
       │       Umbrales τ_bajo, τ_alto (validación)
       │                    │
       │                    ▼
       │       Primera / Segunda / Descarte
       │
       └──► Segmentación (Lab + Otsu) ──► d_norm ──► pequeño / mediano / grande
```

## 6.3 Requisitos

| Recurso | Requisito |
|---|---|
| Hardware | Cualquier computador con CPU (no requiere GPU); cámara web o celular |
| Software | Python ≥ 3.11 y `requirements.txt` |
| Modelo | `experiments/checkpoints/cnn.pt` (~450 KB) + `experiments/results/cnn_norm.json` + `grading_cnn.json` |
| Latencia | Ver `docs/crisp-dm/05_evaluacion.md` (milisegundos por imagen en CPU) |

## 6.4 Protocolo de uso

1. Una fruta por foto, centrada, sobre un fondo liso (cartulina blanca o gris).
2. Cámara a unos 30 cm para que el tamaño sea comparable.
3. Las frutas en **Segunda** se revisan a mano: el sistema apoya al clasificador humano, no lo reemplaza.

## 6.5 Plan de despliegue y trabajo futuro

| Etapa | Descripción |
|---|---|
| Piloto | Uso en un puesto de plaza de mercado con un celular y un portátil, comparando con la clasificación manual del vendedor |
| Línea de empaque (extensión opcional) | Ya simulada en software con `src/app/realtime.py`. En hardware: Raspberry Pi 4 con cámara sobre una banda transportadora con fondo uniforme e iluminación controlada, y un actuador que desvía cada fruta según su grado. La CNN de 113 K parámetros (~450 KB) cabe holgadamente en memoria y se puede exportar a ONNX o TorchScript |
| Mejora continua | Recolectar las fotos que el sistema manda a Segunda, anotarlas y reentrenar periódicamente (aprendizaje activo) |
| Mantenimiento | Vigilar la distribución de P(buena) en producción: si se desplaza, indica un cambio de condiciones (*data drift*) |
