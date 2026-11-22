"""Interfaz gráfica del clasificador de calidad de frutas (Streamlit).

Permite cargar una foto o capturarla con la cámara, y muestra:
- la fruta reconocida,
- el grado de calidad (Primera, Segunda o Descarte) con P(buena),
- el tamaño estimado (pequeño, mediano o grande) con la máscara de segmentación.

Uso:
    streamlit run src/app/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from src.models.predictor import Predictor  # noqa: E402
from src.utils.paths import CHECKPOINTS  # noqa: E402

GRADE_STYLE = {
    "Primera": ("🟢", "Fruta sana: apta para venta de primera."),
    "Segunda": ("🟡", "Dudosa o con defectos leves: revisión manual o venta a menor precio."),
    "Descarte": ("🔴", "Defectos graves: no apta para la venta."),
}

st.set_page_config(page_title="Calidad de frutas", page_icon="🍎", layout="wide")


@st.cache_resource
def get_predictor(name: str) -> Predictor:
    return Predictor(name)


def available_models() -> list[str]:
    models = [p.stem for p in CHECKPOINTS.glob("cnn.pt")]
    models += [p.stem for p in sorted(CHECKPOINTS.glob("*.joblib"))]
    return models


def overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (255, 0, 255), 2)
    return out


st.title("🍎 Clasificador de calidad de frutas")
st.caption("Proyecto final · Algoritmos y Programación III · Universidad Icesi · 2026-2")

models = available_models()
if not models:
    st.error("No hay modelos entrenados en experiments/checkpoints/. Ver README → Cómo reproducir.")
    st.stop()

with st.sidebar:
    st.header("Configuración")
    model_name = st.selectbox("Modelo", models, index=0)
    source = st.radio("Origen de la imagen", ["Cargar archivo", "Cámara"])
    st.markdown(
        "**Recomendaciones de captura**\n"
        "- Una sola fruta, centrada\n"
        "- Fondo liso y uniforme\n"
        "- Buena luz, sin sombras fuertes\n"
        "- Cámara a ~30 cm (para el tamaño)"
    )
    st.info("Frutas soportadas: manzana, banano, guayaba, limón, naranja y granada.")

predictor = get_predictor(model_name)
file = (st.file_uploader("Imagen de la fruta", type=["jpg", "jpeg", "png"])
        if source == "Cargar archivo" else st.camera_input("Tome una foto de la fruta"))

if file is None:
    st.info("Cargue o capture una imagen para clasificarla.")
    st.stop()

t0 = time.perf_counter()
pred = predictor.predict(Image.open(file))
ms = 1000 * (time.perf_counter() - t0)

c1, c2 = st.columns([1, 1.3])
with c1:
    st.image(overlay(pred.image, pred.mask), caption="Imagen procesada (128×128) y segmentación",
             width="stretch")
with c2:
    icon, desc = GRADE_STYLE[pred.grade]
    st.subheader(f"{icon} Calidad: {pred.grade}")
    st.write(desc)
    st.progress(pred.p_good, text=f"P(buena) = {pred.p_good:.1%}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Fruta", pred.fruit_es.capitalize(), f"{pred.fruit_confidence:.0%} confianza",
              delta_color="off")
    m2.metric("Tamaño", pred.size.capitalize(), f"d = {pred.d_norm:.2f}", delta_color="off")
    m3.metric("Tiempo", f"{ms:.0f} ms")
    th = predictor.thresholds
    st.caption(f"Umbrales de grado (validación, precisión ≥ {th.target_precision:.0%}): "
               f"Descarte si P ≤ {th.low:.2f} · Primera si P ≥ {th.high:.2f}")
    if pred.fruit_confidence < 0.5:
        st.warning("Baja confianza en la fruta: puede no ser una de las 6 frutas soportadas.")
    if pred.d_norm < 0.1 or pred.d_norm > 0.95:
        st.warning("La segmentación no es confiable: use un fondo liso y centre la fruta.")
