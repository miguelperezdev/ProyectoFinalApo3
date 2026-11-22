"""Herramienta de anotación manual de recortes (Streamlit).

Muestra los recortes generados por ``src.data.segment_mixed`` uno por uno para
asignarles un grado (Primera, Segunda o Descarte) o marcarlos como inválidos
(no es una fruta, hay varias frutas o el recorte está cortado).

Las anotaciones se guardan en ``data/own/mixed_annotations.csv``, que sí se
versiona en git: es el aporte de anotación manual del grupo.

Criterios de anotación (deben ser los mismos para los tres anotadores):
- Primera:  fruta sana, sin manchas visibles ni golpes, color uniforme.
- Segunda:  defectos leves (manchas pequeñas, rozaduras, madurez irregular)
            que no comprometen el consumo.
- Descarte: podredumbre, moho, golpes profundos, arrugas o piel rota.

Uso:
    streamlit run src/app/annotate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.utils.paths import INTERIM, OWN  # noqa: E402

CROPS = INTERIM / "mixed_crops"
OUT = OWN / "mixed_annotations.csv"
OPTIONS = ["Primera", "Segunda", "Descarte", "Inválido"]

st.set_page_config(page_title="Anotación de frutas", layout="centered")
st.title("Anotación de recortes · carpeta Mixed")

if not (CROPS / "annotation.csv").exists():
    st.error("Primero ejecute: python -m src.data.segment_mixed")
    st.stop()

crops = pd.read_csv(CROPS / "annotation.csv")
done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["crop", "fruit", "grade", "annotator"])
annotator = st.sidebar.selectbox("Anotador", ["miguel", "danna", "mateo"])
st.sidebar.metric("Anotados", f"{len(done)} / {len(crops)}")
with st.sidebar.expander("Criterios"):
    st.markdown(__doc__.split("Criterios de anotación")[1].split("Uso:")[0])

pending = crops[~crops["crop"].isin(done["crop"])]
if pending.empty:
    st.success("¡Todos los recortes están anotados!")
    st.dataframe(done.groupby(["fruit", "grade"]).size().unstack(fill_value=0))
    st.stop()

row = pending.iloc[0]
st.image(str(CROPS / row["crop"]), caption=f"{row['crop']} · fruta: {row['fruit']}", width=360)
cols = st.columns(len(OPTIONS))
for col, opt in zip(cols, OPTIONS):
    if col.button(opt, width="stretch"):
        new = pd.DataFrame([{"crop": row["crop"], "fruit": row["fruit"], "grade": opt,
                             "annotator": annotator}])
        pd.concat([done, new]).to_csv(OUT, index=False)
        st.rerun()
