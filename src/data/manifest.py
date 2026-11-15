"""Construye el índice (manifest) del dataset con etiquetas y grupos de captura.

Cada fila del manifest es una imagen con:
    path      ruta relativa a data/raw/fruit-quality-classification
    fruit     apple, banana, guava, lime, orange, pomegranate
    quality   good, bad o mixed
    capture   identificador de la foto original (timestamp o número de IMG_####)
    group     identificador de la *ráfaga*: fotos de la misma fruta tomadas con
              pocos segundos de diferencia, más sus copias aumentadas
    md5       hash del archivo (para detectar duplicados exactos)

Por qué agrupar: el dataset contiene copias aumentadas de una misma foto
(p. ej. ``20190820_143522.jpg`` y ``20190820_143522_22896.jpg``) y ráfagas de la
misma fruta desde varios ángulos. Si estas fotos quedan repartidas entre
entrenamiento y prueba, el modelo "ve" el examen y las métricas se inflan
(fuga de información). La partición se hace por ``group``, no por imagen.

Uso:
    python -m src.data.manifest
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.paths import MANIFEST_PATH, RAW_DATASET

FOLDER_QUALITY = {
    "Good Quality_Fruits": "good",
    "Bad Quality_Fruits": "bad",
    "Mixed Qualit_Fruits": "mixed",
}
FRUIT_NAMES = {
    "apple": "apple",
    "banana": "banana",
    "guava": "guava",
    "lime": "lime",
    "lemon": "lime",  # en Mixed la carpeta se llama "Lemon"; es la misma fruta
    "orange": "orange",
    "pomegranate": "pomegranate",
}

# Ventana de tiempo que define una ráfaga: fotos tomadas en el mismo minuto
# (misma fruta desde varios ángulos). Se usan ventanas fijas y no "huecos" entre
# fotos porque las sesiones fueron continuas y los huecos encadenaban sesiones enteras.
BURST_WINDOW_SECONDS = 60
# Tamaño del bloque de numeración IMG_#### considerado una misma ráfaga.
BURST_WINDOW_NUMBERS = 5

_TS = re.compile(r"(?:IMG_?)?(\d{8})_?(\d{6})")
_NUM = re.compile(r"IMG_(\d{4})\.", re.IGNORECASE)


def parse_capture(filename: str) -> tuple[str, datetime | int | None]:
    """Extrae el identificador de la foto original y su valor ordenable.

    >>> parse_capture("20190820_143522_22896.jpg")[0]
    'ts:20190820_143522'
    >>> parse_capture("IMG_1234.JPG")
    ('num:1234', 1234)
    """
    m = _TS.match(filename)
    if m:
        stamp = f"{m.group(1)}_{m.group(2)}"
        try:
            return f"ts:{stamp}", datetime.strptime(stamp, "%Y%m%d_%H%M%S")
        except ValueError:
            return f"ts:{stamp}", None
    m = _NUM.match(filename)
    if m:
        n = int(m.group(1))
        return f"num:{n}", n
    return f"file:{Path(filename).stem}", None


def _burst_key(order) -> str:
    """Clave de ráfaga: ventana fija de tiempo o bloque de numeración."""
    if isinstance(order, datetime):
        return f"t{int(order.timestamp()) // BURST_WINDOW_SECONDS}"
    if isinstance(order, int):
        return f"n{order // BURST_WINDOW_NUMBERS}"
    return "x"


def _assign_groups(df: pd.DataFrame) -> pd.Series:
    """Agrupa las capturas de una misma carpeta que caen en la misma ventana.

    Las capturas sin fecha ni número quedan cada una en su propio grupo.
    """
    keys = [
        _burst_key(o) if _burst_key(o) != "x" else f"c{c}"
        for o, c in zip(df["order"], df["capture"])
    ]
    return df["folder"] + "#" + pd.Series(keys, index=df.index)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_manifest(root: Path = RAW_DATASET) -> pd.DataFrame:
    rows = []
    for quality_dir, quality in FOLDER_QUALITY.items():
        for fruit_dir in sorted((root / quality_dir).iterdir()):
            fruit = FRUIT_NAMES[fruit_dir.name.split("_")[0].lower()]
            for f in sorted(fruit_dir.iterdir()):
                capture, order = parse_capture(f.name)
                rows.append(
                    {
                        "path": str(f.relative_to(root)),
                        "fruit": fruit,
                        "quality": quality,
                        "folder": f"{quality}/{fruit}",
                        "capture": capture,
                        "order": order,
                        "md5": _md5(f),
                    }
                )
    df = pd.DataFrame(rows)
    df["group"] = _assign_groups(df)
    return df.drop(columns=["order"])


def main() -> None:
    df = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"{len(df)} imágenes -> {MANIFEST_PATH}")
    print(df.groupby(["quality", "fruit"]).agg(imgs=("path", "size"),
                                               capturas=("capture", "nunique"),
                                               grupos=("group", "nunique")))
    dup = df[df.duplicated("md5", keep=False)]
    print(f"Duplicados exactos (mismo md5): {len(dup)} archivos")


if __name__ == "__main__":
    main()
