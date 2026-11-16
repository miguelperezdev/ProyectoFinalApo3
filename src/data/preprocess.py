"""Preprocesamiento: carga, recorte cuadrado central y redimensionamiento.

Cada imagen se transforma así:
1. Se corrige la orientación según los metadatos EXIF del celular.
2. Se recorta el cuadrado central de lado ``min(alto, ancho)``. Esto conserva
   la relación de aspecto de la fruta (no la deforma) y descarta bordes, que
   suelen ser fondo.
3. Se redimensiona a ``IMG_SIZE × IMG_SIZE`` con interpolación por área
   (antialiasing adecuado al reducir).

El resultado se guarda como arreglo ``uint8`` por conjunto en
``data/processed/images_<split>.npz`` para no releer los JPG en cada experimento.

Uso:
    python -m src.data.preprocess
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.utils.paths import PROCESSED, RAW_DATASET, SPLITS_PATH

IMG_SIZE = 128


def load_image(path: str | Path, size: int = IMG_SIZE) -> np.ndarray:
    """Devuelve la imagen RGB recortada al centro y redimensionada (uint8, HxWx3)."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return square_resize(img, size)


def square_resize(img: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.Resampling.BOX)
    return np.asarray(img, dtype=np.uint8)


def _load(rel: str) -> np.ndarray:
    return load_image(RAW_DATASET / rel)


def cache_path(split: str) -> Path:
    return PROCESSED / f"images_{split}.npz"


def build_cache(splits: pd.DataFrame, workers: int = 8) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for split, sub in splits.groupby("split"):
        with ProcessPoolExecutor(workers) as ex:
            imgs = np.stack(list(ex.map(_load, sub["path"], chunksize=32)))
        np.savez(cache_path(split), images=imgs, index=sub.index.to_numpy())
        print(f"{split}: {imgs.shape} -> {cache_path(split)}")


def load_split(split: str, splits: pd.DataFrame | None = None):
    """Carga imágenes y metadatos de un conjunto (train, val, test o mixed)."""
    if splits is None:
        splits = pd.read_csv(SPLITS_PATH)
    data = np.load(cache_path(split))
    meta = splits.loc[data["index"]].reset_index(drop=True)
    return data["images"], meta


def main() -> None:
    build_cache(pd.read_csv(SPLITS_PATH))


if __name__ == "__main__":
    main()
