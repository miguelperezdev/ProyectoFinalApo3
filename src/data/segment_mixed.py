"""Segmenta las frutas individuales de las fotos de la carpeta *Mixed quality*.

El enunciado pide segmentar individualmente cada fruta presente en las imágenes
con varias frutas para obtener ejemplares individuales. Procedimiento:

1. **Color del fondo.** Se agrupan los colores (CIELab) con k-means (k = 4) y
   se toma como fondo el centro del grupo más grande. Es más robusto que usar
   el borde de la imagen, que a veces contiene ventanas, bordes de mesa u otras
   frutas.
2. **Distancia al fondo.** Para cada píxel, d = ‖(0,3·L, a, b) − fondo‖₂. La
   luminosidad pesa 0,3 (se compararon 0,5, 0,3 y 0,2 sobre una muestra) para
   que las sombras y los reflejos, que cambian sobre todo L, no se confundan
   con frutas.
3. **Binarización** con el umbral de Otsu, apertura y cierre morfológicos y
   relleno de huecos.
4. **Separación de frutas que se tocan** con *watershed*: los máximos de la
   transformada de distancia (el centro de cada fruta, que es aproximadamente
   redonda) son las semillas.
5. **Filtros de forma.** Se conserva cada región con área entre 0,4 % y 25 %
   de la imagen, que no toque el borde (fruta incompleta o parte del fondo),
   con solidez (área / envolvente convexa) ≥ 0,85, relación de aspecto ≤ 3 y
   ocupación de su caja ≥ 0,45 y luminosidad media L* ≥ 40/255. Las frutas son
   compactas, convexas y no son negras; el pasto, las sombras y las franjas
   oscuras del fondo no cumplen alguna de estas condiciones.
6. Se recorta la caja envolvente con un margen del 10 % y se guarda.

Limitaciones: falla en los primeros planos, donde la fruta ocupa casi toda la
imagen (el "fondo" dominante es la fruta), y puede unir frutas muy apiladas o
racimos de banano. Por eso los recortes se revisan y anotan a mano
(``src/app/annotate.py``), con la opción *Inválido*.

Uso:
    python -m src.data.segment_mixed
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from src.utils.paths import INTERIM, RAW_DATASET, SPLITS_PATH

OUT_DIR = INTERIM / "mixed_crops"
WORK_SIZE = 512
MIN_AREA, MAX_AREA = 0.004, 0.25
MIN_SOLIDITY = 0.85
MAX_ASPECT = 3.0
MIN_EXTENT = 0.45
MIN_LIGHTNESS = 40  # L* medio (0-255) mínimo: descarta franjas negras y sombras profundas
L_WEIGHT = 0.3
PAD = 0.10


def background_color(lab: np.ndarray, k: int = 4, seed: int = 0) -> np.ndarray:
    """Centro del grupo de color más grande (k-means sobre una submuestra de píxeles)."""
    cv2.setRNGSeed(seed)
    px = np.ascontiguousarray(lab[::4, ::4].reshape(-1, 3))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return centers[np.bincount(labels.ravel()).argmax()]


def foreground_mask(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(cv2.GaussianBlur(rgb, (7, 7), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= L_WEIGHT
    dist = np.linalg.norm(lab - background_color(lab), axis=2)
    d8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return ndi.binary_fill_holes(mask > 0)


def split_touching(mask: np.ndarray) -> np.ndarray:
    """Watershed sobre la transformada de distancia: una región por fruta."""
    dt = ndi.distance_transform_edt(mask)
    min_dist = max(8, int(0.04 * min(mask.shape)))
    peaks = peak_local_max(dt, min_distance=min_dist, threshold_abs=min_dist,
                           labels=ndi.label(mask)[0], exclude_border=False)
    markers = np.zeros(mask.shape, int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return watershed(-dt, markers, mask=mask)


def fruit_components(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Cajas (x, y, w, h) de las frutas detectadas en una imagen RGB."""
    H, W = rgb.shape[:2]
    regions = split_touching(foreground_mask(rgb))
    lightness = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
    boxes = []
    for r in range(1, regions.max() + 1):
        contours, _ = cv2.findContours((regions == r).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        hull = cv2.contourArea(cv2.convexHull(c))
        if (not MIN_AREA * H * W <= area <= MAX_AREA * H * W
                or x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
                or hull == 0 or area / hull < MIN_SOLIDITY
                or max(w, h) / min(w, h) > MAX_ASPECT
                or area / (w * h) < MIN_EXTENT
                or lightness[regions == r].mean() < MIN_LIGHTNESS):
            continue
        boxes.append((x, y, w, h))
    return boxes


def crop_with_pad(img: Image.Image, box, scale: float) -> Image.Image:
    x, y, w, h = (int(v / scale) for v in box)
    side = int(max(w, h) * (1 + 2 * PAD))
    cx, cy = x + w // 2, y + h // 2
    left, top = max(cx - side // 2, 0), max(cy - side // 2, 0)
    return img.crop((left, top, min(left + side, img.width), min(top + side, img.height)))


def main() -> None:
    splits = pd.read_csv(SPLITS_PATH)
    mixed = splits[splits["split"] == "mixed"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.jpg"):  # recortes de una ejecución anterior
        old.unlink()
    rows = []
    for _, r in mixed.iterrows():
        img = ImageOps.exif_transpose(Image.open(RAW_DATASET / r["path"])).convert("RGB")
        scale = WORK_SIZE / max(img.size)
        small = np.asarray(img.resize((round(img.width * scale), round(img.height * scale))))
        for k, box in enumerate(fruit_components(small)):
            name = f"{r['fruit']}_{r['capture'].split(':')[1]}_{k}.jpg"
            crop_with_pad(img, box, scale).save(OUT_DIR / name, quality=95)
            rows.append({"crop": name, "source": r["path"], "fruit": r["fruit"],
                         "group": r["group"], "grade": ""})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "annotation.csv", index=False)
    print(f"{len(mixed)} fotos -> {len(df)} recortes en {OUT_DIR}")
    print(df.groupby("fruit").size())


if __name__ == "__main__":
    main()
