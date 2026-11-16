"""Extracción de características manuales para los modelos de ML clásico.

Cada imagen RGB de 128×128 se convierte en un vector x ∈ R^d que concatena:

=====================  ====  ==================================================
Bloque                 dim   Justificación
=====================  ====  ==================================================
Histograma HSV         34    El tono (H) separa colores de madurez/podredumbre
                             de forma más robusta a la iluminación que RGB,
                             porque la intensidad queda aislada en V.
Momentos de color      18    Media, desviación y asimetría por canal en HSV y
                             CIELab. Lab es aproximadamente uniforme en
                             percepción: distancias euclidianas ≈ diferencias
                             de color percibidas.
Indicadores de defecto  2    Fracción de píxeles oscuros y fracción marrón:
                             aproximan manchas, golpes y podredumbre.
LBP uniforme           28    Local Binary Patterns (P=8,R=1 y P=16,R=2):
                             textura local invariante a cambios monótonos de
                             brillo (piel lisa vs. arrugada o con moho).
GLCM (Haralick)        12    Contraste, disimilitud, homogeneidad, energía,
                             correlación y ASM a distancias 1 y 3: textura de
                             segundo orden.
HOG                    324   Histogramas de gradientes orientados sobre 64×64:
                             forma y bordes (silueta, grietas).
=====================  ====  ==================================================

Las características se estandarizan después (z-score) dentro del pipeline de
cada modelo, ajustando la media y la desviación SOLO con el conjunto de
entrenamiento.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

from src.utils.paths import PROCESSED

H_BINS, S_BINS, V_BINS = 18, 8, 8
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def _hist(channel: np.ndarray, bins: int, upper: int, mask=None) -> np.ndarray:
    vals = channel[mask] if mask is not None else channel.ravel()
    h, _ = np.histogram(vals, bins=bins, range=(0, upper))
    return h / max(h.sum(), 1)


def color_features(rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hist = np.concatenate([
        _hist(hsv[..., 0], H_BINS, 180, mask),
        _hist(hsv[..., 1], S_BINS, 256, mask),
        _hist(hsv[..., 2], V_BINS, 256, mask),
    ])
    moments = []
    for img in (hsv, lab):
        px = img[mask] if mask is not None else img.reshape(-1, 3)
        px = px.astype(np.float32)
        moments += [px.mean(0) / 255, px.std(0) / 255, np.nan_to_num(skew(px, axis=0))]
    h, s, v = (hsv[..., i][mask] if mask is not None else hsv[..., i].ravel() for i in range(3))
    dark = np.mean(v < 60)
    brown = np.mean((h >= 5) & (h <= 25) & (s > 80) & (v < 150))
    return np.concatenate([hist, np.concatenate(moments), [dark, brown]])


def texture_features(gray: np.ndarray) -> np.ndarray:
    feats = []
    for p, r in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(gray, p, r, method="uniform")
        h, _ = np.histogram(lbp, bins=p + 2, range=(0, p + 2))
        feats.append(h / h.sum())
    q = (gray // 8).astype(np.uint8)  # 32 niveles de gris: matriz GLCM estable
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    feats.append(np.concatenate([graycoprops(glcm, p).mean(axis=1) for p in GLCM_PROPS]))
    return np.concatenate(feats)


def shape_features(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    return hog(small, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
               feature_vector=True)


def extract(rgb: np.ndarray) -> np.ndarray:
    """Vector de características de una imagen RGB uint8."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.concatenate([color_features(rgb), texture_features(gray), shape_features(gray)]
                          ).astype(np.float32)


def border_mask(size: int = 128, width: int = 16) -> np.ndarray:
    m = np.ones((size, size), bool)
    m[width:-width, width:-width] = False
    return m


def extract_background(rgb: np.ndarray) -> np.ndarray:
    """Solo color del borde de la imagen (fondo). Sirve para medir el sesgo de fondo."""
    return color_features(rgb, border_mask(rgb.shape[0])).astype(np.float32)


def feature_names() -> list[str]:
    names = [f"hue_{i}" for i in range(H_BINS)] + [f"sat_{i}" for i in range(S_BINS)]
    names += [f"val_{i}" for i in range(V_BINS)]
    for space in ("hsv", "lab"):
        for stat in ("mean", "std", "skew"):
            names += [f"{space}_{stat}_{c}" for c in range(3)]
    names += ["dark_ratio", "brown_ratio"]
    names += [f"lbp8_{i}" for i in range(10)] + [f"lbp16_{i}" for i in range(18)]
    names += [f"glcm_{p}_d{d}" for p in GLCM_PROPS for d in (1, 3)]
    names += [f"hog_{i}" for i in range(324)]
    return names


def extract_batch(images: np.ndarray, fn=extract, workers: int = 8) -> np.ndarray:
    with ProcessPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(fn, images, chunksize=64)))


def features_path(split: str, kind: str = "full"):
    return PROCESSED / f"features_{kind}_{split}.npz"


def main() -> None:
    from src.data.preprocess import load_split

    for split in ("train", "val", "test", "mixed"):
        imgs, _ = load_split(split)
        for kind, fn in (("full", extract), ("background", extract_background)):
            X = extract_batch(imgs, fn)
            np.savez(features_path(split, kind), X=X)
            print(f"{split} {kind}: {X.shape}")


if __name__ == "__main__":
    main()
