"""Segmentación de la fruta y estimación de su tamaño relativo.

Supuesto de despliegue (enunciado): una fruta sobre un fondo simple y uniforme.

1. **Modelo del fondo.** Se toma el color mediano del borde de la imagen
   (franja de ancho b) en el espacio CIELab: b_lab = mediana(borde).
2. **Distancia de color.** Para cada píxel, d(u,v) = ‖lab(u,v) − b_lab‖₂. Como
   Lab es aproximadamente uniforme en percepción, d mide qué tan distinto
   se ve el píxel respecto del fondo.
3. **Umbral de Otsu.** Se elige el τ que maximiza la varianza entre clases
   σ_B²(τ) = ω₀(τ)·ω₁(τ)·(μ₀(τ) − μ₁(τ))² del histograma de d. Se separa
   fondo/fruta sin fijar un umbral a mano.
4. **Limpieza morfológica** (apertura y cierre) y selección del componente
   conexo más grande, que se rellena para obtener la máscara M.
5. **Diámetro equivalente normalizado**:

       A = Σ M(u,v),   d_eq = 2·√(A/π),   d_norm = d_eq / mín(alto, ancho)

   d_norm ∈ (0, 1] es el diámetro del círculo de igual área, expresado como
   fracción del lado menor de la imagen. Es invariante a la resolución, pero
   **no** a la distancia de la cámara: por eso el protocolo de captura fija la
   distancia (ver data/own/README.md).
6. **Categoría** según umbrales de d_norm por fruta (calibrables con fotos
   anotadas).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Umbrales (d_norm) pequeño/mediano y mediano/grande para fotos tomadas a la
# distancia estándar del protocolo (~30 cm). Valores iniciales razonables; se
# recalibran con calibrate_thresholds() usando las fotos propias anotadas.
SIZE_THRESHOLDS: dict[str, tuple[float, float]] = {
    "apple": (0.40, 0.55),
    "banana": (0.45, 0.65),
    "guava": (0.35, 0.50),
    "lime": (0.22, 0.32),
    "orange": (0.40, 0.55),
    "pomegranate": (0.42, 0.58),
}
DEFAULT_THRESHOLDS = (0.35, 0.55)
SIZE_LABELS = ("pequeño", "mediano", "grande")


@dataclass
class SizeResult:
    mask: np.ndarray
    d_norm: float
    area_fraction: float
    category: str


def segment(rgb: np.ndarray, border: int = 6) -> np.ndarray:
    """Máscara booleana de la fruta (componente más grande distinto del fondo)."""
    lab = cv2.cvtColor(cv2.GaussianBlur(rgb, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    edge = np.concatenate([lab[:border].reshape(-1, 3), lab[-border:].reshape(-1, 3),
                           lab[:, :border].reshape(-1, 3), lab[:, -border:].reshape(-1, 3)])
    dist = np.linalg.norm(lab - np.median(edge, axis=0), axis=2)
    dist8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros(mask.shape, bool)
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    comp = (lab_cc == biggest).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def normalized_diameter(mask: np.ndarray) -> float:
    area = float(mask.sum())
    return 2.0 * np.sqrt(area / np.pi) / min(mask.shape)


def categorize(d_norm: float, fruit: str | None = None) -> str:
    lo, hi = SIZE_THRESHOLDS.get(fruit, DEFAULT_THRESHOLDS)
    return SIZE_LABELS[0] if d_norm < lo else SIZE_LABELS[2] if d_norm > hi else SIZE_LABELS[1]


def estimate_size(rgb: np.ndarray, fruit: str | None = None) -> SizeResult:
    mask = segment(rgb)
    d = normalized_diameter(mask)
    return SizeResult(mask=mask, d_norm=d, area_fraction=float(mask.mean()),
                      category=categorize(d, fruit))


def calibrate_thresholds(d_norm: np.ndarray) -> tuple[float, float]:
    """Umbrales por terciles: divide las frutas observadas en tres grupos iguales."""
    lo, hi = np.quantile(d_norm, [1 / 3, 2 / 3])
    return float(lo), float(hi)
