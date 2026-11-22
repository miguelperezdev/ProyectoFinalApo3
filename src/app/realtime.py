"""Clasificación en tiempo real con la cámara (o simulación de una línea de empaque).

El enunciado pide ver en tiempo real la calidad de una fruta presentada ante la
cámara. Este script lee la cámara cuadro a cuadro con OpenCV, clasifica cada
cuadro y dibuja la fruta, el grado, P(buena) y el tamaño sobre el video.

Para que la etiqueta no "parpadee" entre cuadros, las probabilidades se
suavizan con una media móvil exponencial:

    P̄_t = α · P_t + (1 − α) · P̄_{t−1},   α = 0,4

y el grado se calcula sobre P̄_t. Con una cámara a 15–30 cuadros por segundo,
el valor se estabiliza en menos de medio segundo.

Con ``--fuente <carpeta>`` simula una **línea de empaque** (extensión opcional
del enunciado): recorre las imágenes de la carpeta como si pasaran frente a la
cámara y lleva el conteo por grado.

Uso:
    python -m src.app.realtime                     # cámara 0
    python -m src.app.realtime --fuente 1          # otra cámara
    python -m src.app.realtime --fuente data/interim/mixed_crops --pausa 0.8
    python -m src.app.realtime --fuente data/interim/mixed_crops --salida demo.mp4
    Teclas: q = salir · espacio = pausar
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.data.preprocess import square_resize
from src.evaluation.metrics import FRUIT_ES, FRUITS, p_fruit, p_good
from src.features.size import estimate_size
from src.models.grading import assign_grades
from src.models.predictor import Predictor

ALPHA = 0.4
COLORS = {"Primera": (60, 170, 60), "Segunda": (0, 190, 230), "Descarte": (50, 50, 210)}  # BGR


def frames_from_camera(index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {index}.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def frames_from_folder(folder: Path, pause: float, limit: int = 0):
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    files = files[:limit] if limit else files
    if not files:
        raise SystemExit(f"No hay imágenes en {folder}.")
    for f in files:
        frame = cv2.imread(str(f))
        t_end = time.time() + pause
        while time.time() < t_end:  # la misma "fruta" frente a la cámara durante `pause` segundos
            yield frame
            time.sleep(1 / 30)


def letterbox(frame: np.ndarray, width: int) -> np.ndarray:
    """Ajusta el cuadro a un lienzo fijo de width × 3/4·width sin deformarlo (bandas negras)."""
    height = width * 3 // 4
    scale = min(width / frame.shape[1], height / frame.shape[0])
    small = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    out = np.zeros((height, width, 3), np.uint8)
    y, x = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def draw(frame: np.ndarray, lines: list[str], color) -> np.ndarray:
    out = frame.copy()
    h = 28 * len(lines) + 16
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (30, 30, 30), -1)
    cv2.rectangle(out, (0, 0), (12, h), color, -1)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (22, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuente", default="0", help="índice de cámara o carpeta con imágenes")
    ap.add_argument("--modelo", default="cnn")
    ap.add_argument("--pausa", type=float, default=1.0, help="segundos por imagen al simular la línea")
    ap.add_argument("--ancho", type=int, default=720, help="ancho de la ventana")
    ap.add_argument("--salida", help="guarda el video anotado (.mp4) en lugar de abrir una ventana")
    ap.add_argument("--max-imagenes", type=int, default=0, help="límite de imágenes al simular la línea")
    args = ap.parse_args()

    pred = Predictor(args.modelo)
    folder = Path(args.fuente)
    line = folder.is_dir()
    frames = frames_from_folder(folder, args.pausa, args.max_imagenes) if line else frames_from_camera(int(args.fuente))
    writer, smooth, counts, current, grade = None, None, Counter(), None, None
    t_prev = time.perf_counter()

    for frame in frames:
        if line and frame is not current:  # llega una fruta nueva a la línea
            if grade is not None:
                counts[grade] += 1
            current, smooth = frame, None
        img = square_resize(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        proba = pred.proba(img)
        smooth = proba if smooth is None else ALPHA * proba + (1 - ALPHA) * smooth

        pg = float(p_good(smooth[None])[0])
        pf = p_fruit(smooth[None])[0]
        fruit = FRUITS[int(pf.argmax())]
        grade = str(assign_grades(np.array([pg]), pred.thresholds)[0])
        size = estimate_size(img, fruit)
        now = time.perf_counter()
        fps, t_prev = 1 / max(now - t_prev, 1e-6), now

        lines = [f"{FRUIT_ES[fruit].capitalize()} ({pf.max():.0%}) | {grade} | P(buena) {pg:.0%}",
                 f"Tamano: {size.category} (d = {size.d_norm:.2f}) | {fps:.0f} cuadros/s"]
        if line:
            lines.append("Linea: " + "  ".join(f"{g} {counts[g]}" for g in ("Primera", "Segunda", "Descarte")))
        view = draw(letterbox(frame, args.ancho), lines, COLORS[grade])

        if args.salida:
            if writer is None:
                writer = cv2.VideoWriter(args.salida, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                         (view.shape[1], view.shape[0]))
            writer.write(view)
            continue
        cv2.imshow("Calidad de frutas (q: salir, espacio: pausa)", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            key = 0
            while key not in (ord(" "), ord("q")):
                key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break

    if line and grade is not None:
        counts[grade] += 1
    if writer is not None:
        writer.release()
        print("Video guardado en", args.salida)
    else:
        cv2.destroyAllWindows()
    if counts:
        print("Conteo por grado:", dict(counts))


if __name__ == "__main__":
    main()
