"""Predicción de extremo a extremo para una imagen: fruta, grado de calidad y tamaño.

    imagen → recorte central 128×128 → modelo → P(fruta, calidad)
           → P(buena) → grado (Primera / Segunda / Descarte)
           → segmentación → diámetro normalizado → tamaño
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image, ImageOps

from src.data.preprocess import square_resize
from src.evaluation.metrics import FRUIT_ES, FRUITS, p_fruit, p_good
from src.features.size import estimate_size
from src.models.cnn import FruitCNN
from src.models.grading import GradeThresholds, assign_grades
from src.utils.paths import CHECKPOINTS, RESULTS

DEFAULT_MODEL = "cnn"


@dataclass
class Prediction:
    fruit: str
    fruit_es: str
    fruit_confidence: float
    p_good: float
    grade: str
    size: str
    d_norm: float
    mask: np.ndarray = field(repr=False)
    image: np.ndarray = field(repr=False)
    proba: np.ndarray = field(repr=False)


def load_thresholds(model: str) -> GradeThresholds:
    path = RESULTS / f"grading_{model}.json"
    if path.exists():
        d = json.loads(path.read_text())
        return GradeThresholds(d["low"], d["high"], d["target_precision"])
    return GradeThresholds(0.2, 0.8, 0.98)


class Predictor:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model_name = model
        self.thresholds = load_thresholds(model)
        if model.startswith("cnn"):
            norm = json.loads((RESULTS / "cnn_norm.json").read_text())
            self.mean = torch.tensor(norm["mean"]).view(3, 1, 1)
            self.std = torch.tensor(norm["std"]).view(3, 1, 1)
            self.size = norm.get("size", 128)
            self.net = FruitCNN()
            self.net.load_state_dict(torch.load(CHECKPOINTS / f"{model}.pt", map_location="cpu"))
            self.net.eval()
        else:
            import joblib

            from src.features.handcrafted import extract

            self.clf = joblib.load(CHECKPOINTS / f"{model}.joblib")
            self._extract = extract

    def to_tensor(self, images: np.ndarray) -> torch.Tensor:
        """Lote (N,128,128,3) uint8 → tensor normalizado (N,3,s,s) a la resolución de la CNN."""
        x = torch.from_numpy(np.array(images, copy=True)).permute(0, 3, 1, 2).float() / 255  # copia: PIL da arreglos de solo lectura
        if x.shape[-1] != self.size:
            x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear",
                                                antialias=True, align_corners=False)
        return (x - self.mean) / self.std

    def proba(self, img128: np.ndarray) -> np.ndarray:
        """Probabilidades de las 12 clases para una imagen 128×128 RGB uint8."""
        if self.model_name.startswith("cnn"):
            with torch.no_grad():
                return torch.softmax(self.net(self.to_tensor(img128[None])), 1).numpy()[0]
        return self.clf.predict_proba(self._extract(img128)[None])[0]

    def predict(self, image: Image.Image | np.ndarray) -> Prediction:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        img = square_resize(ImageOps.exif_transpose(image).convert("RGB"))
        pr = self.proba(img)
        pf = p_fruit(pr[None])[0]
        fi = int(pf.argmax())
        pg = float(p_good(pr[None])[0])
        fruit = FRUITS[fi]
        size = estimate_size(img, fruit)
        return Prediction(fruit=fruit, fruit_es=FRUIT_ES[fruit], fruit_confidence=float(pf[fi]),
                          p_good=pg, grade=str(assign_grades(np.array([pg]), self.thresholds)[0]),
                          size=size.category, d_norm=size.d_norm, mask=size.mask, image=img, proba=pr)
