"""Grad-CAM para la CNN: ¿en qué región de la imagen se basa la predicción?

Para la clase c y el mapa de activaciones A^k de la última capa convolucional:

    α_k^c = (1/Z) Σ_{i,j} ∂y^c / ∂A^k_{ij}          (importancia del canal k)
    L^c   = ReLU( Σ_k α_k^c · A^k )                  (mapa de calor)

El mapa se sobremuestrea al tamaño de la imagen. Si la red mira la fruta, el
calor cae sobre ella; si aprendió el atajo del fondo, cae en los bordes
[Selvaraju et al., 2017].

Además se calcula la **fracción de energía dentro de la fruta**,
E = Σ L·M / Σ L, con M la máscara de segmentación.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.models.cnn import FruitCNN


class GradCAM:
    def __init__(self, model: FruitCNN):
        self.model = model.eval()
        self._acts = self._grads = None
        last_conv = model.features[-1][0]
        last_conv.register_forward_hook(lambda m, i, o: setattr(self, "_acts", o))
        last_conv.register_full_backward_hook(lambda m, gi, go: setattr(self, "_grads", go[0]))

    def __call__(self, x: torch.Tensor, target: int | torch.Tensor | None = None) -> np.ndarray:
        """x: tensor (N,3,H,W) normalizado. Devuelve mapas (N,H,W) en [0,1]."""
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if target is None:
            target = logits.argmax(1)
        elif isinstance(target, int):
            target = torch.full((len(x),), target)
        self.model.zero_grad()
        logits.gather(1, target.view(-1, 1)).sum().backward()
        w = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return cam.detach().numpy()


def energy_inside(cam: np.ndarray, mask: np.ndarray) -> float:
    return float((cam * mask).sum() / max(cam.sum(), 1e-8))
