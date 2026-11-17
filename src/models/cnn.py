"""CNN pequeña entrenada desde cero para clasificar fruta × calidad.

Arquitectura (entrada 3×96×96):

    [Conv 5×5, 32]  → BatchNorm → ReLU → MaxPool 2×2   → 32×48×48
    [Conv 3×3, 64]  → BatchNorm → ReLU → MaxPool 2×2   → 64×24×24
    [Conv 3×3, 128] → BatchNorm → ReLU → MaxPool 2×2   → 128×12×12
    Global Average Pooling                              → 128
    Dense 128 → ReLU → Dropout(p)                       → 128
    Dense 12 (logits)                                   → 12

Justificación:
- Las convoluciones comparten pesos y explotan la estructura local de la imagen:
  una mancha o un golpe se detectan igual en cualquier posición (equivarianza
  a la traslación), con muchos menos parámetros que una red densa.
- El primer kernel de 5×5 da un campo receptivo inicial más amplio para captar
  manchas pequeñas sin aumentar la profundidad.
- BatchNorm normaliza las activaciones de cada mini-lote,
  x̂ = (x − μ_B)/√(σ²_B + ε), lo que estabiliza y acelera el entrenamiento.
- Global Average Pooling en lugar de aplanar reduce los parámetros de la
  primera capa densa de 128·12·12·128 ≈ 2,4 M a 128·128 ≈ 16 K, lo que limita
  el sobreajuste con ~8 000 imágenes.
- Dropout apaga neuronas al azar con probabilidad p y actúa como regularizador.

Salida: logits z ∈ R^12; P(y = k | x) = softmax(z)_k = e^{z_k} / Σ_j e^{z_j}.
"""

from __future__ import annotations

import torch
from torch import nn


def conv_block(c_in: int, c_out: int, k: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class FruitCNN(nn.Module):
    def __init__(self, n_classes: int = 12, dropout: float = 0.4, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, width, 5),
            conv_block(width, width * 2, 3),
            conv_block(width * 2, width * 4, 3),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
