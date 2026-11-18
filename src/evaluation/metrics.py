"""Métricas del proyecto.

Los modelos predicen la distribución conjunta P(fruta, calidad) sobre las
12 clases ``<fruta>_<calidad>``. De ella se obtienen por marginalización:

    P(calidad = buena | x) = Σ_f P(f, buena | x)
    P(fruta = f | x)       = P(f, buena | x) + P(f, mala | x)

Métrica principal: **F1-macro**. Es el promedio no ponderado del F1 de cada
clase, F1_k = 2·P_k·R_k / (P_k + R_k), así que una clase minoritaria pesa lo
mismo que una mayoritaria y no se puede "ganar" ignorándola (a diferencia de
la exactitud).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

FRUITS = ["apple", "banana", "guava", "lime", "orange", "pomegranate"]
QUALITIES = ["bad", "good"]
CLASSES = [f"{f}_{q}" for f in FRUITS for q in QUALITIES]  # orden alfabético = orden del LabelEncoder
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
FRUIT_ES = {"apple": "manzana", "banana": "banano", "guava": "guayaba", "lime": "limón",
            "orange": "naranja", "pomegranate": "granada"}


def encode(labels) -> np.ndarray:
    return np.array([CLASS_TO_ID[l] for l in labels])


def p_good(proba: np.ndarray) -> np.ndarray:
    """P(calidad buena | x) a partir de las 12 probabilidades conjuntas."""
    return proba[:, 1::2].sum(axis=1)


def p_fruit(proba: np.ndarray) -> np.ndarray:
    """Matriz n×6 con P(fruta | x)."""
    return proba[:, 0::2] + proba[:, 1::2]


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Métricas de las 12 clases, de la calidad binaria y de la fruta."""
    y_pred = proba.argmax(1)
    q_true = y_true % 2
    pg = p_good(proba)
    q_pred = (pg >= 0.5).astype(int)
    f_true, f_pred = y_true // 2, p_fruit(proba).argmax(1)
    clipped = np.clip(proba, 1e-7, 1)
    clipped /= clipped.sum(1, keepdims=True)
    return {
        "joint_accuracy": accuracy_score(y_true, y_pred),
        "joint_f1_macro": f1_score(y_true, y_pred, average="macro"),
        "joint_log_loss": log_loss(y_true, clipped, labels=range(len(CLASSES))),
        "quality_accuracy": accuracy_score(q_true, q_pred),
        "quality_balanced_accuracy": balanced_accuracy_score(q_true, q_pred),
        "quality_f1_macro": f1_score(q_true, q_pred, average="macro"),
        "quality_roc_auc": roc_auc_score(q_true, pg),
        "fruit_accuracy": accuracy_score(f_true, f_pred),
        "fruit_f1_macro": f1_score(f_true, f_pred, average="macro"),
    }
