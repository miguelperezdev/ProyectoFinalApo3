"""Conversión de P(buena | x) en tres grados comerciales: Primera, Segunda y Descarte.

El dataset solo tiene dos etiquetas de calidad (buena / mala), pero el negocio
necesita al menos tres categorías. Se usa una **regla de decisión con opción de
rechazo** (Chow, 1970): el modelo solo emite un veredicto firme cuando su
probabilidad es suficientemente alta, y los casos dudosos pasan a una categoría
intermedia que se revisa a mano o se vende a menor precio.

    grado(x) = Primera   si p(x) ≥ τ_alto
               Descarte  si p(x) ≤ τ_bajo
               Segunda   en otro caso,          con p(x) = P(buena | x)

Los umbrales se eligen en el conjunto de **validación** para garantizar una
precisión mínima π (por defecto 98 %) en las categorías extremas:

    τ_alto = mín { τ : P(buena | p ≥ τ) ≥ π }    (≤ 5 % de frutas malas vendidas como Primera)
    τ_bajo = máx { τ : P(mala  | p ≤ τ) ≥ π }    (≤ 5 % de frutas buenas descartadas)

Se ajusta con π = 98 % en validación, aunque el criterio de éxito del negocio es
≥ 95 % en prueba: el umbral se elige justo en el borde donde la precisión de
validación alcanza la meta, y en datos nuevos esa precisión suele bajar entre
1 y 3 puntos. Con π = 95 % la categoría Segunda quedaba casi vacía y la
precisión de Primera en prueba caía a ~93 %; el margen de 3 puntos lo corrige.

Además se reporta la *cobertura*: la fracción de frutas que reciben un veredicto
firme (Primera o Descarte). Subir π aumenta la confiabilidad, pero manda más
frutas a revisión manual (compromiso error-rechazo de Chow).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

GRADES = ("Primera", "Segunda", "Descarte")


@dataclass
class GradeThresholds:
    low: float
    high: float
    target_precision: float

    def to_dict(self) -> dict:
        return asdict(self)


def fit_thresholds(p_good: np.ndarray, is_good: np.ndarray, target: float = 0.98) -> GradeThresholds:
    """Busca τ_alto y τ_bajo sobre los valores observados de p (validación)."""
    order = np.sort(np.unique(p_good))
    high = 1.0
    for t in order:  # el menor τ cuya precisión de "buena" alcance la meta
        sel = p_good >= t
        if sel.sum() >= 10 and is_good[sel].mean() >= target:
            high = float(t)
            break
    low = 0.0
    for t in order[::-1]:  # el mayor τ cuya precisión de "mala" alcance la meta
        sel = p_good <= t
        if sel.sum() >= 10 and (1 - is_good[sel]).mean() >= target:
            low = float(t)
            break
    low = min(low, high)
    return GradeThresholds(low=low, high=high, target_precision=target)


def assign_grades(p_good: np.ndarray, th: GradeThresholds) -> np.ndarray:
    p_good = np.asarray(p_good)
    out = np.full(p_good.shape, "Segunda", dtype=object)
    out[p_good >= th.high] = "Primera"
    out[p_good <= th.low] = "Descarte"
    return out


def grade_report(p_good: np.ndarray, is_good: np.ndarray, th: GradeThresholds) -> dict:
    g = assign_grades(p_good, th)
    rep = {"coverage": float(np.mean(g != "Segunda"))}
    for name in GRADES:
        sel = g == name
        rep[f"{name}_fraction"] = float(sel.mean())
        rep[f"{name}_pct_good"] = float(is_good[sel].mean()) if sel.any() else float("nan")
    return rep
