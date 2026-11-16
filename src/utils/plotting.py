"""Estilo común de las figuras y guardado en formato vectorial (SVG) y PNG."""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.utils.paths import FIGURES

QUALITY_COLORS = {"good": "#2a9d8f", "bad": "#c4553a", "mixed": "#8d99ae"}
GRADE_COLORS = {"Primera": "#2a9d8f", "Segunda": "#e9c46a", "Descarte": "#c4553a"}
MODEL_COLORS = {"logreg": "#8d99ae", "svm": "#457b9d", "rf": "#6a994e", "xgb": "#bc6c25",
                "cnn": "#7b2cbf"}


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "svg.fonttype": "none",
    })


def save(fig, name: str) -> None:
    """Guarda la figura como reports/figures/<name>.svg y .png."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
