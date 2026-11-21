"""Prueba de fruta desconocida con las fotos propias de ``data/own/otras_frutas/``.

El modelo solo conoce seis frutas, así que ante una fresa, un lulo, un tomate o
un tomate de árbol siempre responde con alguna de ellas. Esta prueba mide si
se puede detectar ese caso con la confianza de la predicción de fruta,

    c(x) = máx_f P(fruta = f | x),

y una regla de rechazo: si c(x) < τ, la app responde «fruta no reconocida».
El umbral τ se fija en **validación** (dataset de Kaggle) para que el 95 % de
las frutas conocidas pase (percentil 5 de c). Luego se reporta:

- qué fracción de las frutas desconocidas se rechaza (lo deseable: alta);
- qué fracción de las manzanas propias se rechaza por error (lo deseable: baja);
- a qué fruta conocida se parece cada fruta desconocida.

Salidas: ``experiments/results/otras_frutas_pred_<modelo>.csv``,
``experiments/results/otras_frutas_<modelo>.json`` y la figura
``reports/figures/res_otras_frutas.{svg,png}`` (un panel por modelo).

Uso:
    python -m src.evaluation.unknown_fruits                    # cnn (+ cnn_own si existe)
    python -m src.evaluation.unknown_fruits --models cnn
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.data.preprocess import load_split
from src.evaluation.metrics import p_fruit
from src.models.predictor import Predictor
from src.utils import plotting
from src.utils.paths import CHECKPOINTS, OWN, RESULTS

KNOWN_PASS = 0.95  # fracción de frutas conocidas (validación) que debe pasar el umbral


def batch_confidence(pred: Predictor, images: np.ndarray, batch: int = 256) -> np.ndarray:
    """c(x) para un lote de imágenes 128×128 del caché."""
    out = []
    for i in range(0, len(images), batch):
        if pred.model_name.startswith("cnn"):
            import torch
            with torch.no_grad():
                pr = torch.softmax(pred.net(pred.to_tensor(images[i:i + batch])), 1).numpy()
        else:
            pr = np.stack([pred.proba(im) for im in images[i:i + batch]])
        out.append(p_fruit(pr).max(1))
    return np.concatenate(out)


def predict_folder(pred: Predictor, folder, files) -> pd.DataFrame:
    rows = []
    for f in files:
        p = pred.predict(Image.open(folder / f))
        rows.append({"archivo": f, "pred_fruit": p.fruit, "confianza": p.fruit_confidence,
                     "p_good": p.p_good, "pred_grade": p.grade})
    return pd.DataFrame(rows)


NAMES = {"cnn": "CNN original", "cnn_own": "CNN ajustada con fotos propias"}


def run(model: str) -> tuple[dict, dict]:
    pred = Predictor(model)
    Xva, _ = load_split("val")
    conf_val = batch_confidence(pred, Xva)
    tau = float(np.quantile(conf_val, 1 - KNOWN_PASS))

    meta = pd.read_csv(OWN / "otras_frutas.csv")
    unk = predict_folder(pred, OWN / "otras_frutas", meta["archivo"]).merge(meta, on="archivo")
    unk["rechazada"] = unk.confianza < tau
    unk.to_csv(RESULTS / f"otras_frutas_pred_{model}.csv", index=False)

    apples = pd.read_csv(OWN / "labels.csv")
    own = predict_folder(pred, OWN / "images", apples["archivo"])
    own["rechazada"] = own.confianza < tau

    res = {
        "modelo": model,
        "tau": tau,
        "n_desconocidas": len(unk),
        "desconocidas_rechazadas": float(unk.rechazada.mean()),
        "desconocidas_como_manzana": float((unk.pred_fruit == "apple").mean()),
        "rechazo_por_fruta": unk.groupby("fruta").rechazada.mean().round(3).to_dict(),
        "confundida_con": {f: g.pred_fruit.value_counts().to_dict() for f, g in unk.groupby("fruta")},
        "manzanas_propias_rechazadas": float(own.rechazada.mean()),
        "confianza_media": {"validacion": float(conf_val.mean()),
                            "manzanas_propias": float(own.confianza.mean()),
                            "desconocidas": float(unk.confianza.mean())},
    }
    (RESULTS / f"otras_frutas_{model}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    curves = {"val": conf_val, "own": own.confianza.to_numpy(), "unk": unk.confianza.to_numpy(), "tau": tau}
    return res, curves


def figure(curves: dict) -> None:
    """Un panel por modelo con la confianza de fruta en los tres conjuntos."""
    plotting.setup()
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n / (1 if n == 1 else 1.45), 2.1), sharey=True, squeeze=False)
    bins = np.linspace(0, 1, 26)
    for ax, (model, c) in zip(axes[0], curves.items()):
        ax.hist(c["val"], bins, density=True, alpha=0.5, color="#8d99ae", label="Validación (Kaggle)")
        ax.hist(c["own"], bins, density=True, alpha=0.6, color="#2a9d8f", label="Manzanas propias")
        ax.hist(c["unk"], bins, density=True, alpha=0.6, color="#c4553a", label="Frutas desconocidas")
        ax.axvline(c["tau"], color="k", ls="--", lw=1)
        ax.text(c["tau"], ax.get_ylim()[1] * 0.92, f" τ = {c['tau']:.2f}", fontsize=6)
        ax.set_title(NAMES.get(model, model), fontsize=7)
        ax.set_xlabel("máx P(fruta | x)", fontsize=7)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0,25", "0,5", "0,75", "1"])
    axes[0][0].set_ylabel("Densidad")
    axes[0][0].legend(fontsize=5, loc="upper left")
    fig.tight_layout(pad=0.2)
    plotting.save(fig, "res_otras_frutas")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None,
                    help="por defecto: cnn, y cnn_own si existe su checkpoint (src.training.finetune_own --final)")
    args = ap.parse_args()
    models = args.models or ["cnn"] + (["cnn_own"] if (CHECKPOINTS / "cnn_own.pt").exists() else [])
    curves = {}
    for m in models:
        res, curves[m] = run(m)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    figure(curves)


if __name__ == "__main__":
    main()
