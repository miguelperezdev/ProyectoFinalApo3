"""Ajuste fino de la CNN con datos combinados: dataset de Kaggle + fotos propias.

El enunciado pide entrenar con fuentes públicas **y** recolección propia. La CNN
final (``experiments/checkpoints/cnn.pt``) solo vio el dataset de Kaggle; con
las fotos propias acierta la fruta en el 83 %, pero el grado solo coincide en
el 38 % porque su P(buena) cae fuera del dominio de entrenamiento. Este script
mide si sumar las fotos propias al entrenamiento corrige ese cambio de dominio
sin estropear el desempeño en el dataset.

Diseño del experimento
----------------------
1. **Punto de partida:** los pesos de ``cnn.pt`` (no se entrena desde cero).
2. **Datos de entrenamiento:** todo el conjunto de entrenamiento de Kaggle más
   las fotos propias del pliegue de entrenamiento, repetidas ``--repeat`` veces
   para que pesen frente a ~8 000 imágenes. Se usa el mismo aumento de datos que
   en ``train_cnn.py``. Etiquetas: Primera → ``<fruta>_good`` y
   Descarte → ``<fruta>_bad``; Segunda (defecto leve) no tiene equivalente
   binario y solo se usa para evaluar.
3. **Validación cruzada de ``--folds`` pliegues sobre las fotos propias:** cada
   foto se evalúa con un modelo que no la vio. Así la cifra es comparable con
   la de la CNN original sobre las mismas 48 fotos.
4. **Control de regresión:** cada modelo ajustado se evalúa también en el
   conjunto de prueba de Kaggle (F1-macro, exactitud de calidad y de fruta).
5. **Umbrales de grado:** como en el resto del proyecto, se recalculan en la
   validación de Kaggle con meta π = 98 % (``fit_thresholds``).

Con ``--final`` se entrena además un modelo con **todas** las fotos propias y se
guarda como ``experiments/checkpoints/cnn_own.pt`` con sus umbrales
(``grading_cnn_own.json``). La app lo puede usar con ``Predictor("cnn_own")``.
**Nunca se sobrescriben** ``cnn.pt``, ``cnn_norm.json`` ni ``grading_cnn.json``.

Salidas:
- ``experiments/results/finetune_own.json``      resumen: CNN original vs. ajustada
- ``experiments/results/finetune_own_pred.csv``  predicción de cada foto en su pliegue

Tiempo medido en CPU (16 hilos): ~4 min por pliegue con 4 épocas, 21 min en
total con ``--final``.

Uso:
    python -m src.training.finetune_own                   # validación cruzada
    python -m src.training.finetune_own --final           # + modelo final cnn_own
    python -m src.training.finetune_own --max-kaggle 300 --epochs 1 --folds 2 --tag prueba
                                                          # prueba rápida del script (minutos)
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from src.data.preprocess import load_image, load_split
from src.evaluation.metrics import CLASS_TO_ID, FRUITS, encode, evaluate, p_fruit, p_good
from src.models.cnn import FruitCNN
from src.models.grading import assign_grades, fit_thresholds
from src.models.predictor import load_thresholds
from src.training.train_cnn import (ArrayDataset, _worker_init, eval_transform, predict_proba,
                                    train_transform)
from src.utils.paths import CHECKPOINTS, OWN, RESULTS, SPLITS_PATH

SEED = 42
QUALITY_TO_LABEL = {"Primera": "good", "Descarte": "bad"}  # Segunda: solo evaluación


def own_labels(df: pd.DataFrame) -> np.ndarray:
    """Clase conjunta de cada foto propia; −1 si no se usa para entrenar (Segunda)."""
    return np.array([CLASS_TO_ID[f"{r.fruta}_{QUALITY_TO_LABEL[r.calidad]}"]
                     if r.calidad in QUALITY_TO_LABEL else -1 for r in df.itertuples()])


def own_metrics(proba: np.ndarray, df: pd.DataFrame, th) -> dict:
    fruit = np.array(FRUITS)[p_fruit(proba).argmax(1)]
    pg = p_good(proba)
    grade = assign_grades(pg, th)
    return {"n": len(df),
            "fruit_accuracy": float((fruit == df.fruta.to_numpy()).mean()),
            "grade_agreement": float((grade == df.calidad.to_numpy()).mean()),
            "p_good_mean": float(pg.mean())}


class Runner:
    def __init__(self, args):
        self.args = args
        norm = json.loads((RESULTS / "cnn_norm.json").read_text())
        self.mean, self.std, self.size = norm["mean"], norm["std"], norm["size"]
        self.tt = train_transform(self.mean, self.std, self.size)
        self.ev = eval_transform(self.mean, self.std, self.size)
        splits = pd.read_csv(SPLITS_PATH)
        rng = np.random.default_rng(SEED)
        self.data = {}
        for split in ("train", "val", "test"):
            X, m = load_split(split, splits)
            y = encode(m["label"])
            if args.max_kaggle and len(X) > args.max_kaggle:  # solo para probar el script
                idx = rng.choice(len(X), args.max_kaggle, replace=False)
                X, y = X[idx], y[idx]
            self.data[split] = (X, y)
        self.own = pd.read_csv(OWN / "labels.csv")
        self.X_own = np.stack([load_image(OWN / "images" / f) for f in self.own.archivo])
        self.y_own = own_labels(self.own)

    def proba(self, model, X) -> np.ndarray:
        return predict_proba(model, DataLoader(ArrayDataset(X, np.zeros(len(X), int), self.ev),
                                               batch_size=256))

    def base_model(self) -> nn.Module:
        model = FruitCNN(dropout=self.args.dropout).to(memory_format=torch.channels_last)
        model.load_state_dict(torch.load(CHECKPOINTS / "cnn.pt", map_location="cpu"))
        return model

    def fine_tune(self, own_idx: np.ndarray, tag: str) -> nn.Module:
        a = self.args
        torch.manual_seed(SEED)
        Xtr, ytr = self.data["train"]
        own_idx = own_idx[self.y_own[own_idx] >= 0]
        Xo = np.repeat(self.X_own[own_idx], a.repeat, axis=0)
        yo = np.repeat(self.y_own[own_idx], a.repeat)
        ds = ConcatDataset([ArrayDataset(Xtr, ytr, self.tt), ArrayDataset(Xo, yo, self.tt)])
        loader = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                            persistent_workers=a.workers > 0, worker_init_fn=_worker_init)
        counts = np.bincount(ytr, minlength=12).clip(min=1)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(len(ytr) / (12 * counts), dtype=torch.float32))
        model = self.base_model()
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        for epoch in range(1, a.epochs + 1):
            model.train()
            t0, tot, n = time.time(), 0.0, 0
            for x, y in loader:
                opt.zero_grad()
                loss = criterion(model(x.contiguous(memory_format=torch.channels_last)), y)
                loss.backward()
                opt.step()
                tot, n = tot + loss.item() * len(y), n + len(y)
            sched.step()
            print(f"[{tag}] ép {epoch}/{a.epochs} pérdida {tot / n:.3f} "
                  f"({len(own_idx)} fotos propias × {a.repeat}, {time.time() - t0:.0f} s)", flush=True)
        return model

    def thresholds(self, model):
        Xva, yva = self.data["val"]
        return fit_thresholds(p_good(self.proba(model, Xva)), yva % 2)

    def test_metrics(self, model) -> dict:
        Xte, yte = self.data["test"]
        r = evaluate(yte, self.proba(model, Xte))
        return {k: float(r[k]) for k in ("joint_f1_macro", "quality_accuracy", "fruit_accuracy")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4, help="más baja que al entrenar desde cero")
    ap.add_argument("--repeat", type=int, default=20, help="veces que se repite cada foto propia")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-kaggle", type=int, default=None, help="submuestra de Kaggle (solo pruebas)")
    ap.add_argument("--final", action="store_true", help="entrena y guarda cnn_own con todas las fotos")
    ap.add_argument("--tag", default="finetune_own", help="prefijo de los archivos de salida")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t_start = time.time()
    r = Runner(args)
    print(f"Kaggle train {len(r.data['train'][0])} · fotos propias {len(r.own)} "
          f"({int((r.y_own >= 0).sum())} entrenables)", flush=True)

    base = r.base_model()
    base_th = load_thresholds("cnn")
    summary = {"params": vars(args),
               "original": {"own": own_metrics(r.proba(base, r.X_own), r.own, base_th),
                            "test": r.test_metrics(base)}}

    rows, tests = [], []
    folds = KFold(args.folds, shuffle=True, random_state=SEED).split(r.X_own)
    for k, (tr, te) in enumerate(folds, 1):
        model = r.fine_tune(tr, f"pliegue {k}/{args.folds}")
        th = r.thresholds(model)
        pr = r.proba(model, r.X_own[te])
        pg = p_good(pr)
        for i, j in enumerate(te):
            rows.append({"pliegue": k, **r.own.iloc[j].to_dict(),
                         "pred_fruit": FRUITS[p_fruit(pr[i:i + 1]).argmax()], "p_good": float(pg[i]),
                         "pred_grade": assign_grades(pg[i:i + 1], th)[0]})
        tests.append(r.test_metrics(model))
    pred = pd.DataFrame(rows).sort_values("archivo")
    pred.to_csv(RESULTS / f"{args.tag}_pred.csv", index=False)
    summary["ajustada_cv"] = {
        "own": {"n": len(pred),
                "fruit_accuracy": float((pred.pred_fruit == pred.fruta).mean()),
                "grade_agreement": float((pred.pred_grade == pred.calidad).mean()),
                "p_good_mean": float(pred.p_good.mean())},
        "test": {k: float(np.mean([t[k] for t in tests])) for k in tests[0]},
        "test_por_pliegue": tests,
    }

    if args.final:
        model = r.fine_tune(np.arange(len(r.own)), "final")
        name = "cnn_own" if args.tag == "finetune_own" else f"cnn_{args.tag}"
        torch.save(model.state_dict(), CHECKPOINTS / f"{name}.pt")
        th = r.thresholds(model)
        (RESULTS / f"grading_{name}.json").write_text(json.dumps(th.to_dict(), indent=2))
        summary["final"] = {"modelo": name, "umbrales": th.to_dict(), "test": r.test_metrics(model)}

    summary["minutos"] = round((time.time() - t_start) / 60, 1)
    (RESULTS / f"{args.tag}.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "params"}, indent=2, default=float))


if __name__ == "__main__":
    main()
