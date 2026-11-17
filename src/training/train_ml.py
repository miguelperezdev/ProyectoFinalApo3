"""Entrenamiento y ajuste de hiperparámetros de los modelos de ML clásico.

Protocolo:
- Búsqueda en rejilla (``GridSearchCV``) con validación cruzada de 5 particiones
  **agrupadas y estratificadas** sobre el conjunto de entrenamiento. Agrupar es
  necesario: si la misma ráfaga de fotos cae en el pliegue de entrenamiento y en
  el de validación, la CV sobreestima el desempeño.
- Criterio de selección: F1-macro sobre las 12 clases (fruta × calidad).
- El mejor modelo se reentrena en todo el conjunto de entrenamiento y se evalúa
  en validación y en prueba. El conjunto de prueba no se usa para ninguna
  decisión.
- Desbalance: pesos de clase w_k = n / (K · n_k) (``class_weight="balanced"``).

Modelos:
- ``logreg``: regresión logística multinomial (línea base lineal).
- ``svm``: máquina de soporte vectorial con kernel RBF.
- ``rf``: Random Forest.
- ``xgb``: XGBoost (gradient boosting de árboles).

Uso:
    python -m src.training.train_ml                  # todos los modelos
    python -m src.training.train_ml --models svm rf
    python -m src.training.train_ml --features background   # experimento de sesgo de fondo
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.evaluation.metrics import encode, evaluate
from src.features.handcrafted import features_path
from src.utils.paths import CHECKPOINTS, RESULTS, SPLITS_PATH

SEED = 42


def model_specs() -> dict:
    return {
        "logreg": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
            {"clf__C": [0.01, 0.1, 1.0]},
        ),
        "svm": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", SVC(kernel="rbf", class_weight="balanced"))]),
            {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.001]},
        ),
        "rf": (
            RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=1,
                                   random_state=SEED),
            {"max_features": ["sqrt", 0.1], "min_samples_leaf": [1, 3]},
        ),
        "xgb": (
            XGBClassifier(n_estimators=400, subsample=0.8, colsample_bytree=0.5,
                          tree_method="hist", n_jobs=4, random_state=SEED, eval_metric="mlogloss"),
            {"max_depth": [4, 6], "learning_rate": [0.05, 0.1]},
        ),
    }


def load_xy(split: str, kind: str, splits: pd.DataFrame):
    X = np.load(features_path(split, kind))["X"]
    idx = np.load(features_path(split, kind).with_name(f"images_{split}.npz"))["index"]
    meta = splits.loc[idx].reset_index(drop=True)
    return X, meta


def train_one(name: str, kind: str, splits: pd.DataFrame) -> dict:
    estimator, grid = model_specs()[name]
    Xtr, mtr = load_xy("train", kind, splits)
    ytr = encode(mtr["label"])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit_params = {}
    if name == "xgb":
        fit_params["sample_weight"] = compute_sample_weight("balanced", ytr)

    t0 = time.time()
    search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=cv, n_jobs=-1 if name != "xgb" else 4,
                          return_train_score=True, verbose=0)
    search.fit(Xtr, ytr, groups=mtr["group"], **fit_params)
    t_search = time.time() - t0

    best = search.best_estimator_
    if name == "svm":  # probabilidades calibradas (Platt) solo para el modelo final
        best.set_params(clf__probability=True, clf__random_state=SEED)
        best.fit(Xtr, ytr)

    tag = f"{name}" if kind == "full" else f"{name}_{kind}"
    out = {"model": tag, "features": kind, "best_params": search.best_params_,
           "cv_f1_macro": search.best_score_, "search_seconds": round(t_search, 1)}
    cvres = pd.DataFrame(search.cv_results_)
    keep = [c for c in cvres.columns if c.startswith("param_") or c in
            ("mean_test_score", "std_test_score", "mean_train_score", "std_train_score", "mean_fit_time")]
    cvres[keep].to_csv(RESULTS / f"cv_{tag}.csv", index=False)

    out["train"] = evaluate(ytr, best.predict_proba(Xtr))
    for split in ("val", "test"):
        X, m = load_xy(split, kind, splits)
        y = encode(m["label"])
        t0 = time.time()
        proba = best.predict_proba(X)
        ms = 1000 * (time.time() - t0) / len(X)
        out[split] = evaluate(y, proba)
        out[split]["ms_per_image"] = ms
        np.save(RESULTS / f"proba_{tag}_{split}.npy", proba)
    if kind == "full":
        X, _ = load_xy("mixed", kind, splits)
        np.save(RESULTS / f"proba_{tag}_mixed.npy", best.predict_proba(X))
        joblib.dump(best, CHECKPOINTS / f"{tag}.joblib")
    with open(RESULTS / f"metrics_{tag}.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(model_specs()))
    ap.add_argument("--features", default="full", choices=["full", "background"])
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(SPLITS_PATH)
    for name in args.models:
        r = train_one(name, args.features, splits)
        print(f"[{r['model']}] cv={r['cv_f1_macro']:.3f} params={r['best_params']} "
              f"val_f1={r['val']['joint_f1_macro']:.3f} val_q_acc={r['val']['quality_accuracy']:.3f} "
              f"({r['search_seconds']}s)")


if __name__ == "__main__":
    main()
