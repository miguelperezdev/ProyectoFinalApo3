"""Evaluación final y comparación de modelos (CRISP-DM fase 5).

Genera en ``experiments/results/`` y ``reports/figures/``:
- tabla comparativa de modelos (train / val / test) con intervalos de confianza
  bootstrap del 95 % para el F1-macro de prueba;
- prueba de McNemar entre el mejor modelo clásico y la CNN;
- umbrales de grado (Primera / Segunda / Descarte) ajustados en validación y
  su desempeño en prueba, y la curva precisión–cobertura;
- matrices de confusión, curvas de aprendizaje de la CNN, ejemplos de error,
  Grad-CAM y su energía en el borde;
- predicciones sobre los recortes de *Mixed* y, si existen, sobre las fotos
  propias anotadas (``data/own/labels.csv``) y las anotaciones de recortes.

Uso:
    python -m src.evaluation.evaluate
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score

from src.data.preprocess import load_split
from src.evaluation.gradcam import GradCAM
from src.evaluation.metrics import CLASSES, FRUIT_ES, FRUITS, encode, p_fruit, p_good
from src.features.handcrafted import border_mask
from src.models.grading import GRADES, assign_grades, fit_thresholds, grade_report
from src.models.predictor import Predictor
from src.utils import plotting
from src.utils.paths import FIGURES, INTERIM, OWN, RESULTS, SPLITS_PATH

MODELS = ["logreg", "svm", "rf", "xgb", "cnn"]
NAMES = {"logreg": "Reg. logística", "svm": "SVM-RBF", "rf": "Random Forest", "xgb": "XGBoost",
         "cnn": "CNN", "logreg_background": "Reg. log. (solo fondo)", "rf_background": "RF (solo fondo)"}
CLASSES_ES = [f"{FRUIT_ES[c.split('_')[0]]}-{'B' if c.endswith('good') else 'M'}" for c in CLASSES]
SEED = 42
TARGET_PRECISION = 0.98  # meta de precisión de Primera y Descarte en validación


def available(models=MODELS) -> list[str]:
    return [m for m in models if (RESULTS / f"metrics_{m}.json").exists()]


def load_proba(model: str, split: str) -> np.ndarray:
    return np.load(RESULTS / f"proba_{model}_{split}.npy")


def bootstrap_f1(y, pred, n=1000, seed=SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y)))
    vals = [f1_score(y[i], pred[i], average="macro") for i in idx]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Prueba exacta de McNemar: ¿los dos modelos tienen la misma tasa de error?"""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"solo_A_acierta": b, "solo_B_acierta": c, "p_value": float(p)}


# ---------------------------------------------------------------- tablas
def comparison_table(models, y_test) -> pd.DataFrame:
    rows = []
    for m in models + available(["logreg_background", "rf_background"]):
        r = json.loads((RESULTS / f"metrics_{m}.json").read_text())
        row = {"modelo": NAMES[m], "id": m,
               "F1 train": r["train"]["joint_f1_macro"],
               "F1 CV": r.get("cv_f1_macro", np.nan),
               "F1 val": r["val"]["joint_f1_macro"], "F1 test": r["test"]["joint_f1_macro"],
               "Acc. calidad test": r["test"]["quality_accuracy"],
               "AUC calidad test": r["test"]["quality_roc_auc"],
               "Acc. fruta test": r["test"]["fruit_accuracy"],
               "ms/imagen": r["test"]["ms_per_image"]}
        pred = load_proba(m, "test").argmax(1)
        row["IC95 F1 test"] = "[{:.3f}, {:.3f}]".format(*bootstrap_f1(y_test, pred))
        row["brecha train-test"] = row["F1 train"] - row["F1 test"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figuras
def fig_comparison(tab: pd.DataFrame) -> None:
    t = tab[~tab["id"].str.contains("background")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    x = np.arange(len(t))
    for k, (col, lab) in enumerate([("F1 train", "train"), ("F1 val", "val"), ("F1 test", "test")]):
        axes[0].bar(x + (k - 1) * 0.27, t[col], 0.27, label=lab,
                    color=["#d0d5dd", "#98a2b3", "#344054"][k])
    axes[0].set_xticks(x, t["modelo"], fontsize=7, rotation=15)
    axes[0].set_ylim(0.6, 1.1)
    axes[0].set_yticks(np.arange(0.6, 1.01, 0.1))
    axes[0].set_ylabel("F1-macro (12 clases)")
    axes[0].legend(frameon=False, ncol=3, fontsize=7, loc="upper center")
    axes[0].set_title("Desempeño y sobreajuste")
    axes[1].bar(x, t["Acc. calidad test"], color=[plotting.MODEL_COLORS[i] for i in t["id"]])
    for xi, v in zip(x, t["Acc. calidad test"]):
        axes[1].text(xi, v + 0.003, f"{v:.3f}", ha="center", fontsize=7)
    bg = tab[tab["id"] == "rf_background"]
    if len(bg):
        axes[1].axhline(bg["Acc. calidad test"].iloc[0], ls="--", color="#c4553a", lw=1)
        axes[1].text(-0.45, bg["Acc. calidad test"].iloc[0] - 0.012, "solo fondo (RF)",
                     ha="left", fontsize=7, color="#c4553a")
    axes[1].set_xticks(x, t["modelo"], fontsize=7, rotation=15)
    axes[1].set_ylim(0.8, 1.0)
    axes[1].set_ylabel("exactitud buena/mala (test)")
    axes[1].set_title("Calidad binaria")
    for a in axes:
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    plotting.save(fig, "res_comparacion_modelos")
    plt.close(fig)


def fig_confusion(model: str, y_test: np.ndarray) -> None:
    pred = load_proba(model, "test").argmax(1)
    cm = confusion_matrix(y_test, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 6},
                xticklabels=CLASSES_ES, yticklabels=CLASSES_ES, vmin=0, vmax=1)
    ax.set_xlabel("predicción")
    ax.set_ylabel("real")
    ax.tick_params(labelsize=6.5)
    ax.set_title(f"Matriz de confusión normalizada · {NAMES[model]} (test)")
    plotting.save(fig, f"res_confusion_{model}")
    plt.close(fig)


def fig_grades(models, splits_meta) -> pd.DataFrame:
    y_val = encode(splits_meta["val"]["label"])
    y_test = encode(splits_meta["test"]["label"])
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    targets = np.linspace(0.90, 0.995, 40)
    for m in models:
        pv, pt = p_good(load_proba(m, "val")), p_good(load_proba(m, "test"))
        th = fit_thresholds(pv, y_val % 2, TARGET_PRECISION)
        (RESULTS / f"grading_{m}.json").write_text(json.dumps(th.to_dict(), indent=2))
        rep = grade_report(pt, y_test % 2, th)
        rows.append({"modelo": NAMES[m], "id": m, "tau_bajo": th.low, "tau_alto": th.high, **rep})
        cov, prec = [], []
        for t in targets:
            th_t = fit_thresholds(pv, y_val % 2, t)
            r = grade_report(pt, y_test % 2, th_t)
            cov.append(r["coverage"])
            g = assign_grades(pt, th_t)
            firm = g != "Segunda"
            ok = ((g == "Primera") & (y_test % 2 == 1)) | ((g == "Descarte") & (y_test % 2 == 0))
            prec.append(ok[firm].mean() if firm.any() else np.nan)
        axes[0].plot(cov, prec, label=NAMES[m], color=plotting.MODEL_COLORS[m], lw=1.4)
    axes[0].set_xlabel("cobertura (fracción con veredicto firme)")
    axes[0].set_ylabel("exactitud de Primera + Descarte")
    axes[0].set_title("Compromiso error–rechazo (test)")
    axes[0].legend(frameon=False, fontsize=7)

    best = max(rows, key=lambda r: r["coverage"])["id"]
    pt = p_good(load_proba(best, "test"))
    th = json.loads((RESULTS / f"grading_{best}.json").read_text())
    bins = np.linspace(0, 1, 41)
    axes[1].hist(pt[y_test % 2 == 1], bins, alpha=0.75, color=plotting.QUALITY_COLORS["good"], label="buena (real)")
    axes[1].hist(pt[y_test % 2 == 0], bins, alpha=0.75, color=plotting.QUALITY_COLORS["bad"], label="mala (real)")
    for t in (th["low"], th["high"]):
        axes[1].axvline(t, color="#344054", ls="--", lw=1)
    axes[1].text(th["low"] / 2, axes[1].get_ylim()[1] * 0.9, "Descarte", ha="center", fontsize=7)
    axes[1].text((th["low"] + th["high"]) / 2, axes[1].get_ylim()[1] * 0.9, "Segunda", ha="center", fontsize=7)
    axes[1].text((1 + th["high"]) / 2, axes[1].get_ylim()[1] * 0.9, "Primera", ha="center", fontsize=7)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("P(buena | x)")
    axes[1].set_ylabel("imágenes (escala log)")
    axes[1].set_title(f"Grados · {NAMES[best]} (test)")
    axes[1].legend(frameon=False, fontsize=7, loc="center")
    fig.tight_layout()
    plotting.save(fig, "res_grados")
    plt.close(fig)
    return pd.DataFrame(rows)


def fig_learning_curve() -> None:
    p = RESULTS / "history_cnn.csv"
    if not p.exists():
        return
    h = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    axes[0].plot(h.epoch, h.train_loss, label="train", color="#98a2b3")
    axes[0].plot(h.epoch, h.val_loss, label="validación", color="#7b2cbf")
    axes[0].set_ylabel("entropía cruzada")
    axes[1].plot(h.epoch, h.train_acc, label="train (con aumento)", color="#98a2b3")
    axes[1].plot(h.epoch, h.val_acc, label="val (exactitud)", color="#7b2cbf")
    axes[1].plot(h.epoch, h.val_f1_macro, label="val (F1-macro)", color="#7b2cbf", ls="--")
    axes[1].set_ylabel("métrica")
    best = h.loc[h.val_f1_macro.idxmax(), "epoch"]
    for a in axes:
        a.axvline(best, color="#c4553a", ls=":", lw=1)
        a.set_xlabel("época")
        a.legend(frameon=False, fontsize=7)
    fig.suptitle("Curvas de aprendizaje de la CNN (línea punteada: mejor época)", fontsize=9)
    fig.tight_layout()
    plotting.save(fig, "res_curvas_cnn")
    plt.close(fig)


def gradcam_analysis(Xte, mte) -> dict:
    pred = Predictor("cnn")
    cam = GradCAM(pred.net)
    x = pred.to_tensor(Xte)
    cams = np.concatenate([cam(x[i:i + 128]) for i in range(0, len(x), 128)])
    cams = torch.nn.functional.interpolate(torch.from_numpy(cams)[:, None], size=Xte.shape[1:3],
                                           mode="bilinear", align_corners=False)[:, 0].numpy()
    ring = border_mask(128, 16)
    border_energy = (cams * ring).sum((1, 2)) / cams.sum((1, 2)).clip(1e-8)
    rng = np.random.default_rng(1)
    idx = []
    for f in FRUITS:
        for q in ("good", "bad"):
            idx.append(rng.choice(np.where(mte["label"] == f"{f}_{q}")[0]))
    fig, axes = plt.subplots(2, 12, figsize=(7.2, 1.65))
    for j, i in enumerate(idx):
        axes[0, j].imshow(Xte[i])
        axes[1, j].imshow(Xte[i])
        axes[1, j].imshow(cams[i], cmap="jet", alpha=0.45)
        axes[0, j].set_title(CLASSES_ES[CLASS_ID(mte["label"][i])], fontsize=5.5)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle("Grad-CAM de la CNN (fila inferior): regiones que más influyen en la predicción", fontsize=7)
    fig.tight_layout(pad=0.2)
    plotting.save(fig, "res_gradcam")
    plt.close(fig)
    return {"border_ring_area_fraction": float(ring.mean()),
            "mean_border_energy": float(border_energy.mean()),
            "median_border_energy": float(np.median(border_energy)),
            "frac_images_border_energy_gt_area": float((border_energy > ring.mean()).mean())}


def CLASS_ID(label: str) -> int:
    return CLASSES.index(label)


def fig_errors(model: str, Xte, mte) -> None:
    proba = load_proba(model, "test")
    y = encode(mte["label"])
    q_true, pg = y % 2, p_good(proba)
    wrong = np.where((pg >= 0.5).astype(int) != q_true)[0]
    conf = np.abs(pg[wrong] - 0.5)
    # Un ejemplo por ráfaga para no repetir casi la misma foto
    ranked = pd.Series(wrong[np.argsort(-conf)])
    pick = ranked[~mte["group"].iloc[ranked].duplicated().values].iloc[:12].to_numpy()
    fig, axes = plt.subplots(2, 6, figsize=(7.2, 3.3))
    for a, i in zip(axes.ravel(), pick):
        a.imshow(Xte[i])
        a.set_title(f"{FRUIT_ES[mte['fruit'][i]]} · real: {'B' if q_true[i] else 'M'}\nP(buena)={pg[i]:.2f}",
                    fontsize=6)
        a.axis("off")
    fig.suptitle(f"Errores de calidad más confiados · {NAMES[model]} (test)", fontsize=8)
    fig.tight_layout()
    plotting.save(fig, f"res_errores_{model}")
    plt.close(fig)


def per_fruit_table(models, mte) -> pd.DataFrame:
    y = encode(mte["label"])
    rows = []
    for m in models:
        pg = p_good(load_proba(m, "test"))
        ok = (pg >= 0.5).astype(int) == y % 2
        for f in FRUITS:
            sel = (y // 2) == FRUITS.index(f)
            rows.append({"modelo": NAMES[m], "fruta": FRUIT_ES[f], "acc_calidad": ok[sel].mean()})
    return pd.DataFrame(rows).pivot(index="fruta", columns="modelo", values="acc_calidad")


def mixed_crops_eval(model: str) -> dict:
    ann_path = INTERIM / "mixed_crops" / "annotation.csv"
    if not ann_path.exists():
        return {}
    crops = pd.read_csv(ann_path)
    pred = Predictor(model)
    out = []
    for c in crops["crop"]:
        p = pred.predict(Image.open(INTERIM / "mixed_crops" / c))
        out.append({"crop": c, "pred_fruit": p.fruit, "p_good": p.p_good, "pred_grade": p.grade})
    df = crops.merge(pd.DataFrame(out), on="crop")
    df.to_csv(RESULTS / f"mixed_crops_pred_{model}.csv", index=False)
    res = {"n_crops": len(df), "fruit_accuracy_vs_folder": float((df.fruit == df.pred_fruit).mean()),
           "grade_distribution": df.pred_grade.value_counts(normalize=True).round(3).to_dict()}
    manual = OWN / "mixed_annotations.csv"
    if manual.exists():
        a = pd.read_csv(manual)
        a = a[a.grade != "Inválido"].merge(df[["crop", "pred_grade", "p_good"]], on="crop")
        if len(a):
            res["n_annotated_valid"] = len(a)
            res["grade_agreement"] = float((a.grade == a.pred_grade).mean())
            res["grade_confusion"] = pd.crosstab(a.grade, a.pred_grade).to_dict()
    return res


def fig_mixed(model: str) -> None:
    """Recortes de Mixed (sin etiqueta) con el grado predicho: 4 al azar de cada grado."""
    path = RESULTS / f"mixed_crops_pred_{model}.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    pick = pd.concat([df[df.pred_grade == g].sample(min(4, (df.pred_grade == g).sum()), random_state=3)
                      for g in GRADES])
    fig, axes = plt.subplots(1, len(pick), figsize=(7.2, 1.25))
    for a, (_, r) in zip(np.atleast_1d(axes), pick.iterrows()):
        a.imshow(Image.open(INTERIM / "mixed_crops" / r.crop).convert("RGB").resize((128, 128)))
        a.set_title(f"{r.pred_grade}\nP={r.p_good:.2f}", fontsize=6,
                    color={"Primera": "#1f7a6f", "Segunda": "#9a7400", "Descarte": "#b03a2e"}[r.pred_grade])
        a.axis("off")
    fig.suptitle(f"Recortes de Mixed sin etiqueta y grado asignado por {NAMES[model]}", fontsize=7)
    fig.tight_layout(pad=0.2)
    plotting.save(fig, f"res_mixed_{model}")
    plt.close(fig)


def own_photos_eval(model: str) -> dict:
    labels = OWN / "labels.csv"
    df = pd.read_csv(labels) if labels.exists() else pd.DataFrame()
    if df.empty:
        return {"n": 0, "nota": "Aún no hay fotos propias anotadas en data/own/labels.csv"}
    pred = Predictor(model)
    rows = []
    for _, r in df.iterrows():
        p = pred.predict(Image.open(OWN / "images" / r["archivo"]))
        rows.append({**r.to_dict(), "pred_fruit": p.fruit, "p_good": p.p_good, "pred_grade": p.grade,
                     "pred_size": p.size, "d_norm": p.d_norm})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"own_photos_pred_{model}.csv", index=False)
    res = {"n": len(out), "fruit_accuracy": float((out.fruta == out.pred_fruit).mean())}
    if "calidad" in out:
        res["grade_agreement"] = float((out.calidad == out.pred_grade).mean())
    if "tamano" in out:
        norm = out.pred_size.str.replace("ñ", "n")
        res["size_agreement"] = float((out.tamano == norm).mean())
    return res


def main() -> None:
    plotting.setup()
    splits = pd.read_csv(SPLITS_PATH)
    Xte, mte = load_split("test", splits)
    _, mva = load_split("val", splits)
    y_test = encode(mte["label"])
    models = available()
    print("Modelos:", models)

    tab = comparison_table(models, y_test)
    tab.to_csv(RESULTS / "comparacion_modelos.csv", index=False)
    print(tab.drop(columns=["id"]).round(3).to_string(index=False))
    fig_comparison(tab)

    main_models = [m for m in models if not m.endswith("background")]
    best_ml = max([m for m in main_models if m != "cnn"], key=lambda m: tab.set_index("id").loc[m, "F1 val"])
    summary = {"best_classical": best_ml}
    for m in {best_ml, "cnn"} & set(models):
        fig_confusion(m, y_test)
        fig_errors(m, Xte, mte)

    grades = fig_grades(main_models, {"val": mva, "test": mte})
    grades.to_csv(RESULTS / "grados_test.csv", index=False)
    print(grades.round(3).to_string(index=False))
    pf = per_fruit_table(main_models, mte)
    pf.to_csv(RESULTS / "calidad_por_fruta.csv")
    print(pf.round(3))

    if "cnn" in models:
        fig_learning_curve()
        q_true = y_test % 2
        ok_cnn = (p_good(load_proba("cnn", "test")) >= 0.5) == q_true
        ok_ml = (p_good(load_proba(best_ml, "test")) >= 0.5) == q_true
        summary["mcnemar_quality_cnn_vs_" + best_ml] = mcnemar(ok_cnn, ok_ml)
        j_cnn = load_proba("cnn", "test").argmax(1) == y_test
        j_ml = load_proba(best_ml, "test").argmax(1) == y_test
        summary["mcnemar_joint_cnn_vs_" + best_ml] = mcnemar(j_cnn, j_ml)
        summary["gradcam"] = gradcam_analysis(Xte, mte)
        summary["mixed_crops_cnn"] = mixed_crops_eval("cnn")
        fig_mixed("cnn")
        summary["own_photos_cnn"] = own_photos_eval("cnn")
    (RESULTS / "resumen_evaluacion.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    print("Figuras en", FIGURES)


if __name__ == "__main__":
    main()
