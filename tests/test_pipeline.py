"""Pruebas unitarias de las piezas del pipeline que no requieren el dataset."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.manifest import parse_capture
from src.data.split import keep_one_per_capture, make_splits
from src.evaluation.metrics import CLASSES, evaluate, p_fruit, p_good
from src.features.handcrafted import extract, feature_names
from src.features.size import categorize, estimate_size, normalized_diameter
from src.models.cnn import FruitCNN
from src.models.grading import GradeThresholds, assign_grades, fit_thresholds


# ---------- datos ----------
@pytest.mark.parametrize("name,expected", [
    ("20190820_143522.jpg", "ts:20190820_143522"),
    ("20190820_143522_22896.jpg", "ts:20190820_143522"),
    ("IMG_20190824_175206_3.jpg", "ts:20190824_175206"),
    ("IMG20200728175856_01.jpg", "ts:20200728_175856"),
    ("IMG_1234.JPG", "num:1234"),
])
def test_parse_capture_agrupa_copias(name, expected):
    assert parse_capture(name)[0] == expected


def _fake_manifest(n_groups=60):
    rows = []
    for fruit in ("apple", "lime"):
        for q in ("good", "bad"):
            for g in range(n_groups):
                for k in range(3):
                    rows.append({"path": f"{fruit}/{q}/{g}_{k}.jpg", "fruit": fruit, "quality": q,
                                 "folder": f"{q}/{fruit}", "capture": f"c{g}_{k}",
                                 "group": f"{q}/{fruit}#{g}", "md5": f"{fruit}{q}{g}{k}"})
    return pd.DataFrame(rows)


def test_split_sin_fuga_entre_conjuntos():
    s = make_splits(_fake_manifest())
    assert set(s["split"]) == {"train", "val", "test"}
    assert s.groupby("group")["split"].nunique().max() == 1


def test_keep_one_per_capture_conserva_original():
    df = pd.DataFrame({"folder": ["a"] * 3, "capture": ["x"] * 3,
                       "path": ["a/x_123.jpg", "a/x.jpg", "a/x_1.jpg"]})
    assert keep_one_per_capture(df)["path"].tolist() == ["a/x.jpg"]


# ---------- métricas ----------
def test_marginalizacion_suma_uno():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(12), size=50)
    assert np.allclose(p_fruit(proba).sum(1), 1)
    assert np.all((p_good(proba) >= 0) & (p_good(proba) <= 1))


def test_evaluate_prediccion_perfecta():
    y = np.arange(12)
    m = evaluate(y, np.eye(12))
    assert m["joint_f1_macro"] == 1 and m["quality_accuracy"] == 1 and m["fruit_accuracy"] == 1
    assert CLASSES[1] == "apple_good"


# ---------- grados ----------
def test_grados_respetan_umbrales():
    th = GradeThresholds(low=0.2, high=0.8, target_precision=0.95)
    g = assign_grades(np.array([0.1, 0.5, 0.9, 0.2, 0.8]), th)
    assert g.tolist() == ["Descarte", "Segunda", "Primera", "Descarte", "Primera"]


def test_fit_thresholds_alcanza_precision_objetivo():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 2000), 0, 1)
    th = fit_thresholds(p, y, target=0.95)
    assert th.low <= th.high
    assert y[p >= th.high].mean() >= 0.95
    assert (1 - y[p <= th.low]).mean() >= 0.95


# ---------- características y tamaño ----------
def _disk_image(radius=40, size=128):
    img = np.full((size, size, 3), 230, np.uint8)
    yy, xx = np.mgrid[:size, :size]
    img[(yy - 64) ** 2 + (xx - 64) ** 2 <= radius ** 2] = (200, 40, 30)
    return img


def test_extract_dimension():
    assert extract(_disk_image()).shape == (len(feature_names()),)


def test_diametro_normalizado_de_un_circulo():
    r = estimate_size(_disk_image(radius=40))
    assert r.d_norm == pytest.approx(80 / 128, abs=0.03)


def test_categorias_de_tamano():
    assert categorize(0.1, "orange") == "pequeño"
    assert categorize(0.47, "orange") == "mediano"
    assert categorize(0.9, "orange") == "grande"
    assert normalized_diameter(np.ones((10, 10), bool)) == pytest.approx(2 * np.sqrt(100 / np.pi) / 10)


# ---------- CNN ----------
def test_cnn_salida():
    out = FruitCNN()(torch.zeros(4, 3, 128, 128))
    assert out.shape == (4, 12)


# ---------- fotos propias ----------
def test_own_labels_segunda_no_se_entrena():
    from src.evaluation.metrics import CLASS_TO_ID
    from src.training.finetune_own import own_labels
    df = pd.DataFrame({"fruta": ["apple", "apple", "banana"],
                       "calidad": ["Primera", "Segunda", "Descarte"]})
    assert own_labels(df).tolist() == [CLASS_TO_ID["apple_good"], -1, CLASS_TO_ID["banana_bad"]]


@pytest.mark.parametrize("folder,csv", [("images", "labels.csv"), ("otras_frutas", "otras_frutas.csv")])
def test_fotos_propias_coinciden_con_su_csv(folder, csv):
    from src.utils.paths import OWN
    df = pd.read_csv(OWN / csv)
    en_disco = {p.name for p in (OWN / folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    assert set(df["archivo"]) == en_disco
    assert not df.isna().any().any()
