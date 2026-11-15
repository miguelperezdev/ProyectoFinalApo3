"""Partición train / val / test por grupos de captura y estratificada.

Decisiones:
1. Se conserva **una imagen por captura original** (se descartan las copias
   aumentadas del dataset). Las copias no aportan información nueva, inflan
   artificialmente la clase granada-buena (5.940 imágenes para 1.154 fotos) y
   causarían fuga de información si caen en conjuntos distintos.
2. La partición se hace con ``StratifiedGroupKFold``: ningún grupo (ráfaga de la
   misma fruta) se reparte entre conjuntos, y la proporción de las 12 clases
   (fruta × calidad) se mantiene aproximadamente igual en los tres conjuntos.
3. La carpeta *Mixed* no se usa para entrenar: queda como conjunto de
   evaluación externo (fotos con varias frutas y fondos distintos).

Uso:
    python -m src.data.split
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.paths import MANIFEST_PATH, SPLITS_PATH

SEED = 42
TEST_FOLDS = 7  # 1/7 ≈ 14 % para prueba
VAL_FOLDS = 6  # 1/6 del resto ≈ 14 % para validación


def keep_one_per_capture(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva una imagen por captura: la de nombre más corto (la original)."""
    df = df.assign(_len=df["path"].str.len())
    df = df.sort_values(["folder", "capture", "_len", "path"])
    return df.drop_duplicates(["folder", "capture"]).drop(columns="_len")


def _one_fold(df: pd.DataFrame, n_splits: int, seed: int) -> pd.Index:
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, idx = next(sgkf.split(df, df["label"], df["group"]))
    return df.index[idx]


def make_splits(manifest: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    df = keep_one_per_capture(manifest)
    labeled = df[df["quality"] != "mixed"].copy()
    labeled["label"] = labeled["fruit"] + "_" + labeled["quality"]

    test_idx = _one_fold(labeled, TEST_FOLDS, seed)
    rest = labeled.drop(test_idx)
    val_idx = _one_fold(rest, VAL_FOLDS, seed)

    labeled["split"] = "train"
    labeled.loc[test_idx, "split"] = "test"
    labeled.loc[val_idx, "split"] = "val"

    mixed = df[df["quality"] == "mixed"].copy()
    mixed["label"] = mixed["fruit"] + "_mixed"
    mixed["split"] = "mixed"
    return pd.concat([labeled, mixed]).reset_index(drop=True)


def main() -> None:
    splits = make_splits(pd.read_csv(MANIFEST_PATH))
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(SPLITS_PATH, index=False)

    lab = splits[splits["split"] != "mixed"]
    leaks = lab.groupby("group")["split"].nunique().gt(1).sum()
    print(pd.crosstab(splits["label"], splits["split"], margins=True))
    print(f"Grupos repartidos en más de un conjunto: {leaks}")


if __name__ == "__main__":
    main()
