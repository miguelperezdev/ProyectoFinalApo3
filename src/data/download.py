"""Descarga el dataset *Fruit Quality Classification* de Kaggle y lo enlaza en data/raw/.

Fuente: Ryan Park, "Fruit Quality Classification", Kaggle, 2022 (licencia CC0).
https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
Origen original: Meshram & Patil, "FruitNet: Indian fruits dataset with quality for
machine learning applications", Data in Brief, vol. 40, 2022.

Uso:
    python -m src.data.download

No requiere token de Kaggle para este dataset público. Si Kaggle llegara a pedirlo,
coloque su token en ~/.kaggle/kaggle.json (nunca dentro del repositorio).
"""

from pathlib import Path

import kagglehub

DATASET = "ryandpark/fruit-quality-classification"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> Path:
    cache_path = Path(kagglehub.dataset_download(DATASET))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    link = RAW_DIR / "fruit-quality-classification"
    if not link.exists():
        link.symlink_to(cache_path, target_is_directory=True)
    print(f"Dataset disponible en: {link} -> {cache_path}")
    return link


if __name__ == "__main__":
    main()
