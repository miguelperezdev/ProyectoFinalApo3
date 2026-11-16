"""Rutas centralizadas del proyecto."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW_DATASET = DATA / "raw" / "fruit-quality-classification"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
OWN = DATA / "own"

MANIFEST_PATH = INTERIM / "manifest.csv"
SPLITS_PATH = PROCESSED / "splits.csv"

EXPERIMENTS = ROOT / "experiments"
RESULTS = EXPERIMENTS / "results"
CHECKPOINTS = EXPERIMENTS / "checkpoints"
FIGURES = ROOT / "reports" / "figures"
