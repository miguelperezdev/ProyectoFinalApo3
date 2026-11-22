"""Punto de entrada del proyecto: ejecuta las etapas del pipeline en orden.

Cada etapa corresponde a una fase de CRISP-DM y se puede correr por separado
con ``python -m <módulo>``; este script solo las encadena.

Uso:
    python -m src.main                 # todas las etapas, de los datos a la evaluación
    python -m src.main datos           # solo preparación (fase 3)
    python -m src.main ml cnn          # solo entrenamiento (fase 4)
    python -m src.main evaluacion      # solo evaluación (fase 5)
    python -m src.main desconocidas    # prueba de fruta desconocida con las fotos propias
    python -m src.main ajuste          # ajuste fino de la CNN con las fotos propias (~20 min)
    python -m src.main --listar        # muestra las etapas y sus comandos
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

PY = sys.executable

STAGES: dict[str, list[list[str]]] = {
    "descarga": [["src.data.download"]],
    "datos": [
        ["src.data.manifest"],
        ["src.data.split"],
        ["src.data.preprocess"],
        ["src.features.handcrafted"],
        ["src.data.segment_mixed"],
    ],
    "ml": [
        ["src.training.train_ml"],
        ["src.training.train_ml", "--models", "logreg", "rf", "--features", "background"],
    ],
    "cnn": [["src.training.train_cnn", "--tag", "cnn", "--lr", "1e-3", "--dropout", "0.3",
             "--epochs", "35", "--patience", "8"]],
    "evaluacion": [["src.evaluation.evaluate"]],
    # Pruebas con las fotos propias (no forman parte de la ejecución por defecto)
    "desconocidas": [["src.evaluation.unknown_fruits"]],
    "ajuste": [["src.training.finetune_own", "--final"]],
}
DEFAULT = ["datos", "ml", "cnn", "evaluacion"]


def run(stage: str) -> None:
    for module, *args in STAGES[stage]:
        cmd = [PY, "-m", module, *args]
        print(f"\n▶ [{stage}] {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        subprocess.run(cmd, check=True)
        print(f"✓ {module} ({time.time() - t0:.0f} s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapas", nargs="*", choices=list(STAGES), metavar="ETAPA",
                    help="etapas a ejecutar (por defecto: datos, ml, cnn y evaluacion)")
    ap.add_argument("--listar", action="store_true", help="muestra las etapas sin ejecutarlas")
    args = ap.parse_args()
    if args.listar:
        for name, cmds in STAGES.items():
            for module, *extra in cmds:
                print(f"{name:12s} python -m {module} {' '.join(extra)}".rstrip())
        return
    for stage in args.etapas or DEFAULT:
        run(stage)


if __name__ == "__main__":
    main()
