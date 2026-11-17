"""Entrenamiento de la CNN con aumento de datos y parada temprana.

Función de pérdida: entropía cruzada ponderada por clase,

    L(θ) = −(1/N) Σ_i w_{y_i} · log softmax(f_θ(x_i))_{y_i},   w_k = N / (K · N_k)

minimizada con AdamW (Adam con decaimiento de pesos desacoplado, que equivale a
una penalización L2 sobre θ) y tasa de aprendizaje con decaimiento coseno:

    η_t = η_0 · ½ (1 + cos(π t / T)).

Resolución de entrada: 96×96 (se reduce desde la caché de 128×128). Se midió
que entrenar a 128×128 en CPU tarda ~207 s por época frente a ~80 s a 96×96
con formato channels_last; 96 px conservan las manchas y golpes visibles.

Aumento de datos (solo en entrenamiento): recorte aleatorio redimensionado,
volteos, rotaciones y variaciones leves de brillo/contraste/saturación. El tono
casi no se altera (hue = 0.02) porque el color es justamente la señal de
calidad. El aumento obliga a la red a reconocer la fruta en distintas
posiciones, escalas e iluminaciones, lo que reduce la dependencia del fondo.

Parada temprana: se guarda el modelo con mejor F1-macro de validación y se
detiene el entrenamiento si no mejora en ``patience`` épocas.

Uso:
    python -m src.training.train_cnn --lr 1e-3 --dropout 0.4 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from src.data.preprocess import load_split
from src.evaluation.metrics import encode, evaluate
from src.models.cnn import FruitCNN, count_parameters
from src.utils.paths import CHECKPOINTS, RESULTS, SPLITS_PATH

SEED = 42
# Media y desviación por canal del conjunto de entrenamiento (se calculan una vez).
MEAN_STD_PATH = RESULTS / "cnn_norm.json"


def train_transform(mean, std, size=96):
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(25),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean, std),
    ])


def eval_transform(mean, std, size=96):
    return T.Compose([T.ToImage(), T.Resize(size, antialias=True),
                      T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)])


class ArrayDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), int(self.labels[i])


def _worker_init(_):
    # Cada proceso de carga de datos usa un solo hilo: si cada uno abriera tantos
    # hilos como núcleos, se sobresuscribe la CPU y el entrenamiento se vuelve ~100x
    # más lento (se observó una época de >4 h).
    torch.set_num_threads(1)


def channel_stats(images: np.ndarray) -> tuple[list, list]:
    x = images.reshape(-1, 3).astype(np.float64) / 255
    return x.mean(0).tolist(), x.std(0).tolist()


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    return torch.cat([torch.softmax(model(x.contiguous(memory_format=torch.channels_last)), 1)
                      for x, _ in loader]).numpy()


def run(args) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(args.threads)
    splits = pd.read_csv(SPLITS_PATH)
    Xtr, mtr = load_split("train", splits)
    Xva, mva = load_split("val", splits)
    ytr, yva = encode(mtr["label"]), encode(mva["label"])

    mean, std = channel_stats(Xtr)
    MEAN_STD_PATH.write_text(json.dumps({"mean": mean, "std": std, "size": args.size}))

    tr_loader = DataLoader(ArrayDataset(Xtr, ytr, train_transform(mean, std, args.size)),
                           batch_size=args.batch,
                           shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0,
                           worker_init_fn=_worker_init)
    ev = eval_transform(mean, std, args.size)
    va_loader = DataLoader(ArrayDataset(Xva, yva, ev), batch_size=256)

    # channels_last (NHWC) acelera las convoluciones en CPU ~1,4x
    model = FruitCNN(dropout=args.dropout, width=args.width).to(memory_format=torch.channels_last)
    counts = np.bincount(ytr, minlength=12)
    weights = torch.tensor(len(ytr) / (12 * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    tag = args.tag or f"cnn_lr{args.lr:g}_do{args.dropout:g}"
    history, best_f1, best_epoch, wait = [], -1.0, 0, 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for x, y in tr_loader:
            opt.zero_grad()
            logits = model(x.contiguous(memory_format=torch.channels_last))
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        sched.step()

        proba = predict_proba(model, va_loader)
        val_loss = criterion(torch.log(torch.tensor(proba).clamp_min(1e-7)), torch.tensor(yva)).item()
        val_f1 = f1_score(yva, proba.argmax(1), average="macro")
        history.append({"epoch": epoch, "train_loss": tot / n, "train_acc": correct / n,
                        "val_loss": val_loss, "val_acc": float((proba.argmax(1) == yva).mean()),
                        "val_f1_macro": val_f1, "lr": sched.get_last_lr()[0],
                        "seconds": time.time() - t0})
        print(f"[{tag}] ep {epoch:02d} loss {tot / n:.3f} acc {correct / n:.3f} | "
              f"val loss {val_loss:.3f} f1 {val_f1:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if val_f1 > best_f1:
            best_f1, best_epoch, wait = val_f1, epoch, 0
            torch.save(model.state_dict(), CHECKPOINTS / f"{tag}.pt")
        else:
            wait += 1
            if wait >= args.patience:
                break

    pd.DataFrame(history).to_csv(RESULTS / f"history_{tag}.csv", index=False)
    model.load_state_dict(torch.load(CHECKPOINTS / f"{tag}.pt"))

    out = {"model": tag, "params": vars(args), "n_parameters": count_parameters(model),
           "best_epoch": best_epoch, "train_seconds": round(time.time() - t_start, 1)}
    train_eval = DataLoader(ArrayDataset(Xtr, ytr, ev), batch_size=256)
    out["train"] = evaluate(ytr, predict_proba(model, train_eval))
    for split in ("val", "test", "mixed"):
        X, m = load_split(split, splits)
        y = encode(m["label"]) if split != "mixed" else np.zeros(len(m), int)
        loader = DataLoader(ArrayDataset(X, y, ev), batch_size=256)
        t0 = time.time()
        proba = predict_proba(model, loader)
        np.save(RESULTS / f"proba_{tag}_{split}.npy", proba)
        if split != "mixed":
            out[split] = evaluate(y, proba)
            out[split]["ms_per_image"] = 1000 * (time.time() - t0) / len(X)
    with open(RESULTS / f"metrics_{tag}.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--size", type=int, default=96, help="lado de la imagen de entrada")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    r = run(args)
    print(json.dumps({k: r[k] for k in ("model", "best_epoch", "n_parameters")}),
          f"val_f1={r['val']['joint_f1_macro']:.3f} test_f1={r['test']['joint_f1_macro']:.3f}")


if __name__ == "__main__":
    main()
