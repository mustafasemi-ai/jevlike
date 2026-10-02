"""Calibration under real sensor drift, on data where the model works first.

SECOM could not carry the question: no model there is good enough in-domain to
have anything to lose, and 28 later failures are too few to measure calibration.
The UCI Gas Sensor Array Drift dataset is the opposite case. 13,910 measurements
from 16 chemical sensors (128 features), six gases, collected over 36 months in
ten batches, published specifically because the sensors drift. The task is easy
inside a batch and gets harder with time.

    in-domain     : the first `--train-batches` batches, out-of-fold predictions
    out-of-domain : every later batch, reported separately -- a time axis

As in `secom_study.py`, every row gets exactly one prediction from a model that
did not train on it, temperature is fitted in-domain only, and the output is
the JSONL of `evaluate.py --dump-predictions` (source = batch), so
`calibration_study.py` runs on it unchanged with --temperature 1.

Rows inside a batch file are not in time order; the batch is the unit of time.

Data: https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip
      -> data/gas/Dataset/batch{1..10}.dat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from secom_study import Preprocessor, TreeModel, predict_logits, train_mlp
from torch import nn

from jevlike.calibrate import fit_temperature, softmax_rows
from jevlike.metrics import evaluate_probs

N_CLASSES = 6
THRESHOLDS = (0.9, 0.99)


def load_gas(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features [n, 128], gas label 0..5 and batch number 1..10."""
    xs, ys, bs = [], [], []
    for b in range(1, 11):
        with (data_dir / "Dataset" / f"batch{b}.dat").open(encoding="utf-8") as f:
            for line in f:
                label, *cells = line.split()
                row = np.full(128, np.nan)
                for cell in cells:
                    j, v = cell.split(":")
                    row[int(j) - 1] = float(v)
                xs.append(row)
                ys.append(int(label.split(";")[0]) - 1)
                bs.append(b)
    return np.asarray(xs), np.asarray(ys, dtype=np.int64), np.asarray(bs)


def fit(arch: str, x: np.ndarray, y: np.ndarray, seed: int, epochs: int):
    if arch == "mlp":
        return train_mlp(
            x, y, seed=seed, epochs=epochs, hidden=128, weight_decay=1e-2, n_classes=N_CLASSES,
            net_factory=lambda d: nn.Sequential(
                nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, N_CLASSES)
            ),
        )
    return TreeModel(arch, x, y, seed)


def predict_all(a: argparse.Namespace) -> list[dict]:
    x, y, batch = load_gas(a.data_dir)
    rng = np.random.default_rng(a.seed)
    if a.random_split:
        # Control: keep every batch's size, fill it with wafers from any time.
        order = rng.permutation(len(y))
        x, y = x[order], y[order]
    early = np.flatnonzero(batch <= a.train_batches)
    late = np.flatnonzero(batch > a.train_batches)
    early_fold = rng.permutation(len(early)) % a.folds
    late_fold = rng.permutation(len(late)) % a.folds

    rows: list[dict] = []
    for k in range(a.folds):
        fit_idx = early[early_fold != k]
        rng.shuffle(fit_idx)
        n_cal = int(len(fit_idx) * a.cal_frac)
        cal_idx, train_idx = fit_idx[:n_cal], fit_idx[n_cal:]

        prep = Preprocessor(x[train_idx])
        model = fit(a.arch, prep(x[train_idx]), y[train_idx], a.seed + k, a.epochs)
        temp = fit_temperature(
            predict_logits(model, prep(x[cal_idx])), y[cal_idx].tolist()
        ).temperature

        for name, idx in (("in_task", early[early_fold == k]), ("held_out", late[late_fold == k])):
            for i, raw in zip(idx, predict_logits(model, prep(x[idx]))):
                rows.append(
                    {
                        "set": name,
                        "row": int(i),
                        "source": f"batch{batch[i]:02d}",
                        "fold": k,
                        "label": int(y[i]),
                        "raw_logits": raw,
                        "logits": [v / temp for v in raw],
                    }
                )
    return rows


def gate(probs: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    """Coverage and realised accuracy of the decisions above the threshold."""
    released = probs.max(axis=1) >= threshold
    if not released.any():
        return 0.0, float("nan")
    return float(released.mean()), float((probs.argmax(axis=1) == labels)[released].mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibration under sensor drift (gas sensor array)")
    ap.add_argument("--data-dir", type=Path, default=Path("data/gas"))
    ap.add_argument("--out", type=Path, default=Path("runs/gas/preds.jsonl"))
    ap.add_argument("--arch", default="mlp", help="mlp, or any tree/classic name in secom_study")
    ap.add_argument("--train-batches", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cal-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--random-split", action="store_true",
                    help="control: shuffle rows across batches, so later batches are not later")
    a = ap.parse_args()

    rows = predict_all(a)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    split = "random (control)" if a.random_split else "chronological"
    print(f"{a.arch}, trained on batches 1-{a.train_batches}, split: {split}\n")
    head = " | ".join(f"gate {t}: coverage | accuracy" for t in THRESHOLDS)
    print(f"| batch | n | accuracy | mean confidence | ECE | {head} |")
    print("|---|---|---|---|---|" + "---|---|" * len(THRESHOLDS))
    groups = [("in-domain", [r for r in rows if r["set"] == "in_task"])]
    for b in sorted({r["source"] for r in rows if r["set"] == "held_out"}):
        groups.append((b, [r for r in rows if r["source"] == b and r["set"] == "held_out"]))
    for name, rs in groups:
        labels = np.asarray([r["label"] for r in rs])
        probs = softmax_rows([r["logits"] for r in rs], 1.0)
        rep = evaluate_probs(probs, labels.tolist())
        cells = []
        for t in THRESHOLDS:
            cov, acc = gate(np.asarray(probs), labels, t)
            cells.append(f"{cov:.3f} | {acc:.3f}")
        print(
            f"| {name} | {len(rs)} | {rep.accuracy:.3f} | {rep.mean_confidence:.3f} "
            f"| {rep.ece:.3f} | {' | '.join(cells)} |"
        )
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
