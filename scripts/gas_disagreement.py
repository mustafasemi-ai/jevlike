"""Is this drift harmful? Asking the question without labels.

`secom_drift.py` asks "has the data changed?" and the answer is always yes. The
useful question is whether the change is one that breaks the gate. On the gas
data the gate breaks in some later batches and holds in others, so there is
something to tell apart.

The signal tried here is disagreement. Train a few runs of the same model.
One is deployed; the others are only consulted. For every item the deployed
run releases, check whether the companions would have predicted the same
class. Where the runs were pinned down by the training data they agree; where
drift has moved the inputs somewhere the training data did not constrain,
they scatter. No label is needed. (Disagreement between independently trained
models as an estimate of error is a known idea: Jiang et al., 2022.)

It is compared against the two free signals from `gas_audit.py`: the share of
items released, and the drift detector, which is 1.000 on every later batch.

Input: the prediction caches written by `gas_seeds.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gas_study import load_gas
from scipy.stats import spearmanr
from secom_study import auc


def main() -> int:
    ap = argparse.ArgumentParser(description="Label-free test for harmful drift")
    ap.add_argument("--data-dir", type=Path, default=Path("data/gas"))
    ap.add_argument("--cache", type=Path, default=Path("runs/gas/seeds.npz"))
    ap.add_argument("--train-batches", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--companions", type=int, default=4)
    ap.add_argument("--broken-at", type=float, default=0.05)
    ap.add_argument("--min-released", type=int, default=30)
    a = ap.parse_args()

    _, y, batch = load_gas(a.data_dir)
    probs = np.load(a.cache)["probs"]
    n_runs = len(probs)
    pred, conf = probs.argmax(axis=2), probs.max(axis=2)
    groups = [("in-domain", batch <= a.train_batches)]
    groups += [(f"batch {b}", batch == b) for b in range(a.train_batches + 1, 11)]

    # one record per (deployed run, batch)
    records = []
    for i in range(n_runs):
        others = [(i + j) % n_runs for j in range(1, a.companions + 1)]
        for name, m in groups:
            rel = m & (conf[i] >= a.threshold)
            if rel.sum() < a.min_released:
                continue
            records.append({
                "group": name,
                "error": float((pred[i][rel] != y[rel]).mean()),
                "disagreement": float(np.mean([(pred[o][rel] != pred[i][rel]).mean() for o in others])),
                "share_released": float(rel.sum() / m.sum()),
                "confidence": float(conf[i][m].mean()),
            })

    col = lambda rs, k: np.asarray([r[k] for r in rs])
    print(f"{a.cache.name}: {n_runs} runs, gate at {a.threshold}, {a.companions} companions "
          f"per deployed run\n")
    print("| batch | error among released (truth) | disagreement among released (free) "
          "| share released (free) | runs with a broken gate |")
    print("|---|---|---|---|---|")
    for name, _ in groups:
        rs = [r for r in records if r["group"] == name]
        if not rs:
            continue
        err = col(rs, "error")
        print(
            f"| {name} | {err.mean():.3f} | {col(rs, 'disagreement').mean():.3f} "
            f"| {col(rs, 'share_released').mean():.3f} "
            f"| {int((err > a.broken_at).sum())} of {len(rs)} |"
        )

    later = [r for r in records if r["group"] != "in-domain"]
    err = col(later, "error")
    broken = (err > a.broken_at).astype(np.int64)
    print(f"\nAcross all {len(later)} (run, later batch) pairs; {int(broken.sum())} have a broken gate.\n")
    print("| free signal | correlation with the true error | AUC for telling broken from intact |")
    print("|---|---|---|")
    for label, key, sign in (
        ("disagreement among released", "disagreement", 1.0),
        ("share released (low = suspicious)", "share_released", -1.0),
        ("mean confidence (low = suspicious)", "confidence", -1.0),
    ):
        v = sign * col(later, key)
        print(f"| {label} | {spearmanr(v, err).statistic:+.2f} | {auc(v, broken):.3f} |")
    print("| drift detector | constant 1.000 on every later batch | 0.500 |")

    # How good is disagreement as a number, not just a ranking?
    d = col(later, "disagreement")
    print("\nAs an alarm: flag a batch when disagreement among released items exceeds a level.\n")
    print("| alarm level | broken gates caught | intact gates flagged by mistake |")
    print("|---|---|---|")
    for level in (0.003, 0.005, 0.01, 0.02, 0.05):
        caught = (d[broken == 1] > level).mean()
        mistaken = (d[broken == 0] > level).mean()
        print(f"| {level:.3f} | {caught:.1%} | {mistaken:.1%} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
