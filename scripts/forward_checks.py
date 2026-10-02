"""Is the in-domain score itself too kind? Blocked-in-time checks for both datasets.

The main studies score "in-domain" with random folds inside the training
period. If the process drifts inside that period too, random folds let a model
see each test row's neighbours in time, and the score describes a situation
that deployment never offers. These checks only ever predict rows that lie
outside the time span the model was trained on.

SECOM
    forward   : train on every wafer so far, predict the next block, slide on.
    retrain   : on the later period, a model trained once against one
                retrained before every block -- what an alarm would trigger.

Gas sensors
    blocked   : inside the training batches, random folds against leaving a
                whole batch out and against predicting the next batch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gas_study import fit, load_gas
from secom_study import Preprocessor, TreeModel, auc, auc_ci, load_secom, predict_logits


def p_fail(model, x: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(predict_logits(model, x))[:, 1])


def secom(a: argparse.Namespace) -> None:
    x, y, _ = load_secom(a.secom_dir)
    n_early = int(len(y) * a.split_frac)
    later = np.arange(n_early, len(y))

    def predict_blocks(arch: str, first: int) -> tuple[np.ndarray, np.ndarray]:
        """Retrain on all past wafers before each block of `--block` wafers."""
        idx = np.arange(first, len(y))
        score = np.zeros(len(idx))
        for start in range(first, len(y), a.block):
            prep = Preprocessor(x[:start])
            model = TreeModel(arch, prep(x[:start]), y[:start], a.seed)
            blk = np.arange(start, min(start + a.block, len(y)))
            score[blk - first] = p_fail(model, prep(x[blk]))
        return score, y[idx]

    print(f"SECOM. Block = {a.block} wafers. Later period = last {len(later)} wafers, "
          f"{int(y[later].sum())} failures.\n")
    print("| model | forward, from wafer 500 | later period: trained once "
          "| later period: retrained every block |")
    print("|---|---|---|---|")
    for arch in a.secom_archs:
        fwd, fwd_y = predict_blocks(arch, 500)
        prep = Preprocessor(x[:n_early])
        frozen = p_fail(TreeModel(arch, prep(x[:n_early]), y[:n_early], a.seed), prep(x[later]))
        rolled, _ = predict_blocks(arch, n_early)
        cells = []
        for score, labels in ((fwd, fwd_y), (frozen, y[later]), (rolled, y[later])):
            lo, hi = auc_ci(score, labels, n_boot=a.n_boot, seed=a.seed)
            cells.append(f"{auc(score, labels):.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"| {arch} | {' | '.join(cells)} |", flush=True)


def gas(a: argparse.Namespace) -> None:
    x, y, batch = load_gas(a.gas_dir)
    train_batches = list(range(1, a.train_batches + 1))

    def accuracy(arch: str, tr: np.ndarray, te: np.ndarray) -> float:
        prep = Preprocessor(x[tr])
        model = fit(arch, prep(x[tr]), y[tr], a.seed, 30)
        return float((np.argmax(predict_logits(model, prep(x[te])), axis=1) == y[te]).mean())

    early = np.flatnonzero(batch <= a.train_batches)
    fold = np.random.default_rng(a.seed).permutation(len(early)) % 5
    print(f"\nGas sensors, inside training batches 1-{a.train_batches}. Accuracy.\n")
    head = " | ".join(f"batch {b} left out" for b in train_batches)
    print(f"| model | random 5 folds | {head} | train on 1-{a.train_batches - 1}, "
          f"predict {a.train_batches} |")
    print("|---|---|" + "---|" * len(train_batches) + "---|")
    for arch in a.gas_archs:
        random = np.mean([accuracy(arch, early[fold != k], early[fold == k]) for k in range(5)])
        left_out = [
            accuracy(arch, np.flatnonzero((batch <= a.train_batches) & (batch != b)),
                     np.flatnonzero(batch == b))
            for b in train_batches
        ]
        forward = accuracy(arch, np.flatnonzero(batch < a.train_batches),
                           np.flatnonzero(batch == a.train_batches))
        cells = " | ".join(f"{v:.3f}" for v in left_out)
        print(f"| {arch} | {random:.3f} | {cells} | {forward:.3f} |", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Blocked-in-time checks on the in-domain scores")
    ap.add_argument("--secom-dir", type=Path, default=Path("data/secom"))
    ap.add_argument("--gas-dir", type=Path, default=Path("data/gas"))
    ap.add_argument("--secom-archs", nargs="+", default=["gbdt", "adaboost"])
    ap.add_argument("--gas-archs", nargs="+", default=["mlp", "gbdt"])
    ap.add_argument("--split-frac", type=float, default=0.6)
    ap.add_argument("--train-batches", type=int, default=3)
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    secom(a)
    gas(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
