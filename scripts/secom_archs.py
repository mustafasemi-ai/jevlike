"""Does the SECOM result depend on the model? Same protocol, several architectures.

If ranking collapses in the later period for a linear model, an MLP, a
transformer and boosted trees alike, the cause is in the data, not in one
model's inductive bias. The random-split column is the control: the same
models, the same sample sizes, but "later" is no longer later in time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from secom_study import ARCHS, auc, cross_fit, load_secom

from jevlike.calibrate import softmax_rows
from jevlike.metrics import evaluate_probs


def summarise(rows: list[dict], release_at: float) -> dict[str, float]:
    out = {}
    for name, tag in (("in_task", "in"), ("held_out", "out")):
        rs = [r for r in rows if r["set"] == name]
        labels = np.asarray([r["label"] for r in rs])
        probs = softmax_rows([r["logits"] for r in rs], 1.0)
        p_fail = np.asarray(probs)[:, 1]
        rel = (1.0 - p_fail) >= release_at
        out[f"auc_{tag}"] = auc(p_fail, labels)
        rep = evaluate_probs(probs, labels.tolist())
        out[f"ece_{tag}"] = rep.ece
        out[f"acc_{tag}"] = rep.accuracy
        # Accuracy of always answering "pass". A model below this has learned
        # nothing that accuracy can see.
        out[f"majority_{tag}"] = 1.0 - labels.mean()
        # How many times more often held-back wafers fail than released ones.
        # 1.0 means the gate selects nothing.
        if rel.sum() >= 30 and (~rel).sum() >= 30 and labels[rel].mean() > 0:
            out[f"lift_{tag}"] = labels[~rel].mean() / labels[rel].mean()
        else:
            out[f"lift_{tag}"] = float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="SECOM temporal shift across architectures")
    ap.add_argument("--data-dir", type=Path, default=Path("data/secom"))
    ap.add_argument("--archs", nargs="+", choices=ARCHS, default=list(ARCHS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=0,
                    help="keep only the k most informative sensors (0 = all)")
    ap.add_argument("--split-frac", type=float, default=0.6)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cal-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--release-at", type=float, default=0.95)
    a = ap.parse_args()

    x, y, t = load_secom(a.data_dir)
    print(f"sensors: {a.top_k or 'all'}. {a.seeds} seeds, mean [min, max] for AUC. Accuracy in brackets: difference to "
          "always answering 'pass'. lift = fail rate held back / released "
          f"at P(pass) >= {a.release_at}\n")
    print("| architecture | split | acc early | acc later | AUC early | AUC later "
          "| ECE early | ECE later | lift early | lift later |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arch in a.archs:
        for random_split in (False, True):
            runs = []
            for seed in range(1, a.seeds + 1):
                a.arch, a.seed, a.random_split = arch, seed, random_split
                runs.append(summarise(cross_fit(x, y, t, a)[0], a.release_at))
            m = {k: np.nanmean([r[k] for r in runs]) for k in runs[0]}
            rng = {k: (min(r[k] for r in runs), max(r[k] for r in runs))
                   for k in ("auc_in", "auc_out")}
            print(
                f"| {arch if not random_split else ''} "
                f"| {'random (control)' if random_split else 'chronological'} "
                f"| {m['acc_in']:.3f} ({m['acc_in'] - m['majority_in']:+.3f}) "
                f"| {m['acc_out']:.3f} ({m['acc_out'] - m['majority_out']:+.3f}) "
                f"| {m['auc_in']:.3f} [{rng['auc_in'][0]:.2f}, {rng['auc_in'][1]:.2f}] "
                f"| {m['auc_out']:.3f} [{rng['auc_out'][0]:.2f}, {rng['auc_out'][1]:.2f}] "
                f"| {m['ece_in']:.3f} | {m['ece_out']:.3f} "
                f"| {m['lift_in']:.2f} | {m['lift_out']:.2f} |",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
