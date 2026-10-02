"""A drift alarm in front of the predictor: can the failure be announced?

Every model in `secom_archs.py` loses its ranking in the later period, and
nothing in its own output says so. This puts a second model in series whose only
job is to notice that the inputs have changed.

The detector is a classifier two-sample test. Take the wafers the predictor was
trained on and a recent window of production, and train trees to tell which is
which. If they cannot (AUC ~ 0.5) the window looks like the training data. If
they can, the sensors have moved and the predictor's confidences should not be
trusted. It needs no pass/fail labels, so it can run before any wafer from the
window has been physically measured.

Three kinds of window, to tell an alarm from a false alarm:

    null    : random wafers from the training period vs the rest of it.
              Sets the threshold -- this is what "no drift" looks like.
    within  : consecutive wafers from the training period vs the rest of it.
              How much the line already drifts inside the period we trained on.
    later   : consecutive wafers after training vs the whole training period.
              The windows the alarm is for.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from secom_study import Preprocessor, auc, load_secom
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def drift_auc(reference: np.ndarray, window: np.ndarray, seed: int) -> float:
    """Cross-validated AUC of telling `window` rows from `reference` rows."""
    x = np.vstack([reference, window])
    is_window = np.r_[np.zeros(len(reference)), np.ones(len(window))].astype(np.int64)
    model = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, max_depth=3, random_state=seed
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    score = cross_val_predict(model, x, is_window, cv=folds, method="predict_proba")[:, 1]
    return auc(score, is_window)


def main() -> int:
    ap = argparse.ArgumentParser(description="Label-free drift alarm on SECOM")
    ap.add_argument("--data-dir", type=Path, default=Path("data/secom"))
    ap.add_argument("--split-frac", type=float, default=0.6)
    ap.add_argument("--window", type=int, default=100, help="wafers per window")
    ap.add_argument("--null-draws", type=int, default=30)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    x, y, t = load_secom(a.data_dir)
    n_early = int(len(y) * a.split_frac)
    x = Preprocessor(x[:n_early])(x)  # statistics from the training period only
    early = np.arange(n_early)
    rng = np.random.default_rng(a.seed)

    null = []
    for d in range(a.null_draws):
        pick = rng.choice(n_early, size=a.window, replace=False)
        null.append(drift_auc(x[np.setdiff1d(early, pick)], x[pick], a.seed + d))
    threshold = float(np.quantile(null, 0.95))
    print(f"null (random training-period wafers, {a.null_draws} draws): "
          f"mean {np.mean(null):.3f}, max {max(null):.3f}")
    print(f"alarm threshold = 95th percentile of the null = {threshold:.3f}\n")

    print("| kind | window | wafers | fail rate | drift AUC | alarm |")
    print("|---|---|---|---|---|---|")
    counts = {"within": [0, 0], "later": [0, 0]}
    for start in range(0, len(y) - a.window + 1, a.window):
        idx = np.arange(start, start + a.window)
        if idx[-1] < n_early:
            kind, reference = "within", x[np.setdiff1d(early, idx)]
        elif idx[0] >= n_early:
            kind, reference = "later", x[early]
        else:
            continue  # straddles the split; neither clean training nor clean later
        score = drift_auc(reference, x[idx], a.seed)
        alarm = score > threshold
        counts[kind][0] += int(alarm)
        counts[kind][1] += 1
        print(
            f"| {kind} | {t[idx[0]]:%m-%d} .. {t[idx[-1]]:%m-%d} | {len(idx)} "
            f"| {y[idx].mean():.3f} | {score:.3f} | {'ALARM' if alarm else '-'} |",
            flush=True,
        )
    for kind, (fired, total) in counts.items():
        print(f"\n{kind}: alarm on {fired} of {total} windows", end="")

    # Which sensors moved: how well each one alone separates the two periods.
    later = np.arange(n_early, len(y))
    is_later = np.r_[np.zeros(n_early), np.ones(len(later))].astype(np.int64)
    shift = np.array([abs(auc(x[:, j], is_later) - 0.5) for j in range(x.shape[1])])
    print(
        f"\n\nsensors that alone separate training from later period: "
        f"{int((shift > 0.2).sum())} of {x.shape[1]} with AUC beyond 0.7/0.3, "
        f"{int((shift > 0.4).sum())} beyond 0.9/0.1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
