"""Calibration and accuracy metrics.

Accuracy alone is not enough here: the product promise is "state the probability
correctly too". Every evaluation therefore reports ECE and Brier as well.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class EvalReport:
    n: int
    accuracy: float
    ece: float
    """Expected Calibration Error, using equal-mass bins."""
    mce: float
    """Maximum Calibration Error."""
    brier: float
    """Multi-class Brier score (lower is better)."""
    nll: float
    mean_confidence: float
    overconfidence: float
    """mean_confidence - accuracy. Positive means the model is overconfident."""

    def as_dict(self) -> dict[str, float | int]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    def summary(self) -> str:
        return (
            f"n={self.n}  acc={self.accuracy:.4f}  ECE={self.ece:.4f}  "
            f"Brier={self.brier:.4f}  NLL={self.nll:.4f}  "
            f"conf={self.mean_confidence:.4f} (overconfidence {self.overconfidence:+.4f})"
        )


def _as_arrays(
    probs: Sequence[Sequence[float]], labels: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad variable option counts with 0 rather than -inf."""
    n = len(probs)
    if n == 0:
        raise ValueError("Empty evaluation set.")
    width = max(len(p) for p in probs)
    arr = np.zeros((n, width), dtype=np.float64)
    for i, p in enumerate(probs):
        arr[i, : len(p)] = p
    return arr, np.asarray(labels, dtype=np.int64)


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> tuple[float, float]:
    """ECE and MCE using equal-mass (quantile) bins.

    We use equal-mass rather than equal-width bins: confidences pile up near 1,
    so most equal-width bins end up empty and ECE looks artificially small.
    """
    n = len(confidences)
    if n == 0:
        return 0.0, 0.0
    n_bins = max(1, min(n_bins, n))
    order = np.argsort(confidences)
    conf_sorted = confidences[order]
    corr_sorted = correct[order]

    edges = np.linspace(0, n, n_bins + 1).astype(int)
    ece = 0.0
    mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        gap = abs(conf_sorted[lo:hi].mean() - corr_sorted[lo:hi].mean())
        ece += (hi - lo) / n * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def evaluate_probs(
    probs: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    n_bins: int = 15,
    eps: float = 1e-12,
) -> EvalReport:
    """Full report from a list of probabilities plus gold label indices."""
    p, y = _as_arrays(probs, labels)
    n = len(y)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    conf = p[np.arange(n), pred]

    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1.0
    brier = float(((p - onehot) ** 2).sum(axis=1).mean())
    nll = float(-np.log(np.clip(p[np.arange(n), y], eps, 1.0)).mean())
    ece, mce = expected_calibration_error(conf, correct, n_bins=n_bins)

    acc = float(correct.mean())
    mean_conf = float(conf.mean())
    return EvalReport(
        n=n,
        accuracy=acc,
        ece=ece,
        mce=mce,
        brier=brier,
        nll=nll,
        mean_confidence=mean_conf,
        overconfidence=mean_conf - acc,
    )


def bootstrap_ci(
    probs: Sequence[Sequence[float]],
    labels: Sequence[int],
    metric: str = "ece",
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 17,
) -> dict[str, float]:
    """Bootstrap confidence interval for one metric.

    ECE is volatile on small sets: on 800-1000 examples a 0.01-0.02 difference
    between two runs can easily be sampling noise. Reporting only the point
    estimate misleads when comparing models, so we report an interval.
    """
    p, y = _as_arrays(probs, labels)
    n = len(y)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rep = evaluate_probs([p[i].tolist() for i in idx], [int(y[i]) for i in idx])
        vals[b] = getattr(rep, metric)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    point = getattr(evaluate_probs(probs, labels), metric)
    return {
        "metric": metric,
        "value": round(float(point), 6),
        "lo": round(float(lo), 6),
        "hi": round(float(hi), 6),
        "width": round(float(hi - lo), 6),
        "n": n,
        "n_boot": n_boot,
    }


def reliability_table(
    probs: Sequence[Sequence[float]], labels: Sequence[int], n_bins: int = 10
) -> list[dict[str, float]]:
    """Reliability-curve data: mean confidence vs actual accuracy per bin."""
    p, y = _as_arrays(probs, labels)
    pred = p.argmax(axis=1)
    conf = p[np.arange(len(y)), pred]
    correct = (pred == y).astype(np.float64)

    order = np.argsort(conf)
    conf, correct = conf[order], correct[order]
    edges = np.linspace(0, len(y), n_bins + 1).astype(int)
    rows: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        rows.append(
            {
                "n": int(hi - lo),
                "mean_confidence": float(conf[lo:hi].mean()),
                "accuracy": float(correct[lo:hi].mean()),
                "gap": float(conf[lo:hi].mean() - correct[lo:hi].mean()),
            }
        )
    return rows
