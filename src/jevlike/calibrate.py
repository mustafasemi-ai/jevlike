"""Temperature scaling.

Raw LLM probabilities are almost always overconfident. Dividing the logits by a
single scalar T leaves accuracy untouched (argmax is invariant) but typically
cuts ECE substantially. It is cheap, reversible and independent of training, so
it is the first calibration tool to reach for.

T > 1  -> softens the probabilities (fixes overconfidence)
T < 1  -> sharpens them (if the model is too timid)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from .metrics import EvalReport, evaluate_probs


@dataclass
class Calibration:
    temperature: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "temperature": round(self.temperature, 6),
            "nll_before": round(self.nll_before, 6),
            "nll_after": round(self.nll_after, 6),
            "ece_before": round(self.ece_before, 6),
            "ece_after": round(self.ece_after, 6),
            "n": self.n,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def summary(self) -> str:
        return (
            f"T={self.temperature:.4f}  "
            f"NLL {self.nll_before:.4f} -> {self.nll_after:.4f}  "
            f"ECE {self.ece_before:.4f} -> {self.ece_after:.4f}  (n={self.n})"
        )


def softmax_rows(logits: Sequence[Sequence[float]], temperature: float) -> list[list[float]]:
    """Softmax variable-length logit rows with temperature T."""
    out: list[list[float]] = []
    for row in logits:
        a = np.asarray(row, dtype=np.float64) / temperature
        a -= a.max()
        e = np.exp(a)
        out.append((e / e.sum()).tolist())
    return out


def _nll(logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float) -> float:
    total = 0.0
    for row, y in zip(logits, labels):
        a = np.asarray(row, dtype=np.float64) / temperature
        a -= a.max()
        total += float(np.log(np.exp(a).sum()) - a[y])
    return total / len(labels)


def fit_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> Calibration:
    """Find the scalar T minimising validation NLL.

    NLL is unimodal in T, so a bounded scalar search suffices -- no LBFGS needed.
    """
    if len(logits) != len(labels):
        raise ValueError("logits and labels must have the same length.")
    if not labels:
        raise ValueError("Empty calibration set.")

    res = minimize_scalar(
        lambda t: _nll(logits, labels, t), bounds=bounds, method="bounded",
        options={"xatol": 1e-4},
    )
    t_star = float(res.x)

    before: EvalReport = evaluate_probs(softmax_rows(logits, 1.0), labels)
    after: EvalReport = evaluate_probs(softmax_rows(logits, t_star), labels)
    return Calibration(
        temperature=t_star,
        nll_before=before.nll,
        nll_after=after.nll,
        ece_before=before.ece,
        ece_after=after.ece,
        n=len(labels),
    )


def load_temperature(path: str | Path, default: float = 1.0) -> float:
    p = Path(path)
    if not p.exists():
        return default
    return float(json.loads(p.read_text(encoding="utf-8"))["temperature"])
