"""Sequence models: read the last few wafers, not just the current one.

Every other model here treats a wafer as an isolated row. But the line drifts
continuously (`secom_drift.py`), so the wafers just before the current one say
something about the state the tool is in. Sequence models are the one family
that can use that.

Each row becomes a window of the `window` most recent wafers' sensor readings,
oldest first, the wafer being judged last. Only sensor readings enter the
window -- never the earlier wafers' pass/fail labels, which in a real line
arrive late or not at all.

Folds, calibration and scoring are `secom_study.cross_fit`, unchanged, so the
numbers are comparable with `secom_archs.py`. window=1 is the plain MLP's
setting and serves as the reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from secom_archs import summarise
from secom_study import cross_fit, load_secom, train_mlp
from torch import nn

SEQ_ARCHS = ("gru", "cnn1d", "mlp_window")


def windowed(x: np.ndarray, window: int) -> np.ndarray:
    """[n, d] in time order -> [n, window * d]; the first rows repeat row 0."""
    lags = [x[np.maximum(np.arange(len(x)) - lag, 0)] for lag in range(window - 1, -1, -1)]
    return np.concatenate(lags, axis=1)


class SeqNet(nn.Module):
    def __init__(self, arch: str, d_in: int, window: int, hidden: int = 64) -> None:
        super().__init__()
        if d_in % window:
            # Preprocessor dropped a sensor at some lags but not others; the
            # reshape below would silently mix sensors across time steps.
            raise ValueError(f"{d_in} columns do not divide into {window} time steps.")
        self.arch, self.window, self.d = arch, window, d_in // window
        self.proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.d, hidden), nn.ReLU())
        if arch == "gru":
            self.body = nn.GRU(hidden, hidden, batch_first=True)
        elif arch == "cnn1d":
            self.body = nn.Sequential(
                nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
                nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU(),
            )
        elif arch == "mlp_window":
            self.body = nn.Sequential(nn.Linear(hidden * window, hidden), nn.ReLU())
        else:
            raise ValueError(f"Unknown sequence architecture: {arch}")
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x.view(len(x), self.window, self.d))  # [B, L, H]
        if self.arch == "gru":
            h = self.body(h)[0][:, -1]
        elif self.arch == "cnn1d":
            h = self.body(h.transpose(1, 2))[:, :, -1]
        else:
            h = self.body(h.flatten(1))
        return self.head(h)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sequence models on SECOM")
    ap.add_argument("--data-dir", type=Path, default=Path("data/secom"))
    ap.add_argument("--archs", nargs="+", choices=SEQ_ARCHS, default=list(SEQ_ARCHS))
    ap.add_argument("--windows", nargs="+", type=int, default=[4, 16])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--split-frac", type=float, default=0.6)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cal-frac", type=float, default=0.2)
    ap.add_argument("--release-at", type=float, default=0.95)
    a = ap.parse_args()
    a.drop_columns = False  # keep the window aligned; see Preprocessor

    x, y, t = load_secom(a.data_dir)
    # Fix the sensor set before windowing so every time step has the same
    # columns. Decided on the training period's readings only, no labels.
    early = x[: int(len(y) * a.split_frac)]
    # Near-constant sensors go too: one that is constant on a fold's training
    # rows at some lags only would be dropped there and misalign the window.
    filled = np.where(np.isnan(early), np.nanmedian(early, axis=0), early)
    varied = (filled != np.median(filled, axis=0)).mean(axis=0) > 0.1
    x = x[:, (np.isnan(early).mean(axis=0) <= 0.4) & varied]

    print(f"{x.shape[1]} sensors, {a.seeds} seeds, mean [min, max] for AUC\n")
    print("| architecture | window | split | acc early | acc later | AUC early | AUC later "
          "| lift early | lift later |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arch in a.archs:
        for window in a.windows:
            xw = windowed(x, window)

            def trainer(xt: np.ndarray, yt: np.ndarray, seed: int, arch=arch, window=window):
                return train_mlp(
                    xt, yt, seed=seed, epochs=a.epochs, hidden=64, weight_decay=1e-2,
                    net_factory=lambda d: SeqNet(arch, d, window),
                )

            for random_split in (False, True):
                runs = []
                for seed in range(1, a.seeds + 1):
                    a.seed, a.random_split = seed, random_split
                    runs.append(summarise(cross_fit(xw, y, t, a, trainer=trainer)[0], a.release_at))
                m = {k: np.nanmean([r[k] for r in runs]) for k in runs[0]}
                lo = {k: min(r[k] for r in runs) for k in ("auc_in", "auc_out")}
                hi = {k: max(r[k] for r in runs) for k in ("auc_in", "auc_out")}
                print(
                    f"| {arch} | {window} | {'random (control)' if random_split else 'chronological'} "
                    f"| {m['acc_in']:.3f} ({m['acc_in'] - m['majority_in']:+.3f}) "
                    f"| {m['acc_out']:.3f} ({m['acc_out'] - m['majority_out']:+.3f}) "
                    f"| {m['auc_in']:.3f} [{lo['auc_in']:.2f}, {hi['auc_in']:.2f}] "
                    f"| {m['auc_out']:.3f} [{lo['auc_out']:.2f}, {hi['auc_out']:.2f}] "
                    f"| {m['lift_in']:.2f} | {m['lift_out']:.2f} |",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
