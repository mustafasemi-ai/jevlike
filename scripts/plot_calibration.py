"""Render the two figures that carry the finding.

Both panels use the same reference: the 45-degree line is what the model
*promised*. Anything below it is a promise the model did not keep.

  Left  -- reliability: mean confidence vs actual accuracy, per equal-mass bin.
  Right -- confidence gate: threshold vs the accuracy actually realised above it.

Light and dark variants are rendered separately (dark is stepped for the dark
surface, not an automatic inversion of the light one).

Palette: categorical slots 1-2 of the reference palette, validated for
colour-vision deficiency and surface contrast in both modes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jevlike.calibrate import softmax_rows
from jevlike.metrics import reliability_table

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "grid": "#e3e2df",
        "in_domain": "#2a78d6",
        "out_domain": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "grid": "#33322f",
        "in_domain": "#3987e5",
        "out_domain": "#d95926",
    },
}

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]


def load(path: Path) -> dict[str, list[dict]]:
    by_set: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_set[r.get("set") or r["kume"]].append(r)
    return by_set


def probs_labels(rows: Sequence[dict], temperature: float):
    return softmax_rows([r["logits"] for r in rows], temperature), [r["label"] for r in rows]


def gate_points(rows: Sequence[dict], temperature: float) -> tuple[list[float], list[float]]:
    probs, labels = probs_labels(rows, temperature)
    conf = np.array([max(p) for p in probs])
    correct = np.array(
        [int(max(range(len(p)), key=p.__getitem__) == y) for p, y in zip(probs, labels)]
    )
    xs, ys = [], []
    for t in THRESHOLDS:
        m = conf >= t
        if m.sum() == 0:
            continue
        xs.append(t)
        ys.append(float(correct[m].mean()))
    return xs, ys


def style_axes(ax, c: dict, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(c["surface"])
    ax.set_xlim(0.35, 1.02)
    ax.set_ylim(0.35, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, color=c["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=c["ink2"], labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=c["ink2"], fontsize=10)
    ax.set_ylabel(ylabel, color=c["ink2"], fontsize=10)
    ax.set_title(title, color=c["ink"], fontsize=12, fontweight="bold", loc="left", pad=12)
    # the promise line -- recessive, it is a reference not a series
    ax.plot([0, 1.02], [0, 1.02], color=c["ink2"], linewidth=1.0, linestyle=(0, (4, 3)),
            alpha=0.55, zorder=1)


def style_gap_axes(ax, c: dict) -> None:
    """Axes for the promise-gap panel: zero is the reference, above it is failure."""
    ax.set_facecolor(c["surface"])
    ax.set_xlim(0.46, 1.03)
    ax.set_ylim(-0.42, 0.20)
    ax.grid(True, color=c["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=c["ink2"], labelsize=9, length=0)
    ax.set_xlabel("confidence threshold", color=c["ink2"], fontsize=10)
    ax.set_ylabel("threshold  −  realised accuracy", color=c["ink2"], fontsize=10)
    ax.set_title("Promise gap", color=c["ink"], fontsize=12, fontweight="bold",
                 loc="left", pad=12)

    # Everything above zero is a promise the model did not keep.
    ax.axhspan(0.0, 0.20, color=c["out_domain"], alpha=0.07, zorder=0)
    ax.axhline(0.0, color=c["ink2"], linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.55, zorder=1)
    ax.annotate("gate is looser than you think", (0.475, 0.165),
                color=c["ink2"], fontsize=8.5, ha="left", va="center")
    # bottom-right is empty (both curves climb away from it)
    ax.annotate("gate keeps its promise", (1.015, -0.385),
                color=c["ink2"], fontsize=8.5, ha="right", va="center")


def render(by_set: dict, temperature: float, mode: str, out: Path) -> Path:
    c = THEMES[mode]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.9), dpi=200)
    fig.patch.set_facecolor(c["surface"])

    series = [
        ("in_task", "in-domain", c["in_domain"]),
        ("held_out", "out-of-domain", c["out_domain"]),
    ]

    # --- left: reliability -------------------------------------------------
    style_axes(ax1, c, "mean confidence", "actual accuracy", "Reliability")
    for key, label, colour in series:
        if key not in by_set:
            continue
        probs, labels = probs_labels(by_set[key], temperature)
        tbl = reliability_table(probs, labels, n_bins=8)
        xs = [r["mean_confidence"] for r in tbl]
        ys = [r["accuracy"] for r in tbl]
        ax1.plot(xs, ys, color=colour, linewidth=2.0, zorder=3)
        ax1.plot(xs, ys, "o", color=colour, markersize=8, markeredgecolor=c["surface"],
                 markeredgewidth=2, zorder=4)
        # direct label at the last point, so identity is never colour-alone
        ax1.annotate(label, (xs[-1], ys[-1]), textcoords="offset points",
                     xytext=(-6, -16 if key == "held_out" else 10),
                     color=c["ink"], fontsize=10, fontweight="bold",
                     ha="right" if key == "held_out" else "left")

    ax1.annotate("perfect calibration", (0.62, 0.62), textcoords="offset points",
                 xytext=(6, -14), color=c["ink2"], fontsize=8.5, rotation=45,
                 rotation_mode="anchor")

    # --- right: the promise gap -------------------------------------------
    # Plotting raw accuracy against the threshold would be misleading: accuracy
    # above a 0.5 gate is naturally ~0.75, so both curves sit above the diagonal
    # and the chart reads as "both keep their promise". The claim is about the
    # gap, so plot the gap.
    style_gap_axes(ax2, c)
    for key, label, colour in series:
        if key not in by_set:
            continue
        xs, ys = gate_points(by_set[key], temperature)
        gaps = [t - acc for t, acc in zip(xs, ys)]
        ax2.plot(xs, gaps, color=colour, linewidth=2.0, zorder=3)
        ax2.plot(xs, gaps, "o", color=colour, markersize=8, markeredgecolor=c["surface"],
                 markeredgewidth=2, zorder=4)
        # Anchor each label where its own curve has clear space: out-of-domain
        # above its last point, in-domain below a mid-curve point (its tail runs
        # into the zero line and the other series).
        if key == "held_out":
            ax2.annotate(label, (xs[-1], gaps[-1]), textcoords="offset points",
                         xytext=(-8, 12), color=c["ink"], fontsize=10,
                         fontweight="bold", ha="right")
        else:
            i = min(3, len(xs) - 1)
            ax2.annotate(label, (xs[i], gaps[i]), textcoords="offset points",
                         xytext=(0, -20), color=c["ink"], fontsize=10,
                         fontweight="bold", ha="center")

    fig.suptitle(
        "Calibrated where it was trained; overconfident everywhere else",
        color=c["ink"], fontsize=13.5, fontweight="bold", x=0.012, ha="left", y=0.99,
    )
    fig.text(
        0.012, 0.015,
        "Qwen3-1.7B + LoRA, 37 open datasets, temperature fitted in-domain.  "
        "Left: below the line = overconfident.  Right: above the line = the gate "
        "auto-approves more errors than it promised.",
        color=c["ink2"], fontsize=9, ha="left",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.945))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=c["surface"], bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Render the calibration figures")
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out-dir", type=Path, default=Path("docs"))
    a = ap.parse_args()

    by_set = load(a.predictions)
    for mode in ("light", "dark"):
        p = render(by_set, a.temperature, mode, a.out_dir / f"calibration-{mode}.png")
        print(f"  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
