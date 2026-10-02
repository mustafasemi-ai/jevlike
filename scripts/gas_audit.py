"""How many physical measurements does it take to notice the gate has failed?

A confidence gate exists to skip measurements: whatever the model is sure about
is released unmeasured. `gas_study.py` shows the gate failing silently as the
sensors drift. The only direct evidence that it has failed is ground truth, and
ground truth is exactly what the gate was installed to avoid buying.

So the practical question is a budget: audit a fraction of the *released*
items anyway, and ask how small that fraction can be.

    stream  : in-domain items first (the gate works), then batches 4..10 in
              order (it does not). The change point is known, so false alarms
              and detection delay can both be counted.
    audit   : each released item is measured with probability f.
    monitor : a Bernoulli CUSUM on the audited outcomes. It accumulates
              evidence that the error rate among released items is p1 rather
              than the promised p0, and alarms at a fixed threshold.

Two label-free signals are reported next to it, per batch, because they cost
nothing: the share of items the gate releases, and how well a classifier can
tell the batch from the training data (`secom_drift.drift_auc`).

This is a known idea, not a new one: sequential risk monitoring (Podkopaev and
Ramdas, ICLR 2022) in machine learning, sampling decision schemes in virtual
metrology. Those come with guarantees on the false alarm rate; the CUSUM here
has none, its false alarm rate is only measured.

Input: one or more JSONL files written by `gas_study.py` (one per seed). How
far the gate degrades depends on the seed, so the tables report the spread.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from gas_study import load_gas
from secom_drift import drift_auc
from secom_study import Preprocessor

from jevlike.calibrate import softmax_rows


def load_stream(path: Path, threshold: float) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    probs = np.asarray(softmax_rows([r["logits"] for r in rows], 1.0))
    labels = np.asarray([r["label"] for r in rows])
    return {
        "released": probs.max(axis=1) >= threshold,
        "correct": probs.argmax(axis=1) == labels,
        "later": np.asarray([r["set"] == "held_out" for r in rows]),
        "batch": np.asarray([int(r["source"][-2:]) for r in rows]),
    }


def cusum_alarm(errors: np.ndarray, p0: float, p1: float, h: float) -> int:
    """Index of the audited item at which the CUSUM alarms, or -1."""
    up, down = math.log(p1 / p0), math.log((1 - p1) / (1 - p0))
    s = 0.0
    for i, e in enumerate(errors):
        s = max(0.0, s + (up if e else down))
        if s >= h:
            return i
    return -1


def simulate(s: dict[str, np.ndarray], f: float, a: argparse.Namespace, rng) -> dict:
    """One pass over a freshly shuffled stream with a fresh audit draw."""
    early = rng.permutation(np.flatnonzero(~s["later"]))
    late = np.concatenate([
        rng.permutation(np.flatnonzero(s["later"] & (s["batch"] == b)))
        for b in sorted(set(s["batch"][s["later"]]))
    ])
    order = np.concatenate([early, late])
    change = len(early)

    released = np.flatnonzero(s["released"][order])  # positions in the stream
    audited = released[rng.random(len(released)) < f]
    hit = cusum_alarm(~s["correct"][order][audited], a.p0, a.p1, a.h)
    if hit < 0:
        return {"outcome": "missed"}
    pos = audited[hit]
    if pos < change:
        return {"outcome": "false_alarm"}
    after = (released >= change) & (released <= pos)
    return {
        "outcome": "detected",
        "audits_after": int((audited[: hit + 1] >= change).sum()),
        "delay_released": int(after.sum()),
        "escaped": int((~s["correct"][order][released[after]]).sum()),
    }


def spread(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f} [{min(values):.3f}, {max(values):.3f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit budget needed to catch a failed gate")
    ap.add_argument("--predictions", type=Path, nargs="+", default=[Path("runs/gas/preds.jsonl")])
    ap.add_argument("--data-dir", type=Path, default=Path("data/gas"))
    ap.add_argument("--train-batches", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--p0", type=float, default=0.01,
                    help="error rate among released items that the gate promises")
    ap.add_argument("--p1", type=float, default=0.05, help="error rate worth an alarm")
    ap.add_argument("--h", type=float, default=math.log(100), help="CUSUM alarm threshold")
    ap.add_argument("--fractions", nargs="+", type=float,
                    default=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0])
    ap.add_argument("--repeats", type=int, default=500, help="shuffled streams per seed")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--skip-drift", action="store_true")
    a = ap.parse_args()

    streams = [load_stream(p, a.threshold) for p in a.predictions]
    rng = np.random.default_rng(a.seed)

    # --- what each signal says, batch by batch ---
    x, _, batch = load_gas(a.data_dir)
    ref = batch <= a.train_batches
    x = Preprocessor(x[ref])(x)
    print(f"gate: release when confidence >= {a.threshold}; promised error <= {a.p0}. "
          f"{len(streams)} seed(s): mean [min, max]\n")
    print("| batch | error among released (needs labels) | share released (free) "
          "| drift AUC (free) |")
    print("|---|---|---|---|")
    for b in [0, *range(a.train_batches + 1, 11)]:
        errs, shares = [], []
        for s in streams:
            m = ~s["later"] if b == 0 else s["later"] & (s["batch"] == b)
            rel = m & s["released"]
            errs.append(1.0 - s["correct"][rel].mean() if rel.any() else float("nan"))
            shares.append(rel.sum() / m.sum())
        if a.skip_drift:
            drift = "-"
        elif b == 0:
            pick = rng.choice(np.flatnonzero(ref), size=300, replace=False)
            rest = np.setdiff1d(np.flatnonzero(ref), pick)
            drift = f"{drift_auc(x[rest], x[pick], a.seed):.3f} (random 300)"
        else:
            drift = f"{drift_auc(x[ref], x[batch == b], a.seed):.3f}"
        name = "in-domain" if b == 0 else f"batch{b:02d}"
        print(f"| {name} | {spread(errs)} | {spread(shares)} | {drift} |", flush=True)

    # --- the audit budget ---
    print(f"\nCUSUM on audited released items: p0={a.p0}, p1={a.p1}, h={a.h:.2f}; "
          f"{a.repeats} shuffled streams per seed. Brackets: range of the per-seed medians.\n")
    print("| audited share of released | false alarm before the change | detected "
          "| wrong items released before the alarm: median | audits spent after the change: median |")
    print("|---|---|---|---|---|")
    for f in a.fractions:
        per_seed = [[simulate(s, f, a, rng) for _ in range(a.repeats)] for s in streams]
        runs = [r for seed_runs in per_seed for r in seed_runs]
        det = [r for r in runs if r["outcome"] == "detected"]
        false = sum(r["outcome"] == "false_alarm" for r in runs) / len(runs)
        if det:
            meds = [
                np.median([r["escaped"] for r in sr if r["outcome"] == "detected"])
                for sr in per_seed if any(r["outcome"] == "detected" for r in sr)
            ]
            cells = (
                f"{np.median([r['escaped'] for r in det]):.0f} [{min(meds):.0f}, {max(meds):.0f}] "
                f"| {np.median([r['audits_after'] for r in det]):.0f}"
            )
        else:
            cells = "- | -"
        print(f"| {f:.1%} | {false:.1%} | {len(det) / len(runs):.1%} | {cells} |", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
