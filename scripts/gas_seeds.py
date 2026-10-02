"""Which training run's gate will break? Seed instability of a confidence gate.

`gas_study.py` over six seeds showed something the averages hide. Accuracy and
ECE barely move from seed to seed, in-domain or after drift. The gate does: in
the same later batch one run releases items that are 89% right and another
99.9%. In-domain all runs look identical, so validation cannot tell them apart.

That the out-of-distribution behaviour of equally validated models differs is
known as underspecification (D'Amour et al., 2020). This script measures the
narrow case that matters for a gate, and asks three things:

    spread      : how much does the error among released items vary across
                  runs, next to how much accuracy and ECE vary?
    predictable : does anything measurable in-domain, or anything label-free
                  after deployment, say which runs will break?
    ensembles   : does averaging a few runs remove the lottery?

A run = one seed: its own folds, calibration split, initialisation and batch
order. Predictions are cached, so the analysis can be rerun without training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gas_study import load_gas, predict_all
from scipy.stats import spearmanr

from jevlike.metrics import expected_calibration_error


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def collect(a: argparse.Namespace) -> dict[str, np.ndarray]:
    """probs [seeds, rows, classes] and temperature [seeds], in dataset row order."""
    if a.cache.exists():
        return dict(np.load(a.cache))
    probs, temps = [], []
    for seed in range(1, a.seeds + 1):
        a.seed = seed
        rows = sorted(predict_all(a), key=lambda r: r["row"])
        scaled = np.asarray([r["logits"] for r in rows])
        raw = np.asarray([r["raw_logits"] for r in rows])
        probs.append(softmax(scaled))
        # logits = raw / T per fold; recover the mean T from any non-zero entry
        temps.append(float(np.median(np.abs(raw).sum(axis=1) / np.abs(scaled).sum(axis=1))))
        print(f"seed {seed}/{a.seeds}", flush=True)
    out = {"probs": np.asarray(probs, dtype=np.float32), "temp": np.asarray(temps)}
    a.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.cache, **out)
    return out


def describe(p: np.ndarray, y: np.ndarray, m: np.ndarray, threshold: float) -> dict[str, float]:
    """Metrics of one model's probabilities on the rows selected by mask m."""
    conf, ok = p[m].max(axis=1), p[m].argmax(axis=1) == y[m]
    rel = conf >= threshold
    return {
        "accuracy": float(ok.mean()),
        "ece": expected_calibration_error(conf.astype(np.float64), ok.astype(np.float64))[0],
        "confidence": float(conf.mean()),
        "share_released": float(rel.mean()),
        "gate_error": float(1.0 - ok[rel].mean()) if rel.sum() >= 30 else float("nan"),
    }


def row(name: str, v: np.ndarray, fmt: str = ".3f") -> str:
    v = v[~np.isnan(v)]
    return (f"| {name} | {np.mean(v):{fmt}} | {np.std(v):{fmt}} | {np.min(v):{fmt}} "
            f"| {np.median(v):{fmt}} | {np.max(v):{fmt}} |")


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed instability of the confidence gate")
    ap.add_argument("--data-dir", type=Path, default=Path("data/gas"))
    ap.add_argument("--cache", type=Path, default=Path("runs/gas/seeds.npz"))
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--train-batches", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cal-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--broken-at", type=float, default=0.05,
                    help="error among released items above which a gate counts as broken")
    ap.add_argument("--ensemble-size", type=int, default=5)
    a = ap.parse_args()
    a.random_split = False

    _, y, batch = load_gas(a.data_dir)
    data = collect(a)
    probs, temp = data["probs"], data["temp"]
    n_seeds = len(probs)
    early, later = batch <= a.train_batches, batch > a.train_batches

    ind = [describe(p, y, early, a.threshold) for p in probs]
    ood = [describe(p, y, later, a.threshold) for p in probs]
    col = lambda runs, k: np.asarray([r[k] for r in runs])

    # --- 1. spread ---
    print(f"{n_seeds} runs, gate at {a.threshold}\n")
    print("## 1. Spread across runs\n")
    print("| quantity | mean | std | min | median | max |")
    print("|---|---|---|---|---|---|")
    print(row("in-domain accuracy", col(ind, "accuracy")))
    print(row("in-domain ECE", col(ind, "ece")))
    print(row("in-domain error among released", col(ind, "gate_error"), ".4f"))
    print(row("later accuracy", col(ood, "accuracy")))
    print(row("later ECE", col(ood, "ece")))
    print(row("later error among released", col(ood, "gate_error")))

    print(f"\n| batch | accuracy: mean [min, max] | error among released: median [min, max] "
          f"| runs with a broken gate (> {a.broken_at}) |")
    print("|---|---|---|---|")
    for b in range(a.train_batches + 1, 11):
        per = [describe(p, y, batch == b, a.threshold) for p in probs]
        acc, ge = col(per, "accuracy"), col(per, "gate_error")
        valid = ge[~np.isnan(ge)]
        if len(valid) == 0:
            print(f"| {b} | {acc.mean():.3f} [{acc.min():.3f}, {acc.max():.3f}] | too few released | - |")
            continue
        print(
            f"| {b} | {acc.mean():.3f} [{acc.min():.3f}, {acc.max():.3f}] "
            f"| {np.median(valid):.3f} [{valid.min():.3f}, {valid.max():.3f}] "
            f"| {int((valid > a.broken_at).sum())} of {len(valid)} |"
        )

    # --- 2. predictability ---
    target = col(ood, "gate_error")
    print("\n## 2. Does anything say in advance which runs break?\n")
    print("Spearman correlation with the later error among released items.\n")
    print("| signal | available | correlation | p |")
    print("|---|---|---|---|")
    signals = [
        ("in-domain accuracy", "before deployment", col(ind, "accuracy")),
        ("in-domain ECE", "before deployment", col(ind, "ece")),
        ("in-domain mean confidence", "before deployment", col(ind, "confidence")),
        ("in-domain share released", "before deployment", col(ind, "share_released")),
        ("in-domain error among released", "before deployment", col(ind, "gate_error")),
        ("temperature", "before deployment", temp),
        ("later mean confidence", "after, no labels", col(ood, "confidence")),
        ("later share released", "after, no labels", col(ood, "share_released")),
        ("later accuracy", "after, needs labels", col(ood, "accuracy")),
        ("later ECE", "after, needs labels", col(ood, "ece")),
    ]
    for name, when, v in signals:
        ok = ~np.isnan(v) & ~np.isnan(target)
        rho = spearmanr(v[ok], target[ok])
        print(f"| {name} | {when} | {rho.statistic:+.2f} | {rho.pvalue:.3f} |")

    # --- 3. ensembles ---
    print("\n## 3. Does averaging runs remove the lottery?\n")
    print("| model | n | later accuracy | later error among released: median [min, max] "
          "| later share released |")
    print("|---|---|---|---|---|")

    def summarise(name: str, models: list[np.ndarray]) -> None:
        d = [describe(p, y, later, a.threshold) for p in models]
        ge = col(d, "gate_error")
        print(
            f"| {name} | {len(models)} | {col(d, 'accuracy').mean():.3f} "
            f"| {np.nanmedian(ge):.3f} [{np.nanmin(ge):.3f}, {np.nanmax(ge):.3f}] "
            f"| {col(d, 'share_released').mean():.3f} |"
        )

    summarise("single run", list(probs))
    k = a.ensemble_size
    summarise(f"average of {k} runs", [probs[i:i + k].mean(axis=0) for i in range(0, n_seeds - k + 1, k)])
    summarise(f"average of all {n_seeds}", [probs.mean(axis=0)])

    # An ensemble is less confident, so at a fixed threshold it releases less.
    # Compare again with every model releasing the same share of later items.
    def error_at(p: np.ndarray, share: float) -> float:
        conf, ok = p[later].max(axis=1), p[later].argmax(axis=1) == y[later]
        top = np.argsort(-conf)[: int(round(share * later.sum()))]
        return float(1.0 - ok[top].mean())

    print("\nSame comparison with every model releasing the same share of later items:\n")
    print(f"| share released | single run: median [min, max] | average of {k}: median [min, max] "
          f"| average of all {n_seeds} |")
    print("|---|---|---|---|")
    for share in (0.3, 0.2, 0.1):
        s = np.asarray([error_at(p, share) for p in probs])
        e = np.asarray([error_at(probs[i:i + k].mean(axis=0), share)
                        for i in range(0, n_seeds - k + 1, k)])
        print(
            f"| {share:.0%} | {np.median(s):.3f} [{s.min():.3f}, {s.max():.3f}] "
            f"| {np.median(e):.3f} [{e.min():.3f}, {e.max():.3f}] "
            f"| {error_at(probs.mean(axis=0), share):.3f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
