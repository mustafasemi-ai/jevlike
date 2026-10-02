"""Simulated wafers with a known failure mechanism and a chosen kind of shift.

SECOM shows a symptom -- ranking collapses in the later period while accuracy
and ECE look fine -- but cannot say what caused it. A simulator can: here the
failure mechanism is ours, so each kind of shift can be switched on alone and
its signature read off.

    control      : nothing changes; the later period is more of the same
    base_rate    : failures become rarer, the mechanism is untouched
    sensor_drift : the causal sensors read with an offset and a changed gain;
                   the physics is untouched, the readings are not
    mechanism    : failures start to come from a different set of sensors
    mechanism+base_rate : both at once -- what SECOM's symptom would need,
                   since there accuracy rises while ranking collapses

This says nothing about any real line. The model is only ever asked to recover
a rule we wrote, so the numbers are a map from cause to symptom, not evidence.

Training and scoring are `secom_study.py`'s, unchanged.
"""

from __future__ import annotations

import argparse

import numpy as np
from secom_study import Preprocessor, auc, predict_logits, train_mlp

from jevlike.calibrate import fit_temperature, softmax_rows
from jevlike.metrics import evaluate_probs

SCENARIOS = ("control", "base_rate", "sensor_drift", "mechanism", "mechanism+base_rate")
N_SENSORS, N_FACTORS, N_CAUSAL = 60, 8, 5


def simulate(scenario: str, n: int, split_frac: float, seed: int):
    """Readings, labels and the number of early rows. Rows are in time order."""
    rng = np.random.default_rng(seed)
    n_early = int(n * split_frac)
    late = np.arange(n) >= n_early

    # Sensors on a real tool are correlated: a few process factors drive many
    # readings. Independent columns would make the task unrealistically easy.
    mixing = rng.normal(size=(N_FACTORS, N_SENSORS)) / np.sqrt(N_FACTORS)
    true = rng.normal(size=(n, N_FACTORS)) @ mixing + 0.7 * rng.normal(size=(n, N_SENSORS))

    causal = rng.choice(N_SENSORS, size=2 * N_CAUSAL, replace=False)
    early_set, late_set = causal[:N_CAUSAL], causal[N_CAUSAL:]
    weights = rng.choice([-1.0, 1.0], size=N_CAUSAL) * rng.uniform(0.6, 1.2, size=N_CAUSAL)

    def logit(cols: np.ndarray, bias: float) -> np.ndarray:
        # One term is an out-of-range penalty: a sensor being far from nominal
        # in either direction is bad, which a linear model cannot see.
        return bias + true[:, cols] @ weights + 0.8 * (np.abs(true[:, cols[0]]) - 0.8)

    z = logit(early_set, -3.6)
    if scenario == "base_rate":
        z = np.where(late, z - 0.9, z)
    elif scenario == "mechanism":
        z = np.where(late, logit(late_set, -3.6), z)
    elif scenario == "mechanism+base_rate":
        z = np.where(late, logit(late_set, -3.6) - 0.9, z)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(np.int64)

    readings = true.copy()
    if scenario == "sensor_drift":
        offset = rng.choice([-1.5, 1.5], size=N_CAUSAL)
        gain = rng.uniform(0.5, 1.5, size=N_CAUSAL)
        readings[np.ix_(late, early_set)] = true[np.ix_(late, early_set)] * gain + offset
    return readings, y, n_early


def run(scenario: str, a: argparse.Namespace, seed: int) -> dict:
    x, y, n_early = simulate(scenario, a.n, a.split_frac, seed)
    rng = np.random.default_rng(seed)
    early = rng.permutation(n_early)
    n_test, n_cal = int(n_early * 0.25), int(n_early * 0.15)
    test_idx, cal_idx, train_idx = early[:n_test], early[n_test:n_test + n_cal], early[n_test + n_cal:]

    prep = Preprocessor(x[train_idx])
    model = train_mlp(prep(x[train_idx]), y[train_idx], seed=seed, epochs=a.epochs,
                      hidden=64, weight_decay=1e-2)
    temp = fit_temperature(
        predict_logits(model, prep(x[cal_idx])), y[cal_idx].tolist()
    ).temperature

    out: dict = {}
    for name, idx in (("in", test_idx), ("out", np.arange(n_early, len(y)))):
        probs = softmax_rows(predict_logits(model, prep(x[idx])), temp)
        labels = y[idx]
        p_fail = np.asarray(probs)[:, 1]
        rel = (1.0 - p_fail) >= a.release_at
        rep = evaluate_probs(probs, labels.tolist())
        out[name] = {
            "fail": labels.mean(),
            "acc": rep.accuracy,
            "ece": rep.ece,
            "auc": auc(p_fail, labels),
            "released": labels[rel].mean() if rel.any() else float("nan"),
            "held": labels[~rel].mean() if (~rel).any() else float("nan"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Which shift produces which symptom")
    ap.add_argument("--n", type=int, default=20000,
                    help="wafers per run; 1567 reproduces SECOM's sample size")
    ap.add_argument("--split-frac", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--release-at", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    keys = ("fail", "acc", "ece", "auc", "released", "held")
    print(f"n={a.n}, {a.repeats} repeats, mean (std of AUC in brackets). "
          f"Gate: release when P(pass) >= {a.release_at}\n")
    print("| scenario | period | fail rate | accuracy | ECE | AUC | released: fail | held back: fail |")
    print("|---|---|---|---|---|---|---|---|")
    for scenario in SCENARIOS:
        runs = [run(scenario, a, a.seed + r) for r in range(a.repeats)]
        for period, label in (("in", "early"), ("out", "later")):
            m = {k: np.nanmean([r[period][k] for r in runs]) for k in keys}
            sd = np.std([r[period]["auc"] for r in runs])
            print(
                f"| {scenario if period == 'in' else ''} | {label} | {m['fail']:.3f} | "
                f"{m['acc']:.3f} | {m['ece']:.3f} | {m['auc']:.3f} [{sd:.3f}] | "
                f"{m['released']:.3f} | {m['held']:.3f} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
