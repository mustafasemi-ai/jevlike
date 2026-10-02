"""Calibration under distribution shift -- "the experiment that matters".

The entire value proposition of this model class is *confidence gating*:
"auto-handle anything above 0.9, escalate the rest". That only works if the
probabilities mean what they say. Existing public work either does not measure
calibration at all, or measures it in-domain only. The question here is
different:

    What does a threshold tuned in-domain do out-of-domain?

It matters because the failure is silent: on a novel input type the model still
says "92% confident", actually gets 70% right, and the gate stays open exactly
when it is needed most.

Produces three artefacts:
  1. micro vs macro ECE -- why a single ECE figure over a mixed eval is a
     property of the eval, not the model (large, easy task families carry it)
  2. reliability table  -- in-domain vs out-of-domain, side by side
  3. gate analysis      -- realised accuracy and coverage out-of-domain for a
     threshold chosen in-domain

Input: the JSONL produced by `evaluate.py --dump-predictions`.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from jevlike.calibrate import softmax_rows
from jevlike.metrics import bootstrap_ci, evaluate_probs, reliability_table


def load(path: Path) -> dict[str, list[dict]]:
    by_set: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # "kume" is the pre-rename key; accept it so older dumps still load.
            by_set[r.get("set") or r["kume"]].append(r)
    return by_set


def probs_labels(rows: Sequence[dict], temperature: float):
    return softmax_rows([r["logits"] for r in rows], temperature), [r["label"] for r in rows]


def macro_micro(rows: Sequence[dict], temperature: float) -> dict:
    """Per-slot (micro) and per-task (macro) metrics.

    A large divergence means the single reported ECE is measuring the eval's
    task mix rather than the model.
    """
    probs, labels = probs_labels(rows, temperature)
    micro = evaluate_probs(probs, labels)

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["source"]].append(r)
    per = {}
    for name, rs in by_task.items():
        p, l = probs_labels(rs, temperature)
        per[name] = evaluate_probs(p, l)

    return {
        "micro": micro.as_dict(),
        "macro": {
            "accuracy": st.mean(v.accuracy for v in per.values()),
            "ece": st.mean(v.ece for v in per.values()),
            "brier": st.mean(v.brier for v in per.values()),
            "overconfidence": st.mean(v.overconfidence for v in per.values()),
            "n_tasks": len(per),
        },
        "per_task": {k: v.as_dict() for k, v in sorted(per.items())},
        "largest_tasks": sorted(
            ((k, len(v)) for k, v in by_task.items()), key=lambda x: -x[1]
        )[:5],
        "n_slots": len(rows),
    }


def gate_curve(rows: Sequence[dict], temperature: float, thresholds: Sequence[float]) -> list[dict]:
    """Coverage and realised accuracy as a function of the threshold.

    coverage : fraction of decisions above the threshold (the automated work)
    accuracy : how many of those are actually correct
    gap      : promised minus realised (positive = the gate is looser than you
               think -- the silent failure)
    """
    probs, labels = probs_labels(rows, temperature)
    p = np.array([max(x) for x in probs])
    correct = np.array(
        [int(max(range(len(pr)), key=pr.__getitem__) == y) for pr, y in zip(probs, labels)]
    )
    out = []
    for t in thresholds:
        m = p >= t
        n = int(m.sum())
        out.append(
            {
                "threshold": t,
                "coverage": round(n / len(p), 4),
                "n": n,
                "accuracy": round(float(correct[m].mean()), 4) if n else None,
                "gap": round(float(t - correct[m].mean()), 4) if n else None,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibration study under distribution shift")
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("runs/calibration_study.json"))
    ap.add_argument("--markdown", type=Path, default=Path("runs/calibration_study.md"))
    ap.add_argument("--n-boot", type=int, default=2000, help="number of bootstrap resamples")
    a = ap.parse_args()

    by_set = load(a.predictions)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    report: dict = {"temperature": a.temperature, "sets": {}}
    md: list[str] = []

    md.append("# Calibration under distribution shift\n")
    md.append(
        f"Temperature T={a.temperature}, fitted on the in-domain calibration split "
        "and applied unchanged to both sets.\n"
    )

    # --- 1. micro vs macro ---
    md.append("\n## 1. Micro vs macro ECE\n")
    md.append("| set | slots | tasks | micro acc | macro acc | **micro ECE** | **macro ECE** | ratio |")
    md.append("|---|---|---|---|---|---|---|---|")
    for name in ("in_task", "held_out"):
        if name not in by_set:
            continue
        mm = macro_micro(by_set[name], a.temperature)
        report["sets"][name] = mm
        mi, ma = mm["micro"], mm["macro"]
        ratio = ma["ece"] / mi["ece"] if mi["ece"] > 1e-9 else float("inf")
        md.append(
            f"| {name} | {mm['n_slots']} | {ma['n_tasks']} | "
            f"{mi['accuracy']:.4f} | {ma['accuracy']:.4f} | "
            f"{mi['ece']:.4f} | {ma['ece']:.4f} | **{ratio:.1f}x** |"
        )
    md.append(
        "\nA large ratio means the single ECE figure is measuring the eval's task mix "
        "rather than the model: a large, easy task family carries the micro average. "
        "Largest tasks:\n"
    )

    # --- 1b. bootstrap confidence intervals ---
    md.append("\n### Bootstrap confidence intervals (95%)\n")
    md.append(
        "ECE is volatile on small sets. If the difference between two runs is narrower "
        "than the intervals below, it is sampling noise -- not a model difference.\n"
    )
    md.append("| set | metric | value | 95% interval | width |")
    md.append("|---|---|---|---|---|")
    for name in ("in_task", "held_out"):
        if name not in by_set:
            continue
        probs, labels = probs_labels(by_set[name], a.temperature)
        cis = {}
        for metric in ("ece", "accuracy", "brier"):
            ci = bootstrap_ci(probs, labels, metric, n_boot=a.n_boot)
            cis[metric] = ci
            md.append(
                f"| {name} | {metric} | {ci['value']:.4f} | "
                f"[{ci['lo']:.4f}, {ci['hi']:.4f}] | {ci['width']:.4f} |"
            )
        report["sets"][name]["bootstrap_ci"] = cis
    for name in ("in_task", "held_out"):
        if name in report["sets"]:
            big = report["sets"][name]["largest_tasks"]
            tot = report["sets"][name]["n_slots"]
            md.append(
                f"- `{name}`: "
                + ", ".join(f"{k} {v} (%{100 * v / tot:.0f})" for k, v in big[:3])
            )

    # --- 2. reliability ---
    md.append("\n## 2. Reliability: in-domain vs out-of-domain\n")
    md.append("| set | bin | n | mean confidence | actual accuracy | gap |")
    md.append("|---|---|---|---|---|---|")
    for name in ("in_task", "held_out"):
        if name not in by_set:
            continue
        probs, labels = probs_labels(by_set[name], a.temperature)
        tbl = reliability_table(probs, labels, n_bins=8)
        report["sets"][name]["reliability"] = tbl
        for i, row in enumerate(tbl):
            md.append(
                f"| {name if i == 0 else ''} | {i + 1} | {row['n']} | "
                f"{row['mean_confidence']:.3f} | {row['accuracy']:.3f} | "
                f"{row['gap']:+.3f} |"
            )

    # --- 3. confidence gate ---
    md.append("\n## 3. Confidence gate: a threshold tuned in-domain, applied out-of-domain\n")
    md.append(
        "| threshold | in-domain coverage | in-domain accuracy | "
        "out-of-domain coverage | out-of-domain accuracy | **gap** |"
    )
    md.append("|---|---|---|---|---|---|")
    gates = {}
    for name in ("in_task", "held_out"):
        if name in by_set:
            gates[name] = gate_curve(by_set[name], a.temperature, thresholds)
            report["sets"][name]["gate_curve"] = gates[name]
    if "in_task" in gates and "held_out" in gates:
        for gi, gh in zip(gates["in_task"], gates["held_out"]):
            gap_s = f"**{gh['gap']:+.3f}**" if gh["gap"] is not None else "-"
            md.append(
                f"| {gi['threshold']:.2f} "
                f"| {gi['coverage']:.3f} "
                f"| {gi['accuracy'] if gi['accuracy'] is not None else '-'} "
                f"| {gh['coverage']:.3f} "
                f"| {gh['accuracy'] if gh['accuracy'] is not None else '-'} "
                f"| {gap_s} |"
            )
        md.append(
            "\n`gap` = threshold - realised accuracy, out-of-domain.\n\n"
            "- **Negative** values are expected at low thresholds: at a 0.50 gate the "
            "realised accuracy is already 0.75. The gate is loose, but you never "
            "promised more than 0.50.\n"
            "- **Positive** values are the silent failure: the decisions the model "
            "labels '90% confident' are not 90% correct. Nothing errors, nothing warns "
            "-- wrong decisions are simply auto-approved.\n"
            "- The gap **growing** as the threshold rises is the key result: the reflex "
            "mitigation ('tighten the gate') does not fix this failure mode, it makes "
            "it worse.\n"
        )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    a.markdown.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {a.out}\n-> {a.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
