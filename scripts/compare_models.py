"""Put several prediction dumps through one identical calibration protocol.

For each model: fit a single temperature on the `cal` split, apply it unchanged
to `in_task` and `held_out`, and report micro/macro accuracy and ECE, the
overconfidence, and the promise gap at strict thresholds.

Usage:
    uv run python scripts/compare_models.py ours=runs/preds_v2.jsonl \
        laya_en=laya_eval/preds_laya_en.jsonl laya_ml=laya_eval/preds_laya_ml.jsonl
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from jevlike.calibrate import fit_temperature, softmax_rows
from jevlike.metrics import bootstrap_ci, evaluate_probs, expected_calibration_error


def load(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        out[r.get("set") or r["kume"]].append(r)
    return out


def summarise(rows: list[dict], T: float) -> dict:
    probs = softmax_rows([r["logits"] for r in rows], T)
    labels = [r["label"] for r in rows]
    micro = evaluate_probs(probs, labels)
    by_task: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_task[r["source"]].append(i)
    per = {k: evaluate_probs([probs[i] for i in ix], [labels[i] for i in ix])
           for k, ix in by_task.items()}
    conf = np.array([max(p) for p in probs])
    correct = np.array([int(int(np.argmax(p)) == y) for p, y in zip(probs, labels)])
    gaps = {}
    for t in (0.9, 0.95, 0.99):
        m = conf >= t
        gaps[t] = (float(t - correct[m].mean()), float(m.mean())) if m.any() else (None, 0.0)
    return {
        "micro_acc": micro.accuracy, "micro_ece": micro.ece, "overconf": micro.overconfidence,
        "macro_acc": st.mean(v.accuracy for v in per.values()),
        "macro_ece": st.mean(v.ece for v in per.values()),
        "gaps": gaps, "probs": probs, "labels": labels, "correct": correct,
    }


def main() -> int:
    specs = [a.split("=", 1) for a in sys.argv[1:]]
    if not specs:
        print(__doc__)
        return 2
    for name, path in specs:
        d = load(Path(path))
        T = fit_temperature([r["logits"] for r in d["cal"]], [r["label"] for r in d["cal"]]).temperature
        print(f"\n{'=' * 78}\n{name}   (T fitted on cal = {T:.3f})\n{'=' * 78}")
        for split in ("in_task", "held_out"):
            raw, cal = summarise(d[split], 1.0), summarise(d[split], T)
            ci = bootstrap_ci(cal["probs"], cal["labels"], "ece", n_boot=300)
            print(f"  [{split}]  n={len(d[split])}")
            print(f"    as shipped (T=1)   micro acc {raw['micro_acc']:.4f}  micro ECE {raw['micro_ece']:.4f}"
                  f"  overconf {raw['overconf']:+.4f}")
            print(f"    single T           micro acc {cal['micro_acc']:.4f}  micro ECE {cal['micro_ece']:.4f}"
                  f" [{ci['lo']:.4f},{ci['hi']:.4f}]  overconf {cal['overconf']:+.4f}")
            print(f"                       macro acc {cal['macro_acc']:.4f}  macro ECE {cal['macro_ece']:.4f}")
            g = "  ".join(
                f"@{t}: gap {v[0]:+.3f} cov {v[1]:.2f}" if v[0] is not None else f"@{t}: -"
                for t, v in cal["gaps"].items()
            )
            print(f"    promise gap        {g}")
            lc = [r.get("laya_confidence") for r in d[split]]
            if all(c is not None for c in lc):
                ece, _ = expected_calibration_error(np.array(lc, dtype=float), raw["correct"])
                print(f"    laya `confidence` field ECE {ece:.4f}  (mean {np.mean(lc):.3f} vs acc {raw['micro_acc']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
