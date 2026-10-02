"""Put two evaluation reports side by side (e.g. base vs LoRA).

Usage:
    uv run python scripts/compare.py runs/eval_base.json runs/eval_lora.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(v: float, width: int = 7, digits: int = 4) -> str:
    return f"{v:>{width}.{digits}f}"


def delta(a: float, b: float, *, lower_is_better: bool = False) -> str:
    d = b - a
    good = (d < 0) if lower_is_better else (d > 0)
    sign = "+" if d > 0 else ""
    flag = "  " if abs(d) < 1e-6 else (" v" if good else " ^")
    return f"{sign}{d:.4f}{flag}"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))

    print(f"A = {a_path.name}  (adapter: {a.get('adapter') or 'none'})")
    print(f"B = {b_path.name}  (adapter: {b.get('adapter') or 'none'})")
    t_a = a["calibration"]["temperature"]
    t_b = b["calibration"]["temperature"]
    print(f"\ntemperature: A T={t_a:.3f}  B T={t_b:.3f}")

    metrics = [
        ("accuracy", False),
        ("ece", True),
        ("brier", True),
        ("nll", True),
        ("mean_confidence", False),
        ("overconfidence", True),
    ]

    for split in ("in_task", "held_out"):
        if split not in a.get("sets", {}) or split not in b.get("sets", {}):
            continue
        print(f"\n{'=' * 62}\n{split}  (after temperature scaling)\n{'=' * 62}")
        print(f"{'metric':<18}{'A':>9}{'B':>9}   delta")
        ka = a["sets"][split]["after_temperature"]
        kb = b["sets"][split]["after_temperature"]
        for m, lower in metrics:
            print(f"{m:<18}{fmt(ka[m], 9)}{fmt(kb[m], 9)}   {delta(ka[m], kb[m], lower_is_better=lower)}")

        # biggest per-task movements
        ga, gb = a["sets"][split]["per_task"], b["sets"][split]["per_task"]
        shared = sorted(set(ga) & set(gb), key=lambda k: gb[k]["accuracy"] - ga[k]["accuracy"])
        if shared:
            print("\n  per-task accuracy (6 largest movements):")
            for k in shared[:3] + shared[-3:]:
                print(
                    f"    {k:<22}{fmt(ga[k]['accuracy'], 8)}{fmt(gb[k]['accuracy'], 8)}"
                    f"   {delta(ga[k]['accuracy'], gb[k]['accuracy'])}"
                    f"   (n={gb[k]['n']})"
                )

    for name, rep in (("A", a), ("B", b)):
        if "latency" in rep:
            print(f"\nlatency {name}: {json.dumps(rep['latency'], ensure_ascii=False)}")

    print("\n(v = better, ^ = worse)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
