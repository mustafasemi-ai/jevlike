"""Turn the registry's datasets into a single JSONL mixture.

Two augmentations happen here and both are load-bearing:

1. MENU SUBSAMPLING. Showing a 77-class intent dataset with all 77 options every
   time produces a fixed classifier. Instead, each example gets a random-sized
   submenu that always contains the gold answer. The goal is a decision engine
   that "takes its categories as input": at inference it must work on a menu it
   has never seen.

2. ORDER SHUFFLING. Options are re-ordered per example, otherwise the model
   learns a position prior like "A is usually right" -- which inflates accuracy
   and wrecks calibration.

Output line:
    {"state":..., "questions":[{id,prompt,kind,options,label,values}],
     "source":..., "group":..., "lang":...}
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from collections import Counter
from pathlib import Path

from ..schema import MAX_OPTIONS
from .registry import HELD_OUT, REGISTRY, QSpec, Record, TaskSpec

warnings.filterwarnings("ignore", category=UserWarning)


def subsample_options(
    q: QSpec, rng: random.Random, *, min_k: int = 2, max_k: int = MAX_OPTIONS
) -> QSpec:
    """Shrink and shuffle the menu. The gold answer always stays in it.

    Skipped for score questions: subsetting or shuffling ordered levels is
    meaningless (expected value only makes sense in order).
    """
    n = len(q.options)
    if q.kind == "score":
        return q
    if q.kind == "bool":
        # For bool the menu is fixed; only the order carries meaning
        # (options[1] is the "true" side), so leave it alone.
        return q

    hi = min(n, max_k)
    lo = min(min_k, hi)
    k = rng.randint(lo, hi) if hi > lo else hi

    gold = q.options[q.label]
    others = [o for i, o in enumerate(q.options) if i != q.label]
    rng.shuffle(others)
    menu = [gold] + others[: k - 1]
    rng.shuffle(menu)
    return QSpec(
        id=q.id, prompt=q.prompt, kind=q.kind, options=menu, label=menu.index(gold), values=q.values
    )


def record_to_json(rec: Record) -> dict:
    return {
        "state": rec.state,
        "questions": [
            {
                "id": q.id,
                "prompt": q.prompt,
                "kind": q.kind,
                "options": q.options,
                "label": q.label,
                **({"values": q.values} if q.values else {}),
            }
            for q in rec.questions
        ],
        "source": rec.source,
        "group": rec.group,
        "lang": rec.lang,
    }


def harvest(
    spec: TaskSpec,
    split: str,
    limit: int,
    rng: random.Random,
    *,
    augment: bool,
    from_end: bool = False,
) -> list[dict]:
    """Load one dataset and convert it into records.

    from_end: when a dataset has no separate test split, take the evaluation
    rows from the END of the same split. Otherwise train and eval share the same
    first N rows and the scores are inflated by leakage.
    """
    from datasets import load_dataset

    # Converters drop some examples (ambiguous label, missing field), so pull a
    # bit more than the target.
    take = min(limit * 3, 60_000)
    window = f"{split}[-{take}:]" if from_end else f"{split}[:{take}]"
    ds = load_dataset(spec.hf_id, spec.config, split=window)

    rows: list[dict] = []
    for ex in ds:
        for rec in spec.convert(ds, ex):
            rec.source = spec.name
            rec.group = spec.group
            rec.lang = spec.lang
            if augment:
                rec.questions = [subsample_options(q, rng) for q in rec.questions]
            if any(len(q.options) > MAX_OPTIONS for q in rec.questions):
                continue
            # If the same option text appears twice the question is unanswerable:
            # which copy is correct is undefined. (Some multiple-choice datasets
            # contain duplicated choices.) Drop it rather than train on an
            # ambiguous target.
            if any(len(set(q.options)) != len(q.options) for q in rec.questions):
                continue
            rows.append(record_to_json(rec))
            if len(rows) >= limit:
                return rows
    return rows


def build(
    out_dir: Path,
    *,
    seed: int = 17,
    train_scale: float = 1.0,
    eval_scale: float = 1.0,
    only: list[str] | None = None,
) -> dict:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [s for s in REGISTRY if not only or s.name in only]
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    report: dict[str, dict] = {}

    for spec in specs:
        entry: dict = {"group": spec.group, "held_out": spec.group in HELD_OUT}
        # training side: held-out groups are never trained on
        if spec.group not in HELD_OUT:
            n = int(spec.max_train * train_scale)
            try:
                rows = harvest(spec, spec.split, n, rng, augment=True)
                train_rows += rows
                entry["train"] = len(rows)
            except Exception as e:  # dataset may have moved or broken; do not stop the mixture
                entry["train_error"] = str(e).splitlines()[0][:160]
                print(f"  ! {spec.name} train: {entry['train_error']}")
        else:
            entry["train"] = 0

        # evaluation side
        split = spec.eval_split or spec.split
        # If there is no separate eval split (or it has the same name) and this
        # task is trained on, take eval rows from the end -> no leakage.
        from_end = (split == spec.split) and spec.group not in HELD_OUT
        entry["eval_from_end"] = from_end
        n = int(spec.max_eval * eval_scale)
        try:
            rows = harvest(spec, split, n, rng, augment=True, from_end=from_end)
            eval_rows += rows
            entry["eval"] = len(rows)
        except Exception as e:
            entry["eval_error"] = str(e).splitlines()[0][:160]
            print(f"  ! {spec.name} eval: {entry['eval_error']}")

        report[spec.name] = entry
        print(
            f"  {spec.name:22s} group={spec.group:16s} "
            f"train={entry.get('train', 0):6d} eval={entry.get('eval', 0):5d}"
            f"{'  [HELD-OUT]' if entry['held_out'] else ''}"
        )

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    # The calibration split comes from the eval tail (same distribution as the
    # trained groups), not from training data: fitting the temperature on data
    # the model has memorised would be meaningless.
    in_task = [r for r in eval_rows if r["group"] not in HELD_OUT]
    held = [r for r in eval_rows if r["group"] in HELD_OUT]
    n_cal = max(1, len(in_task) // 3)
    cal_rows, in_task_rows = in_task[:n_cal], in_task[n_cal:]

    written = {}
    for name, rows in [
        ("train", train_rows),
        ("cal", cal_rows),
        ("eval_in_task", in_task_rows),
        ("eval_held_out", held),
    ]:
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        written[name] = len(rows)
        print(f"  -> {path.name}: {len(rows)} rows")

    n_slots = Counter()
    for r in train_rows:
        n_slots[len(r["questions"])] += 1
    summary = {
        "seed": seed,
        "written": written,
        "per_task": report,
        "train_slot_histogram": dict(sorted(n_slots.items())),
        "train_total_slots": sum(len(r["questions"]) for r in train_rows),
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Build the decision data mixture")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--train-scale", type=float, default=1.0)
    p.add_argument("--eval-scale", type=float, default=1.0)
    p.add_argument("--only", nargs="*", default=None, help="only these tasks")
    a = p.parse_args()

    s = build(
        a.out, seed=a.seed, train_scale=a.train_scale, eval_scale=a.eval_scale, only=a.only
    )
    print("\n=== summary ===")
    print(json.dumps(s["written"], ensure_ascii=False, indent=2))
    print(f"total training questions (slots): {s['train_total_slots']}")
    print(f"slots-per-example distribution: {s['train_slot_histogram']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
