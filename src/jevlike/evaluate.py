"""Evaluation: accuracy + calibration + latency, broken down by task and group.

Three things are reported separately because each can hide the others:

- in_task   : unseen examples from task families that were trained on
- held_out  : task families absent from training entirely (generalisation)
- calibration: ECE before and after temperature scaling

`--latency` additionally compares "K questions in one pass" against "one pass
per question", to measure what encoding the shared state once actually costs in
accuracy. If the speed is not free, that is worth knowing.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .calibrate import fit_temperature, softmax_rows
from .dataset import DecisionDataset, collate, length_grouped_batches, read_jsonl
from .metrics import evaluate_probs, reliability_table
from .scorer import ScorerConfig, letter_logits, letter_weight_matrix, load_model, mask_to_options


def collect_logits(
    model: Any,
    tokenizer: Any,
    letter_w: torch.Tensor,
    rows: Sequence[dict],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[list[float]], list[int], list[str], list[str], float]:
    """Collect raw (pre-temperature) option logits for every slot.

    Returning logits rather than probabilities lets us try different
    temperatures later without re-running any forward passes.
    """
    ds = DecisionDataset(rows, tokenizer, max_length=max_length)
    if ds.n_dropped:
        print(f"  ({ds.n_dropped} examples dropped: too long)")
    if ds.n_invalid:
        print(f"  ({ds.n_invalid} examples dropped: invalid schema)")
        for reason, count in sorted(ds.invalid_reasons.items(), key=lambda x: -x[1])[:3]:
            print(f"      {count}x {reason}")

    logits_out: list[list[float]] = []
    labels_out: list[int] = []
    sources: list[str] = []
    groups: list[str] = []
    total_ms = 0.0

    model.eval()
    with torch.inference_mode():
        for idx in length_grouped_batches(ds, batch_size):
            batch = collate([ds[i] for i in idx], tokenizer.pad_token_id).to(device)
            t0 = time.perf_counter()
            lg = letter_logits(
                model, batch.input_ids, batch.attention_mask, batch.slot_positions, letter_w
            )
            lg = mask_to_options(lg, batch.option_counts)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_ms += (time.perf_counter() - t0) * 1000.0

            lab = batch.labels
            for i in range(lab.size(0)):
                for k in range(lab.size(1)):
                    y = int(lab[i, k].item())
                    if y == -100:
                        continue
                    n = int(batch.option_counts[i, k].item())
                    logits_out.append(lg[i, k, :n].float().tolist())
                    labels_out.append(y)
                    sources.append(batch.sources[i])
                    groups.append(batch.groups[i])
    return logits_out, labels_out, sources, groups, total_ms


def per_key_report(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    keys: Sequence[str],
    temperature: float,
) -> dict[str, dict]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        buckets[k].append(i)
    out: dict[str, dict] = {}
    for k, idxs in sorted(buckets.items()):
        probs = softmax_rows([logits[i] for i in idxs], temperature)
        out[k] = evaluate_probs(probs, [labels[i] for i in idxs]).as_dict()
    return out


def latency_benchmark(scorer, rows: Sequence[dict], n: int = 30) -> dict:
    """Single-request latency: K questions in one pass vs one pass per question."""
    from .dataset import question_from_json

    samples = [r for r in rows if len(r["questions"]) >= 3][:n]
    if not samples:
        samples = list(rows[:n])

    batched, per_q = [], []
    for r in samples:
        qs = [question_from_json(q) for q in r["questions"]]
        scorer.decide(r["state"], qs)  # warm-up
        d = scorer.decide(r["state"], qs)
        batched.append(d.latency_ms)

        t0 = time.perf_counter()
        for q in qs:
            scorer.decide(r["state"], [q])
        per_q.append((time.perf_counter() - t0) * 1000.0)

    batched.sort()
    per_q.sort()
    mid = len(batched) // 2
    k_avg = sum(len(r["questions"]) for r in samples) / len(samples)
    return {
        "n_samples": len(samples),
        "mean_questions": round(k_avg, 2),
        "single_pass_p50_ms": round(batched[mid], 2),
        "per_question_p50_ms": round(per_q[mid], 2),
        "speedup": round(per_q[mid] / max(batched[mid], 1e-6), 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="jevlike evaluation")
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--adapter", default=None, help="path to the LoRA adapter (omit for the untrained baseline)")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="at most N rows per set")
    p.add_argument("--latency", action="store_true", help="also run the latency benchmark")
    p.add_argument(
        "--dump-predictions",
        type=Path,
        default=None,
        help="write per-slot raw logits/labels as JSONL (for the calibration study)",
    )
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ScorerConfig(model_id=a.model_id, adapter_path=a.adapter, device=str(device))
    print(f"model: {a.model_id}  adapter: {a.adapter or '(none, untrained baseline)'}")
    model, tokenizer = load_model(cfg)
    letter_w = letter_weight_matrix(model, tokenizer)

    limit = a.limit or None
    sets = {
        "cal": read_jsonl(a.data_dir / "cal.jsonl", limit),
        "in_task": read_jsonl(a.data_dir / "eval_in_task.jsonl", limit),
        "held_out": read_jsonl(a.data_dir / "eval_held_out.jsonl", limit),
    }

    collected = {}
    for name, rows in sets.items():
        print(f"\n[{name}] processing {len(rows)} rows...")
        collected[name] = collect_logits(
            model, tokenizer, letter_w, rows,
            max_length=a.max_length, batch_size=a.batch_size, device=device,
        )
        print(f"  {len(collected[name][1])} slots")

    # Temperature is fitted on the cal split only, then applied to the eval sets.
    cal_logits, cal_labels = collected["cal"][0], collected["cal"][1]
    calib = fit_temperature(cal_logits, cal_labels)
    print(f"\n[calibration] {calib.summary()}")

    report: dict[str, Any] = {
        "model": a.model_id,
        "adapter": a.adapter,
        "calibration": calib.as_dict(),
        "sets": {},
    }

    for name in ("in_task", "held_out"):
        lg, lb, srcs, grps, ms = collected[name]
        if not lb:
            continue
        raw = evaluate_probs(softmax_rows(lg, 1.0), lb)
        cal_ = evaluate_probs(softmax_rows(lg, calib.temperature), lb)
        report["sets"][name] = {
            "before_temperature": raw.as_dict(),
            "after_temperature": cal_.as_dict(),
            "per_task": per_key_report(lg, lb, srcs, calib.temperature),
            "per_group": per_key_report(lg, lb, grps, calib.temperature),
            "reliability": reliability_table(
                softmax_rows(lg, calib.temperature), lb, n_bins=10
            ),
            "batched_forward_ms": round(ms, 1),
        }
        print(f"\n=== {name} ===")
        print(f"  T=1.0 : {raw.summary()}")
        print(f"  T={calib.temperature:.3f} : {cal_.summary()}")

    if a.latency:
        from .scorer import LetterSlotScorer

        scorer = LetterSlotScorer(cfg, model=model, tokenizer=tokenizer)
        scorer.cfg.temperature = calib.temperature
        report["latency"] = latency_benchmark(scorer, sets["in_task"])
        print(f"\n[latency] {json.dumps(report['latency'], ensure_ascii=False)}")

    out = a.out or Path("runs") / f"eval_{'lora' if a.adapter else 'base'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport: {out}")

    if a.dump_predictions:
        # We write raw logits, not probabilities, so the temperature can be
        # changed later without re-running any forward pass.
        a.dump_predictions.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with a.dump_predictions.open("w", encoding="utf-8") as f:
            for name, (lg, lb, srcs, grps, _) in collected.items():
                for logits, label, src, grp in zip(lg, lb, srcs, grps):
                    f.write(
                        json.dumps(
                            {
                                "set": name,
                                "source": src,
                                "group": grp,
                                "label": label,
                                "logits": [round(x, 5) for x in logits],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n += 1
        print(f"predictions: {a.dump_predictions} ({n} slots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
