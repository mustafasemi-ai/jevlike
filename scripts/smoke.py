"""Smoke test: prompt construction, slot alignment, single-pass decision, latency."""

from __future__ import annotations

import sys
import time

import torch

from jevlike.schema import Question
from jevlike.scorer import LetterSlotScorer, ScorerConfig


def main() -> int:
    cfg = ScorerConfig()
    print(f"model: {cfg.model_id}")
    t0 = time.perf_counter()
    scorer = LetterSlotScorer(cfg)
    print(f"load: {time.perf_counter() - t0:.1f}s")
    print(f"vram : {torch.cuda.memory_allocated() / 1e9:.2f} GB\n")

    state = {
        "kanal": "email",
        "konu": "Kartimdan iki kere cekim yapilmis",
        "govde": (
            "Merhaba, dun aksam premium abonelige gectim ama ekstremde ayni tutar "
            "iki kez gorunuyor. Fazla cekilen tutarin iadesini istiyorum. "
            "Ayrica bu ay icinde uyeligi iptal edersem ne olur?"
        ),
        "musteri_kidemi_ay": 14,
    }

    questions = [
        Question.choice(
            "departman",
            "Bu talep hangi departmana yonlendirilmeli?",
            ["faturalama", "teknik_destek", "satis", "hesap_yonetimi", "spam"],
        ),
        Question.boolean("iade_talebi", "Musteri acikca para iadesi istiyor mu?"),
        Question.boolean("iptal_riski", "Musteri uyeligi iptal etmeyi dusunuyor mu?"),
        Question.score(
            "aciliyet", "Bu talebin aciliyeti nedir?", lo=1, hi=5
        ),
        Question.choice(
            "duygu", "Musterinin tonu nasil?", ["ofkeli", "hayal_kirikligi", "notr", "memnun"]
        ),
    ]

    # Is slot alignment right? Inspect the tail of the prompt.
    built = scorer.build(state, questions)
    print(f"prompt: {len(built)} tokens, {len(built.slot_positions)} slots")
    print(f"slots : {built.slot_positions}")
    for k, pos in enumerate(built.slot_positions):
        tok = scorer.tokenizer.decode([built.input_ids[pos]])
        assert "(" in tok, f"slot {k} does not end in '(': {tok!r}"
    print("slot alignment: OK\n")

    # warm-up + measurement
    for _ in range(2):
        scorer.decide(state, questions)
    lat = []
    for _ in range(10):
        d = scorer.decide(state, questions)
        lat.append(d.latency_ms)
    lat.sort()

    print("--- decision (T=1.0, untrained) ---")
    for q in questions:
        a = d[q.id]
        top = sorted(zip(a.options, a.probs), key=lambda x: -x[1])[:3]
        extra = ""
        if q.kind == "score":
            extra = f"  E[x]={a.expected_value(q.values):.2f}"
        print(f"  {q.id:14s} -> {a.best:18s} p={a.confidence:.3f} margin={a.margin:.3f}{extra}")
        print(f"                  {', '.join(f'{o}:{p:.3f}' for o, p in top)}")

    print(f"\nlatency: p50={lat[4]:.1f}ms  p90={lat[8]:.1f}ms  min={lat[0]:.1f}ms")
    print(f"5 questions in one pass -> ~{lat[4] / len(questions):.1f}ms per question")
    return 0


if __name__ == "__main__":
    sys.exit(main())
