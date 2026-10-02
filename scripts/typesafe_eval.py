"""Run TypeSafe's public workflow eval against our model.

Goal: a number on the SAME ruler as Jev (~87%) and system-one-open (76.7%). To
stay comparable with their published figures, the reference is constructed the
way TypeSafe defines it.

CAUTION -- this eval does not measure "accuracy":

  The reference answers are not ground truth; they are the mean of GPT-6 Astra
  and Claude Fable 5.1 probabilities ("consensus"). The metric is therefore "how
  close are you to what the frontier models said", not "how often are you right".
  A high score means resembling frontier models, not being correct.

Also: evals.typesafe.ai publishes only 5 example cases per workflow (the full
security_incidents set is 240 cases). This is therefore a small subset of the
real eval, with wide confidence intervals.

Usage:
    uv run python scripts/typesafe_eval.py --adapter runs/qwen3-1.7b-2k/adapter
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from jevlike.metrics import evaluate_probs
from jevlike.schema import Question, SchemaError
from jevlike.scorer import LetterSlotScorer, ScorerConfig

WORKFLOWS = [
    "security_incidents",
    "agent_trace_observability",
    "invoice_processing",
    "customer_service",
]
BASE_URL = "https://evals.typesafe.ai/{wf}-cases.js"


def fetch(wf: str, cache: Path) -> dict:
    """Download the viewer data (stripping the JS wrapper) and cache it."""
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{wf}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    req = urllib.request.Request(
        BASE_URL.format(wf=wf), headers={"User-Agent": "Mozilla/5.0"}
    )
    text = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    data = json.loads(text[text.index("(") + 1 : text.rindex(")")])
    f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def build_question(qid: str, spec: dict) -> Question | None:
    """Convert a TypeSafe question definition into our schema object.

    The criteria text goes into the prompt: option labels are short keys (e.g.
    "money_back") and the model needs to know what they mean.
    """
    kind = spec["type"]
    instr = spec["instructions"].strip()
    crit = spec.get("criteria")

    if kind == "noul":
        desc = ""
        if isinstance(crit, dict):
            desc = f"\n   evet ise: {crit.get('true', '')}\n   hayir ise: {crit.get('false', '')}"
        # our convention: options[1] is the "true" side
        return Question(id=qid, prompt=instr + desc, kind="bool", options=("false", "true"))

    if kind == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            return None
        desc = "\n" + "\n".join(f"   {i}: {c}" for i, c in enumerate(crit))
        return Question(
            id=qid,
            prompt=instr + desc,
            kind="score",
            options=tuple(str(i) for i in range(len(crit))),
            values=tuple(float(i) for i in range(len(crit))),
        )

    if kind == "choice":
        if not isinstance(crit, dict) or len(crit) < 2:
            return None
        desc = "\n" + "\n".join(f"   {k}: {v}" for k, v in crit.items())
        return Question(id=qid, prompt=instr + desc, kind="choice", options=tuple(crit.keys()))

    return None


def consensus(sets: list[dict], options: tuple[str, ...]) -> tuple[int | None, list[float] | None]:
    """Derive the target label (and distribution if available) from the reference sets.

    TypeSafe's definition: "mean of Astra and Fable probabilities for every
    question". Where probabilities are absent (the invoice workflow) we fall back
    to a majority vote.
    """
    # mean of the probabilities
    probs_acc: dict[str, float] = defaultdict(float)
    n_prob = 0
    for s in sets:
        pr = s.get("probabilities")
        if isinstance(pr, dict) and pr:
            for k, v in pr.items():
                probs_acc[str(k)] += float(v)
            n_prob += 1
    if n_prob:
        mean = {k: v / n_prob for k, v in probs_acc.items()}
        vec = [mean.get(o, 0.0) for o in options]
        total = sum(vec)
        if total > 0:
            vec = [v / total for v in vec]
            return max(range(len(vec)), key=vec.__getitem__), vec

    # no probabilities -> majority of the values
    votes: dict[str, int] = defaultdict(int)
    for s in sets:
        v = s.get("value")
        votes[str(v).lower() if isinstance(v, bool) else str(v)] += 1
    if not votes:
        return None, None
    best = max(votes, key=votes.get)
    if best not in options:
        return None, None
    return options.index(best), None


def collect_pairs(ev: dict) -> list[dict]:
    """Extract the (state, question, reference) triples for one workflow."""
    questions = ev["questions"]
    documents = ev["documents"]
    pairs: list[dict] = []

    for case_id, case in ev["cases"].items():
        ref_by_node = case.get("reference_answers") or {}
        # The qid -> question-index and node -> document mappings come from the
        # model records; the reference block does not carry them.
        qindex: dict[str, int] = {}
        docindex: dict[str, int] = {}
        for m in case.get("models", {}).values():
            for node in m.get("nodes", []):
                nm = node.get("node")
                if node.get("doc") is not None:
                    docindex.setdefault(nm, node["doc"])
                for qid, idx in (node.get("questions") or {}).items():
                    qindex.setdefault(qid, idx)

        for node_name, qids in ref_by_node.items():
            doc_i = docindex.get(node_name)
            if doc_i is None or doc_i >= len(documents):
                continue
            for qid, ref in qids.items():
                qi = qindex.get(qid)
                if qi is None or qi >= len(questions):
                    continue
                q = build_question(qid, questions[qi])
                if q is None:
                    continue
                label, ref_probs = consensus(ref.get("sets") or [], q.options)
                if label is None:
                    continue
                pairs.append(
                    {
                        "case_id": case_id,
                        "node": node_name,
                        "state": documents[doc_i],
                        "question": q,
                        "label": label,
                        "ref_probs": ref_probs,
                    }
                )
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description="TypeSafe public workflow eval")
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--adapter", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--cache", type=Path, default=Path("data/typesafe_eval"))
    p.add_argument("--out", type=Path, default=Path("runs/typesafe_eval.json"))
    p.add_argument(
        "--max-length",
        type=int,
        default=16384,
        help="invoice_processing prompts are ~10.4k tokens; the 4096 default drops them",
    )
    a = p.parse_args()

    print("fetching eval data...", flush=True)
    per_wf: dict[str, list[dict]] = {}
    for wf in WORKFLOWS:
        ev = fetch(wf, a.cache)["eval"]
        pairs = collect_pairs(ev)
        per_wf[wf] = pairs
        print(f"  {wf:<28} {len(pairs):>4} pairs  ({len(ev["cases"])} cases, n_cases={ev["n_cases"]})")
    total = sum(len(v) for v in per_wf.values())
    print(f"  total: {total} pairs\n", flush=True)

    cfg = ScorerConfig(model_id=a.model_id, adapter_path=a.adapter, max_length=a.max_length)
    if a.temperature:
        cfg.temperature = a.temperature
    print(f"model: {a.model_id}  adapter: {a.adapter or '(yok)'}  T={cfg.temperature}", flush=True)
    scorer = LetterSlotScorer(cfg)

    report: dict[str, Any] = {
        "model": a.model_id,
        "adapter": a.adapter,
        "temperature": cfg.temperature,
        "caution": (
            "Reference = mean of GPT-6 Astra + Fable 5.1 probabilities (consensus), "
            "NOT ground truth. The score measures closeness to frontier models."
        ),
        "workflows": {},
    }
    all_probs: list[list[float]] = []
    all_labels: list[int] = []

    for wf, pairs in per_wf.items():
        probs: list[list[float]] = []
        labels: list[int] = []
        reasons: dict[str, int] = defaultdict(int)
        for it in pairs:
            try:
                d = scorer.decide(it["state"], [it["question"]])
            except (SchemaError, ValueError) as e:
                # Count and report skips: dropping silently can make a whole
                # workflow vanish and inflate the overall score.
                reasons[type(e).__name__ + ": " + str(e)[:80]] += 1
                continue
            ans = d[it["question"].id]
            probs.append(list(ans.probs))
            labels.append(it["label"])

        n_skip = sum(reasons.values())
        if not labels:
            report["workflows"][wf] = {
                "pairs": 0, "skipped": n_skip, "skip_reasons": dict(reasons)
            }
            print(f"  {wf:<28} ALL SKIPPED ({n_skip} pairs)", flush=True)
            for r, c in sorted(reasons.items(), key=lambda x: -x[1])[:2]:
                print(f"      {c}x {r}", flush=True)
            continue

        rep = evaluate_probs(probs, labels)
        report["workflows"][wf] = {
            **rep.as_dict(), "skipped": n_skip, "pairs": len(labels),
            "skip_reasons": dict(reasons),
        }
        all_probs += probs
        all_labels += labels
        note = f"  ({n_skip} skipped)" if n_skip else ""
        print(f"  {wf:<28} {rep.summary()}{note}", flush=True)

    overall = evaluate_probs(all_probs, all_labels)
    report["overall"] = overall.as_dict()
    print(f"\n  {'OVERALL':<28} {overall.summary()}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
