"""HTTP service.

Two surfaces:

- POST /decide      -- this project's own contract
- POST /v1/systemone -- a TypeSafe System One compatible body, so you can
                        point an SDK at it by changing base_url

The model is loaded once per process; reloading per request would make the
latency numbers meaningless.
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .schema import Question, SchemaError
from .scorer import LetterSlotScorer, ScorerConfig

STATE: dict[str, Any] = {}


class QuestionIn(BaseModel):
    id: str
    prompt: str
    kind: Literal["choice", "score", "bool"] = "choice"
    options: list[str] | None = None
    values: list[float] | None = None


class DecideRequest(BaseModel):
    state: Any = Field(..., description="Text, JSON object, or list")
    questions: list[QuestionIn]
    temperature: float | None = None


def to_question(q: QuestionIn) -> Question:
    if q.kind == "bool":
        return Question.boolean(q.id, q.prompt, tuple(q.options or ("hayir", "evet")))
    if q.kind == "score":
        if q.options:
            return Question.score(q.id, q.prompt, levels=q.options, values=q.values)
        return Question.score(q.id, q.prompt)
    if not q.options:
        raise SchemaError(f"[{q.id}] a choice question requires options.")
    return Question.choice(q.id, q.prompt, q.options)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = ScorerConfig(
        model_id=os.environ.get("JEVLIKE_MODEL", "Qwen/Qwen3-1.7B"),
        adapter_path=os.environ.get("JEVLIKE_ADAPTER") or None,
    )
    t0 = time.perf_counter()
    scorer = LetterSlotScorer(cfg)
    temp_path = os.environ.get("JEVLIKE_TEMPERATURE_FILE")
    if temp_path and os.path.exists(temp_path):
        t = scorer.load_temperature(temp_path)
        print(f"loaded calibration temperature: T={t:.4f}")
    STATE["scorer"] = scorer
    print(f"model ready ({time.perf_counter() - t0:.1f}s), adapter={cfg.adapter_path or 'none'}")
    yield
    STATE.clear()


app = FastAPI(title="jevlike", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    s: LetterSlotScorer = STATE["scorer"]
    return {
        "status": "ready",
        "model": s.cfg.model_id,
        "adapter": s.cfg.adapter_path,
        "temperature": s.temperature,
    }


def _decide(req: DecideRequest) -> dict:
    scorer: LetterSlotScorer = STATE["scorer"]
    try:
        questions = [to_question(q) for q in req.questions]
    except SchemaError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    try:
        decision = scorer.decide(req.state, questions, temperature=req.temperature)
    except (ValueError, SchemaError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return decision.as_dict(questions)


@app.post("/decide")
def decide(req: DecideRequest) -> dict:
    return _decide(req)


@app.post("/v1/systemone")
def systemone(req: DecideRequest) -> dict:
    """System One style body. Field names kept close to the TypeSafe SDK."""
    out = _decide(req)
    answers = {}
    for qid, a in out["answers"].items():
        entry = {"best": a["best"], "confidence": a["confidence"], "probs": a["probs"]}
        if "p_true" in a:
            entry["p_true"] = a["p_true"]
        if "expected_value" in a:
            entry["expected_value"] = a["expected_value"]
        answers[qid] = entry
    return {"answers": answers, "usage": out["usage"]}


def main() -> int:
    import uvicorn

    p = argparse.ArgumentParser(description="jevlike HTTP service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--adapter", default=None)
    p.add_argument("--temperature-file", default=None)
    a = p.parse_args()

    if a.adapter:
        os.environ["JEVLIKE_ADAPTER"] = a.adapter
    if a.temperature_file:
        os.environ["JEVLIKE_TEMPERATURE_FILE"] = a.temperature_file
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
