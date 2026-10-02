"""Typed decision primitives.

All three question kinds rest on one core operation: a categorical distribution
over a fixed label set. `bool` is the two-option case, `score` the ordered-level
case, `choice` the free-category case.

Because the schema is fixed in advance the model cannot return anything outside
it. The "no type errors" guarantee comes from here, not from the model -- and it
guarantees format validity, not truth.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

QuestionKind = Literal["choice", "score", "bool"]

MAX_OPTIONS = 52  # capacity of the single-token letter slot: A-Z + a-z


class SchemaError(ValueError):
    """Invalid question or schema definition."""


@dataclass(frozen=True)
class Question:
    """A single typed question.

    Attributes:
        id: Key used in the answer dict.
        prompt: The text shown to the model.
        kind: "choice" | "score" | "bool".
        options: Labels. Defaults to ["hayir", "evet"] for bool (Turkish for
            no/yes -- part of the trained interface), derived from `levels`
            for score.
        values: For `score` only: the numeric value of each level. Expected
            value is computed from these.
    """

    id: str
    prompt: str
    kind: QuestionKind = "choice"
    options: tuple[str, ...] = ()
    values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise SchemaError("Question id must not be empty.")
        if not self.prompt.strip():
            raise SchemaError(f"[{self.id}] prompt must not be empty.")
        if len(self.options) < 2:
            raise SchemaError(f"[{self.id}] needs at least 2 options, got {len(self.options)}.")
        if len(self.options) > MAX_OPTIONS:
            raise SchemaError(
                f"[{self.id}] at most {MAX_OPTIONS} options are supported in one pass, "
                f"got {len(self.options)}. For higher cardinality, chunk the menu."
            )
        if len(set(self.options)) != len(self.options):
            raise SchemaError(f"[{self.id}] options must be unique.")
        if self.kind == "score":
            if self.values is None or len(self.values) != len(self.options):
                raise SchemaError(f"[{self.id}] score question needs values the same length as options.")
        elif self.values is not None:
            raise SchemaError(f"[{self.id}] values is only valid for score questions.")

    # -- constructors ------------------------------------------------------

    @staticmethod
    def choice(id: str, prompt: str, options: Sequence[str]) -> Question:
        return Question(id=id, prompt=prompt, kind="choice", options=tuple(options))

    @staticmethod
    def boolean(id: str, prompt: str, labels: Sequence[str] = ("hayir", "evet")) -> Question:
        if len(labels) != 2:
            raise SchemaError(f"[{id}] a bool question needs exactly 2 labels.")
        return Question(id=id, prompt=prompt, kind="bool", options=tuple(labels))

    @staticmethod
    def score(
        id: str,
        prompt: str,
        levels: Sequence[str] | None = None,
        values: Sequence[float] | None = None,
        lo: int = 1,
        hi: int = 5,
    ) -> Question:
        """An ordered-level score question.

        If `levels` is omitted, integer levels lo..hi are generated.
        """
        if levels is None:
            levels = [str(v) for v in range(lo, hi + 1)]
            values = [float(v) for v in range(lo, hi + 1)]
        if values is None:
            values = [float(i) for i in range(len(levels))]
        return Question(
            id=id, prompt=prompt, kind="score", options=tuple(levels), values=tuple(float(v) for v in values)
        )


@dataclass
class Answer:
    """The answer to a single question."""

    id: str
    kind: QuestionKind
    options: tuple[str, ...]
    probs: tuple[float, ...]
    """Probabilities aligned with `options`; they sum to 1."""

    @property
    def best_index(self) -> int:
        return max(range(len(self.probs)), key=self.probs.__getitem__)

    @property
    def best(self) -> str:
        return self.options[self.best_index]

    @property
    def confidence(self) -> float:
        """The top probability. If calibrated, "the chance this answer is right"."""
        return self.probs[self.best_index]

    @property
    def margin(self) -> float:
        """Gap between best and runner-up. Useful for threshold-based escalation."""
        if len(self.probs) < 2:
            return 1.0
        top2 = sorted(self.probs, reverse=True)[:2]
        return top2[0] - top2[1]

    def expected_value(self, values: Sequence[float] | None = None) -> float:
        """Expected value, for `score` questions."""
        if values is None:
            raise SchemaError(f"[{self.id}] expected_value requires values.")
        return sum(p * v for p, v in zip(self.probs, values))

    def as_dict(self, values: Sequence[float] | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "best": self.best,
            "best_index": self.best_index,
            "confidence": round(self.confidence, 6),
            "margin": round(self.margin, 6),
            "probs": {o: round(p, 6) for o, p in zip(self.options, self.probs)},
        }
        if self.kind == "score" and values is not None:
            out["expected_value"] = round(self.expected_value(values), 6)
        if self.kind == "bool":
            # convention: options[1] is the "true" side
            out["p_true"] = round(self.probs[1], 6)
        return out


@dataclass
class Decision:
    """All answers for one request, plus timing."""

    answers: dict[str, Answer]
    latency_ms: float = 0.0
    n_forward_passes: int = 0
    prompt_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Answer:
        return self.answers[key]

    def as_dict(self, questions: Sequence[Question] | None = None) -> dict[str, Any]:
        values_by_id = {q.id: q.values for q in questions} if questions else {}
        return {
            "answers": {
                k: a.as_dict(values_by_id.get(k)) for k, a in self.answers.items()
            },
            "usage": {
                "latency_ms": round(self.latency_ms, 3),
                "forward_passes": self.n_forward_passes,
                "prompt_tokens": self.prompt_tokens,
            },
            "meta": self.meta,
        }


def format_state(state: Any) -> str:
    """Render an arbitrary state into text for embedding in the prompt."""
    if isinstance(state, str):
        return state.strip()
    if isinstance(state, (dict, list, tuple)):
        return json.dumps(state, ensure_ascii=False, indent=2, default=str)
    return str(state)
