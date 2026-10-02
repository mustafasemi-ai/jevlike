"""Prompt construction and answer-slot positioning.

The idea: write the state once and embed every question in the same sequence.
Each question ends in an "answer slot"; the logits at that position predict the
next token, i.e. the option letter. Causal masking means one forward pass yields
the logits for every slot simultaneously -- no decode loop.

WHY THE SENTINEL EXISTS: Qwen3's pre-tokenizer merges runs of punctuation
followed by whitespace. Writing a plain `"1) (\\n"` produces `" (\\n"` as a
*single* token, so the logits at that position predict what comes after the
newline rather than the option letter. This fails silently and corrupts every
downstream number. Putting a non-whitespace character right after the slot keeps
`" ("` as its own token. The sentinel is a dash meaning "not answered yet"; the
real answer is never written into the sequence, only read out as logits.

Slot positions are located via the fast tokenizer's offset mapping rather than
string search, so a tokenizer change cannot shift them silently.

NOTE ON LANGUAGE: the prompt scaffolding below is Turkish. This is not
cosmetic -- the released adapter was trained with these exact strings, so
changing them would silently degrade the model. Treat them as part of the
trained interface, not as user-facing copy.
"""

from __future__ import annotations

import string
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .schema import MAX_OPTIONS, Question, SchemaError, format_state

# A-Z then a-z: 52 slots. Order is fixed and must match between train and inference.
LETTERS: tuple[str, ...] = tuple(string.ascii_uppercase) + tuple(string.ascii_lowercase)

# The "not answered yet" marker placed after each slot. Must not be whitespace
# (see module docstring) and must not be a valid option letter.
SENTINEL = "-"

# Part of the trained interface -- see the language note in the module docstring.
SYSTEM_PROMPT = (
    "Sen bir karar motorusun. Sana bir durum ve numarali sorular verilir. "
    "Her soru icin yalnizca verilen secenek harflerinden birini secersin. "
    "Aciklama yazmazsin."
)


class PromptError(RuntimeError):
    """Prompt/tokenizer incompatibility."""


@dataclass(frozen=True)
class BuiltPrompt:
    """A tokenized prompt plus each question's slot position and option count."""

    text: str
    input_ids: tuple[int, ...]
    slot_positions: tuple[int, ...]
    """slot_positions[k] = token index of question k's slot (logits are read here)."""
    option_counts: tuple[int, ...]
    """option_counts[k] = number of options for question k.

    Option j always maps to LETTERS[j], so token ids need not be carried
    separately; the first n letters are that question's valid set.
    """
    question_ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.input_ids)


@lru_cache(maxsize=8)
def _letter_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Verify every letter is a single token and return the ids.

    Letters follow a "(" directly, so there is no leading space; we encode the
    bare character.
    """
    ids: list[int] = []
    for i, ch in enumerate(LETTERS):
        enc = tokenizer.encode(ch, add_special_tokens=False)
        if len(enc) != 1:
            raise PromptError(
                f"Letter '{ch}' is not a single token in this tokenizer ({enc}). "
                f"The single-token letter slot cannot be used with this model; "
                f"lower MAX_OPTIONS to {i} or score options by likelihood instead."
            )
        ids.append(enc[0])
    if len(set(ids)) != len(ids):
        raise PromptError("Letters collide on the same token id; unexpected tokenizer.")
    return tuple(ids)


def render_questions(questions: Sequence[Question]) -> str:
    """Render questions and their option letters as text."""
    lines: list[str] = []
    for k, q in enumerate(questions, start=1):
        if len(q.options) > MAX_OPTIONS:
            raise SchemaError(f"[{q.id}] {len(q.options)} options > {MAX_OPTIONS}")
        # Part of the trained interface -- do not translate.
        suffix = {"score": " (puan)", "bool": " (evet/hayir)"}.get(q.kind, "")
        lines.append(f"{k}. {q.prompt.strip()}{suffix}")
        for j, opt in enumerate(q.options):
            lines.append(f"   ({LETTERS[j]}) {opt}")
    return "\n".join(lines)


def build_prompt(
    tokenizer: Any,
    state: Any,
    questions: Sequence[Question],
    *,
    use_chat_template: bool = True,
    system_prompt: str = SYSTEM_PROMPT,
) -> BuiltPrompt:
    """Turn state + questions into one tokenized sequence and mark the slots."""
    if not questions:
        raise SchemaError("At least one question is required.")

    _letter_token_ids(tokenizer)  # fail fast on tokenizer incompatibility
    body = (
        "<state>\n"
        f"{format_state(state)}\n"
        "</state>\n\n"
        "Sorular:\n"  # trained interface -- do not translate
        f"{render_questions(questions)}"
    )

    if use_chat_template and getattr(tokenizer, "chat_template", None):
        head = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": body},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        head = f"{system_prompt}\n\n{body}\n\n"

    # Answer block: one slot per question. The slot ends at "("; the logits there
    # predict the option letter. The sentinel stops "(" from merging with the
    # line break into a single token.
    #
    # The sentinel is technically redundant on the last question (nothing follows
    # to merge with), but we keep it uniform so that tokenization is byte-for-byte
    # identical between training and inference regardless of question count.
    parts = [head, "Cevaplar:\n"]  # trained interface -- do not translate
    slot_char_ends: list[int] = []
    cursor = sum(len(p) for p in parts)
    for k in range(len(questions)):
        seg = f"{k + 1}) ("
        parts.append(seg)
        cursor += len(seg)
        slot_char_ends.append(cursor)  # position just past the "("
        tail = f"{SENTINEL}\n"
        parts.append(tail)
        cursor += len(tail)

    text = "".join(parts)

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids: list[int] = enc["input_ids"]
    offsets: list[tuple[int, int]] = enc["offset_mapping"]

    slot_positions = [_slot_position(offsets, end, text) for end in slot_char_ends]

    return BuiltPrompt(
        text=text,
        input_ids=tuple(input_ids),
        slot_positions=tuple(slot_positions),
        option_counts=tuple(len(q.options) for q in questions),
        question_ids=tuple(q.id for q in questions),
    )


def _slot_position(offsets: Sequence[tuple[int, int]], char_end: int, text: str) -> int:
    """Find the index of the token that ends at the "(" character.

    The token must end exactly at char_end. If it does not, BPE has merged "("
    with what follows, and the logits there no longer predict the option letter
    -- fail loudly rather than return a silently wrong result.
    """
    for idx, (start, end) in enumerate(offsets):
        if end == char_end and start < char_end:
            return idx
    context = text[max(0, char_end - 24) : char_end + 8].replace("\n", "\\n")
    raise PromptError(
        f"Answer slot at character {char_end} does not land on a token boundary "
        f"(...{context}...). The tokenizer is merging '(' with what follows; "
        f"change SENTINEL to a different non-whitespace character."
    )
