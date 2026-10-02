"""Turn the JSONL mixture into slot-aligned tensors.

Training and evaluation share this code; prompt construction lives in exactly
one place so tokenization cannot drift between training and inference.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .prompt import build_prompt
from .schema import Question, SchemaError


def question_from_json(d: dict) -> Question:
    """Convert a JSONL question into a schema object."""
    return Question(
        id=d["id"],
        prompt=d["prompt"],
        kind=d["kind"],
        options=tuple(d["options"]),
        values=tuple(d["values"]) if d.get("values") else None,
    )


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


@dataclass
class Example:
    input_ids: list[int]
    slot_positions: list[int]
    option_counts: list[int]
    labels: list[int]
    source: str
    group: str


class DecisionDataset(Dataset):
    """Tokenize records; drop the ones exceeding max_length.

    We do not truncate: the answer slots live at the END of the sequence, so
    truncating from the right destroys them, and truncating from the left drops
    the start of the state. Dropping a long example is more honest than silently
    corrupting it -- the count of dropped examples is reported.
    """

    def __init__(
        self,
        rows: Sequence[dict],
        tokenizer: Any,
        *,
        max_length: int = 1024,
        use_chat_template: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chat_template = use_chat_template
        self.examples: list[Example] = []
        self.n_dropped = 0
        self.n_invalid = 0
        self.dropped_by_source: dict[str, int] = {}
        self.invalid_reasons: dict[str, int] = {}

        for row in rows:
            src = row.get("source", "?")
            try:
                ex = self._encode(row)
            except SchemaError as e:
                # One bad row must not take down the whole run: count, skip, report.
                self.n_invalid += 1
                key = f"{src}: {e}"
                self.invalid_reasons[key] = self.invalid_reasons.get(key, 0) + 1
                continue
            if ex is None:
                self.n_dropped += 1
                self.dropped_by_source[src] = self.dropped_by_source.get(src, 0) + 1
            else:
                self.examples.append(ex)

    def _encode(self, row: dict) -> Example | None:
        qs = [question_from_json(q) for q in row["questions"]]
        built = build_prompt(
            self.tokenizer, row["state"], qs, use_chat_template=self.use_chat_template
        )
        if len(built) > self.max_length:
            return None
        return Example(
            input_ids=list(built.input_ids),
            slot_positions=list(built.slot_positions),
            option_counts=list(built.option_counts),
            labels=[q["label"] for q in row["questions"]],
            source=row.get("source", "?"),
            group=row.get("group", "?"),
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]

    def lengths(self) -> list[int]:
        return [len(e.input_ids) for e in self.examples]


@dataclass
class Batch:
    input_ids: torch.Tensor  # [B, L]
    attention_mask: torch.Tensor  # [B, L]
    slot_positions: torch.Tensor  # [B, K]
    option_counts: torch.Tensor  # [B, K]  (0 = padding slot)
    labels: torch.Tensor  # [B, K]  (-100 = padding)
    sources: list[str]
    groups: list[str]

    def to(self, device: torch.device | str) -> Batch:
        return Batch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            slot_positions=self.slot_positions.to(device),
            option_counts=self.option_counts.to(device),
            labels=self.labels.to(device),
            sources=self.sources,
            groups=self.groups,
        )

    @property
    def n_slots(self) -> int:
        return int((self.labels != -100).sum().item())


def collate(items: Sequence[Example], pad_id: int) -> Batch:
    b = len(items)
    max_len = max(len(x.input_ids) for x in items)
    max_k = max(len(x.slot_positions) for x in items)

    input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((b, max_len), dtype=torch.long)
    slots = torch.zeros((b, max_k), dtype=torch.long)
    counts = torch.zeros((b, max_k), dtype=torch.long)
    labels = torch.full((b, max_k), -100, dtype=torch.long)

    for i, x in enumerate(items):
        n = len(x.input_ids)
        input_ids[i, :n] = torch.tensor(x.input_ids, dtype=torch.long)
        attn[i, :n] = 1
        k = len(x.slot_positions)
        slots[i, :k] = torch.tensor(x.slot_positions, dtype=torch.long)
        counts[i, :k] = torch.tensor(x.option_counts, dtype=torch.long)
        labels[i, :k] = torch.tensor(x.labels, dtype=torch.long)

    return Batch(
        input_ids=input_ids,
        attention_mask=attn,
        slot_positions=slots,
        option_counts=counts,
        labels=labels,
        sources=[x.source for x in items],
        groups=[x.group for x in items],
    )


def length_grouped_batches(
    dataset: DecisionDataset, batch_size: int, *, shuffle_seed: int | None = None, megabatch: int = 64
) -> Iterator[list[int]]:
    """Group similar lengths into the same batch -> less padding waste.

    Full sorting would kill variety across epochs; instead we shuffle first and
    sort within large blocks.
    """
    import random

    idx = list(range(len(dataset)))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(idx)

    lens = dataset.lengths()
    block = batch_size * megabatch
    batches: list[list[int]] = []
    for start in range(0, len(idx), block):
        chunk = sorted(idx[start : start + block], key=lambda i: lens[i])
        for bs in range(0, len(chunk), batch_size):
            batches.append(chunk[bs : bs + batch_size])

    if shuffle_seed is not None:
        random.Random(shuffle_seed + 1).shuffle(batches)
    yield from batches
