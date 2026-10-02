"""Typed decisions in a single forward pass.

No decode loop. The state is encoded once, every question's answer slot lives in
the same sequence, and all their logits are read out in one pass.

Two memory shortcuts, both resting on the same observation: the answer is always
one of 52 letters.

1. The full [B, L, V] logit tensor is never materialised (V=151936 for Qwen3;
   that is ~600 MB for a single 2k-token example). We take the body's final
   hidden states and gather only the slot positions -> [B, K, H].
2. Only the 52 letter rows of lm_head are used, not all of it -> [B, K, 52].
   The vocabulary dimension leaves the computation entirely.

The same path is used during training; gradients are only needed at the slots
anyway.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .prompt import BuiltPrompt, _letter_token_ids, build_prompt
from .schema import Answer, Decision, Question

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
NEG_INF = -1e4  # stands in for -inf under bf16: softmax gives 0 without NaNs


@dataclass
class ScorerConfig:
    model_id: str = DEFAULT_MODEL
    adapter_path: str | None = None
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    temperature: float = 1.0
    """Calibration temperature. 1.0 = raw model probabilities (usually overconfident)."""
    use_chat_template: bool = True
    attn_implementation: str = "sdpa"
    max_length: int = 4096


def load_model(cfg: ScorerConfig) -> tuple[Any, Any]:
    """Load model + tokenizer. If adapter_path is given, apply the LoRA on top."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        dtype=cfg.dtype,
        attn_implementation=cfg.attn_implementation,
    )
    if cfg.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, cfg.adapter_path)
        model = model.merge_and_unload()  # no LoRA overhead at inference

    model.to(cfg.device)
    model.eval()
    return model, tokenizer


def decoder_body(model: Any) -> Any:
    """Return the transformer body (without the LM head).

    Reaching in via `model.model` gives the wrong layer under a PEFT wrapper:
    PeftModel.model is a *ForCausalLM*, not the body, and calling it produces
    the full [B, L, V] logits -- destroying every memory saving in this module.
    get_decoder() returns the right layer for both plain and wrapped models.
    """
    if hasattr(model, "get_decoder"):
        return model.get_decoder()
    return getattr(model, "model", model)  # pragma: no cover - older versions


def letter_weight_matrix(model: Any, tokenizer: Any) -> torch.Tensor:
    """Only the 52 letter rows of lm_head: [52, H]."""
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if lm_head is None:  # pragma: no cover - fail loudly if the architecture changes
        raise AttributeError("Model has no output embedding; this scorer does not support it.")
    if getattr(lm_head, "bias", None) is not None:
        raise NotImplementedError("lm_head has a bias; the letter-matrix path ignores it.")
    ids = torch.tensor(_letter_token_ids(tokenizer), device=lm_head.weight.device)
    # detach: the letter rows are a fixed readout basis, not a trained parameter.
    # Without detach the tensor carries an autograd graph tied to lm_head.weight
    # and every step attempts a second backward through that same graph.
    return lm_head.weight.index_select(0, ids).detach()


def letter_logits(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    slot_positions: torch.Tensor,
    letter_w: torch.Tensor,
) -> torch.Tensor:
    """The 52 letter logits at each slot position.

    Args:
        input_ids: [B, L]
        attention_mask: [B, L]
        slot_positions: [B, K]
        letter_w: [52, H]

    Returns:
        [B, K, 52]
    """
    body = decoder_body(model)
    out = body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = out.last_hidden_state  # [B, L, H]

    idx = slot_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
    gathered = hidden.gather(dim=1, index=idx)  # [B, K, H]
    return gathered @ letter_w.t().to(gathered.dtype)  # [B, K, 52]


def mask_to_options(logits: torch.Tensor, option_counts: torch.Tensor) -> torch.Tensor:
    """Suppress the letters that are not valid options for each question.

    Args:
        logits: [B, K, 52]
        option_counts: [B, K] -- number of valid options (0 = padding slot)

    Returns:
        [B, K, 52] -- invalid positions set to NEG_INF
    """
    ar = torch.arange(logits.size(-1), device=logits.device)
    valid = ar.view(1, 1, -1) < option_counts.unsqueeze(-1)
    return logits.masked_fill(~valid, NEG_INF)


class LetterSlotScorer:
    """The core engine: typed decisions in a single pass."""

    def __init__(self, cfg: ScorerConfig | None = None, model: Any = None, tokenizer: Any = None):
        self.cfg = cfg or ScorerConfig()
        if model is None or tokenizer is None:
            model, tokenizer = load_model(self.cfg)
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(self.cfg.device)
        self.letter_w = letter_weight_matrix(model, tokenizer)

    # -- calibration -------------------------------------------------------

    @property
    def temperature(self) -> float:
        return self.cfg.temperature

    def load_temperature(self, path: str | Path) -> float:
        """Load the temperature written by calibrate.py."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.cfg.temperature = float(data["temperature"])
        return self.cfg.temperature

    # -- decisions ---------------------------------------------------------

    def build(self, state: Any, questions: Sequence[Question]) -> BuiltPrompt:
        return build_prompt(
            self.tokenizer, state, questions, use_chat_template=self.cfg.use_chat_template
        )

    def _pack(self, built: Sequence[BuiltPrompt]) -> tuple[torch.Tensor, ...]:
        """Batched tensors with right padding."""
        pad_id = self.tokenizer.pad_token_id
        b = len(built)
        max_len = max(len(x) for x in built)
        max_k = max(len(x.slot_positions) for x in built)

        ids = torch.full((b, max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((b, max_len), dtype=torch.long)
        slots = torch.zeros((b, max_k), dtype=torch.long)
        counts = torch.zeros((b, max_k), dtype=torch.long)
        for i, x in enumerate(built):
            ids[i, : len(x)] = torch.tensor(x.input_ids, dtype=torch.long)
            mask[i, : len(x)] = 1
            k = len(x.slot_positions)
            slots[i, :k] = torch.tensor(x.slot_positions, dtype=torch.long)
            counts[i, :k] = torch.tensor(x.option_counts, dtype=torch.long)
            # leftover slots: position 0, count 0 -> fully masked out
        return (
            ids.to(self.device),
            mask.to(self.device),
            slots.to(self.device),
            counts.to(self.device),
        )

    def _answers(
        self, logits: torch.Tensor, questions: Sequence[Question], temp: float
    ) -> dict[str, Answer]:
        """[K, 52] masked logits -> dict of answers."""
        answers: dict[str, Answer] = {}
        for k, q in enumerate(questions):
            ql = logits[k, : len(q.options)].float()
            probs = F.softmax(ql / temp, dim=-1)
            answers[q.id] = Answer(
                id=q.id, kind=q.kind, options=q.options, probs=tuple(probs.tolist())
            )
        return answers

    @torch.inference_mode()
    def decide(
        self,
        state: Any,
        questions: Sequence[Question],
        *,
        temperature: float | None = None,
    ) -> Decision:
        """Answer N questions about one state in a single forward pass."""
        t0 = time.perf_counter()
        built = self.build(state, questions)
        if len(built) > self.cfg.max_length:
            raise ValueError(
                f"Prompt {len(built)} token, limit {self.cfg.max_length}. "
                f"Shorten the state or raise max_length."
            )
        ids, mask, slots, counts = self._pack([built])
        logits = letter_logits(self.model, ids, mask, slots, self.letter_w)
        logits = mask_to_options(logits, counts)

        temp = self.temperature if temperature is None else temperature
        answers = self._answers(logits[0], questions, temp)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return Decision(
            answers=answers,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            n_forward_passes=1,
            prompt_tokens=len(built),
            meta={"temperature": temp, "model": self.cfg.model_id},
        )

    @torch.inference_mode()
    def decide_many(
        self,
        items: Sequence[tuple[Any, Sequence[Question]]],
        *,
        batch_size: int = 8,
        temperature: float | None = None,
    ) -> list[Decision]:
        """Batched execution, for evaluation."""
        temp = self.temperature if temperature is None else temperature
        results: list[Decision] = []

        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            built = [self.build(s, qs) for s, qs in chunk]
            ids, mask, slots, counts = self._pack(built)

            t0 = time.perf_counter()
            logits = letter_logits(self.model, ids, mask, slots, self.letter_w)
            logits = mask_to_options(logits, counts)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000.0

            for i, (b, (_, questions)) in enumerate(zip(built, chunk)):
                results.append(
                    Decision(
                        answers=self._answers(logits[i], questions, temp),
                        latency_ms=elapsed / len(chunk),
                        n_forward_passes=1,
                        prompt_tokens=len(b),
                        meta={"temperature": temp, "batched": True},
                    )
                )
        return results
