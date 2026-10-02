"""Fine-tune the decision LLM itself on SECOM and put it through the same test.

The tabular models in `secom_archs.py` are trained from scratch. This asks the
obvious follow-up: does the pretrained model this repo is about -- Qwen3-1.7B,
optionally with the released adapter merged in -- do any better when the wafer
is written out as text and it is fine-tuned on that?

A wafer becomes a `state` of its most informative sensors as z-scores, and one
yes/no question. Everything downstream is the repo's own path: the same prompt,
the same single-pass letter readout, the same CE + Brier loss. Folds, splits and
temperature fitting are `secom_study.cross_fit`, so the numbers sit next to the
tabular ones without an asterisk.

Sensors are selected inside each fold, on its training rows only. All 590 would
not fit the context, and selecting on the full data would leak the later period.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from secom_archs import summarise
from secom_study import Boosted, cross_fit, load_secom

from jevlike.dataset import DecisionDataset, collate
from jevlike.scorer import (
    ScorerConfig,
    letter_logits,
    letter_weight_matrix,
    load_model,
    mask_to_options,
)
from jevlike.train import decision_loss

# Turkish like the rest of the trained interface (see prompt.py).
QUESTION = {
    "id": "hatali",
    "prompt": "Bu wafer kalite testinde basarisiz olur mu?",
    "kind": "bool",
    "options": ["hayir", "evet"],
}


def top_sensors(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """The k columns whose mean differs most between failing and passing wafers."""
    gap = np.abs(x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0))
    return np.sort(np.argsort(-gap)[:k])


def as_rows(x: np.ndarray, y: np.ndarray | None, cols: np.ndarray) -> list[dict]:
    rows = []
    for i, values in enumerate(x[:, cols]):
        state = {f"s{c:03d}": f"{v:+.1f}" for c, v in zip(cols, values)}
        label = int(y[i]) if y is not None else 0
        rows.append({"state": state, "questions": [{**QUESTION, "label": label}], "source": "secom"})
    return rows


class WaferLLM:
    """One LoRA fitted on one fold's training rows."""

    def __init__(self, base, tokenizer, letter_w, x: np.ndarray, y: np.ndarray, seed: int, a) -> None:
        from peft import LoraConfig, get_peft_model

        self.tokenizer, self.letter_w = tokenizer, letter_w
        self.cols = top_sensors(x, y, a.top_k)
        ds = self._dataset(x, y)

        torch.manual_seed(seed)
        self.model = get_peft_model(
            base,
            LoraConfig(
                r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias="none",
                task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95))
        gen = torch.Generator().manual_seed(seed)
        for _ in range(a.llm_epochs):
            for idx in torch.randperm(len(ds), generator=gen).split(a.batch_size):
                batch = collate([ds[i] for i in idx.tolist()], tokenizer.pad_token_id).to(base.device)
                logits = letter_logits(
                    self.model, batch.input_ids, batch.attention_mask, batch.slot_positions, letter_w
                )
                loss, _ = decision_loss(
                    mask_to_options(logits, batch.option_counts), batch.labels,
                    batch.option_counts, brier_weight=0.5,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        self.model.eval()

    def _dataset(self, x: np.ndarray, y: np.ndarray | None) -> DecisionDataset:
        ds = DecisionDataset(as_rows(x, y, self.cols), self.tokenizer, max_length=2048)
        if ds.n_dropped or ds.n_invalid:
            # A dropped row would shift every later prediction onto the wrong wafer.
            raise RuntimeError(f"{ds.n_dropped} rows too long, {ds.n_invalid} invalid.")
        return ds

    @torch.no_grad()
    def logits(self, x: np.ndarray) -> list[list[float]]:
        ds = self._dataset(x, None)
        out: list[list[float]] = []
        for start in range(0, len(ds), 32):
            batch = collate(
                [ds[i] for i in range(start, min(start + 32, len(ds)))], self.tokenizer.pad_token_id
            ).to(self.model.device)
            lg = letter_logits(
                self.model, batch.input_ids, batch.attention_mask, batch.slot_positions, self.letter_w
            )
            out.extend(lg[:, 0, :2].float().cpu().tolist())
        return out


class TopKBoosted:
    """Boosted trees restricted to the sensors the LLM gets to see."""

    def __init__(self, x: np.ndarray, y: np.ndarray, seed: int, k: int) -> None:
        self.cols = top_sensors(x, y, k)
        self.inner = Boosted(x[:, self.cols], y, seed)

    def logits(self, x: np.ndarray) -> list[list[float]]:
        return self.inner.logits(x[:, self.cols])


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune the decision LLM on SECOM")
    ap.add_argument("--data-dir", type=Path, default=Path("data/secom"))
    ap.add_argument("--adapter", default="runs/qwen3-1.7b-v2/adapter",
                    help="adapter merged into the base before fine-tuning; 'none' for plain Qwen3")
    ap.add_argument("--model", choices=("llm", "gbdt_topk"), default="llm")
    ap.add_argument("--top-k", type=int, default=30, help="sensors written into the prompt")
    ap.add_argument("--llm-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--split-frac", type=float, default=0.6)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cal-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--random-split", action="store_true")
    ap.add_argument("--release-at", type=float, default=0.95)
    ap.add_argument("--out", type=Path, default=Path("runs/secom/preds_llm.jsonl"))
    a = ap.parse_args()

    x, y, t = load_secom(a.data_dir)
    if a.model == "gbdt_topk":
        trainer = lambda xt, yt, seed: TopKBoosted(xt, yt, seed, a.top_k)
    else:
        base, tokenizer = load_model(
            ScorerConfig(adapter_path=None if a.adapter == "none" else a.adapter)
        )
        letter_w = letter_weight_matrix(base, tokenizer).clone()
        # Without checkpointing a batch of 8 wafers overflows 12 GB, and on
        # Windows the overflow spills into system memory instead of failing.
        # The cap turns any remaining overflow into an ordinary CUDA OOM.
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.85)
        live: list[WaferLLM] = []

        def trainer(xt: np.ndarray, yt: np.ndarray, seed: int) -> WaferLLM:
            nonlocal base
            if live:  # strip the previous fold's LoRA so folds stay independent
                base = live.pop().model.unload()
            t0 = time.perf_counter()
            live.append(WaferLLM(base, tokenizer, letter_w, xt, yt, seed, a))
            print(f"  fine-tuned on {len(yt)} wafers in {time.perf_counter() - t0:.0f}s", flush=True)
            return live[-1]

    rows = cross_fit(x, y, t, a, verbose=True, trainer=trainer)[0]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    s = summarise(rows, a.release_at)
    split = "random (control)" if a.random_split else "chronological"
    print(f"\n{a.model} adapter={a.adapter} top_k={a.top_k} split={split}")
    for tag, name in (("in", "early"), ("out", "later")):
        print(
            f"{name:6s} acc={s[f'acc_{tag}']:.3f} ({s[f'acc_{tag}'] - s[f'majority_{tag}']:+.3f} "
            f"vs always-pass)  AUC={s[f'auc_{tag}']:.3f}  ECE={s[f'ece_{tag}']:.3f}  "
            f"lift={s[f'lift_{tag}']:.2f}"
        )
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
