"""Calibrated decision training with LoRA.

Loss = CE + lambda * Brier.

Why not cross-entropy alone: CE pushes up the probability of the correct option
and is never satisfied -- going from p=0.99 to p=0.999 is still rewarded. The
model therefore learns confidence far beyond its accuracy; the p=1.000 we see
from the untrained base model is exactly this. Brier is a bounded *proper
scoring rule*: it pulls the entire distribution toward reality and penalises
over-sharpening.

This is the open-source stand-in for TypeSafe's unpublished RLCD. No claim of
equivalence is made; the measurable goal is the same -- probabilities that track
accuracy.

Temperature scaling runs AFTER training as a separate step (calibrate.py). The
two are complementary: Brier fixes the model's internal distribution, the
temperature closes the residual systematic offset.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import Batch, DecisionDataset, collate, length_grouped_batches, read_jsonl
from .metrics import evaluate_probs
from .scorer import letter_logits, letter_weight_matrix, mask_to_options


@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen3-1.7B"
    data_dir: str = "data"
    out_dir: str = "runs/qwen3-1.7b-lora"
    max_length: int = 1024
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    epochs: float = 1.0
    max_steps: int = -1
    brier_weight: float = 0.5
    label_smoothing: float = 0.0
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    grad_checkpointing: bool = True
    seed: int = 17
    train_limit: int = 0
    eval_limit: int = 800
    eval_every: int = 250
    log_every: int = 25


def decision_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    option_counts: torch.Tensor,
    *,
    brier_weight: float,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-slot CE + Brier.

    Args:
        logits: [B, K, 52] -- invalid letters are already masked
        labels: [B, K] -- -100 marks padding
        option_counts: [B, K]
    """
    b, k, n_letters = logits.shape
    flat_logits = logits.reshape(b * k, n_letters).float()
    flat_labels = labels.reshape(b * k)
    flat_counts = option_counts.reshape(b * k)

    keep = flat_labels != -100
    if not keep.any():
        zero = logits.sum() * 0.0
        return zero, {"ce": 0.0, "brier": 0.0, "acc": 0.0, "conf": 0.0}

    lg = flat_logits[keep]
    lb = flat_labels[keep]
    cnt = flat_counts[keep]

    ce = F.cross_entropy(lg, lb, label_smoothing=label_smoothing)

    probs = F.softmax(lg, dim=-1)
    onehot = F.one_hot(lb, num_classes=n_letters).to(probs.dtype)
    # Probability on invalid letters is already ~0; keep masked slots out of Brier
    ar = torch.arange(n_letters, device=lg.device)
    valid = ar.view(1, -1) < cnt.unsqueeze(-1)
    brier = (((probs - onehot) ** 2) * valid).sum(dim=-1).mean()

    loss = ce + brier_weight * brier
    with torch.no_grad():
        pred = lg.argmax(dim=-1)
        acc = (pred == lb).float().mean().item()
        conf = probs.max(dim=-1).values.mean().item()
    return loss, {"ce": ce.item(), "brier": brier.item(), "acc": acc, "conf": conf}


@torch.no_grad()
def run_eval(model, letter_w, dataset: DecisionDataset, cfg: TrainConfig, device) -> dict:
    """Accuracy + calibration on a small slice."""
    model.eval()
    pad_id = dataset.tokenizer.pad_token_id
    all_probs: list[list[float]] = []
    all_labels: list[int] = []

    for batch_idx in length_grouped_batches(dataset, cfg.batch_size * 2):
        batch = collate([dataset[i] for i in batch_idx], pad_id).to(device)
        logits = letter_logits(
            model, batch.input_ids, batch.attention_mask, batch.slot_positions, letter_w
        )
        logits = mask_to_options(logits, batch.option_counts)
        probs = F.softmax(logits.float(), dim=-1)

        lab = batch.labels
        for i in range(lab.size(0)):
            for k in range(lab.size(1)):
                y = int(lab[i, k].item())
                if y == -100:
                    continue
                n = int(batch.option_counts[i, k].item())
                all_probs.append(probs[i, k, :n].tolist())
                all_labels.append(y)

    model.train()
    return evaluate_probs(all_probs, all_labels).as_dict()


def main() -> int:
    p = argparse.ArgumentParser(description="jevlike LoRA training")
    for f, v in asdict(TrainConfig()).items():
        if isinstance(v, bool):
            p.add_argument(f"--{f.replace('_', '-')}", type=lambda s: s.lower() == "true", default=v)
        else:
            p.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    args = p.parse_args()
    cfg = TrainConfig(**vars(args))

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading model: {cfg.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    # lm_head is not trained; take the letter matrix before LoRA, as a constant.
    letter_w = letter_weight_matrix(model, tokenizer).to(device).clone()

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.to(device)
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M ({100 * trainable / total:.2f}%)")

    print("tokenizing data...")
    t0 = time.perf_counter()
    train_rows = read_jsonl(Path(cfg.data_dir) / "train.jsonl", cfg.train_limit or None)
    eval_rows = read_jsonl(Path(cfg.data_dir) / "cal.jsonl", cfg.eval_limit or None)
    train_ds = DecisionDataset(train_rows, tokenizer, max_length=cfg.max_length)
    eval_ds = DecisionDataset(eval_rows, tokenizer, max_length=cfg.max_length)
    train_ds.tokenizer = tokenizer
    eval_ds.tokenizer = tokenizer
    print(
        f"  train {len(train_ds)} examples ({train_ds.n_dropped} dropped, too long), "
        f"eval {len(eval_ds)} ({eval_ds.n_dropped} dropped) -- {time.perf_counter() - t0:.0f}s"
    )
    if train_ds.dropped_by_source:
        worst = sorted(train_ds.dropped_by_source.items(), key=lambda x: -x[1])[:5]
        print(f"  most-dropped sources: {worst}")

    steps_per_epoch = math.ceil(len(train_ds) / cfg.batch_size / cfg.grad_accum)
    total_steps = cfg.max_steps if cfg.max_steps > 0 else int(steps_per_epoch * cfg.epochs)
    warmup = max(1, int(total_steps * cfg.warmup_ratio))
    print(f"total optimizer steps: {total_steps} (warmup {warmup})")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: s / warmup
        if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
    )

    pad_id = tokenizer.pad_token_id
    history: list[dict] = []
    step = 0
    micro = 0
    running: dict[str, float] = {}
    t_start = time.perf_counter()
    done = False

    for epoch in range(math.ceil(cfg.epochs) if cfg.max_steps <= 0 else 10**6):
        if done:
            break
        for batch_idx in length_grouped_batches(
            train_ds, cfg.batch_size, shuffle_seed=cfg.seed + epoch
        ):
            batch: Batch = collate([train_ds[i] for i in batch_idx], pad_id).to(device)
            logits = letter_logits(
                model, batch.input_ids, batch.attention_mask, batch.slot_positions, letter_w
            )
            logits = mask_to_options(logits, batch.option_counts)
            loss, parts = decision_loss(
                logits,
                batch.labels,
                batch.option_counts,
                brier_weight=cfg.brier_weight,
                label_smoothing=cfg.label_smoothing,
            )
            (loss / cfg.grad_accum).backward()

            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            running["loss"] = running.get("loss", 0.0) + loss.item()
            micro += 1

            if micro % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % cfg.log_every == 0:
                    n = cfg.log_every * cfg.grad_accum
                    el = time.perf_counter() - t_start
                    msg = (
                        f"step {step:5d}/{total_steps}  "
                        f"loss={running['loss'] / n:.4f}  ce={running['ce'] / n:.4f}  "
                        f"brier={running['brier'] / n:.4f}  acc={running['acc'] / n:.4f}  "
                        f"conf={running['conf'] / n:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                        f"{el / 60:.1f}min  vram={torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
                    )
                    print(msg, flush=True)
                    history.append({"step": step, **{k: v / n for k, v in running.items()}})
                    running = {}

                if cfg.eval_every and step % cfg.eval_every == 0:
                    rep = run_eval(model, letter_w, eval_ds, cfg, device)
                    print(
                        f"  [eval] step {step}: acc={rep['accuracy']:.4f} "
                        f"ECE={rep['ece']:.4f} Brier={rep['brier']:.4f} "
                        f"conf={rep['mean_confidence']:.4f} "
                        f"overconfidence={rep['overconfidence']:+.4f}",
                        flush=True,
                    )
                    history.append({"step": step, "eval": rep})
                    (out / "history.json").write_text(
                        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    model.save_pretrained(str(out / "adapter"))

                if step >= total_steps:
                    done = True
                    break

    rep = run_eval(model, letter_w, eval_ds, cfg, device)
    print(f"\nfinal eval: {rep}")
    history.append({"step": step, "eval_final": rep})

    model.save_pretrained(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    (out / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "train_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved: {out / 'adapter'}")
    print(f"total time: {(time.perf_counter() - t_start) / 60:.1f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
