"""Train the student judge: a ModernBERT sequence classifier.

Stage A learns from public labels (WildGuardTrain + Aegis 2.0). Stage B distills the
Claude teacher: it trains on teacher verdicts as soft targets, where the teacher's
confidence (1-5) sets how far the target sits from 0.5.

    python -m llm_eval_forge.distill.train --stage A --out models/student-A
    python -m llm_eval_forge.distill.train --stage B --init models/student-A --out models/student-AB
    python -m llm_eval_forge.distill.train --stage B --out models/student-B
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_eval_forge.distill.data import bucket, read_split
from llm_eval_forge.distill.metrics import auroc, best_f1_threshold, log_loss

BASE_MODEL = "answerdotai/ModernBERT-base"
PUBLIC_TRAIN = {"wildguardtrain", "aegis2_train"}
GOLD_VAL = {"wildguardtrain", "aegis2_val"}


@dataclass
class TrainConfig:
    stage: str
    out: str
    init: str = BASE_MODEL
    data: str = "data"
    epochs: float = 2.0
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup: float = 0.1
    max_len: int = 1536
    tokens_per_batch: int = 12288
    max_batch_size: int = 32
    evals_per_epoch: int = 2
    seed: int = 13
    limit: int | None = None
    vram_fraction: float = 0.85
    grad_ckpt: bool = False  # recompute activations in backward; needed for ModernBERT-large on 12 GB
    train_sources: tuple[str, ...] = tuple(sorted(PUBLIC_TRAIN))
    val_sources: tuple[str, ...] = tuple(sorted(GOLD_VAL))


# --------------------------------------------------------------------------- data

def format_pair(prompt: str, response: str) -> tuple[str, str]:
    """Model input as a sentence pair, so truncation trims the longer side first."""
    return f"User: {prompt or ''}", f"Assistant: {response or ''}"


def soft_target(verdict: dict) -> float:
    """Teacher verdict -> P(harmful). Confidence 5 -> 1.0 / 0.0; confidence 1 -> 0.6 / 0.4."""
    shift = 0.1 * int(verdict["confidence"])
    return 0.5 + shift if verdict["response_harmful"] else 0.5 - shift


def pool_rows(data_dir: Path) -> list[dict]:
    path = data_dir / "teacher" / "sets" / "pool.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def holdout_part(row_id: str) -> str:
    """Split the teacher holdout 50/50: 'val' picks Stage B checkpoints and the teacher-matching
    threshold; 'test' is only used to report agreement with the teacher."""
    return "val" if bucket(row_id, salt="teacher-holdout") < 0.5 else "test"


def teacher_holdout_rows(data_dir: Path, part: str) -> list[dict]:
    labels = teacher_labels(data_dir)
    return [
        {**r, "label": int(labels[r["id"]]["verdict"]["response_harmful"]), "gold": r.get("label")}
        for r in pool_rows(data_dir)
        if r.get("teacher_holdout") and r["id"] in labels and holdout_part(r["id"]) == part
    ]


def holdout_ids(data_dir: Path) -> set[str]:
    """Pool rows reserved for measuring student-vs-teacher agreement; never trained on."""
    return {r["id"] for r in pool_rows(data_dir) if r.get("teacher_holdout")}


def teacher_labels(data_dir: Path) -> dict[str, dict]:
    path = data_dir / "teacher" / "labels" / "pool.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r["status"] == "ok"}


def stage_a_examples(data_dir: Path, sources: set[str] = PUBLIC_TRAIN) -> list[tuple[str, str, float]]:
    held = holdout_ids(data_dir)
    rows = read_split(data_dir, "train", set(sources))
    return [(r["prompt"], r["response"], float(r["label"])) for r in rows if r["label"] is not None and r["id"] not in held]


def stage_b_examples(data_dir: Path) -> list[tuple[str, str, float]]:
    labels = teacher_labels(data_dir)
    out = []
    for r in pool_rows(data_dir):
        rec = labels.get(r["id"])
        if rec is None or r.get("teacher_holdout"):
            continue
        out.append((r["prompt"], r["response"], soft_target(rec["verdict"])))
    return out


def gold_val_examples(data_dir: Path, sources: set[str] = GOLD_VAL) -> tuple[list[tuple[str, str, float]], list[str]]:
    """Gold-labeled validation examples and the source of each."""
    rows = [r for r in read_split(data_dir, "val", set(sources)) if r["label"] is not None]
    return [(r["prompt"], r["response"], float(r["label"])) for r in rows], [r["source"] for r in rows]


def token_budget_batches(lengths: list[int], tokens_per_batch: int, max_batch_size: int,
                         rng: random.Random, shuffle: bool = True) -> list[list[int]]:
    """Group similar lengths together so padding is minimal; each batch's padded size
    (len(batch) * longest) stays within the token budget."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if cur and (new_max * (len(cur) + 1) > tokens_per_batch or len(cur) >= max_batch_size):
            batches.append(cur)
            cur, new_max = [], lengths[i]
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


# --------------------------------------------------------------------------- model

def load_model(path: str, device: str = "cuda"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=2, attn_implementation="sdpa",
        id2label={0: "safe", 1: "harmful"}, label2id={"safe": 0, "harmful": 1},
    )
    return tok, model.to(device), torch


def encode(tok, examples, max_len: int) -> tuple[list[list[int]], list[float]]:
    first = [format_pair(p, r)[0] for p, r, _ in examples]
    second = [format_pair(p, r)[1] for p, r, _ in examples]
    enc = tok(first, second, truncation="longest_first", max_length=max_len)
    return enc["input_ids"], [y for _, _, y in examples]


def collate(torch, ids_batch: list[list[int]], pad_id: int, device: str):
    width = max(len(x) for x in ids_batch)
    input_ids = torch.full((len(ids_batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids_batch), width), dtype=torch.long)
    for j, x in enumerate(ids_batch):
        input_ids[j, : len(x)] = torch.tensor(x)
        mask[j, : len(x)] = 1
    return input_ids.to(device), mask.to(device)


def predict_proba(model, tok, torch, examples, max_len: int = 2048, tokens_per_batch: int = 32768,
                  device: str = "cuda") -> list[float]:
    """P(harmful) for each (prompt, response, _) example, in input order."""
    ids, _ = encode(tok, examples, max_len)
    batches = token_budget_batches([len(x) for x in ids], tokens_per_batch, 64, random.Random(0), shuffle=False)
    probs = [0.0] * len(ids)
    model.eval()
    with torch.inference_mode():
        for b in batches:
            input_ids, mask = collate(torch, [ids[i] for i in b], tok.pad_token_id, device)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=mask).logits.float()
            for i, p in zip(b, logits.softmax(-1)[:, 1].tolist()):
                probs[i] = p
    return probs


# --------------------------------------------------------------------------- training

def train(cfg: TrainConfig) -> dict:
    data_dir = Path(cfg.data)
    rng = random.Random(cfg.seed)
    examples = stage_a_examples(data_dir, set(cfg.train_sources)) if cfg.stage == "A" else stage_b_examples(data_dir)
    if cfg.limit:
        rng.shuffle(examples)
        examples = examples[: cfg.limit]
    val, val_sources = gold_val_examples(data_dir, set(cfg.val_sources))
    # Stage A is selected on gold labels; Stage B (distillation) on agreement with the teacher.
    select_on = "gold"
    if cfg.stage == "B":
        tval = teacher_holdout_rows(data_dir, "val")
        val += [(r["prompt"], r["response"], float(r["label"])) for r in tval]
        val_sources += ["teacher_val"] * len(tval)
        select_on = "teacher_val"
    if not examples:
        raise SystemExit(f"No training examples for stage {cfg.stage}")

    tok, model, torch = load_model(cfg.init)
    # On Windows (WDDM) the CUDA cache can grow past physical VRAM and silently spill into shared
    # system memory, slowing training ~10x. A hard cap makes the allocator free cached blocks instead.
    torch.cuda.set_per_process_memory_fraction(cfg.vram_fraction)
    if cfg.grad_ckpt:
        model.gradient_checkpointing_enable()
    torch.manual_seed(cfg.seed)
    ids, targets = encode(tok, examples, cfg.max_len)
    lengths = [len(x) for x in ids]
    steps_per_epoch = len(token_budget_batches(lengths, cfg.tokens_per_batch, cfg.max_batch_size, rng))
    total_steps = math.ceil(steps_per_epoch * cfg.epochs)
    warmup = int(total_steps * cfg.warmup)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warmup) if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    eval_every = max(1, steps_per_epoch // cfg.evals_per_epoch)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"stage {cfg.stage}: {len(examples):,} examples, {steps_per_epoch} steps/epoch, {total_steps} steps, "
          f"val {len(val):,}")

    # Select checkpoints by validation AUROC: it ignores calibration and threshold, which drift
    # apart from accuracy as the model grows confident (val loss rose while F1 improved).
    history, best_auroc, step, t0, seen_tokens = [], -1.0, 0, time.time(), 0
    torch.cuda.reset_peak_memory_stats()
    while step < total_steps:
        for b in token_budget_batches(lengths, cfg.tokens_per_batch, cfg.max_batch_size, rng):
            if step >= total_steps:
                break
            model.train()
            input_ids, mask = collate(torch, [ids[i] for i in b], tok.pad_token_id, "cuda")
            y = torch.tensor([targets[i] for i in b], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=mask).logits.float()
            logp = logits.log_softmax(-1)
            loss = -(y * logp[:, 1] + (1 - y) * logp[:, 0]).mean()  # soft-target cross-entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            seen_tokens += int(mask.sum())
            if step % 100 == 0:
                rate = seen_tokens / (time.time() - t0)
                print(f"step {step}/{total_steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{rate:,.0f} tok/s reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB", flush=True)
            if step % eval_every == 0 or step == total_steps:
                # Keep eval batches near the training budget: large inference batches grow the CUDA
                # cache on top of optimizer state and spill into shared memory on Windows (WDDM).
                probs = predict_proba(model, tok, torch, val, tokens_per_batch=2 * cfg.tokens_per_batch)
                torch.cuda.empty_cache()
                labels = [int(y) for _, _, y in val]
                gold = [i for i, src in enumerate(val_sources) if src != "teacher_val"]
                thr, f1 = best_f1_threshold([labels[i] for i in gold], [probs[i] for i in gold])
                gold_auroc = auroc([labels[i] for i in gold], [probs[i] for i in gold])
                by_source = {
                    src: round(auroc([y for y, s in zip(labels, val_sources) if s == src],
                                     [p for p, s in zip(probs, val_sources) if s == src]) or 0.0, 4)
                    for src in sorted(set(val_sources))
                }
                val_auroc = by_source["teacher_val"] if select_on == "teacher_val" else gold_auroc
                entry = {"step": step, "val_loss": round(log_loss([labels[i] for i in gold], [probs[i] for i in gold]), 4),
                         "val_auroc": round(val_auroc, 4), "gold_auroc": round(gold_auroc, 4), "selected_on": select_on,
                         "val_f1": round(f1, 4), "threshold": thr, "auroc_by_source": by_source}
                history.append(entry)
                print(f"  eval {entry}", flush=True)
                if val_auroc > best_auroc:
                    best_auroc = val_auroc
                    model.save_pretrained(out)
                    tok.save_pretrained(out)

    summary = {
        "config": asdict(cfg),
        "examples": len(examples),
        "steps": total_steps,
        "train_seconds": round(time.time() - t0, 1),
        "tokens_per_second": round(seen_tokens / (time.time() - t0)),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "best_val_auroc": best_auroc,
        "history": history,
        "best": max(history, key=lambda h: h["val_auroc"]),
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"done in {summary['train_seconds']}s; best val AUROC {best_auroc:.4f} at step "
          f"{summary['best']['step']}; saved {out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the student judge")
    parser.add_argument("--stage", choices=["A", "B"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init", default=BASE_MODEL)
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-len", type=int, default=1536)
    parser.add_argument("--tokens-per-batch", type=int, default=12288)
    parser.add_argument("--limit", type=int, default=None, help="Train on a random subset (smoke tests)")
    parser.add_argument("--evals-per-epoch", type=int, default=4)
    parser.add_argument("--grad-ckpt", action="store_true", help="Gradient checkpointing (larger models)")
    parser.add_argument("--train-sources", default=",".join(sorted(PUBLIC_TRAIN)),
                        help="Comma-separated source names under <data>/train (Stage A)")
    parser.add_argument("--val-sources", default=",".join(sorted(GOLD_VAL)),
                        help="Comma-separated source names under <data>/val")
    args = parser.parse_args()
    defaults = {"A": (2.0, 3e-5), "B": (2.0, 2e-5)}[args.stage]
    cfg = TrainConfig(
        stage=args.stage, out=args.out, init=args.init, data=args.data,
        epochs=args.epochs or defaults[0], lr=args.lr or defaults[1],
        max_len=args.max_len, tokens_per_batch=args.tokens_per_batch, limit=args.limit,
        evals_per_epoch=args.evals_per_epoch, grad_ckpt=args.grad_ckpt,
        train_sources=tuple(args.train_sources.split(",")), val_sources=tuple(args.val_sources.split(",")),
    )
    train(cfg)


if __name__ == "__main__":
    main()
