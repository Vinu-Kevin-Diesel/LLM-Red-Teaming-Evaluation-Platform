"""Inference runtimes for the student judge.

Every runtime maps a list of (prompt, response) pairs to P(harmful). `OnnxRuntime` needs only
onnxruntime + tokenizers, so the CPU serving image stays small; `TorchRuntime` (GPU) imports
torch and transformers lazily.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Protocol

from llm_eval_forge.distill.train import format_pair, token_budget_batches


class Runtime(Protocol):
    name: str
    model_version: str

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def load_serve_config(model_dir: Path) -> dict:
    path = Path(model_dir) / "serve_config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    name = Path(model_dir).name
    # Response judges keep metrics in results/metrics, input guards in results/metrics_prompt,
    # tool-call guards in results/metrics_toolcall.
    m = {}
    for metrics in (Path("results/metrics") / f"{name}.json", Path("results/metrics_prompt") / f"{name}.json",
                    Path("results/metrics_toolcall") / f"{name}.json"):
        if metrics.exists():
            m = json.loads(metrics.read_text(encoding="utf-8"))
            break
    gold = m.get("threshold", 0.5)
    return {"model_version": name, "max_len": 2048,
            "thresholds": {"teacher": m.get("teacher_threshold", gold), "gold": gold},
            "uncertain_band": [0.2, 0.8]}


class OnnxRuntime:
    """CPU runtime over an exported artifact directory (see serve/export.py)."""

    def __init__(self, model_dir: Path, onnx_file: str = "model.int8.onnx", threads: int | None = None,
                 tokens_per_batch: int = 8192):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.dir = Path(model_dir)
        self.config = load_serve_config(self.dir)
        self.model_version = self.config["model_version"]
        self.name = f"onnx-cpu-{'int8' if 'int8' in onnx_file else 'fp32'}"
        self.tokens_per_batch = tokens_per_batch
        self.tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        self.tok.enable_truncation(max_length=self.config.get("max_len", 2048), strategy="longest_first")
        self.tok.no_padding()
        self.pad_id = self.tok.token_to_id("[PAD]") or 0
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or int(os.environ.get("JUDGE_THREADS", 0)) or max(1, (os.cpu_count() or 2) // 2)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.dir / onnx_file), opts, providers=["CPUExecutionProvider"])

    def encode(self, pairs: list[tuple[str, str]]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch([format_pair(p, r) for p, r in pairs])]

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        import numpy as np

        ids = self.encode(pairs)
        probs = [0.0] * len(ids)
        for b in token_budget_batches([len(x) for x in ids], self.tokens_per_batch, 32, random.Random(0), shuffle=False):
            width = max(len(ids[i]) for i in b)
            input_ids = np.full((len(b), width), self.pad_id, dtype=np.int64)
            mask = np.zeros((len(b), width), dtype=np.int64)
            for j, i in enumerate(b):
                input_ids[j, : len(ids[i])] = ids[i]
                mask[j, : len(ids[i])] = 1
            logits = self.session.run(None, {"input_ids": input_ids, "attention_mask": mask})[0].astype(np.float64)
            z = logits - logits.max(-1, keepdims=True)
            p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
            for i, v in zip(b, p[:, 1].tolist()):
                probs[i] = v
        return probs


class TorchRuntime:
    """GPU (or CPU) runtime over a Hugging Face checkpoint directory; bf16 on CUDA."""

    def __init__(self, model_dir: Path, device: str = "cuda", tokens_per_batch: int = 16384):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.device, self.tokens_per_batch = torch, device, tokens_per_batch
        self.dir = Path(model_dir)
        self.config = load_serve_config(self.dir)
        self.model_version = self.config["model_version"]
        self.name = f"torch-{device}-{'bf16' if device == 'cuda' else 'fp32'}"
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        model = AutoModelForSequenceClassification.from_pretrained(self.dir, attn_implementation="sdpa").eval()
        self.model = model.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.85)

    def encode(self, pairs: list[tuple[str, str]]) -> list[list[int]]:
        first = [format_pair(p, r)[0] for p, r in pairs]
        second = [format_pair(p, r)[1] for p, r in pairs]
        return self.tok(first, second, truncation="longest_first", max_length=self.config.get("max_len", 2048))["input_ids"]

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        torch = self.torch
        ids = self.encode(pairs)
        probs = [0.0] * len(ids)
        with torch.inference_mode():
            for b in token_budget_batches([len(x) for x in ids], self.tokens_per_batch, 64, random.Random(0), shuffle=False):
                width = max(len(ids[i]) for i in b)
                input_ids = torch.full((len(b), width), self.tok.pad_token_id, dtype=torch.long)
                mask = torch.zeros((len(b), width), dtype=torch.long)
                for j, i in enumerate(b):
                    input_ids[j, : len(ids[i])] = torch.tensor(ids[i])
                    mask[j, : len(ids[i])] = 1
                logits = self.model(input_ids=input_ids.to(self.device), attention_mask=mask.to(self.device)).logits
                for i, v in zip(b, logits.float().softmax(-1)[:, 1].tolist()):
                    probs[i] = v
        return probs


class GuardRuntime:
    """Wrap a baseline guard (e.g. Qwen3Guard) as a runtime, for escalation tiers and benchmarks."""

    def __init__(self, baseline: str):
        from llm_eval_forge.distill.baselines import BASELINES

        self.guard = BASELINES[baseline]()
        self.name = f"guard-{baseline}"
        self.model_version = baseline

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self.guard([{"prompt": p, "response": r} for p, r in pairs])


def load_runtime(kind: str, model_dir: str | Path, **kwargs) -> Runtime:
    """kind: onnx-int8 | onnx-fp32 | torch-cuda | torch-cpu"""
    if kind == "onnx-int8":
        return OnnxRuntime(Path(model_dir), "model.int8.onnx", **kwargs)
    if kind == "onnx-fp32":
        return OnnxRuntime(Path(model_dir), "model.onnx", **kwargs)
    if kind in ("torch-cuda", "torch-cpu"):
        return TorchRuntime(Path(model_dir), device=kind.split("-")[1], **kwargs)
    raise ValueError(f"Unknown runtime {kind!r}")
