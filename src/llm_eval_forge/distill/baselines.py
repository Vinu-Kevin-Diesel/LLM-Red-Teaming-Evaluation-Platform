"""Open guard models as baselines, scored from the first verdict token in one forward pass.

Generating the verdict text and parsing it gives the same decision as comparing the
probabilities of the verdict's first token, but a single batched forward pass is far
faster and also yields a probability (needed for AUROC and calibration).
"""
from __future__ import annotations

import random
from typing import Callable

from llm_eval_forge.distill.train import token_budget_batches

MAX_FIELD_CHARS = 8000


class FirstTokenGuard:
    """P(unsafe) = mass on the unsafe verdict tokens / mass on all verdict tokens."""

    model_id: str
    unsafe_ids: list[int]
    safe_ids: list[int]
    tokens_per_batch = 16384

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16).to(device).eval()

    def build(self, prompt: str, response: str) -> str:
        raise NotImplementedError

    def decide_text(self, prompt: str, response: str, max_new_tokens: int = 12) -> str:
        """Generated verdict text, used to check that the first-token shortcut agrees."""
        ids = self.tok(self.build(prompt, response), return_tensors="pt", add_special_tokens=False).to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def __call__(self, rows: list[dict]) -> list[float]:
        torch = self.torch
        texts = [self.build(r["prompt"][:MAX_FIELD_CHARS], r["response"][:MAX_FIELD_CHARS]) for r in rows]
        ids = [self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        batches = token_budget_batches([len(x) for x in ids], self.tokens_per_batch, 32, random.Random(0), shuffle=False)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        probs = [0.0] * len(rows)
        with torch.inference_mode():
            for b in batches:
                width = max(len(ids[i]) for i in b)
                input_ids = torch.full((len(b), width), pad, dtype=torch.long)
                mask = torch.zeros((len(b), width), dtype=torch.long)
                for j, i in enumerate(b):  # left padding: the next-token logits sit at the last position
                    input_ids[j, width - len(ids[i]):] = torch.tensor(ids[i])
                    mask[j, width - len(ids[i]):] = 1
                position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
                logits = self.model(input_ids=input_ids.to(self.device), attention_mask=mask.to(self.device),
                                    position_ids=position_ids.to(self.device)).logits[:, -1, :].float()
                p = logits.softmax(-1)
                unsafe = p[:, self.unsafe_ids].sum(-1)
                safe = p[:, self.safe_ids].sum(-1)
                for i, v in zip(b, (unsafe / (unsafe + safe)).tolist()):
                    probs[i] = v
        return probs


class Qwen3GuardGen(FirstTokenGuard):
    """Qwen/Qwen3Guard-Gen-0.6B in strict mode (Controversial counts as unsafe).
    Its reply starts 'Safety: Safe|Unsafe|Controversial'."""

    model_id = "Qwen/Qwen3Guard-Gen-0.6B"
    unsafe_ids = [73067, 2093]  # " Unsafe", first token of " Controversial"
    safe_ids = [22291]  # " Safe"

    def build(self, prompt: str, response: str) -> str:
        chat = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], tokenize=False)
        return chat + "Safety:"


class GraniteGuardian(FirstTokenGuard):
    """ibm-granite/granite-guardian-3.2-3b-a800m with the 'harm' risk on the assistant turn.
    Its reply starts 'Yes' (harmful) or 'No'."""

    model_id = "ibm-granite/granite-guardian-3.2-3b-a800m"
    unsafe_ids = [10922]  # "Yes"
    safe_ids = [2023]  # "No"
    tokens_per_batch = 8192

    def build(self, prompt: str, response: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            guardian_config={"risk_name": "harm"}, add_generation_prompt=True, tokenize=False)


BASELINES: dict[str, Callable[[], FirstTokenGuard]] = {
    "qwen3guard-0.6b": Qwen3GuardGen,
    "granite-guardian-3.2-3b": GraniteGuardian,
}
