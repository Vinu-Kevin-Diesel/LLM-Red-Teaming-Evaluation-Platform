"""HuggingFace Inference API provider."""

from __future__ import annotations

import os
import time

from huggingface_hub import AsyncInferenceClient

from llm_eval_forge.providers.base import LLMProvider, LLMResponse
from llm_eval_forge.providers.registry import register_provider


@register_provider("huggingface")
class HuggingFaceProvider(LLMProvider):
    name = "huggingface"

    DEFAULT_MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]

    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.3",
                 api_key: str | None = None):
        self.model = model
        token = api_key or os.environ.get("HF_TOKEN", "")
        self.client = AsyncInferenceClient(model=model, token=token)

    async def complete(self, prompt: str, system: str | None = None,
                       temperature: float = 0.0, max_tokens: int = 2048) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.complete_chat(messages, temperature, max_tokens)

    async def complete_chat(self, messages: list[dict],
                            temperature: float = 0.0, max_tokens: int = 2048) -> LLMResponse:
        start = time.perf_counter()
        response = await self.client.chat_completion(
            messages=messages,
            temperature=temperature if temperature > 0 else None,
            max_tokens=max_tokens,
        )
        elapsed = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model=self.model,
            provider=self.name,
            latency_ms=round(elapsed, 1),
            token_count=response.usage.total_tokens if response.usage else None,
            raw={"id": response.id, "model": response.model},
        )
