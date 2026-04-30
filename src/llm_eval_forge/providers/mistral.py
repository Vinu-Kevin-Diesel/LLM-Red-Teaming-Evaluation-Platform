"""Mistral provider — Mistral Medium 3.5 via NVIDIA NIM, free tier, OpenAI-compatible API."""

from __future__ import annotations

import os
import time

from openai import AsyncOpenAI

from llm_eval_forge.providers.base import LLMProvider, LLMResponse
from llm_eval_forge.providers.registry import register_provider


@register_provider("mistral")
class MistralProvider(LLMProvider):
    name = "mistral"

    DEFAULT_MODELS = ["mistralai/mistral-medium-3.5-128b"]

    def __init__(self, model: str = "mistralai/mistral-medium-3.5-128b",
                 api_key: str | None = None):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("NVIDIA_API_KEY", ""),
            base_url="https://integrate.api.nvidia.com/v1",
        )

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
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
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
            raw=response.model_dump(),
        )
