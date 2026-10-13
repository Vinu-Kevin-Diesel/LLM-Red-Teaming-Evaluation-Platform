"""Groq provider — OpenAI-compatible API."""

from __future__ import annotations

import time

from openai import AsyncOpenAI

from llm_eval_forge.providers.base import LLMProvider, LLMResponse, resolve_api_key
from llm_eval_forge.providers.registry import register_provider


@register_provider("groq")
class GroqProvider(LLMProvider):
    name = "groq"

    DEFAULT_MODELS = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
    ]

    def __init__(self, model: str = "openai/gpt-oss-20b", api_key: str | None = None):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=resolve_api_key(api_key, "GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
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
