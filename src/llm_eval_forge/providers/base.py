"""Abstract base class for LLM providers."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

# openai>=3 rejects an empty api_key at construction time. A placeholder keeps
# providers constructible without keys (tests, dry runs); real calls then fail
# with a clear 401 from the provider.
MISSING_API_KEY = "missing-api-key"


def resolve_api_key(explicit: Optional[str], env_var: str) -> str:
    """Return the explicit key, else the env var, else a placeholder."""
    return explicit or os.environ.get(env_var) or MISSING_API_KEY


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str
    model: str
    provider: str
    latency_ms: float
    token_count: Optional[int] = None
    raw: Optional[dict] = field(default=None, repr=False)


class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    name: str

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send a single prompt and get a completion."""
        ...

    @abstractmethod
    async def complete_chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send a chat-format message list and get a completion."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model!r})"
