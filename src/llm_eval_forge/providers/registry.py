"""Provider registry — decorator-based factory for LLM providers."""

from __future__ import annotations

from typing import Type

from llm_eval_forge.providers.base import LLMProvider

_PROVIDERS: dict[str, Type[LLMProvider]] = {}


def register_provider(name: str):
    """Class decorator to register a provider implementation."""
    def wrapper(cls: Type[LLMProvider]) -> Type[LLMProvider]:
        _PROVIDERS[name] = cls
        return cls
    return wrapper


def get_provider(name: str, **kwargs) -> LLMProvider:
    """Instantiate a registered provider by name."""
    if name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys()) or "(none)"
        raise ValueError(f"Unknown provider: {name!r}. Available: {available}")
    return _PROVIDERS[name](**kwargs)


def list_providers() -> list[str]:
    """Return names of all registered providers."""
    return list(_PROVIDERS.keys())
