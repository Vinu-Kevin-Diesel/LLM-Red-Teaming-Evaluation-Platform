"""Tests for provider registry and base classes."""

from llm_eval_forge.providers.registry import get_provider, list_providers
from llm_eval_forge.providers.base import LLMResponse

import pytest


def test_list_providers():
    providers = list_providers()
    assert "groq" in providers
    assert "kimi" in providers
    assert "huggingface" in providers


def test_get_groq_provider():
    provider = get_provider("groq", model="llama-3.3-70b-versatile")
    assert provider.name == "groq"
    assert provider.model == "llama-3.3-70b-versatile"


def test_get_kimi_provider():
    provider = get_provider("kimi", model="moonshotai/kimi-k2-5")
    assert provider.name == "kimi"
    assert provider.model == "moonshotai/kimi-k2-5"


def test_get_huggingface_provider():
    provider = get_provider("huggingface", model="mistralai/Mistral-7B-Instruct-v0.3")
    assert provider.name == "huggingface"


def test_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("nonexistent")


def test_llm_response_dataclass():
    resp = LLMResponse(
        text="Hello",
        model="test-model",
        provider="test",
        latency_ms=100.0,
        token_count=10,
    )
    assert resp.text == "Hello"
    assert resp.latency_ms == 100.0
