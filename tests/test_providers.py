"""Tests for provider registry and base classes."""

from llm_eval_forge.providers.registry import get_provider, list_providers
from llm_eval_forge.providers.base import LLMResponse

import pytest


def test_list_providers():
    providers = list_providers()
    assert "groq" in providers
    assert "cerebras" in providers
    assert "mistral" in providers
    assert "huggingface" in providers
    assert "kimi" not in providers  # removed in favour of Cerebras


def test_get_groq_provider():
    provider = get_provider("groq", model="qwen/qwen3.8-27b")
    assert provider.name == "groq"
    assert provider.model == "qwen/qwen3.8-27b"


def test_provider_defaults_are_live_models():
    # Retired upstream in 2026: qwen/qwen3-32b, llama-3.x on Groq; llama3.1-8b on Cerebras.
    retired = {"qwen/qwen3-32b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3.1-8b"}
    for name in ("groq", "cerebras"):
        provider = get_provider(name)
        assert provider.model not in retired
        assert not retired & set(provider.DEFAULT_MODELS)


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
