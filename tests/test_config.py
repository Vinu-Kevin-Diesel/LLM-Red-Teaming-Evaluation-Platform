"""Tests for config loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from llm_eval_forge.config import load_config, EvalConfig


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def test_load_minimal_config(tmp_path):
    data = {
        "providers": [{"name": "groq", "models": ["openai/gpt-oss-20b"]}],
        "dimensions": [{"name": "hallucination"}],
    }
    config = load_config(_write_config(tmp_path, data))
    assert isinstance(config, EvalConfig)
    assert len(config.providers) == 1
    assert config.providers[0].name == "groq"


def test_load_full_config(tmp_path):
    data = {
        "eval": {"name": "test-run", "output_dir": "./test-results"},
        "providers": [
            {"name": "groq", "models": ["openai/gpt-oss-20b"], "temperature": 0.0},
            {"name": "cerebras", "models": ["gpt-oss-120b"]},
        ],
        "judge": {"provider": "groq", "model": "openai/gpt-oss-20b"},
        "dimensions": [
            {
                "name": "hallucination",
                "enabled": True,
                "rubric_weights": {
                    "factual_accuracy": 0.4,
                    "fabrication_avoidance": 0.35,
                    "uncertainty_expression": 0.25,
                },
            },
        ],
        "redteam": {"enabled": True, "strategies": ["roleplay_injection"]},
        "rate_limits": {"groq": {"requests_per_minute": 25}},
    }
    config = load_config(_write_config(tmp_path, data))
    assert config.name == "test-run"
    assert len(config.providers) == 2
    assert config.redteam.enabled is True


def test_invalid_dimension(tmp_path):
    data = {
        "providers": [{"name": "groq"}],
        "dimensions": [{"name": "nonexistent"}],
    }
    with pytest.raises(ValueError, match="Unknown dimension"):
        load_config(_write_config(tmp_path, data))


def test_invalid_rubric_weights(tmp_path):
    data = {
        "providers": [{"name": "groq"}],
        "dimensions": [{
            "name": "hallucination",
            "rubric_weights": {"factual_accuracy": 0.5, "fabrication_avoidance": 0.1},
        }],
    }
    with pytest.raises(ValueError, match="sum to"):
        load_config(_write_config(tmp_path, data))


def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_empty_config(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    with pytest.raises(ValueError, match="Empty config"):
        load_config(path)


def test_default_config_uses_live_models_and_claude_judge():
    config = load_config(Path(__file__).resolve().parent.parent / "config" / "default.yaml")
    models = {m for p in config.providers for m in p.models}
    assert "qwen/qwen3-32b" not in models and "llama3.1-8b" not in models
    assert config.judge.provider == "claude"
    assert config.judge.model == "claude-sonnet-5-5"
    assert config.judge.effort == "low"
    assert config.judge.max_tokens >= 2048  # thinking counts toward max_tokens
