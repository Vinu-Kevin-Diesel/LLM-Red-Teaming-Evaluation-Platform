"""YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 30
    delay_between_requests_ms: int = 2000


@dataclass
class ProviderConfig:
    name: str
    models: list[str] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 2048


@dataclass
class JudgeConfig:
    provider: str = "claude"
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 4096  # Claude counts thinking toward max_tokens, so leave headroom
    effort: str | None = "low"  # Claude only: low | medium | high | xhigh | max


@dataclass
class DimensionConfig:
    name: str
    enabled: bool = True
    test_cases: str = ""
    rubric_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class RedTeamConfig:
    enabled: bool = True
    strategies: list[str] = field(default_factory=lambda: [
        "roleplay_injection",
        "encoding_attack",
        "instruction_smuggling",
        "context_manipulation",
        "fewshot_poisoning",
        "semantic_trick",
    ])
    mutations_per_prompt: int = 2


@dataclass
class EvalConfig:
    name: str = "evaluation-run"
    output_dir: str = "./results"
    providers: list[ProviderConfig] = field(default_factory=list)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    dimensions: list[DimensionConfig] = field(default_factory=list)
    redteam: RedTeamConfig = field(default_factory=RedTeamConfig)
    rate_limits: dict[str, RateLimitConfig] = field(default_factory=dict)


def load_config(path: str | Path) -> EvalConfig:
    """Load and validate a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError(f"Empty config file: {path}")

    eval_section = raw.get("eval", {})
    config = EvalConfig(
        name=eval_section.get("name", "evaluation-run"),
        output_dir=eval_section.get("output_dir", "./results"),
    )

    # Providers
    for p in raw.get("providers", []):
        config.providers.append(ProviderConfig(
            name=p["name"],
            models=p.get("models", []),
            temperature=p.get("temperature", 0.0),
            max_tokens=p.get("max_tokens", 2048),
        ))

    # Judge
    judge_raw = raw.get("judge", {})
    config.judge = JudgeConfig(
        provider=judge_raw.get("provider", "claude"),
        model=judge_raw.get("model", "claude-sonnet-5-5"),
        max_tokens=judge_raw.get("max_tokens", 4096),
        effort=judge_raw.get("effort", "low"),
    )

    # Dimensions
    for d in raw.get("dimensions", []):
        config.dimensions.append(DimensionConfig(
            name=d["name"],
            enabled=d.get("enabled", True),
            test_cases=d.get("test_cases", ""),
            rubric_weights=d.get("rubric_weights", {}),
        ))

    # Red team
    rt = raw.get("redteam", {})
    if rt:
        config.redteam = RedTeamConfig(
            enabled=rt.get("enabled", True),
            strategies=rt.get("strategies", config.redteam.strategies),
            mutations_per_prompt=rt.get("mutations_per_prompt", 2),
        )

    # Rate limits
    for name, rl in raw.get("rate_limits", {}).items():
        config.rate_limits[name] = RateLimitConfig(
            requests_per_minute=rl.get("requests_per_minute", 30),
            delay_between_requests_ms=rl.get("delay_between_requests_ms", 2000),
        )

    _validate_config(config)
    return config


def _validate_config(config: EvalConfig) -> None:
    """Validate config for common errors."""
    if not config.providers:
        raise ValueError("At least one provider must be configured")

    valid_dimensions = {"hallucination", "instruction", "reasoning", "adversarial"}
    for dim in config.dimensions:
        if dim.name not in valid_dimensions:
            raise ValueError(f"Unknown dimension: {dim.name}. Valid: {valid_dimensions}")

    for dim in config.dimensions:
        if dim.rubric_weights:
            total = sum(dim.rubric_weights.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Rubric weights for '{dim.name}' sum to {total}, expected 1.0"
                )
