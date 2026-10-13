"""Abstract base class for evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TestCase:
    id: str
    prompt: str
    system: Optional[str] = None
    expected: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    test_case_id: str
    dimension: str
    provider: str
    model: str
    prompt_sent: str
    response_text: str
    latency_ms: float = 0.0
    judge_score: Optional[float] = None
    judge_reasoning: Optional[str] = None
    judge_scores_detail: dict = field(default_factory=dict)
    judge_labels: dict = field(default_factory=dict)
    judge_model: Optional[str] = None
    judge_error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class Evaluator(ABC):
    """Abstract base for all evaluation dimensions."""

    dimension: str

    @abstractmethod
    def load_test_cases(self, path: str | Path) -> list[TestCase]:
        """Load test cases from a YAML file."""
        ...

    @abstractmethod
    def build_judge_payload(self, test_case: TestCase, response_text: str) -> dict:
        """Build context dict sent to the Claude judge alongside prompt and response."""
        ...

    def _load_yaml_cases(self, path: str | Path) -> list[dict]:
        """Helper to load raw YAML test case data."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Test case file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return data.get("test_cases", [])
