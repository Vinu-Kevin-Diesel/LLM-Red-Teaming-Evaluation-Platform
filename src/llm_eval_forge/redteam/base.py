"""Abstract base class for mutation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MutatedPrompt:
    original: str
    mutated: str
    strategy: str
    description: str
    metadata: dict = field(default_factory=dict)


class MutationStrategy(ABC):
    name: str
    description: str

    @abstractmethod
    def mutate(self, prompt: str, target_behavior: str) -> list[MutatedPrompt]:
        """Generate adversarial mutations of the prompt."""
        ...
