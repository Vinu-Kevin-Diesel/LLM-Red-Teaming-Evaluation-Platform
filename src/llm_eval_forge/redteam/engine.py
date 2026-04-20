"""Mutation engine — orchestrates all adversarial strategies."""

from __future__ import annotations

from llm_eval_forge.redteam.base import MutationStrategy, MutatedPrompt
from llm_eval_forge.redteam.roleplay import RolePlayInjection
from llm_eval_forge.redteam.encoding import EncodingAttack
from llm_eval_forge.redteam.smuggling import InstructionSmuggling
from llm_eval_forge.redteam.context import ContextManipulation
from llm_eval_forge.redteam.fewshot import FewShotPoisoning
from llm_eval_forge.redteam.semantic import SemanticTrick

ALL_STRATEGIES: dict[str, type[MutationStrategy]] = {
    "roleplay_injection": RolePlayInjection,
    "encoding_attack": EncodingAttack,
    "instruction_smuggling": InstructionSmuggling,
    "context_manipulation": ContextManipulation,
    "fewshot_poisoning": FewShotPoisoning,
    "semantic_trick": SemanticTrick,
}


class MutationEngine:
    """Orchestrates adversarial mutation strategies."""

    def __init__(self, strategies: list[str] | None = None):
        names = strategies or list(ALL_STRATEGIES.keys())
        self.strategies: list[MutationStrategy] = []
        for name in names:
            if name not in ALL_STRATEGIES:
                raise ValueError(f"Unknown strategy: {name!r}")
            self.strategies.append(ALL_STRATEGIES[name]())

    def generate_attacks(
        self,
        clean_prompt: str,
        target_behavior: str,
        strategies: list[str] | None = None,
        max_per_strategy: int | None = None,
    ) -> list[MutatedPrompt]:
        """Apply selected (or all) strategies to a clean prompt."""
        results = []
        for strategy in self.strategies:
            if strategies and strategy.name not in strategies:
                continue
            mutations = strategy.mutate(clean_prompt, target_behavior)
            if max_per_strategy:
                mutations = mutations[:max_per_strategy]
            results.extend(mutations)
        return results

    def list_strategies(self) -> list[str]:
        return [s.name for s in self.strategies]
