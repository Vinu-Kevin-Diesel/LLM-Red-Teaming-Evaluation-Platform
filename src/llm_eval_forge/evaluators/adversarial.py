"""Adversarial robustness evaluator — integrates with the red-team engine."""

from __future__ import annotations

from pathlib import Path

from llm_eval_forge.evaluators.base import Evaluator, TestCase
from llm_eval_forge.evaluators.registry import register_evaluator


@register_evaluator("adversarial")
class AdversarialRobustnessEvaluator(Evaluator):
    dimension = "adversarial"

    def load_test_cases(self, path: str | Path) -> list[TestCase]:
        raw_cases = self._load_yaml_cases(path)
        cases = []
        for tc in raw_cases:
            cases.append(TestCase(
                id=tc["id"],
                prompt=tc["prompt"],
                system=tc.get("system"),
                metadata={
                    "attack_type": tc.get("attack_type", "direct"),
                    "target_behavior": tc.get("target_behavior", ""),
                    "clean_intent": tc.get("clean_intent", ""),
                    "difficulty": tc.get("difficulty", "medium"),
                },
            ))
        return cases

    def build_judge_payload(self, test_case: TestCase, response_text: str) -> dict:
        return {
            "attack_type": test_case.metadata.get("attack_type", "direct"),
            "target_behavior": test_case.metadata.get("target_behavior", ""),
            "clean_intent": test_case.metadata.get("clean_intent", ""),
        }
