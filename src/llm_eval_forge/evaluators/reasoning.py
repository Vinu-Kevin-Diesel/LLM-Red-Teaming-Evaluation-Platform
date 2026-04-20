"""Reasoning consistency evaluator."""

from __future__ import annotations

from pathlib import Path

from llm_eval_forge.evaluators.base import Evaluator, TestCase
from llm_eval_forge.evaluators.registry import register_evaluator


@register_evaluator("reasoning")
class ReasoningConsistencyEvaluator(Evaluator):
    dimension = "reasoning"

    def load_test_cases(self, path: str | Path) -> list[TestCase]:
        raw_cases = self._load_yaml_cases(path)
        cases = []
        for tc in raw_cases:
            cases.append(TestCase(
                id=tc["id"],
                prompt=tc["prompt"],
                system=tc.get("system"),
                expected=tc.get("expected"),
                metadata={
                    "reasoning_type": tc.get("reasoning_type", "deductive"),
                    "num_steps": tc.get("num_steps", 1),
                    "difficulty": tc.get("difficulty", "medium"),
                },
            ))
        return cases

    def build_judge_payload(self, test_case: TestCase, response_text: str) -> dict:
        context = {
            "reasoning_type": test_case.metadata.get("reasoning_type", "deductive"),
            "expected_steps": test_case.metadata.get("num_steps", 1),
        }
        if test_case.expected:
            context["correct_answer"] = test_case.expected
        return context
