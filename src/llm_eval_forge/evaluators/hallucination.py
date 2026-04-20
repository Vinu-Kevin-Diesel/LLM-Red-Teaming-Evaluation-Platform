"""Hallucination detection evaluator."""

from __future__ import annotations

from pathlib import Path

from llm_eval_forge.evaluators.base import Evaluator, TestCase
from llm_eval_forge.evaluators.registry import register_evaluator


@register_evaluator("hallucination")
class HallucinationEvaluator(Evaluator):
    dimension = "hallucination"

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
                    "category": tc.get("category", "factual"),
                    "difficulty": tc.get("difficulty", "medium"),
                },
            ))
        return cases

    def build_judge_payload(self, test_case: TestCase, response_text: str) -> dict:
        context = {"category": test_case.metadata.get("category", "factual")}
        if test_case.expected:
            context["ground_truth"] = test_case.expected
        return context
