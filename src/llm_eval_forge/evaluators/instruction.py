"""Instruction-following evaluator."""

from __future__ import annotations

from pathlib import Path

from llm_eval_forge.evaluators.base import Evaluator, TestCase
from llm_eval_forge.evaluators.registry import register_evaluator


@register_evaluator("instruction")
class InstructionFollowingEvaluator(Evaluator):
    dimension = "instruction"

    def load_test_cases(self, path: str | Path) -> list[TestCase]:
        raw_cases = self._load_yaml_cases(path)
        cases = []
        for tc in raw_cases:
            cases.append(TestCase(
                id=tc["id"],
                prompt=tc["prompt"],
                system=tc.get("system"),
                metadata={
                    "constraints": tc.get("constraints", []),
                    "difficulty": tc.get("difficulty", "medium"),
                },
            ))
        return cases

    def build_judge_payload(self, test_case: TestCase, response_text: str) -> dict:
        return {
            "constraints": ", ".join(test_case.metadata.get("constraints", [])),
            "num_constraints": len(test_case.metadata.get("constraints", [])),
        }
