"""Tests for evaluators."""

from pathlib import Path

import pytest

from llm_eval_forge.evaluators.registry import get_evaluator, list_evaluators


def test_list_evaluators():
    evaluators = list_evaluators()
    assert "hallucination" in evaluators
    assert "instruction" in evaluators
    assert "reasoning" in evaluators
    assert "adversarial" in evaluators


def test_get_evaluator():
    evaluator = get_evaluator("hallucination")
    assert evaluator.dimension == "hallucination"


def test_unknown_evaluator():
    with pytest.raises(ValueError, match="Unknown evaluator"):
        get_evaluator("nonexistent")


def test_hallucination_load_test_cases():
    evaluator = get_evaluator("hallucination")
    cases = evaluator.load_test_cases("prompts/testcases/hallucination/factual_claims.yaml")
    assert len(cases) > 0
    assert cases[0].id == "hal-001"
    assert cases[0].prompt  # not empty


def test_instruction_load_test_cases():
    evaluator = get_evaluator("instruction")
    cases = evaluator.load_test_cases("prompts/testcases/instruction/complex_instructions.yaml")
    assert len(cases) > 0
    assert "constraints" in cases[0].metadata


def test_reasoning_load_test_cases():
    evaluator = get_evaluator("reasoning")
    cases = evaluator.load_test_cases("prompts/testcases/reasoning/multistep_logic.yaml")
    assert len(cases) > 0
    assert "reasoning_type" in cases[0].metadata


def test_adversarial_load_test_cases():
    evaluator = get_evaluator("adversarial")
    cases = evaluator.load_test_cases("prompts/testcases/adversarial/injection_attempts.yaml")
    assert len(cases) > 0
    assert "attack_type" in cases[0].metadata


def test_hallucination_judge_payload():
    evaluator = get_evaluator("hallucination")
    cases = evaluator.load_test_cases("prompts/testcases/hallucination/factual_claims.yaml")
    payload = evaluator.build_judge_payload(cases[0], "Python was released in 1991.")
    assert "category" in payload


def test_missing_test_cases():
    evaluator = get_evaluator("hallucination")
    with pytest.raises(FileNotFoundError):
        evaluator.load_test_cases("/nonexistent/path.yaml")
