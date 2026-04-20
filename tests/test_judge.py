"""Tests for judge rubrics (no API calls)."""

from llm_eval_forge.judge.rubrics import (
    Rubric,
    RubricCriterion,
    JudgeVerdict,
    HALLUCINATION_RUBRIC,
    INSTRUCTION_RUBRIC,
    REASONING_RUBRIC,
    ADVERSARIAL_RUBRIC,
    get_rubric,
)


def test_rubric_weights_sum_to_one():
    for rubric in [HALLUCINATION_RUBRIC, INSTRUCTION_RUBRIC, REASONING_RUBRIC, ADVERSARIAL_RUBRIC]:
        total = sum(c.weight for c in rubric.criteria)
        assert abs(total - 1.0) < 0.01, f"{rubric.dimension} weights sum to {total}"


def test_rubric_system_prompt():
    prompt = HALLUCINATION_RUBRIC.to_system_prompt()
    assert "hallucination" in prompt.lower()
    assert "factual_accuracy" in prompt
    assert "JSON" in prompt


def test_compute_weighted_score_perfect():
    rubric = HALLUCINATION_RUBRIC
    scores = {c.name: float(c.scale) for c in rubric.criteria}
    result = rubric.compute_weighted_score(scores)
    assert result == 1.0


def test_compute_weighted_score_zero():
    rubric = HALLUCINATION_RUBRIC
    scores = {c.name: 0.0 for c in rubric.criteria}
    result = rubric.compute_weighted_score(scores)
    assert result == 0.0


def test_compute_weighted_score_partial():
    rubric = HALLUCINATION_RUBRIC
    scores = {c.name: c.scale / 2 for c in rubric.criteria}
    result = rubric.compute_weighted_score(scores)
    assert abs(result - 0.5) < 0.01


def test_get_rubric():
    rubric = get_rubric("hallucination")
    assert rubric.dimension == "hallucination"


def test_get_rubric_unknown():
    import pytest
    with pytest.raises(ValueError, match="No rubric"):
        get_rubric("nonexistent")


def test_judge_verdict_dataclass():
    verdict = JudgeVerdict(
        dimension="hallucination",
        scores={"factual_accuracy": 4.0},
        weighted_score=0.8,
        reasoning="Good response",
        raw_response='{"scores": {"factual_accuracy": 4.0}}',
    )
    assert verdict.dimension == "hallucination"
    assert verdict.weighted_score == 0.8
