"""Tests for the LLM judge's verdict parsing and failure handling (no API calls)."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from llm_eval_forge.judge.claude_judge import (
    ADVERSARIAL_LABEL_FIELDS,
    LLMJudge,
    build_verdict_model,
)
from llm_eval_forge.judge.rubrics import get_rubric


def _judge(backend: str = "claude", fake_client=None) -> LLMJudge:
    judge = LLMJudge.__new__(LLMJudge)
    judge.backend, judge.model, judge.max_tokens, judge.effort = backend, "test-model", 4096, "low"
    judge.client = fake_client
    return judge


class _FakeMessages:
    def __init__(self, response):
        self.response, self.calls = response, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _fake_client(text: str, stop_reason: str = "end_turn", category: str | None = None):
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if category else None,
        model="claude-sonnet-5-5",
    )
    messages = _FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


ADV_JSON = json.dumps({
    "scores": {"refusal_appropriateness": 3, "information_leakage": 5, "consistency_under_pressure": 3},
    "reasoning": "Partial decode, no harmful content.",
    "response_harmful": False,
    "attack_succeeded": False,
    "refusal": False,
})


@pytest.mark.parametrize("dimension", ["hallucination", "instruction", "reasoning", "adversarial"])
def test_verdict_model_matches_rubric(dimension):
    rubric = get_rubric(dimension)
    model = build_verdict_model(rubric)
    score_fields = set(model.model_fields["scores"].annotation.model_fields)
    assert score_fields == {c.name for c in rubric.criteria}
    has_labels = set(ADVERSARIAL_LABEL_FIELDS) <= set(model.model_fields)
    assert has_labels == (dimension == "adversarial")


def test_verdict_model_rejects_out_of_scale_scores():
    model = build_verdict_model(get_rubric("adversarial"))
    bad = json.loads(ADV_JSON)
    bad["scores"]["information_leakage"] = 6
    with pytest.raises(ValidationError):
        model.model_validate(bad)


def test_claude_structured_verdict_parsed_with_labels():
    client, messages = _fake_client(ADV_JSON)
    verdict = _judge(fake_client=client).score("adversarial", "p", "r", get_rubric("adversarial"))
    assert verdict.ok
    assert verdict.weighted_score == pytest.approx(0.74, abs=1e-4)
    assert verdict.labels == {"response_harmful": False, "attack_succeeded": False, "refusal": False}
    assert verdict.judge_model == "claude-sonnet-5-5"
    sent = messages.calls[0]
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["output_config"]["effort"] == "low"
    assert sent["extra_body"] == {"fallbacks": "default"}


def test_claude_refusal_is_no_verdict_not_zero():
    client, _ = _fake_client("", stop_reason="refusal", category="general_harms")
    verdict = _judge(fake_client=client).score("adversarial", "p", "r", get_rubric("adversarial"))
    assert verdict.weighted_score is None
    assert verdict.error == "refusal:general_harms"
    assert not verdict.ok


def test_claude_truncated_json_is_no_verdict():
    client, _ = _fake_client(ADV_JSON[:40], stop_reason="max_tokens")
    verdict = _judge(fake_client=client).score("adversarial", "p", "r", get_rubric("adversarial"))
    assert verdict.weighted_score is None
    assert verdict.error == "unparsed:max_tokens"


def test_openai_compat_recovers_json_after_prose():
    # The original bug: prose before the JSON was scored 0.0.
    raw = "Let me analyze the response against each criterion.\n\n**Word count**: fine.\n\n" + ADV_JSON
    verdict = _judge("groq")._parse_verdict("adversarial", get_rubric("adversarial"), raw)
    assert verdict.weighted_score == pytest.approx(0.74, abs=1e-4)
    assert verdict.labels["response_harmful"] is False


def test_openai_compat_garbage_is_no_verdict_not_zero():
    verdict = _judge("groq")._parse_verdict("adversarial", get_rubric("adversarial"), "I cannot score this.")
    assert verdict.weighted_score is None
    assert verdict.error == "unparsed"
