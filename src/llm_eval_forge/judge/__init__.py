"""Judge module — LLM-based scoring with weighted rubrics."""

from llm_eval_forge.judge.claude_judge import LLMJudge, ClaudeJudge
from llm_eval_forge.judge.rubrics import (
    JudgeVerdict,
    Rubric,
    RubricCriterion,
    HALLUCINATION_RUBRIC,
    INSTRUCTION_RUBRIC,
    REASONING_RUBRIC,
    ADVERSARIAL_RUBRIC,
    get_rubric,
)

__all__ = [
    "ClaudeJudge",
    "JudgeVerdict",
    "Rubric",
    "RubricCriterion",
    "HALLUCINATION_RUBRIC",
    "INSTRUCTION_RUBRIC",
    "REASONING_RUBRIC",
    "ADVERSARIAL_RUBRIC",
    "get_rubric",
]
