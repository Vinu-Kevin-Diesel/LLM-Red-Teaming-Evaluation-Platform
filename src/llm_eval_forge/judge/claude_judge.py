"""LLM judge for scoring responses — supports Claude (Anthropic), Groq, and NVIDIA NIM."""

from __future__ import annotations

import json
import os

import anthropic
from openai import OpenAI

from llm_eval_forge.judge.rubrics import JudgeVerdict, Rubric

# Preconfigured judge backends
JUDGE_BACKENDS = {
    "claude": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-6",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "default_model": "moonshotai/kimi-k2-5",
    },
}


class LLMJudge:
    """Scores LLM responses using a judge model with weighted rubrics."""

    def __init__(
        self,
        backend: str = "claude",
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ):
        if backend not in JUDGE_BACKENDS:
            raise ValueError(f"Unknown judge backend: {backend!r}. Available: {list(JUDGE_BACKENDS)}")

        cfg = JUDGE_BACKENDS[backend]
        self.model = model or cfg["default_model"]
        self.backend = backend
        self.max_tokens = max_tokens

        api_key = api_key or os.environ.get(cfg["api_key_env"], "")
        if backend == "claude":
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = OpenAI(api_key=api_key, base_url=cfg["base_url"])

    def score(
        self,
        dimension: str,
        prompt: str,
        response: str,
        rubric: Rubric,
        context: dict | None = None,
    ) -> JudgeVerdict:
        """Score a response using the judge model with the given rubric."""
        system_prompt = rubric.to_system_prompt()
        user_message = self._format_scoring_request(dimension, prompt, response, context)

        if self.backend == "claude":
            raw_text = self._score_claude(system_prompt, user_message)
        else:
            raw_text = self._score_openai_compat(system_prompt, user_message)

        return self._parse_verdict(dimension, rubric, raw_text)

    def _score_claude(self, system_prompt: str, user_message: str) -> str:
        # Cache the rubric system prompt — same rubric is reused across every test
        # case for a given dimension, so this avoids re-billing it each call.
        result = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_message}],
        )
        return next((b.text for b in result.content if b.type == "text"), "")

    def _score_openai_compat(self, system_prompt: str, user_message: str) -> str:
        result = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return result.choices[0].message.content or ""

    def _format_scoring_request(
        self,
        dimension: str,
        prompt: str,
        response: str,
        context: dict | None = None,
    ) -> str:
        parts = [
            f"## Evaluation Dimension: {dimension}",
            "",
            "## Original Prompt",
            prompt,
            "",
            "## Model Response",
            response,
        ]
        if context:
            parts.append("")
            parts.append("## Additional Context")
            for k, v in context.items():
                parts.append(f"- {k}: {v}")

        parts.extend([
            "",
            "## Instructions",
            "Score the model response using the rubric criteria.",
            'Return ONLY valid JSON: {"scores": {"criterion": score, ...}, "reasoning": "..."}',
        ])
        return "\n".join(parts)

    def _parse_verdict(self, dimension: str, rubric: Rubric, raw: str) -> JudgeVerdict:
        """Parse the JSON response into a JudgeVerdict."""
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return JudgeVerdict(
                dimension=dimension,
                scores={},
                weighted_score=0.0,
                reasoning=f"Failed to parse judge response: {raw[:200]}",
                raw_response=raw,
            )

        scores = {k: float(v) for k, v in data.get("scores", {}).items()}
        weighted = rubric.compute_weighted_score(scores)

        return JudgeVerdict(
            dimension=dimension,
            scores=scores,
            weighted_score=weighted,
            reasoning=data.get("reasoning", ""),
            raw_response=raw,
        )


# Backward-compatible alias
ClaudeJudge = LLMJudge
