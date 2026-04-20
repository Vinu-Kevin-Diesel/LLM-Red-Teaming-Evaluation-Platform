"""JSON report output."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from llm_eval_forge.evaluators.base import EvalResult


def save_json_report(results: list[EvalResult], path: str | Path) -> Path:
    """Save evaluation results as a structured JSON report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_evaluations": len(results),
            "dimensions": list({r.dimension for r in results}),
            "providers": list({r.provider for r in results}),
            "models": list({r.model for r in results}),
        },
        "summary": _build_summary(results),
        "results": [asdict(r) for r in results],
    }

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return path


def _build_summary(results: list[EvalResult]) -> dict:
    """Build aggregate summary statistics."""
    scored = [r for r in results if r.judge_score is not None]
    if not scored:
        return {"note": "No scored results"}

    # By provider
    by_provider: dict[str, list[float]] = {}
    for r in scored:
        key = f"{r.provider}/{r.model}"
        by_provider.setdefault(key, []).append(r.judge_score)

    # By dimension
    by_dimension: dict[str, list[float]] = {}
    for r in scored:
        by_dimension.setdefault(r.dimension, []).append(r.judge_score)

    # By provider x dimension
    by_provider_dim: dict[str, dict[str, list[float]]] = {}
    for r in scored:
        key = f"{r.provider}/{r.model}"
        by_provider_dim.setdefault(key, {}).setdefault(r.dimension, []).append(r.judge_score)

    def _avg(scores: list[float]) -> float:
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    return {
        "overall_average": _avg([r.judge_score for r in scored]),
        "by_provider": {k: _avg(v) for k, v in by_provider.items()},
        "by_dimension": {k: _avg(v) for k, v in by_dimension.items()},
        "by_provider_dimension": {
            provider: {dim: _avg(scores) for dim, scores in dims.items()}
            for provider, dims in by_provider_dim.items()
        },
        "total_errors": sum(1 for r in results if r.response_text.startswith("ERROR:")),
        "average_latency_ms": round(
            sum(r.latency_ms for r in results) / len(results), 1
        ) if results else 0,
    }
