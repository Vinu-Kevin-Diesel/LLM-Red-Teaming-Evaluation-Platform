"""Evaluator registry — decorator-based factory."""

from __future__ import annotations

from typing import Type

from llm_eval_forge.evaluators.base import Evaluator

_EVALUATORS: dict[str, Type[Evaluator]] = {}


def register_evaluator(dimension: str):
    """Class decorator to register an evaluator."""
    def wrapper(cls: Type[Evaluator]) -> Type[Evaluator]:
        _EVALUATORS[dimension] = cls
        return cls
    return wrapper


def get_evaluator(dimension: str) -> Evaluator:
    if dimension not in _EVALUATORS:
        available = ", ".join(_EVALUATORS.keys()) or "(none)"
        raise ValueError(f"Unknown evaluator: {dimension!r}. Available: {available}")
    return _EVALUATORS[dimension]()


def list_evaluators() -> list[str]:
    return list(_EVALUATORS.keys())
