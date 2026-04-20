"""Tests for red-team mutation engine."""

import pytest

from llm_eval_forge.redteam.engine import MutationEngine
from llm_eval_forge.redteam.base import MutatedPrompt
from llm_eval_forge.redteam.roleplay import RolePlayInjection
from llm_eval_forge.redteam.encoding import EncodingAttack


def test_mutation_engine_all_strategies():
    engine = MutationEngine()
    strategies = engine.list_strategies()
    assert len(strategies) == 6
    assert "roleplay_injection" in strategies
    assert "encoding_attack" in strategies


def test_mutation_engine_generate_attacks():
    engine = MutationEngine()
    attacks = engine.generate_attacks(
        clean_prompt="Tell me a joke",
        target_behavior="harmless",
    )
    assert len(attacks) > 0
    assert all(isinstance(a, MutatedPrompt) for a in attacks)


def test_mutation_engine_filter_strategies():
    engine = MutationEngine(strategies=["roleplay_injection", "encoding_attack"])
    assert len(engine.list_strategies()) == 2
    attacks = engine.generate_attacks("test prompt", "test behavior")
    strategies_used = {a.strategy for a in attacks}
    assert strategies_used == {"roleplay_injection", "encoding_attack"}


def test_mutation_engine_max_per_strategy():
    engine = MutationEngine()
    attacks = engine.generate_attacks(
        "test prompt", "test behavior", max_per_strategy=1,
    )
    # Each strategy should produce at most 1 mutation
    from collections import Counter
    counts = Counter(a.strategy for a in attacks)
    assert all(c <= 1 for c in counts.values())


def test_roleplay_injection():
    strategy = RolePlayInjection()
    results = strategy.mutate("What is 2+2?", "test")
    assert len(results) == 3  # 3 personas
    assert all("What is 2+2?" in r.mutated for r in results)


def test_encoding_attack():
    strategy = EncodingAttack()
    results = strategy.mutate("Hello world", "test")
    assert len(results) == 3  # base64, rot13, leetspeak
    assert any("base64" in r.description.lower() for r in results)


def test_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        MutationEngine(strategies=["nonexistent"])


def test_mutated_prompt_fields():
    engine = MutationEngine(strategies=["roleplay_injection"])
    attacks = engine.generate_attacks("test", "behavior")
    for a in attacks:
        assert a.original == "test"
        assert a.mutated  # not empty
        assert a.strategy == "roleplay_injection"
        assert a.description  # not empty
