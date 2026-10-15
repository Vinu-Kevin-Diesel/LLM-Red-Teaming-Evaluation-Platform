"""Tests for student training data, batching and evaluation metrics (no GPU, no network)."""

import json
import random

import pytest

from llm_eval_forge.distill import evaluate as E
from llm_eval_forge.distill import metrics as M
from llm_eval_forge.distill import train as TR
from llm_eval_forge.distill.data import write_jsonl


def test_token_budget_batches_cover_all_and_respect_budget():
    rng = random.Random(0)
    lengths = [rng.randint(10, 600) for _ in range(500)]
    batches = TR.token_budget_batches(lengths, tokens_per_batch=2048, max_batch_size=16, rng=rng)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(500))
    for b in batches:
        assert len(b) <= 16
        assert len(b) == 1 or max(lengths[i] for i in b) * len(b) <= 2048


def test_token_budget_batches_single_long_example_gets_own_batch():
    batches = TR.token_budget_batches([5000, 10, 10], 1024, 8, random.Random(0), shuffle=False)
    assert [sorted(b) for b in batches] == [[1, 2], [0]]


@pytest.mark.parametrize("harmful,conf,expected", [(True, 5, 1.0), (True, 1, 0.6), (False, 5, 0.0), (False, 3, 0.2)])
def test_soft_target(harmful, conf, expected):
    assert TR.soft_target({"response_harmful": harmful, "confidence": conf}) == pytest.approx(expected)


def test_format_pair():
    assert TR.format_pair("hi", "there") == ("User: hi", "Assistant: there")
    assert TR.format_pair(None, None) == ("User: ", "Assistant: ")


def test_best_f1_threshold_finds_separating_point():
    labels = [0, 0, 0, 1, 1]
    probs = [0.05, 0.10, 0.30, 0.40, 0.90]
    thr, f1 = M.best_f1_threshold(labels, probs)
    assert f1 == pytest.approx(1.0)
    assert 0.30 < thr <= 0.40


def _write_pool(tmp_path):
    pool = [
        {"id": "a", "prompt": "p", "response": "r", "label": 1, "teacher_holdout": False},
        {"id": "b", "prompt": "p", "response": "r", "label": 0, "teacher_holdout": True},
        {"id": "c", "prompt": "p", "response": "r", "label": None, "teacher_holdout": False},
    ]
    labels = [
        {"id": "a", "status": "ok", "verdict": {"response_harmful": True, "confidence": 4}},
        {"id": "b", "status": "ok", "verdict": {"response_harmful": False, "confidence": 5}},
        {"id": "c", "status": "refusal:bio", "verdict": None},
    ]
    write_jsonl(tmp_path / "teacher" / "sets" / "pool.jsonl", pool)
    write_jsonl(tmp_path / "teacher" / "labels" / "pool.jsonl", labels)
    write_jsonl(tmp_path / "train" / "wildguardtrain.jsonl", [
        {"id": "a", "source": "wildguardtrain", "prompt": "p", "response": "r", "label": 1},
        {"id": "b", "source": "wildguardtrain", "prompt": "p", "response": "r", "label": 0},
        {"id": "z", "source": "wildguardtrain", "prompt": "p", "response": "r", "label": 0},
    ])


def test_stage_b_uses_teacher_soft_labels_and_skips_holdout_and_refusals(tmp_path):
    _write_pool(tmp_path)
    assert TR.stage_b_examples(tmp_path) == [("p", "r", pytest.approx(0.9))]


def test_stage_a_excludes_teacher_holdout(tmp_path):
    _write_pool(tmp_path)
    assert [y for _, _, y in TR.stage_a_examples(tmp_path)] == [1.0, 0.0]  # 'b' is held out


def test_auroc_known_values():
    assert M.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert M.auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)
    assert M.auroc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)  # ties
    assert M.auroc([1, 1], [0.2, 0.3]) is None


def test_ece_zero_when_confident_and_correct():
    assert M.ece([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0]) == pytest.approx(0.0)
    assert M.ece([0, 1], [1.0, 0.0]) == pytest.approx(1.0)


def test_temperature_scaling_softens_overconfident_scores():
    labels = [1, 0] * 50 + [1] * 10
    probs = [0.99, 0.01] * 50 + [0.01] * 10  # 10 confident mistakes
    t = M.fit_temperature(labels, probs)
    assert t > 1.0
    scaled = M.apply_temperature([0.99], t)[0]
    assert 0.5 < scaled < 0.99


def _metrics(name, kind, f1s, kappa=None):
    sets = {b: {"f1": f, "auroc": 0.9, "ece_scaled": 0.05, "n": 100} for b, f in f1s.items()}
    if kappa is not None:
        sets["teacher_test"] = {"kappa": kappa, "f1": 0.8, "auroc": 0.9, "ece_scaled": 0.05, "n": 50}
    return {"name": name, "kind": kind, "sets": sets, "avg_response_f1": sum(f1s.values()) / len(f1s),
            "pilot_subset": {"harmbench_val": {"f1": 0.8, "kappa": 0.7}}}


def test_render_table_includes_students_baselines_published_and_teacher_subset():
    metrics = [
        _metrics("student-AB", "student", {"harmbench_val": 0.85, "wildguardtest": 0.75}, kappa=0.72),
        _metrics("qwen3guard-0.6b", "baseline", {"harmbench_val": 0.84, "wildguardtest": 0.70}),
        {"name": "claude-sonnet-5-5-teacher", "kind": "teacher", "sets": {},
         "pilot_subset": {"harmbench_val": {"f1": 0.91, "kappa": 0.83}}},
    ]
    text = E.render_table(metrics)
    assert "| Judge | WGTest | HarmB | Avg F1 |" in text
    assert "| **student-AB** | 75.0 | 85.0 | 80.0 |" in text
    assert "0.720" in text  # kappa vs teacher
    assert "Qwen3Guard-Gen-0.6B (published)" in text
    assert "claude-sonnet-5-5-teacher | –" not in text
    assert "| claude-sonnet-5-5-teacher | 91.0 |" in text
    student_line = next(line for line in text.splitlines() if "student-AB" in line and "**" in line)
    qwen_line = next(line for line in text.splitlines() if line.startswith("| qwen3guard"))
    assert text.index(student_line) < text.index(qwen_line)  # sorted by average F1


def test_eval_sets_teacher_holdout_uses_teacher_label(tmp_path):
    _write_pool(tmp_path)
    write_jsonl(tmp_path / "test" / "harmbench_val.jsonl",
                [{"id": "h1", "source": "harmbench_val", "prompt": "p", "response": "r", "label": 1}])
    write_jsonl(tmp_path / "val" / "aegis2_val.jsonl",
                [{"id": "v1", "source": "aegis2_val", "prompt": "p", "response": "r", "label": 0}])
    sets = E.eval_sets(tmp_path)
    assert [r["id"] for r in sets["harmbench_val"]] == ["h1"]
    assert [r["id"] for r in sets["val"]] == ["v1"]
    held = sets["teacher_val"] + sets["teacher_test"]
    assert [(r["id"], r["label"], r["gold"]) for r in held] == [("b", 0, 0)]


def test_holdout_part_is_deterministic_and_roughly_half():
    parts = [TR.holdout_part(f"wildguardtrain_{i:06d}") for i in range(2000)]
    assert parts == [TR.holdout_part(f"wildguardtrain_{i:06d}") for i in range(2000)]
    assert 0.45 < parts.count("val") / len(parts) < 0.55


def test_best_kappa_threshold_matches_teacher_decisions():
    labels = [0, 0, 0, 1, 1, 1]
    probs = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]
    t = E.best_kappa_threshold(labels, probs)
    assert 0.3 < t <= 0.6
