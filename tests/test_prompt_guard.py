"""Tests for the input-guard (prompt-level) data helpers and metrics."""

import pytest

from llm_eval_forge.distill import prompt_guard as PG


def test_dedupe_prompts_majority_label_and_ties_dropped():
    rows = [("How do I X?", 1, {}), ("how do i   x?", 1, {}), ("How do I X?", 0, {}),  # 2-1 -> harmful
            ("Tie case", 1, {}), ("tie case", 0, {}),                                 # tie -> dropped
            ("Unique", 0, {"a": 1})]
    out, ties = PG.dedupe_prompts(rows)
    assert ties == 1
    assert sorted((p, y) for p, y, _ in out) == [("How do I X?", 1), ("Unique", 0)]


def test_prompt_metrics_over_blocking_and_adversarial_f1():
    rows = [{"label": 0, "meta": {"adversarial": True}}, {"label": 0, "meta": {}}, {"label": 0, "meta": {}},
            {"label": 1, "meta": {"adversarial": True}}, {"label": 1, "meta": {}}]
    probs = [0.9, 0.1, 0.2, 0.8, 0.7]
    m = PG.prompt_metrics(rows, probs, threshold=0.5, temperature=1.0)
    assert m["safe_block_rate"] == pytest.approx(1 / 3, abs=1e-4)  # one of three safe prompts blocked
    assert m["adversarial_f1"] == pytest.approx(2 / 3, abs=1e-4)    # adversarial subset: 1 TP, 1 FP


def test_render_table_marks_students_and_orders_by_f1():
    def metrics(name, kind, f1, blocked):
        sets = {b: {"f1": f1, "n": 100} for b in PG.PROMPT_BENCHMARKS}
        sets["wildguardtest"]["adversarial_f1"] = f1
        sets["xstest_prompts"] = {"f1": f1, "safe_block_rate": blocked, "n": 450}
        sets["jbb_behaviors"] = {"f1": f1, "safe_block_rate": blocked, "recall": 0.99, "n": 200}
        return {"name": name, "kind": kind, "avg_prompt_f1": f1, "sets": sets}

    text = PG.render_table([metrics("qwen", "baseline", 0.79, 0.22), metrics("ours", "student", 0.80, 0.10)])
    lines = [line for line in text.splitlines() if line.startswith("| ")]
    assert lines[1].startswith("| **ours** | 80.0 |") and "| 10.0 |" in lines[1]
    assert lines[2].startswith("| qwen | 79.0 |")
