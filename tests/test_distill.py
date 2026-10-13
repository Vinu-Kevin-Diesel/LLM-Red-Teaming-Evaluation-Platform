"""Tests for the distillation data builder and Claude teacher plumbing (no network, no API)."""

import json
from types import SimpleNamespace

import pytest

from llm_eval_forge.distill import data as D
from llm_eval_forge.distill import teacher as T


def _ex(source, split, prompt, response="r", label=0):
    return D.Example(id=f"{source}_{prompt}", source=source, split=split, prompt=prompt,
                     response=response, label=label)


def test_normalize_and_base_case_id():
    assert D.normalize("  Hello\n\tWORLD  ") == "hello world"
    assert D.base_case_id("adv-003_mut_encoding_attack_3") == "adv-003"
    assert D.base_case_id("adv-003") == "adv-003"


def test_bucket_is_deterministic_and_in_range():
    values = [D.bucket(f"k{i}", salt="s") for i in range(200)]
    assert all(0 <= v < 1 for v in values)
    assert values == [D.bucket(f"k{i}", salt="s") for i in range(200)]
    assert D.bucket("k1", salt="a") != D.bucket("k1", salt="b")


def test_harmbench_majority_vote():
    raw = {"b1": [
        {"test_case": "p1", "generation": "g1", "human_0": "1", "human_1": "1", "human_2": "0"},
        {"test_case": "p2", "generation": "g2", "human_0": "1", "human_1": "0", "human_2": "0"},
    ]}
    rows = D.harmbench_examples(raw)
    assert [r.label for r in rows] == [1, 0]
    assert rows[0].id == "harmbench_val_000000"


def test_saferlhf_keeps_only_differing_pairs_and_balances():
    rows = [
        {"prompt": f"p{i}", "response_0": f"a{i}", "response_1": f"b{i}",
         "is_response_0_safe": i % 3 != 0, "is_response_1_safe": i % 3 == 0}
        for i in range(30)
    ] + [{"prompt": "same", "response_0": "x", "response_1": "y",
          "is_response_0_safe": True, "is_response_1_safe": True}]
    out = D.saferlhf_examples(rows, per_class=10, seed=42)
    assert sum(e.label for e in out) == 10 and len(out) == 20
    assert all(e.prompt != "same" for e in out)
    assert [e.response for e in out] == [e.response for e in D.saferlhf_examples(rows, per_class=10, seed=42)]


def test_drop_contaminated_removes_train_rows_with_test_prompts():
    examples = [
        _ex("wg", "train", "How do I bake bread?"),
        _ex("wg", "train", "Leaked prompt"),
        _ex("aegis", "val", "leaked   PROMPT"),
        _ex("hb", "test", "Leaked prompt"),
    ]
    kept, dropped = D.drop_contaminated(examples)
    assert [e.prompt for e in kept] == ["How do I bake bread?", "Leaked prompt"]
    assert kept[1].split == "test"
    assert dropped == {"wg": 1, "aegis": 1}


def test_own_redteam_split_groups_mutations(tmp_path):
    rows = []
    for base in range(20):
        for m in ("", "_mut_roleplay_0", "_mut_encoding_attack_1"):
            rows.append({"test_case_id": f"adv-{base:03d}{m}", "dimension": "adversarial", "model": "m",
                         "prompt_sent": f"p{base}{m}", "response_text": "ok", "judge_score": 0.5,
                         "metadata": {"attack_type": "x"}})
    rows.append({"test_case_id": "adv-999", "dimension": "adversarial", "model": "m", "prompt_sent": "p",
                 "response_text": "ERROR: 429", "judge_score": None, "metadata": {}})
    rows.append({"test_case_id": "hal-001", "dimension": "hallucination", "model": "m", "prompt_sent": "p",
                 "response_text": "ok", "judge_score": 1.0, "metadata": {}})
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"results": rows}), encoding="utf-8")

    out = D.own_redteam_examples(path)
    assert len(out) == 60  # error row and non-adversarial row skipped
    splits_by_base = {}
    for ex in out:
        splits_by_base.setdefault(D.base_case_id(ex.meta["test_case_id"]), set()).add(ex.split)
    assert all(len(s) == 1 for s in splits_by_base.values())
    assert all(ex.label is None and ex.label_source == "pending_teacher" for ex in out)


def _write_split(data_dir, split, source, rows):
    D.write_jsonl(data_dir / split / f"{source}.jsonl", rows)


def _rows(source, split, n_pos, n_neg):
    return [{"id": f"{source}_{i:06d}", "source": source, "split": split, "prompt": f"p{i}",
             "response": f"r{i}", "label": int(i < n_pos), "meta": {}} for i in range(n_pos + n_neg)]


def test_select_pilot_is_balanced_per_source(tmp_path):
    _write_split(tmp_path, "test", "harmbench_val", _rows("harmbench_val", "test", 100, 100))
    _write_split(tmp_path, "test", "wildguardtest", _rows("wildguardtest", "test", 100, 300))
    _write_split(tmp_path, "test", "beavertails_test", _rows("beavertails_test", "test", 50, 50))
    pilot = T.select_pilot(tmp_path)
    assert len(pilot) == 300
    for source in ("harmbench_val", "wildguardtest"):
        labels = [r["label"] for r in pilot if r["source"] == source]
        assert sum(labels) == 75 and len(labels) == 150


def test_select_pool_marks_holdout(tmp_path):
    _write_split(tmp_path, "train", "wildguardtrain", _rows("wildguardtrain", "train", 2000, 2000))
    _write_split(tmp_path, "train", "aegis2_train", _rows("aegis2_train", "train", 1500, 1500))
    own = _rows("own_redteam", "train", 0, 5) + _rows("own_redteam", "test", 0, 2)
    for r in own:
        r["label"] = None
    _write_split(tmp_path, "train", "own_redteam", own[:5])
    _write_split(tmp_path, "test", "own_redteam", own[5:])
    pool = T.select_pool(tmp_path)
    public = [r for r in pool if r["source"] != "own_redteam"]
    assert len(public) == 5600 and sum(r["teacher_holdout"] for r in public) == 560
    assert [r["teacher_holdout"] for r in pool if r["source"] == "own_redteam"] == [False] * 5 + [True] * 2


def test_user_message_truncates_long_fields():
    msg = T.user_message("p" * 10, "x" * 7000, max_chars=6000)
    assert "[... truncated 1000 characters ...]" in msg
    assert "<assistant>" in msg and "</conversation>" in msg


def test_request_params_use_structured_output_and_effort():
    params = T.request_params({"prompt": "p", "response": "r"})
    fmt = params["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert set(fmt["schema"]["required"]) == set(T.SafetyVerdict.model_fields)
    assert params["output_config"]["effort"] == "low"
    assert params["model"] == "claude-sonnet-5-5"


def _batch_result(text="", stop_reason="end_turn", category=None, rtype="succeeded"):
    msg = SimpleNamespace(
        model="claude-sonnet-5-5", stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if category else None,
        usage=SimpleNamespace(input_tokens=900, output_tokens=150),
        content=[SimpleNamespace(type="text", text=text)],
    )
    return SimpleNamespace(type=rtype, message=msg)


GOOD = json.dumps({"response_harmful": True, "refusal": False, "harm_category": "weapons",
                   "confidence": 4, "rationale": "Gives assembly steps."})


def test_parse_result_ok_refusal_and_errors():
    ok = T.parse_result("a", _batch_result(GOOD))
    assert ok["status"] == "ok" and ok["verdict"]["response_harmful"] is True
    assert ok["usage"] == {"input_tokens": 900, "output_tokens": 150}
    ref = T.parse_result("b", _batch_result(stop_reason="refusal", category="bio"))
    assert ref["status"] == "refusal:bio" and ref["verdict"] is None
    cut = T.parse_result("c", _batch_result(GOOD[:30], stop_reason="max_tokens"))
    assert cut["status"] == "unparsed:max_tokens" and cut["verdict"] is None
    err = T.parse_result("d", SimpleNamespace(type="errored"))
    assert err["status"] == "errored" and err["verdict"] is None


def test_binary_metrics_known_values():
    m = T.binary_metrics([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0])
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(6 / 8)
    # pe = (3/8)(3/8) + (5/8)(5/8) = 34/64; kappa = (48/64 - 34/64) / (30/64) = 14/30
    assert m["kappa"] == pytest.approx(14 / 30)
    assert T.binary_metrics([1, 0], [1, 0])["kappa"] == pytest.approx(1.0)


def test_parse_message_live_path_and_cost_multiplier():
    msg = _batch_result(GOOD).message
    rec = T.parse_message("x", msg)
    assert rec["status"] == "ok" and rec["policy"] == T.POLICY_VERSION
    batch_rec = {**rec, "usage": {"input_tokens": 1_000_000, "output_tokens": 0}}
    live_rec = {**batch_rec, "live": True}
    assert T.batch_cost([batch_rec], "claude-sonnet-5-5") == pytest.approx(1.0)
    assert T.batch_cost([live_rec], "claude-sonnet-5-5") == pytest.approx(2.0)


def test_batches_for_filters_by_policy(tmp_path):
    T.log_batch(tmp_path, {"set": "pilot", "batch_id": "old"})  # pre-versioning entries count as v1
    T.log_batch(tmp_path, {"set": "pilot", "batch_id": "new", "policy": T.POLICY_VERSION})
    T.log_batch(tmp_path, {"set": "pool", "batch_id": "other", "policy": T.POLICY_VERSION})
    assert [e["batch_id"] for e in T.batches_for(tmp_path, "pilot")] == ["new"]
    assert [e["batch_id"] for e in T.batches_for(tmp_path, "pilot", policy="v1")] == ["old"]
