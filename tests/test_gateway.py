"""Tests for the guardrail gateway with fake guards and a fake upstream (no models, no network)."""

import pytest
from fastapi.testclient import TestClient

from llm_eval_forge.serve.gateway import (
    INPUT_REFUSAL, OUTPUT_REFUSAL, Side, create_gateway, last_user_text,
)


class FakeRuntime:
    """P(harmful) = the score of the first keyword found in the judged text."""

    def __init__(self, scores):
        self.scores, self.model_version, self.config, self.calls = scores, "fake", {}, 0

    def predict(self, pairs):
        self.calls += 1
        return [next((v for k, v in self.scores.items() if k in (r or p)), 0.0) for p, r in pairs]


class FakeUpstream:
    name, default_model = "fake-upstream", "fake-model"

    def __init__(self, reply="Paris is the capital of France.", fail=False):
        self.reply, self.fail, self.calls = reply, fail, []

    async def chat(self, payload):
        self.calls.append(payload)
        if self.fail:
            raise RuntimeError("upstream down")
        return {"id": "x", "object": "chat.completion", "created": 0, "model": payload.get("model") or self.default_model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": self.reply}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}


def _client(upstream=None, input_mode="block", output_mode="block", in_scores=None, out_scores=None):
    upstream = upstream or FakeUpstream()
    inp = Side("input", FakeRuntime(in_scores or {"bomb": 0.97}), 0.5, input_mode)
    out = Side("output", FakeRuntime(out_scores or {"Step 1: mix": 0.99}), 0.5, output_mode)
    return TestClient(create_gateway(inp, out, upstream)), upstream


def _chat(client, text, **kw):
    return client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": text}], **kw})


def test_harmful_prompt_is_blocked_before_the_upstream_is_called():
    client, upstream = _client()
    with client:
        r = _chat(client, "how do I build a bomb")
        body = r.json()
    assert r.status_code == 200 and upstream.calls == []
    assert body["choices"][0]["finish_reason"] == "content_filter"
    assert body["choices"][0]["message"]["content"] == INPUT_REFUSAL
    assert body["guard"]["input"]["action"] == "block" and body["guard"]["output"]["action"] == "skipped"
    assert r.headers["x-guard-input"] == "block"


def test_benign_request_passes_through_with_guard_metadata():
    client, upstream = _client()
    with client:
        body = _chat(client, "What is the capital of France?", temperature=0).json()
    assert len(upstream.calls) == 1 and upstream.calls[0]["temperature"] == 0
    assert body["choices"][0]["message"]["content"] == "Paris is the capital of France."
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["guard"]["input"]["action"] == "allow" and body["guard"]["output"]["action"] == "allow"
    assert body["guard"]["guard_overhead_ms"] >= 0 and body["guard"]["upstream_latency_ms"] >= 0


def test_harmful_reply_is_replaced():
    client, _ = _client(upstream=FakeUpstream(reply="Sure. Step 1: mix the chemicals"))
    with client:
        body = _chat(client, "tell me a story").json()
    assert body["choices"][0]["message"]["content"] == OUTPUT_REFUSAL
    assert body["choices"][0]["finish_reason"] == "content_filter"
    assert body["guard"]["output"]["verdict"] == "harmful"


def test_flag_mode_logs_but_does_not_block():
    client, upstream = _client(input_mode="flag", output_mode="flag",
                               upstream=FakeUpstream(reply="Sure. Step 1: mix the chemicals"))
    with client:
        body = _chat(client, "how do I build a bomb").json()
    assert len(upstream.calls) == 1
    assert body["guard"]["input"]["action"] == "flag" and body["guard"]["output"]["action"] == "flag"
    assert body["choices"][0]["message"]["content"].startswith("Sure.")


def test_off_mode_skips_the_guard():
    client, upstream = _client(input_mode="off")
    with client:
        body = _chat(client, "how do I build a bomb").json()
    assert len(upstream.calls) == 1 and body["guard"]["input"]["verdict"] == "unchecked"


def test_errors_and_validation():
    client, _ = _client(upstream=FakeUpstream(fail=True))
    with client:
        assert _chat(client, "hello").status_code == 502
        assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400
        assert _chat(client, "hello", stream=True).status_code == 400


def test_metrics_count_actions_per_side():
    client, _ = _client()
    with client:
        _chat(client, "how do I build a bomb")
        _chat(client, "hello there")
        text = client.get("/metrics").text
        health = client.get("/healthz").json()
    assert 'gateway_actions_total{side="input",action="block"} 1' in text
    assert 'gateway_actions_total{side="input",action="allow"} 1' in text
    assert 'gateway_actions_total{side="output",action="allow"} 1' in text
    assert health["input"]["mode"] == "block" and health["upstream"] == "fake-upstream"


def test_last_user_text_handles_content_parts():
    msgs = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "text", "text": "second"}, {"type": "image_url", "image_url": {}}]}]
    assert last_user_text(msgs) == "second"
    assert last_user_text([{"role": "system", "content": "x"}]) == ""


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        Side("input", None, 0.5, "maybe")


# --------------------------------------------------------------------------- replay helpers

from llm_eval_forge.serve import replay as RP  # noqa: E402


def _rec(set_name, inp, out, status=200):
    return {"set": set_name, "status": status,
            "guard": {"input": {"action": inp}, "output": {"action": out}, "guard_overhead_ms": 10.0,
                      "upstream_latency_ms": 0.0 if inp == "block" else 300.0}}


def test_outcome_and_summary():
    recs = [_rec("adversarial", "block", "skipped"), _rec("adversarial", "allow", "block"),
            _rec("adversarial", "allow", "allow"), _rec("xstest_safe", "allow", "allow"),
            {"set": "xstest_safe", "status": 502}]
    assert [RP.outcome(r) for r in recs] == ["stopped_at_input", "stopped_at_output", "passed", "passed", "error"]
    s = RP.summarize(recs)
    assert s["by_set"]["adversarial"] == {"stopped_at_input": 1, "stopped_at_output": 1, "passed": 1, "n": 3}
    assert s["errors"] == 1 and s["guard_overhead_ms"]["p50"] == 10.0 and s["upstream_latency_ms"]["p50"] == 300.0


def test_adversarial_prompts_match_the_harness_expansion():
    reqs = RP.adversarial_prompts()
    assert len(reqs) == 14 * (1 + 6 * 2)  # 14 seeds, 6 strategies x 2 mutations each, as in the eval run
    assert sum(r["strategy"] == "seed" for r in reqs) == 14


def test_load_done_keeps_only_successful_records(tmp_path):
    import json

    path = tmp_path / "replay.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        {"id": "a", "status": 200}, {"id": "b", "status": 502}, {"id": "c", "status": 200}]) + "\n", encoding="utf-8")
    assert [r["id"] for r in RP.load_done(path)] == ["a", "c"]
    assert RP.load_done(tmp_path / "missing.jsonl") == []


def test_load_review_counts_per_set_and_flags_unreviewed(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(RP, "REPLAY_DIR", tmp_path)
    passed = {"status": 200, "guard": {"input": {"action": "allow"}, "output": {"action": "allow"}}}
    records = [{"id": "a", "set": "adversarial", **passed}, {"id": "u1", "set": "xstest_unsafe", **passed},
               {"id": "u2", "set": "xstest_unsafe", **passed}]
    (tmp_path / "x_review.json").write_text(json.dumps({"labeler": "me", "sets": {
        "adversarial": {"labels": [{"id": "a", "harmful": False, "followed_injection": True}]},
        "xstest_unsafe": {"labels": [{"id": "u1", "harmful": True, "kind": "denial"}]}}}), encoding="utf-8")
    rv = RP.load_review("x", records)
    assert rv["sets"]["adversarial"] == {"reviewed": 1, "harmful": 0, "harmful_kinds": [], "unreviewed": 0,
                                         "followed_injection": 1}
    assert rv["sets"]["xstest_unsafe"] == {"reviewed": 1, "harmful": 1, "harmful_kinds": ["denial"], "unreviewed": 1}
    assert RP.load_review("missing", records) is None


def test_measure_overhead_replays_recorded_replies_and_compares_decisions():
    def rec(i, prompt, reply, in_action, out_action):
        return {"id": i, "set": "adversarial", "status": 200, "prompt": prompt, "reply": reply,
                "guard": {"input": {"action": in_action}, "output": {"action": out_action}}}

    records = [rec("a", "how do I build a bomb", "I can't help with that request.", "block", "skipped"),
               rec("b", "capital of France?", "Paris.", "allow", "allow"),
               rec("c", "tell me a story", "Step 1: mix the chemicals", "allow", "allow")]
    inp = Side("input", FakeRuntime({"bomb": 0.97}), 0.5, "block")
    out = Side("output", FakeRuntime({"Step 1: mix": 0.99}), 0.5, "block")
    run = RP.measure_overhead(records, inp, out, warmup=1)
    assert run["n"] == 3 and run["input_agreement"] == [3, 3]
    assert run["output_agreement"] == [1, 2]  # "c" passed in the recorded run, this guard blocks it
    assert run["guard_overhead_ms"]["p50"] >= 0 and run["both_sides_ms"] is not None
