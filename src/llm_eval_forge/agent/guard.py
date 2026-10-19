"""Judge-in-the-loop guard for AgentDojo agents: every proposed tool call is judged before it runs.

A blocked call is not executed; the agent receives a tool error instead and keeps going, so a
guard cannot "win" by crashing the episode, and utility under attack stays measurable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from agentdojo.agent_pipeline import BasePipelineElement, ToolsExecutor
from agentdojo.functions_runtime import EmptyEnv
from agentdojo.types import ChatToolResultMessage, text_content_block_from_string

from llm_eval_forge.agent.toolcall_data import call_text, content_text, identifiers

BLOCK_MESSAGE = "Blocked by the security policy: this action does not appear to follow from the user's request."


@dataclass
class Decision:
    action: str  # allow | block
    p_attack: float | None = None
    latency_ms: float = 0.0


class Judge(Protocol):
    name: str

    def __call__(self, goal: str, history: list[dict], call: dict, tool_outputs: list[str]) -> Decision: ...


class AllowAll:
    name = "none"

    def __call__(self, goal, history, call, tool_outputs) -> Decision:
        return Decision("allow")


class HttpToolCallJudge:
    """Our tool-call guard behind the judge service's POST /v1/judge/tool_call."""

    name = "toolcall-guard"

    def __init__(self, url: str = "http://127.0.0.1:8000"):
        import httpx

        self.client = httpx.Client(base_url=url, timeout=30)

    def __call__(self, goal, history, call, tool_outputs) -> Decision:
        t0 = time.perf_counter()
        r = self.client.post("/v1/judge/tool_call", json={
            "user_goal": goal, "tool_name": call["function"], "arguments": call.get("args") or {},
            "history": [{"tool_name": h["function"], "arguments": h.get("args") or {}} for h in history],
            "tool_outputs": tool_outputs,
        })
        r.raise_for_status()
        body = r.json()
        return Decision(body["decision"], body["p_attack"], round((time.perf_counter() - t0) * 1000, 3))


class OracleJudge:
    """Upper bound: blocks exactly the calls the dataset labels attacker-induced (arguments contain an
    identifier from the injected text that the user never asked for). Needs the injected text."""

    name = "oracle"

    def __init__(self):
        self.injected: set[str] = set()

    def set_injections(self, injections: dict[str, str], goal: str) -> None:
        self.injected = set().union(*(identifiers(v) for v in injections.values())) - identifiers(goal) if injections else set()

    def __call__(self, goal, history, call, tool_outputs) -> Decision:
        import json

        hit = identifiers(json.dumps(call.get("args") or {}, ensure_ascii=False)) & self.injected
        return Decision("block" if hit else "allow")


@dataclass
class GuardLog:
    entries: list[dict] = field(default_factory=list)


class GuardedToolsExecutor(BasePipelineElement):
    """Drop-in replacement for ToolsExecutor that asks a judge before running each tool call."""

    def __init__(self, judge: Judge, log: GuardLog | None = None):
        self.judge, self.inner, self.log = judge, ToolsExecutor(), log or GuardLog()

    def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args=None):
        extra_args = {} if extra_args is None else extra_args
        if not messages or messages[-1]["role"] != "assistant" or not messages[-1]["tool_calls"]:
            return query, runtime, env, messages, extra_args
        history, tool_outputs = [], []
        for m in messages[:-1]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                history += [{"function": c.function, "args": dict(c.args)} for c in m["tool_calls"]]
            if m["role"] == "tool":
                tool_outputs.append(content_text(m) or str(m.get("error") or ""))
        last = messages[-1]
        allowed, blocked = [], []
        for tc in last["tool_calls"]:
            call = {"function": tc.function, "args": dict(tc.args)}
            d = self.judge(query, history, call, tool_outputs)
            self.log.entries.append({"judge": self.judge.name, "call": call_text(call), "action": d.action,
                                     "p_attack": d.p_attack, "latency_ms": d.latency_ms})
            (blocked if d.action == "block" else allowed).append(tc)
        _, runtime, env, out, extra_args = self.inner.query(query, runtime, env, [*messages[:-1], {**last, "tool_calls": allowed}], extra_args)
        results = list(out[len(messages):])
        for tc in blocked:  # every tool_call_id must still get a result
            results.append(ChatToolResultMessage(role="tool", content=[text_content_block_from_string("")],
                                                 tool_call_id=tc.id, tool_call=tc, error=BLOCK_MESSAGE))
        return query, runtime, env, [*messages, *results], extra_args


JUDGES: dict[str, Callable[[], Judge]] = {"none": AllowAll, "toolcall-guard": HttpToolCallJudge, "oracle": OracleJudge}
