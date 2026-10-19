"""Guardrail gateway: an OpenAI-compatible proxy that checks every request on both sides.

    user/app ─► POST /v1/chat/completions ─► input guard ─► upstream LLM ─► output guard ─► user/app

Point any OpenAI-SDK client's `base_url` at the gateway instead of at the provider. A harmful
prompt is answered with a refusal and `finish_reason: "content_filter"` without calling the
upstream model; a harmful reply is replaced the same way. Every response carries a `guard`
block (verdicts, probabilities, latencies) and `x-guard-*` headers.

    uvicorn llm_eval_forge.serve.gateway:create_gateway --factory --port 8000

Configuration (environment):
    GATEWAY_INPUT_MODEL_DIR / GATEWAY_INPUT_RUNTIME     input guard (default models/prompt-guard-A, torch-cuda)
    GATEWAY_OUTPUT_MODEL_DIR / GATEWAY_OUTPUT_RUNTIME   output guard (default models/student-large-AB, torch-cuda)
    GATEWAY_INPUT_POLICY / GATEWAY_OUTPUT_POLICY        threshold policy: gold | teacher (default gold / teacher)
    GATEWAY_INPUT_MODE / GATEWAY_OUTPUT_MODE            block | flag (log only) | off   (default block)
    UPSTREAM_BASE_URL   OpenAI-compatible endpoint (default Groq)
    UPSTREAM_API_KEY    defaults to GROQ_API_KEY
    UPSTREAM_MODEL      used when the request names no model (default qwen/qwen3.8-27b)
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import Counter, deque
from contextlib import asynccontextmanager
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

INPUT_REFUSAL = "I can't help with that request."
OUTPUT_REFUSAL = "I can't share that response because it may be harmful."
MODES = ("block", "flag", "off")


class Upstream(Protocol):
    name: str

    async def chat(self, payload: dict) -> dict: ...


class OpenAIUpstream:
    """Any OpenAI-compatible chat endpoint (Groq, Cerebras, Ollama, vLLM, ...)."""

    def __init__(self, base_url: str, api_key: str, default_model: str):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key or "missing-api-key", max_retries=4)
        self.default_model = default_model
        self.name = f"{base_url} ({default_model})"

    async def chat(self, payload: dict) -> dict:
        payload = {**payload, "model": payload.get("model") or self.default_model}
        resp = await self.client.chat.completions.create(**payload)
        return resp.model_dump()


class Side:
    """One guard side (input or output): runtime, decision threshold and mode."""

    def __init__(self, name: str, runtime, threshold: float, mode: str):
        if mode not in MODES:
            raise ValueError(f"{name} mode must be one of {MODES}, got {mode!r}")
        self.name, self.runtime, self.threshold, self.mode = name, runtime, threshold, mode
        self.lock = asyncio.Lock()

    async def check(self, prompt: str, response: str) -> dict:
        if self.mode == "off" or self.runtime is None:
            return {"verdict": "unchecked", "action": "allow", "p_harmful": None, "latency_ms": 0.0}
        t0 = time.perf_counter()
        async with self.lock:  # one forward pass at a time per model on the GPU
            p = (await asyncio.get_running_loop().run_in_executor(None, self.runtime.predict, [(prompt, response)]))[0]
        harmful = p >= self.threshold
        action = ("block" if self.mode == "block" else "flag") if harmful else "allow"
        return {"verdict": "harmful" if harmful else "safe", "action": action, "p_harmful": round(p, 6),
                "threshold": self.threshold, "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                "model_version": getattr(self.runtime, "model_version", None)}


def last_user_text(messages: list[dict]) -> str:
    """Text of the latest user turn; OpenAI content may be a string or a list of parts."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content") or ""
            if isinstance(content, list):
                return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
            return str(content)
    return ""


def refusal_completion(model: str, text: str) -> dict:
    return {
        "id": f"guard-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "content_filter"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class GatewayMetrics:
    def __init__(self, window: int = 10_000):
        self.actions: Counter = Counter()
        self.overhead: deque = deque(maxlen=window)
        self.upstream: deque = deque(maxlen=window)

    @staticmethod
    def _q(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0

    def render(self) -> str:
        lines = ["# TYPE gateway_actions_total counter"]
        lines += [f'gateway_actions_total{{side="{s}",action="{a}"}} {n}' for (s, a), n in sorted(self.actions.items())]
        for name, xs in (("gateway_guard_overhead_ms", self.overhead), ("gateway_upstream_latency_ms", self.upstream)):
            lines.append(f"# TYPE {name} summary")
            lines += [f'{name}{{quantile="{q}"}} {self._q(xs, q):.3f}' for q in (0.5, 0.95, 0.99)]
            lines.append(f"{name}_count {len(xs)}")
        return "\n".join(lines) + "\n"


def _env_side(name: str, model_dir: str, runtime_kind: str, policy: str, mode: str) -> Side:
    from llm_eval_forge.serve.runtime import load_runtime

    rt = None
    if mode != "off":
        rt = load_runtime(runtime_kind, model_dir)
        rt.predict([("warm-up", "")])
    threshold = float((getattr(rt, "config", {}) or {}).get("thresholds", {}).get(policy, 0.5)) if rt else 0.5
    return Side(name, rt, threshold, mode)


def create_gateway(input_side: Side | None = None, output_side: Side | None = None,
                   upstream: Upstream | None = None) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["input"] = input_side or _env_side(
            "input", os.environ.get("GATEWAY_INPUT_MODEL_DIR", "models/prompt-guard-A"),
            os.environ.get("GATEWAY_INPUT_RUNTIME", "torch-cuda"), os.environ.get("GATEWAY_INPUT_POLICY", "gold"),
            os.environ.get("GATEWAY_INPUT_MODE", "block"))
        state["output"] = output_side or _env_side(
            "output", os.environ.get("GATEWAY_OUTPUT_MODEL_DIR", "models/student-large-AB"),
            os.environ.get("GATEWAY_OUTPUT_RUNTIME", "torch-cuda"), os.environ.get("GATEWAY_OUTPUT_POLICY", "teacher"),
            os.environ.get("GATEWAY_OUTPUT_MODE", "block"))
        state["upstream"] = upstream or OpenAIUpstream(
            os.environ.get("UPSTREAM_BASE_URL", "https://api.groq.com/openai/v1"),
            os.environ.get("UPSTREAM_API_KEY") or os.environ.get("GROQ_API_KEY", ""),
            os.environ.get("UPSTREAM_MODEL", "qwen/qwen3.8-27b"))
        state["metrics"] = GatewayMetrics()
        yield

    app = FastAPI(title="Guardrail gateway", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "upstream": state["upstream"].name,
                **{side: {"mode": state[side].mode, "threshold": state[side].threshold,
                          "model_version": getattr(state[side].runtime, "model_version", None),
                          "runtime": getattr(state[side].runtime, "name", None)}
                   for side in ("input", "output")}}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        payload = await request.json()
        if payload.get("stream"):
            raise HTTPException(status_code=400, detail="Streaming is not supported by the gateway yet.")
        messages = payload.get("messages") or []
        if not messages:
            raise HTTPException(status_code=400, detail="'messages' is required.")
        prompt = last_user_text(messages)
        metrics: GatewayMetrics = state["metrics"]
        guard: dict[str, Any] = {}

        guard["input"] = await state["input"].check(prompt, "")
        metrics.actions[("input", guard["input"]["action"])] += 1
        model = payload.get("model") or getattr(state["upstream"], "default_model", "upstream")
        if guard["input"]["action"] == "block":
            body = refusal_completion(model, INPUT_REFUSAL)
            guard["output"] = {"verdict": "skipped", "action": "skipped"}
            guard["upstream_latency_ms"] = 0.0
            guard["guard_overhead_ms"] = guard["input"]["latency_ms"]
            metrics.overhead.append(guard["guard_overhead_ms"])
            return _respond(body, guard)

        t0 = time.perf_counter()
        try:
            body = await state["upstream"].chat(payload)
        except Exception as exc:  # surface upstream failures as 502, not as a guard verdict
            raise HTTPException(status_code=502, detail=f"Upstream error: {type(exc).__name__}: {exc}") from exc
        guard["upstream_latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        metrics.upstream.append(guard["upstream_latency_ms"])

        reply = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        guard["output"] = await state["output"].check(prompt, reply)
        metrics.actions[("output", guard["output"]["action"])] += 1
        if guard["output"]["action"] == "block":
            body["choices"][0]["message"]["content"] = OUTPUT_REFUSAL
            body["choices"][0]["finish_reason"] = "content_filter"
        guard["guard_overhead_ms"] = round(guard["input"]["latency_ms"] + guard["output"]["latency_ms"], 3)
        metrics.overhead.append(guard["guard_overhead_ms"])
        return _respond(body, guard)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        return state["metrics"].render()

    return app


def _respond(body: dict, guard: dict) -> JSONResponse:
    body = {**body, "guard": guard}
    headers = {"x-guard-input": str(guard["input"]["action"]), "x-guard-output": str(guard["output"]["action"]),
               "x-guard-overhead-ms": str(guard.get("guard_overhead_ms", ""))}
    return JSONResponse(body, headers=headers)

