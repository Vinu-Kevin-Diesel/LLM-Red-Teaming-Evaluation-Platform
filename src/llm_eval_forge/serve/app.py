"""FastAPI service for the distilled judge.

    uvicorn llm_eval_forge.serve.app:create_app --factory --port 8000

Configuration (environment):
    JUDGE_MODEL_DIR    artifact or checkpoint directory   (default artifacts/student-AB)
    JUDGE_RUNTIME      onnx-int8 | onnx-fp32 | torch-cuda | torch-cpu   (default onnx-int8)
    JUDGE_POLICY       teacher (match Claude) | gold (public-benchmark F1)  (default teacher)
    JUDGE_MAX_BATCH    micro-batch size cap (default 32; use 1 on CPU, see results/latency.md)
    JUDGE_MAX_WAIT_MS  micro-batch wait window (default 3)
    JUDGE_ESCALATE     optional second tier for uncertain verdicts, e.g. qwen3guard-0.6b
    JUDGE_UNCERTAIN_BAND  "lo,hi" probability band that counts as uncertain (default from serve_config);
                          results/latency.md lists the band for each escalation rate
"""
from __future__ import annotations

import asyncio
import os
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from llm_eval_forge.serve.batcher import MicroBatcher

MAX_CHARS = 100_000


class JudgeRequest(BaseModel):
    prompt: str = Field(max_length=MAX_CHARS, description="The user turn shown to the model")
    response: str = Field(max_length=MAX_CHARS, description="The model response to judge")


class JudgeResponse(BaseModel):
    verdict: Literal["harmful", "safe"]
    p_harmful: float
    threshold: float
    uncertain: bool = Field(description="p_harmful fell inside the configured uncertain band")
    escalated: bool = False
    escalated_to: Optional[str] = None
    latency_ms: float
    model_version: str
    runtime: str


class ToolCallRequest(BaseModel):
    user_goal: str
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    history: list[dict] = Field(default_factory=list)


class Metrics:
    def __init__(self, window: int = 10_000):
        self.requests = Counter()
        self.escalations = 0
        self.uncertain = 0
        self.latencies: deque = deque(maxlen=window)

    def observe(self, verdict: str, latency_ms: float, uncertain: bool, escalated: bool) -> None:
        self.requests[verdict] += 1
        self.latencies.append(latency_ms)
        self.uncertain += uncertain
        self.escalations += escalated

    def quantile(self, q: float) -> float:
        if not self.latencies:
            return 0.0
        xs = sorted(self.latencies)
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    def render(self, batch_mean: float, model_version: str, runtime: str) -> str:
        lines = [
            "# HELP judge_requests_total Judged requests by verdict.",
            "# TYPE judge_requests_total counter",
            *[f'judge_requests_total{{verdict="{v}"}} {n}' for v, n in sorted(self.requests.items())],
            "# HELP judge_request_latency_ms Server-side latency over the recent window.",
            "# TYPE judge_request_latency_ms summary",
            *[f'judge_request_latency_ms{{quantile="{q}"}} {self.quantile(q):.3f}' for q in (0.5, 0.95, 0.99)],
            f"judge_request_latency_ms_count {len(self.latencies)}",
            "# TYPE judge_uncertain_total counter",
            f"judge_uncertain_total {self.uncertain}",
            "# TYPE judge_escalations_total counter",
            f"judge_escalations_total {self.escalations}",
            "# TYPE judge_batch_size_mean gauge",
            f"judge_batch_size_mean {batch_mean:.3f}",
            f'judge_info{{model_version="{model_version}",runtime="{runtime}"}} 1',
        ]
        return "\n".join(lines) + "\n"


def _env_runtime():
    from llm_eval_forge.serve.runtime import load_runtime

    return load_runtime(os.environ.get("JUDGE_RUNTIME", "onnx-int8"),
                        os.environ.get("JUDGE_MODEL_DIR", "artifacts/student-AB"))


def _env_escalator():
    name = os.environ.get("JUDGE_ESCALATE", "").strip()
    if not name:
        return None
    from llm_eval_forge.serve.runtime import GuardRuntime

    return GuardRuntime(name)


def _band(cfg: dict) -> tuple[float, float]:
    raw = os.environ.get("JUDGE_UNCERTAIN_BAND", "").strip()
    if raw:
        lo, hi = (float(x) for x in raw.split(","))
        return lo, hi
    lo, hi = cfg.get("uncertain_band", [0.2, 0.8])
    return float(lo), float(hi)


def create_app(runtime=None, escalator=None, policy: str | None = None,
               max_batch: int | None = None, max_wait_ms: float | None = None) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        rt = runtime or _env_runtime()
        cfg = getattr(rt, "config", {}) or {}
        pol = policy or os.environ.get("JUDGE_POLICY", "teacher")
        state.update(
            runtime=rt,
            escalator=escalator if escalator is not None else _env_escalator(),
            threshold=float(cfg.get("thresholds", {}).get(pol, 0.5)),
            band=_band(cfg),
            batcher=MicroBatcher(rt, max_batch or int(os.environ.get("JUDGE_MAX_BATCH", 32)),
                                 max_wait_ms if max_wait_ms is not None else float(os.environ.get("JUDGE_MAX_WAIT_MS", 3))),
            escalation_lock=asyncio.Lock(),
            metrics=Metrics(),
        )
        rt.predict([("warm-up", "warm-up")])  # first call pays graph/kernel initialization
        state["batcher"].start()
        yield
        await state["batcher"].stop()

    app = FastAPI(title="Distilled guard judge", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict:
        rt = state["runtime"]
        return {"status": "ok", "model_version": rt.model_version, "runtime": rt.name,
                "threshold": state["threshold"], "escalation": getattr(state["escalator"], "name", None)}

    @app.post("/v1/judge", response_model=JudgeResponse)
    async def judge(req: JudgeRequest) -> JudgeResponse:
        t0 = time.perf_counter()
        p = await state["batcher"].submit(req.prompt, req.response)
        lo, hi = state["band"]
        uncertain = lo <= p < hi
        verdict_p, escalated, threshold = p, False, state["threshold"]
        esc = state["escalator"]
        if uncertain and esc is not None:
            async with state["escalation_lock"]:
                verdict_p = (await asyncio.get_running_loop().run_in_executor(
                    None, esc.predict, [(req.prompt, req.response)]))[0]
            escalated, threshold = True, 0.5
        verdict = "harmful" if verdict_p >= threshold else "safe"
        latency_ms = (time.perf_counter() - t0) * 1000
        state["metrics"].observe(verdict, latency_ms, uncertain, escalated)
        rt = state["runtime"]
        return JudgeResponse(verdict=verdict, p_harmful=round(verdict_p, 6), threshold=threshold,
                             uncertain=uncertain, escalated=escalated,
                             escalated_to=esc.name if escalated else None, latency_ms=round(latency_ms, 3),
                             model_version=rt.model_version, runtime=rt.name)

    @app.post("/v1/judge/tool_call", status_code=501)
    async def judge_tool_call(req: ToolCallRequest) -> dict:
        raise HTTPException(status_code=501, detail="Tool-call judging is not implemented yet.")

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        rt = state["runtime"]
        return state["metrics"].render(state["batcher"].stats.mean_batch_size, rt.model_version, rt.name)

    return app
