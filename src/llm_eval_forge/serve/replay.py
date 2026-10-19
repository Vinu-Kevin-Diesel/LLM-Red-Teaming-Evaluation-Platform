"""Replay red-team attacks and safe controls through the guardrail gateway.

    python -m llm_eval_forge.serve.replay --upstream groq      # aligned production model (Groq)
    python -m llm_eval_forge.serve.replay --upstream ollama --ollama-model <name>   # local model
    python -m llm_eval_forge.serve.replay overhead             # same requests, recorded replies, GPU guards
    python -m llm_eval_forge.serve.replay report               # -> results/gateway.md

Request sets:
    adversarial    the harness's 14 seed attacks + their mutations (as `llm-eval-forge run` builds them)
    xstest_safe    250 safe prompts that sound dangerous (must NOT be blocked)
    xstest_unsafe  200 unsafe prompts

Per request the gateway reports where it was stopped (input / output) or that it passed.
Full responses stay in data/gateway_replay/ (gitignored); results/gateway.json holds counts only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from llm_eval_forge.serve.bench import load_results, pct, wait_for_idle_gpu

GATEWAY_JSON = Path("results/gateway.json")
REPLAY_DIR = Path("data/gateway_replay")
UPSTREAMS = {
    "groq": {"UPSTREAM_BASE_URL": "https://api.groq.com/openai/v1", "UPSTREAM_MODEL": "qwen/qwen3.8-27b"},
    "ollama": {"UPSTREAM_BASE_URL": "http://127.0.0.1:11434/v1", "UPSTREAM_API_KEY": "ollama"},
}


def adversarial_prompts(mutations_per_strategy: int = 2) -> list[dict]:
    import yaml

    from llm_eval_forge.redteam.engine import MutationEngine

    seeds = yaml.safe_load(Path("prompts/testcases/adversarial/injection_attempts.yaml").read_text(encoding="utf-8"))
    engine = MutationEngine()
    out = []
    for tc in seeds["test_cases"]:
        out.append({"id": tc["id"], "set": "adversarial", "prompt": tc["prompt"], "strategy": "seed"})
        for i, m in enumerate(engine.generate_attacks(tc["prompt"], tc.get("target_behavior", ""),
                                                      max_per_strategy=mutations_per_strategy)):
            out.append({"id": f"{tc['id']}_mut_{m.strategy}_{i}", "set": "adversarial", "prompt": m.mutated,
                        "strategy": m.strategy})
    return out


def xstest_prompts() -> list[dict]:
    from llm_eval_forge.distill.data import read_split

    rows = read_split(Path("data/prompts"), "test", {"xstest_prompts"})
    return [{"id": r["id"], "set": "xstest_unsafe" if r["label"] else "xstest_safe", "prompt": r["prompt"],
             "strategy": r["meta"].get("type")} for r in rows]


def load_done(out_path: Path) -> list[dict]:
    """Successful records from an earlier (interrupted) replay; failed requests are retried."""
    if not out_path.exists():
        return []
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("status") == 200]


async def _send_all(url: str, requests: list[dict], concurrency: int, max_tokens: int, out_path: Path,
                    resume: bool = False) -> list[dict]:
    import httpx

    sem = asyncio.Semaphore(concurrency)
    previous = load_done(out_path) if resume else []
    done_ids = {r["id"] for r in previous}
    requests = [r for r in requests if r["id"] not in done_ids]
    if resume:
        print(f"resuming: {len(previous)} requests already done, {len(requests)} to go", flush=True)
        # rewrite the file with only the successful records, then append
        out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in previous), encoding="utf-8")
    done: list[dict] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a" if resume else "w", encoding="utf-8") as sink:
        async with httpx.AsyncClient(timeout=180) as client:
            async def one(req):
                async with sem:
                    t = time.perf_counter()
                    r = await client.post(f"{url}/v1/chat/completions", json={
                        "messages": [{"role": "user", "content": req["prompt"]}],
                        "max_tokens": max_tokens, "temperature": 0})
                    rec = {**req, "status": r.status_code, "client_ms": round((time.perf_counter() - t) * 1000, 1)}
                    if r.status_code == 200:
                        body = r.json()
                        rec["guard"] = body["guard"]
                        rec["reply"] = body["choices"][0]["message"]["content"]
                        rec["finish_reason"] = body["choices"][0]["finish_reason"]
                    else:
                        rec["error"] = r.text[:300]
                    sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    sink.flush()
                    done.append(rec)
                    if len(done) % 50 == 0:
                        print(f"  {len(done)}/{len(requests)}", flush=True)

            await asyncio.gather(*[one(q) for q in requests])
    return previous + done


def outcome(rec: dict) -> str:
    if rec.get("status") != 200:
        return "error"
    g = rec["guard"]
    if g["input"]["action"] == "block":
        return "stopped_at_input"
    if g["output"]["action"] == "block":
        return "stopped_at_output"
    return "passed"


def load_review(safe_name: str, records: list[dict]) -> dict | None:
    """Counts from the manual review of replies that passed both guards; the labels themselves stay local.

    The review file holds one label list per request set. Passed replies that have no label are
    reported as `unreviewed`, so a replay that grew after the review cannot look fully reviewed."""
    path = REPLAY_DIR / f"{safe_name}_review.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    sets = {}
    for name, entry in r["sets"].items():
        labels = entry["labels"]
        labeled = {lab["id"] for lab in labels}
        passed = {rec["id"] for rec in records if rec["set"] == name and outcome(rec) == "passed"}
        sets[name] = {"reviewed": len(labels), "harmful": sum(lab["harmful"] for lab in labels),
                      "harmful_kinds": sorted({lab["kind"] for lab in labels if lab["harmful"]}),
                      "unreviewed": len(passed - labeled)}
        if any("followed_injection" in lab for lab in labels):
            sets[name]["followed_injection"] = sum(lab.get("followed_injection", False) for lab in labels)
    return {"labeler": r["labeler"], "sets": sets}


def summarize(records: list[dict]) -> dict:
    by_set: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        by_set[r["set"]][outcome(r)] += 1
    ok = [r for r in records if r.get("status") == 200]
    overhead = [r["guard"]["guard_overhead_ms"] for r in ok]
    upstream = [r["guard"]["upstream_latency_ms"] for r in ok if r["guard"]["upstream_latency_ms"]]
    return {
        "by_set": {s: dict(c) | {"n": sum(c.values())} for s, c in by_set.items()},
        "guard_overhead_ms": {"p50": round(pct(overhead, 0.5), 1), "p95": round(pct(overhead, 0.95), 1),
                              "p99": round(pct(overhead, 0.99), 1)} if overhead else None,
        "upstream_latency_ms": {"p50": round(pct(upstream, 0.5), 1), "p95": round(pct(upstream, 0.95), 1),
                                "p99": round(pct(upstream, 0.99), 1)} if upstream else None,
        "errors": sum(1 for r in records if r.get("status") != 200),
    }


class RecordedUpstream:
    """Answers each prompt with the reply recorded by an earlier replay, so the guards can be timed on
    another runtime without calling the provider again (and without its latency or rate limits)."""

    name, default_model = "recorded replies", "recorded"

    def __init__(self, records: list[dict]):
        self.replies = {r["prompt"]: r.get("reply") or "" for r in records}

    async def chat(self, payload: dict) -> dict:
        from llm_eval_forge.serve.gateway import last_user_text

        reply = self.replies.get(last_user_text(payload["messages"]), "")
        return {"id": "recorded", "object": "chat.completion", "created": 0, "model": self.default_model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}]}


def measure_overhead(records: list[dict], input_side=None, output_side=None, warmup: int = 10) -> dict:
    """Send the recorded requests one at a time through an in-process gateway (guards from the
    GATEWAY_* environment, torch-cuda by default) and compare its decisions with the recorded ones."""
    from fastapi.testclient import TestClient

    from llm_eval_forge.serve.gateway import create_gateway

    app = create_gateway(input_side, output_side, RecordedUpstream(records))
    seen = []
    with TestClient(app) as client:
        health = client.get("/healthz").json()
        for i, rec in enumerate(records[:warmup] + records):
            body = client.post("/v1/chat/completions",
                               json={"messages": [{"role": "user", "content": rec["prompt"]}]}).json()
            if i >= warmup:
                seen.append((rec, body["guard"]))
    overhead = [g["guard_overhead_ms"] for _, g in seen]
    both = [g["guard_overhead_ms"] for _, g in seen if g["output"]["action"] != "skipped"]
    in_agree = sum((g["input"]["action"] == "block") == (rec["guard"]["input"]["action"] == "block") for rec, g in seen)
    # Output decisions are comparable only where the recorded reply is the model's own (it passed).
    out_pairs = [(rec, g) for rec, g in seen if outcome(rec) == "passed" and g["output"]["action"] != "skipped"]
    out_agree = sum(g["output"]["action"] != "block" for _, g in out_pairs)
    q = lambda xs: {"p50": round(pct(xs, 0.5), 1), "p95": round(pct(xs, 0.95), 1), "p99": round(pct(xs, 0.99), 1)}
    device = None
    try:
        import torch

        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return {"input_runtime": health["input"].get("runtime"), "output_runtime": health["output"].get("runtime"),
            "device": device, "n": len(seen), "guard_overhead_ms": q(overhead) if overhead else None,
            "both_sides_ms": q(both) if both else None,
            "input_agreement": [in_agree, len(seen)], "output_agreement": [out_agree, len(out_pairs)]}


def run_overhead(label: str) -> dict:
    results = json.loads(GATEWAY_JSON.read_text(encoding="utf-8"))
    safe_name = label.replace(":", "_").replace("/", "_")
    records = load_done(REPLAY_DIR / f"{safe_name}.jsonl")
    wait_for_idle_gpu()
    run = measure_overhead(records)
    key = f"{run['input_runtime']}+{run['output_runtime']}"
    results[label].setdefault("overhead_runs", {})[key] = run
    GATEWAY_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(run, indent=2))
    return run


def run_replay(upstream: str, ollama_model: str | None, concurrency: int, max_tokens: int, port: int,
               sets: list[str], resume: bool = False) -> dict:
    import httpx

    env = {**os.environ, **UPSTREAMS[upstream]}
    if upstream == "ollama":
        if not ollama_model:
            raise SystemExit("--ollama-model is required for the ollama upstream")
        env["UPSTREAM_MODEL"] = ollama_model
    requests = [r for r in adversarial_prompts() + xstest_prompts() if r["set"] in sets]
    label = upstream if upstream == "groq" else f"ollama:{ollama_model}"
    wait_for_idle_gpu()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "llm_eval_forge.serve.gateway:create_gateway",
                             "--factory", "--port", str(port), "--log-level", "warning"],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(300):
            try:
                if httpx.get(f"{url}/healthz", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if proc.poll() is not None:
                raise RuntimeError(f"gateway exited: {proc.stderr.read()[-2000:]}")
            time.sleep(1)
        health = httpx.get(f"{url}/healthz", timeout=5).json()
        print(f"gateway up: {health}")
        safe_name = label.replace(":", "_").replace("/", "_")
        records = asyncio.run(_send_all(url, requests, concurrency, max_tokens, REPLAY_DIR / f"{safe_name}.jsonl",
                                        resume=resume))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    summary = summarize(records) | {"upstream": label, "max_tokens": max_tokens,
                                    "review": load_review(safe_name, records), "guards": {
        side: {k: health[side].get(k) for k in ("mode", "threshold", "model_version", "runtime")}
        for side in ("input", "output")}}
    results = json.loads(GATEWAY_JSON.read_text(encoding="utf-8")) if GATEWAY_JSON.exists() else {}
    results[label] = summary
    GATEWAY_JSON.parent.mkdir(parents=True, exist_ok=True)
    GATEWAY_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def resummarize(label: str) -> dict:
    """Rebuild one upstream's summary from its saved records and review file (no requests sent)."""
    results = json.loads(GATEWAY_JSON.read_text(encoding="utf-8")) if GATEWAY_JSON.exists() else {}
    safe_name = label.replace(":", "_").replace("/", "_")
    records = load_done(REPLAY_DIR / f"{safe_name}.jsonl")
    previous = results.get(label, {})
    results[label] = summarize(records) | {"upstream": label, "max_tokens": previous.get("max_tokens"),
                                           "review": load_review(safe_name, records), "guards": previous.get("guards"),
                                           "overhead_runs": previous.get("overhead_runs", {})}
    GATEWAY_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results[label]


SET_LABELS = {"adversarial": "Red-team attacks (14 seeds + mutations)", "xstest_unsafe": "XSTest unsafe prompts",
              "xstest_safe": "XSTest safe prompts (should pass)"}


def _runtime(guard: dict) -> str:
    return f"`{guard['runtime']}`, " if guard.get("runtime") else ""


def render(results: dict) -> str:
    lat = load_results().get("runtimes", {})
    lines = ["# Guardrail gateway", "",
             "Requests go through `POST /v1/chat/completions` on the gateway: input guard → upstream model → output "
             "guard. *Stopped at input* never reaches the model; *stopped at output* means the model answered and the "
             "reply was replaced. For safe prompts, any stop is a false block. Generated by "
             "`python -m llm_eval_forge.serve.replay report`; full replies stay local.", ""]
    for label, s in results.items():
        g = s["guards"]
        lines += [f"## Upstream: `{label}`", "",
                  f"Input guard `{g['input']['model_version']}` ({_runtime(g['input'])}threshold "
                  f"{g['input']['threshold']}), output guard `{g['output']['model_version']}` "
                  f"({_runtime(g['output'])}threshold {g['output']['threshold']}), upstream max_tokens "
                  f"{s['max_tokens']}.", "",
                  "| Requests | n | Stopped at input | Stopped at output | Passed |", "|---|---:|---:|---:|---:|"]
        for set_name in ("adversarial", "xstest_unsafe", "xstest_safe"):
            c = s["by_set"].get(set_name)
            if not c:
                continue
            n = c["n"]
            cells = [f"{100 * c.get(k, 0) / n:.1f}% ({c.get(k, 0)})" for k in ("stopped_at_input", "stopped_at_output", "passed")]
            lines.append(f"| {SET_LABELS[set_name]} | {n} | " + " | ".join(cells) + " |")
        rv = s.get("review")
        if rv:
            lines += ["", f"**Manual review** of the replies that passed both guards ({rv['labeler']}):", ""]
            for set_name, c in rv["sets"].items():
                line = f"- {SET_LABELS[set_name]}: {c['reviewed']} reviewed, **{c['harmful']} harmful**"
                if c["harmful_kinds"]:
                    line += f" ({'; '.join(c['harmful_kinds'])})"
                if c.get("followed_injection"):
                    line += (f"; {c['followed_injection']} followed a harmless injected instruction "
                             f"(e.g. wrote a poem instead of translating)")
                if c["unreviewed"]:
                    line += f"; {c['unreviewed']} passed replies not reviewed yet"
                lines.append(line + ".")
        oh, up = s["guard_overhead_ms"], s["upstream_latency_ms"]
        lines += ["", f"Guard overhead (input + output checks, CPU guards above): p50 **{oh['p50']} ms**, "
                      f"p95 {oh['p95']} ms, p99 **{oh['p99']} ms**." if oh else "",
                  f"Upstream model latency for comparison: p50 {up['p50']} ms, p99 {up['p99']} ms." if up else "",
                  f"Errors: {s['errors']}."]
        for o in (s.get("overhead_runs") or {}).values():
            ia, oa = o["input_agreement"], o["output_agreement"]
            lines += ["", f"Same {o['n']} requests through `{o['input_runtime']}` / `{o['output_runtime']}` guards"
                          f"{' on ' + o['device'] if o.get('device') else ''} (recorded replies, sequential, no provider "
                          f"calls): overhead p50 **{o['guard_overhead_ms']['p50']} ms**, p99 "
                          f"**{o['guard_overhead_ms']['p99']} ms**; with both sides checked p50 "
                          f"{o['both_sides_ms']['p50']} ms, p99 {o['both_sides_ms']['p99']} ms. Decisions agree with "
                          f"the CPU guards on {ia[0]}/{ia[1]} prompts and {oa[0]}/{oa[1]} replies."]
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay attacks and safe controls through the gateway")
    parser.add_argument("cmd", nargs="?", default="run", choices=["run", "summarize", "overhead", "report"])
    parser.add_argument("--upstream", choices=sorted(UPSTREAMS), default="groq")
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--port", type=int, default=8130)
    parser.add_argument("--sets", default="adversarial,xstest_safe,xstest_unsafe")
    parser.add_argument("--resume", action="store_true", help="Skip requests already answered in an earlier run")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv

    load_dotenv()
    label = args.upstream if args.upstream == "groq" else f"ollama:{args.ollama_model}"
    if args.cmd == "summarize":
        resummarize(label)
    if args.cmd == "overhead":
        run_overhead(label)
    if args.cmd == "run":
        run_replay(args.upstream, args.ollama_model, args.concurrency, args.max_tokens, args.port, args.sets.split(","),
                   resume=args.resume)
    results = json.loads(GATEWAY_JSON.read_text(encoding="utf-8")) if GATEWAY_JSON.exists() else {}
    text = render(results)
    Path("results/gateway.md").write_text(text, encoding="utf-8")
    if args.cmd == "report":
        print(text)


if __name__ == "__main__":
    main()
