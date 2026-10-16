"""Latency and throughput benchmarks for the judge runtimes and the HTTP service.

    python -m llm_eval_forge.serve.bench runtimes      # in-process latency by runtime x input length
    python -m llm_eval_forge.serve.bench batch         # GPU batch-size throughput
    python -m llm_eval_forge.serve.bench http          # end-to-end HTTP latency under concurrency
    python -m llm_eval_forge.serve.bench quality       # CPU INT8 artifact on the full test suite
    python -m llm_eval_forge.serve.bench report        # -> results/latency.md

Results accumulate in results/latency.json. GPU numbers are only meaningful on an idle GPU;
every GPU section records the utilization seen just before it ran.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LATENCY_JSON = Path("results/latency.json")
LENGTHS = [128, 512, 1024, 2048]


def pct(values: list[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(ms: list[float]) -> dict:
    return {"n": len(ms), "p50": round(pct(ms, 0.5), 2), "p95": round(pct(ms, 0.95), 2),
            "p99": round(pct(ms, 0.99), 2), "mean": round(sum(ms) / len(ms), 2)}


def load_results() -> dict:
    return json.loads(LATENCY_JSON.read_text(encoding="utf-8")) if LATENCY_JSON.exists() else {}


def save_results(section: str, data) -> None:
    results = load_results()
    results[section] = data
    results["machine"] = machine_info()
    LATENCY_JSON.parent.mkdir(parents=True, exist_ok=True)
    LATENCY_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")


def machine_info() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count()}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        info["gpu"] = out
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def gpu_utilization() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def wait_for_idle_gpu(max_util: int = 10, timeout_s: float = 30.0) -> int | None:
    """nvidia-smi reports utilization over its last sample window, so a reading taken right after
    the previous benchmark still shows that benchmark. Wait until the GPU reads idle."""
    deadline = time.time() + timeout_s
    util = gpu_utilization()
    while util is not None and util > max_util and time.time() < deadline:
        time.sleep(1.0)
        util = gpu_utilization()
    return util


def synthetic_pair(tokenizer_len: int) -> tuple[str, str]:
    """A (prompt, response) pair of roughly `tokenizer_len` tokens (40% prompt / 60% response)."""
    words = max(1, tokenizer_len - 8)
    return " ".join(["word"] * int(words * 0.4)), " ".join(["word"] * (words - int(words * 0.4)))


def time_calls(fn, arg, n: int, warmup: int, sync=None) -> list[float]:
    for _ in range(warmup):
        fn(arg)
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn(arg)
        if sync:
            sync()
        out.append((time.perf_counter() - t) * 1000)
    return out


# --------------------------------------------------------------------------- in-process runtimes

RUNTIME_CONFIGS = [
    # (label, kind, model dir, device class)
    ("student-large-AB · torch bf16 · GPU", "torch-cuda", "models/student-large-AB", "gpu"),
    ("student-AB · torch bf16 · GPU", "torch-cuda", "models/student-AB", "gpu"),
    ("student-AB · ONNX INT8 · CPU", "onnx-int8", "artifacts/student-AB", "cpu"),
    ("student-AB · ONNX fp32 · CPU", "onnx-fp32", "artifacts/student-AB", "cpu"),
    ("student-large-AB · ONNX INT8 · CPU", "onnx-int8", "artifacts/student-large-AB", "cpu"),
]


def bench_runtimes(n_gpu: int = 200, n_cpu: int = 30) -> dict:
    from llm_eval_forge.serve.runtime import GuardRuntime, load_runtime

    results = {}
    for label, kind, model_dir, device in RUNTIME_CONFIGS:
        if not Path(model_dir).exists():
            print(f"skip {label}: {model_dir} missing")
            continue
        rt = load_runtime(kind, model_dir)
        sync = None
        if device == "gpu":
            import torch

            sync = torch.cuda.synchronize
        entry = {"kind": kind, "model_dir": model_dir, "device": device,
                 "gpu_util_before": wait_for_idle_gpu() if device == "gpu" else None, "by_length": {}}
        for length in LENGTHS:
            pair = [synthetic_pair(length)]
            actual = len(rt.encode(pair)[0])
            ms = time_calls(rt.predict, pair, n_gpu if device == "gpu" else n_cpu, 20 if device == "gpu" else 5, sync)
            entry["by_length"][str(length)] = {"tokens": actual, **summarize(ms)}
            print(f"{label:40s} {length:5d} tok: {entry['by_length'][str(length)]}", flush=True)
        results[label] = entry
        del rt
        _free_gpu()

    # Baseline guard: one forward pass for the first verdict token, and full verdict generation.
    guard = GuardRuntime("qwen3guard-0.6b")
    import torch

    entry = {"kind": "guard", "device": "gpu", "gpu_util_before": wait_for_idle_gpu(), "by_length": {},
             "generate_by_length": {}}
    for length in LENGTHS:
        pair = synthetic_pair(length)
        ms = time_calls(guard.predict, [pair], n_gpu // 4, 5, torch.cuda.synchronize)
        entry["by_length"][str(length)] = summarize(ms)
        gen = time_calls(lambda pr: guard.guard.decide_text(*pr, max_new_tokens=12), pair, 20, 3, torch.cuda.synchronize)
        entry["generate_by_length"][str(length)] = summarize(gen)
        print(f"{'Qwen3Guard-Gen-0.6B · bf16 · GPU':40s} {length:5d} tok: first-token {entry['by_length'][str(length)]}"
              f" | generate {entry['generate_by_length'][str(length)]}", flush=True)
    results["Qwen3Guard-Gen-0.6B · bf16 · GPU"] = entry
    save_results("runtimes", results)
    return results


def _free_gpu() -> None:
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except ImportError:
        pass


def bench_batch(sizes=(1, 8, 32), length: int = 512, n: int = 50) -> dict:
    import torch

    from llm_eval_forge.serve.runtime import load_runtime

    results = {}
    for model_dir in ("models/student-large-AB", "models/student-AB"):
        rt = load_runtime("torch-cuda", model_dir, tokens_per_batch=64 * 2048)
        entry = {"gpu_util_before": wait_for_idle_gpu(), "length": length}
        for bs in sizes:
            batch = [synthetic_pair(length)] * bs
            ms = time_calls(rt.predict, batch, n, 5, torch.cuda.synchronize)
            s = summarize(ms)
            s["items_per_s"] = round(bs / (s["p50"] / 1000), 1)
            entry[str(bs)] = s
            print(f"{model_dir:26s} bs={bs:3d} @ {length} tok: {s}", flush=True)
        results[Path(model_dir).name] = entry
        del rt
        _free_gpu()
    save_results("batch", results)
    return results


# --------------------------------------------------------------------------- HTTP service

HTTP_CONFIGS = [
    # (label, env, concurrency levels)
    ("GPU student-large-AB, micro-batching (32 / 3 ms)",
     {"JUDGE_RUNTIME": "torch-cuda", "JUDGE_MODEL_DIR": "models/student-large-AB", "JUDGE_MAX_BATCH": "32",
      "JUDGE_MAX_WAIT_MS": "3"}, (1, 8, 32)),
    ("GPU student-large-AB, no batching",
     {"JUDGE_RUNTIME": "torch-cuda", "JUDGE_MODEL_DIR": "models/student-large-AB", "JUDGE_MAX_BATCH": "1",
      "JUDGE_MAX_WAIT_MS": "0"}, (1, 8, 32)),
    ("CPU student-AB ONNX INT8, micro-batching (8 / 3 ms)",
     {"JUDGE_RUNTIME": "onnx-int8", "JUDGE_MODEL_DIR": "artifacts/student-AB", "JUDGE_MAX_BATCH": "8",
      "JUDGE_MAX_WAIT_MS": "3"}, (1, 8)),
    # On CPU one request already uses every core, and batching mixed-length requests pads them all
    # to the longest one, so batching lowers throughput; serve CPU requests one at a time.
    ("CPU student-AB ONNX INT8, no batching",
     {"JUDGE_RUNTIME": "onnx-int8", "JUDGE_MODEL_DIR": "artifacts/student-AB", "JUDGE_MAX_BATCH": "1",
      "JUDGE_MAX_WAIT_MS": "0"}, (1, 8)),
]


def http_requests(n: int = 400, seed: int = 0) -> list[dict]:
    """Real test-set (prompt, response) pairs, so request lengths follow the benchmark mix."""
    from llm_eval_forge.distill.data import read_split

    rows = read_split(Path("data"), "test", {"wildguardtest", "harmbench_val", "beavertails_test", "aegis2_test",
                                            "saferlhf_test"})
    random.Random(seed).shuffle(rows)
    return [{"prompt": r["prompt"], "response": r["response"]} for r in rows[:n]]


async def _load_test(url: str, payloads: list[dict], concurrency: int) -> dict:
    import httpx

    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors = 0

    async with httpx.AsyncClient(timeout=60) as client:
        async def one(p):
            nonlocal errors
            async with sem:
                t = time.perf_counter()
                r = await client.post(f"{url}/v1/judge", json=p)
                if r.status_code != 200:
                    errors += 1
                latencies.append((time.perf_counter() - t) * 1000)

        t0 = time.perf_counter()
        await asyncio.gather(*[one(p) for p in payloads])
        wall = time.perf_counter() - t0
        metrics = (await client.get(f"{url}/metrics")).text
    batch_mean = next((float(line.split()[-1]) for line in metrics.splitlines()
                       if line.startswith("judge_batch_size_mean")), None)
    return {**summarize(latencies), "throughput_rps": round(len(payloads) / wall, 1), "errors": errors,
            "mean_batch_size": batch_mean}


def bench_http(n: int = 400, port: int = 8123, only: str | None = None) -> dict:
    import httpx

    payloads = http_requests(n)
    results = load_results().get("http", {})  # re-running a subset keeps the other configs
    for label, env, levels in HTTP_CONFIGS:
        if only and only not in label:
            continue
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "llm_eval_forge.serve.app:create_app", "--factory",
             "--port", str(port), "--log-level", "warning"],
            env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        url = f"http://127.0.0.1:{port}"
        try:
            for _ in range(240):
                try:
                    if httpx.get(f"{url}/healthz", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited: {proc.stderr.read()[-2000:]}")
                time.sleep(1)
            asyncio.run(_load_test(url, payloads[:50], 8))  # warm-up
            entry = {"env": env, "gpu_util_before": wait_for_idle_gpu(), "by_concurrency": {}}
            for c in levels:
                entry["by_concurrency"][str(c)] = asyncio.run(_load_test(url, payloads, c))
                print(f"{label:55s} c={c:3d}: {entry['by_concurrency'][str(c)]}", flush=True)
            results[label] = entry
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    save_results("http", results)
    return results


# --------------------------------------------------------------------------- quality of the CPU artifact

def bench_quality(artifact: str = "artifacts/student-AB") -> dict:
    """Run the INT8 artifact through the full evaluation (same sets, same metrics), and measure
    decision agreement with the bf16 checkpoint at the serving threshold."""
    from llm_eval_forge.distill.evaluate import run
    from llm_eval_forge.serve.runtime import OnnxRuntime

    rt = OnnxRuntime(Path(artifact), "model.int8.onnx")
    name = f"{Path(artifact).name}-onnx-int8"
    metrics = run(name, lambda rows: rt.predict([(r["prompt"], r["response"]) for r in rows]),
                  Path("data"), True, "student")
    source = Path(artifact).name
    thr = rt.config["thresholds"]["teacher"]
    agree, total = 0, 0
    for path in sorted((Path("data/preds") / name).glob("*.jsonl")):
        ref_path = Path("data/preds") / source / path.name
        if not ref_path.exists():
            continue
        ref = {r["id"]: r["p"] for r in map(json.loads, ref_path.read_text(encoding="utf-8").splitlines())}
        for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
            if r["id"] in ref:
                agree += (r["p"] >= thr) == (ref[r["id"]] >= thr)
                total += 1
    src_metrics = json.loads((Path("results/metrics") / f"{source}.json").read_text(encoding="utf-8"))
    out = {"artifact": artifact, "threshold": thr, "rows_compared": total,
           "decision_agreement_vs_bf16": round(agree / total, 4) if total else None,
           "avg_response_f1": {"bf16": src_metrics["avg_response_f1"], "int8": metrics["avg_response_f1"]},
           "kappa_vs_teacher": {"bf16": src_metrics["sets"]["teacher_test"]["kappa"],
                                "int8": metrics["sets"]["teacher_test"]["kappa"]}}
    save_results("quality", out)
    print(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- report

def render_report(results: dict) -> str:
    lines = ["# Judge latency and throughput", "",
             f"Generated by `python -m llm_eval_forge.serve.bench report` on "
             f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}. Machine: {results.get('machine', {}).get('gpu', '?')}, "
             f"{results.get('machine', {}).get('cpu_count', '?')} logical CPUs, {results.get('machine', {}).get('platform', '')}.",
             "All latencies in milliseconds. GPU sections list the GPU utilization seen just before they ran "
             "(should be ~0 for clean numbers).", ""]
    rt = results.get("runtimes", {})
    gpu = rt.get("student-large-AB · torch bf16 · GPU", {}).get("by_length", {})
    cpu = rt.get("student-AB · ONNX INT8 · CPU", {}).get("by_length", {})
    if gpu and cpu:
        g99 = max(gpu[str(n)]["p99"] for n in (128, 512, 1024))
        c50 = cpu["512"]["p50"]
        lines += ["## Targets", "",
                  f"- GPU tier (student-large-AB, bf16): p99 at ≤1,024 tokens = **{g99:.1f} ms** (target < 50 ms) "
                  f"{'✅' if g99 < 50 else '❌'}",
                  f"- CPU tier (student-AB, INT8): p50 at 512 tokens = **{c50:.1f} ms** (target < 150 ms) "
                  f"{'✅' if c50 < 150 else '❌'}", ""]
    if rt:
        lines += ["## Single request, in process (batch size 1)", "",
                  "| Runtime | " + " | ".join(f"{n} tok p50 / p99" for n in LENGTHS) + " | GPU util before |",
                  "|---|" + "---:|" * (len(LENGTHS) + 1)]
        for label, e in rt.items():
            cells = [f"{e['by_length'][str(n)]['p50']:.1f} / {e['by_length'][str(n)]['p99']:.1f}"
                     if str(n) in e["by_length"] else "–" for n in LENGTHS]
            util = e.get("gpu_util_before")
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {util if e['device'] == 'gpu' else '–'}"
                         + ("%" if e["device"] == "gpu" and util is not None else "") + " |")
            if "generate_by_length" in e:
                cells = [f"{e['generate_by_length'][str(n)]['p50']:.1f} / {e['generate_by_length'][str(n)]['p99']:.1f}"
                         for n in LENGTHS]
                lines.append("| Qwen3Guard-Gen-0.6B · generate verdict (12 tokens) | " + " | ".join(cells) + " | |")
        large, guard = rt.get("student-large-AB · torch bf16 · GPU"), rt.get("Qwen3Guard-Gen-0.6B · bf16 · GPU")
        if large and guard:
            ratios = [guard["generate_by_length"][str(n)]["p50"] / large["by_length"][str(n)]["p50"] for n in LENGTHS]
            lines += ["", f"Qwen3Guard producing its verdict (generation) takes {min(ratios):.0f}–{max(ratios):.0f}× "
                          f"longer than student-large-AB at batch size 1."]
        lines.append("")
    b = results.get("batch", {})
    if b:
        lines += ["## GPU batch throughput (512-token inputs)", "", "| Model | Batch | p50 ms | items/s |",
                  "|---|---:|---:|---:|"]
        for model, e in b.items():
            for bs in ("1", "8", "32"):
                if bs in e:
                    lines.append(f"| {model} | {bs} | {e[bs]['p50']:.1f} | {e[bs]['items_per_s']:,.0f} |")
        lines.append("")
    h = results.get("http", {})
    if h:
        lines += ["## End-to-end HTTP (FastAPI + uvicorn, real benchmark requests)", "",
                  "Client and server run on the same machine. Requests are 400 real (prompt, response) pairs "
                  "from the test suite, so their lengths follow the benchmark mix.", "",
                  "| Service | Concurrency | p50 | p95 | p99 | Throughput (req/s) | Mean batch |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for label, e in h.items():
            for c, s in e["by_concurrency"].items():
                lines.append(f"| {label} | {c} | {s['p50']:.1f} | {s['p95']:.1f} | {s['p99']:.1f} | "
                             f"{s['throughput_rps']:,.1f} | {s['mean_batch_size'] or 0:.2f} |")
        lines.append("")
    q = results.get("quality")
    if q:
        lines += ["## CPU INT8 artifact quality", "",
                  f"`{q['artifact']}/model.int8.onnx`: dynamic INT8 with each MLP output projection kept in fp32. "
                  "Plain dynamic INT8 flipped ~11% of decisions.", "",
                  f"- Decision agreement with the bf16 checkpoint at the serving threshold ({q['threshold']}): "
                  f"**{100 * q['decision_agreement_vs_bf16']:.1f}%** over {q['rows_compared']:,} rows",
                  f"- Avg response F1: bf16 {100 * q['avg_response_f1']['bf16']:.1f} → INT8 {100 * q['avg_response_f1']['int8']:.1f}",
                  f"- κ vs Claude teacher: bf16 {q['kappa_vs_teacher']['bf16']:.3f} → INT8 {q['kappa_vs_teacher']['int8']:.3f}", ""]
    c = results.get("cascade")
    if c:
        lines += render_cascade(c)
    return "\n".join(lines) + "\n"


def render_cascade(c: dict) -> list[str]:
    lines = ["## Cascade: escalate uncertain verdicts to a larger judge", "",
             f"Tier 1 is `{c['tier1']}`. The least-confident fraction of requests (closest to the operating threshold "
             f"in logit space) is escalated to tier 2. Decisions are replayed from stored predictions; latency is "
             f"the measured tier-1 latency plus the measured tier-2 latency for escalated requests, at each request's "
             f"own length.",
             "",
             "At the Claude-matching threshold, public-benchmark F1 is lower than the gold-threshold numbers in "
             "`benchmark_table.md`: Claude labels fewer responses harmful than the public datasets do. Escalating "
             "to a guard trained on public labels therefore raises public F1, but erodes agreement with Claude once "
             "it decides a large share of traffic. The escalation tier should share the policy it serves.", ""]
    for tier2, rows in c["tiers"].items():
        has_band = any(r.get("band") for r in rows["points"])
        lines += [f"### Tier 2: {tier2}", "", rows["note"], "",
                  "| Escalated | " + " | ".join(rows["metric_names"]) + " | p50 ms | p99 ms |" +
                  (" `JUDGE_UNCERTAIN_BAND` |" if has_band else ""),
                  "|---:|" + "---:|" * (len(rows["metric_names"]) + 2) + ("---|" if has_band else "")]
        for r in rows["points"]:
            band = r.get("band")
            lines.append(f"| {100 * r['escalated']:.0f}% | " + " | ".join(r["metrics"]) +
                         f" | {r['p50_ms']:.1f} | {r['p99_ms']:.1f} |" +
                         ((f" `{band[0]},{band[1]}` |" if band else " – |") if has_band else ""))
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark judge runtimes and the HTTP service")
    parser.add_argument("cmd", choices=["runtimes", "batch", "http", "quality", "report", "all"])
    parser.add_argument("--only", default=None, help="http: run only configs whose label contains this text")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.cmd in ("runtimes", "all"):
        bench_runtimes()
    if args.cmd in ("batch", "all"):
        bench_batch()
    if args.cmd in ("http", "all"):
        bench_http(only=args.only)
    if args.cmd in ("quality", "all"):
        bench_quality()
    if args.cmd in ("report", "all"):
        text = render_report(load_results())
        Path("results/latency.md").write_text(text, encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
