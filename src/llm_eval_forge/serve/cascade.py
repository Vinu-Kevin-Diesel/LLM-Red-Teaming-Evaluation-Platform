"""Cascade simulation: escalate the least-confident tier-1 verdicts to a larger judge.

Tier-1 and tier-2 decisions are replayed from stored predictions (data/preds/*), so a sweep
costs nothing. Latency per request = measured tier-1 latency at that request's token length,
plus measured tier-2 latency if it was escalated (results/latency.json).

    python -m llm_eval_forge.serve.cascade      # adds a `cascade` section to results/latency.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from llm_eval_forge.distill.evaluate import RESPONSE_BENCHMARKS
from llm_eval_forge.distill.teacher import binary_metrics
from llm_eval_forge.serve.bench import LATENCY_JSON, load_results, pct, save_results

FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
CLAUDE_LATENCY_MS = 6000.0  # live claude-sonnet-5-5 judge calls measured on Day 1: 5.6 s and 6.4 s


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def escalate_mask(probs: list[float], threshold: float, fraction: float) -> list[bool]:
    """Mark the `fraction` of items whose tier-1 score is closest to the threshold (logit space)."""
    k = round(len(probs) * fraction)
    if k <= 0:
        return [False] * len(probs)
    t = logit(threshold)
    order = sorted(range(len(probs)), key=lambda i: abs(logit(probs[i]) - t))
    chosen = set(order[:k])
    return [i in chosen for i in range(len(probs))]


def band_for_fraction(probs: list[float], threshold: float, fraction: float) -> tuple[float, float] | None:
    """The probability band [lo, hi) that escalates `fraction` of these scores (symmetric in logit space)."""
    k = round(len(probs) * fraction)
    if k <= 0:
        return None
    t = logit(threshold)
    c = sorted(abs(logit(p) - t) for p in probs)[k - 1]
    sig = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731
    return round(sig(t - c), 4), round(sig(t + c), 4)


def cascade_decisions(p1: list[float], thr1: float, p2: list[float], thr2: float, mask: list[bool]) -> list[int]:
    return [int((b if m else a) >= (thr2 if m else thr1)) for a, b, m in zip(p1, p2, mask)]


def interp_latency(by_length: dict, tokens: int) -> float:
    """Piecewise-linear p50 latency at a token length (clamped to the measured range)."""
    pts = sorted((int(k), v["p50"]) for k, v in by_length.items())
    if tokens <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if tokens <= x1:
            return y0 + (y1 - y0) * (tokens - x0) / (x1 - x0)
    return pts[-1][1]


def _preds(name: str, set_name: str) -> dict[str, dict]:
    path = Path("data/preds") / name / f"{set_name}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def _token_lengths(artifact: str, rows_by_id: dict[str, dict]) -> dict[str, int]:
    from tokenizers import Tokenizer

    from llm_eval_forge.distill.train import format_pair

    tok = Tokenizer.from_file(str(Path(artifact) / "tokenizer.json"))
    tok.enable_truncation(max_length=2048, strategy="longest_first")
    ids = list(rows_by_id)
    encs = tok.encode_batch([format_pair(rows_by_id[i]["prompt"], rows_by_id[i]["response"]) for i in ids])
    return {i: len(e.ids) for i, e in zip(ids, encs)}


def simulate(tier1: str = "student-large-AB", tier1_runtime: str = "student-large-AB · torch bf16 · GPU",
             artifact: str = "artifacts/student-large-AB") -> dict:
    from llm_eval_forge.distill.data import read_split
    from llm_eval_forge.distill.evaluate import eval_sets, pilot_subset

    latency = load_results().get("runtimes", {})
    t1_lat = latency[tier1_runtime]["by_length"]
    q_lat = latency["Qwen3Guard-Gen-0.6B · bf16 · GPU"]["by_length"]
    m1 = json.loads((Path("results/metrics") / f"{tier1}.json").read_text(encoding="utf-8"))
    thr = m1["teacher_threshold"]  # production policy: match Claude
    sets = eval_sets(Path("data"))
    rows_by_id = {r["id"]: r for name in RESPONSE_BENCHMARKS + ["teacher_test"] for r in sets.get(name, [])}
    lengths = _token_lengths(artifact, rows_by_id)

    # Tier 2 = Qwen3Guard-0.6B on every benchmark (escalation ranked over the pooled requests)
    pooled = []  # (set, id, label, p1, p2)
    for name in [b for b in RESPONSE_BENCHMARKS if b in sets] + ["teacher_test"]:
        a, b = _preds(tier1, name), _preds("qwen3guard-0.6b", name)
        pooled += [(name, i, a[i]["label"], a[i]["p"], b[i]["p"]) for i in a if i in b]
    points = []
    for f in FRACTIONS:
        mask = escalate_mask([x[3] for x in pooled], thr, f)
        dec = cascade_decisions([x[3] for x in pooled], thr, [x[4] for x in pooled], 0.5, mask)
        f1s = []
        for name in [b for b in RESPONSE_BENCHMARKS if b in sets]:
            idx = [k for k, x in enumerate(pooled) if x[0] == name]
            f1s.append(binary_metrics([pooled[k][2] for k in idx], [dec[k] for k in idx])["f1"])
        tt = [k for k, x in enumerate(pooled) if x[0] == "teacher_test"]
        kappa = binary_metrics([pooled[k][2] for k in tt], [dec[k] for k in tt])["kappa"]
        lat = [interp_latency(t1_lat, lengths[x[1]]) + (interp_latency(q_lat, lengths[x[1]]) if m else 0.0)
               for x, m in zip(pooled, mask)]
        points.append({"escalated": f, "metrics": [f"{100 * sum(f1s) / len(f1s):.1f}", f"{kappa:.3f}"],
                       "p50_ms": pct(lat, 0.5), "p99_ms": pct(lat, 0.99),
                       "band": band_for_fraction([x[3] for x in pooled], thr, f)})

    # Tier 2 = Claude (the policy owner), only where Claude verdicts exist: the human-gold pilot rows
    pilot = pilot_subset(Path("data"))
    test = {r["id"]: r for r in read_split(Path("data"), "test", {"harmbench_val", "wildguardtest"})}
    t1 = {**_preds(tier1, "harmbench_val"), **_preds(tier1, "wildguardtest")}
    rows = [(i, test[i]["label"], t1[i]["p"], 1.0 if pilot[i]["verdict"]["response_harmful"] else 0.0)
            for i in pilot if i in t1 and i in test]
    claude_points = []
    for f in FRACTIONS:
        mask = escalate_mask([r[2] for r in rows], thr, f)
        dec = cascade_decisions([r[2] for r in rows], thr, [r[3] for r in rows], 0.5, mask)
        m = binary_metrics([r[1] for r in rows], dec)
        lat = [interp_latency(t1_lat, lengths.get(r[0], 512)) + (CLAUDE_LATENCY_MS if e else 0.0)
               for r, e in zip(rows, mask)]
        claude_points.append({"escalated": f, "metrics": [f"{100 * m['f1']:.1f}", f"{m['kappa']:.3f}"],
                              "p50_ms": pct(lat, 0.5), "p99_ms": pct(lat, 0.99)})

    out = {
        "tier1": f"{tier1} (threshold {thr}, the Claude-matching operating point)",
        "tiers": {
            "Qwen3Guard-Gen-0.6B (local GPU, first-token scoring)": {
                "note": f"All {len([x for x in pooled if x[0] != 'teacher_test']):,} benchmark rows plus "
                        f"{len([x for x in pooled if x[0] == 'teacher_test'])} teacher-holdout rows, ranked together.",
                "metric_names": ["Avg response F1", "κ vs Claude"], "points": points},
            "Claude Sonnet 5.5 (the teacher; API)": {
                "note": f"Only the {len(rows)} human-labeled pilot rows (HarmBench + WildGuardTest) have Claude verdicts. "
                        f"Metrics are against human labels. Claude latency is {CLAUDE_LATENCY_MS / 1000:.0f} s per call "
                        f"(live calls measured on Day 1).",
                "metric_names": ["F1 vs human", "κ vs human"], "points": claude_points},
        },
    }
    save_results("cascade", out)
    return out


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = simulate()
    for tier, data in out["tiers"].items():
        print(tier)
        for p in data["points"]:
            print(f"  esc {100 * p['escalated']:4.0f}%  {data['metric_names']} {p['metrics']}  "
                  f"p50 {p['p50_ms']:.1f} ms  p99 {p['p99_ms']:.1f} ms")
    print(f"saved to {LATENCY_JSON}")


if __name__ == "__main__":
    main()
