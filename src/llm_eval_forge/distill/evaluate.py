"""Evaluate judges (student checkpoints and baseline guards) on the response-harm suite.

    python -m llm_eval_forge.distill.evaluate run --model models/student-A --name student-A
    python -m llm_eval_forge.distill.evaluate run --baseline qwen3guard-0.6b
    python -m llm_eval_forge.distill.evaluate teacher        # Claude teacher on its pilot rows
    python -m llm_eval_forge.distill.evaluate table          # -> results/benchmark_table.md

Students use the decision threshold that maximizes F1 on the gold validation set; baseline
guards use their native decision (P(unsafe) >= 0.5). AUROC is threshold-free.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional

from llm_eval_forge.distill.data import read_split, write_jsonl
from llm_eval_forge.distill.metrics import apply_temperature, auroc, best_f1_threshold, ece, fit_temperature
from llm_eval_forge.distill.teacher import binary_metrics
from llm_eval_forge.distill.train import GOLD_VAL, soft_target, teacher_holdout_rows

# Standard response-harmfulness benchmarks (averaged); jbb_judge measures "jailbroken" instead.
RESPONSE_BENCHMARKS = ["wildguardtest", "harmbench_val", "beavertails_test", "saferlhf_test", "aegis2_test", "xstest_resp"]
OTHER_SETS = ["jbb_judge", "teacher_test"]
METRICS_DIR = Path("results/metrics")

# Qwen3Guard tech report (arXiv 2510.14276), Table 3, response-harmfulness F1 (strict mode for Qwen3Guard).
PUBLISHED = {
    "Qwen3Guard-Gen-0.6B (published)": {"harmbench_val": 85.0, "saferlhf_test": 66.6, "beavertails_test": 86.1,
                                        "xstest_resp": 89.7, "aegis2_test": 84.2, "wildguardtest": 76.3},
    "WildGuard-7B (published)": {"harmbench_val": 86.3, "saferlhf_test": 64.2, "beavertails_test": 84.4,
                                 "xstest_resp": 94.7, "aegis2_test": 83.2, "wildguardtest": 75.4},
    "Llama-Guard-3-8B (published)": {"harmbench_val": 84.5, "saferlhf_test": 45.2, "beavertails_test": 67.9,
                                     "xstest_resp": 89.8, "aegis2_test": 66.1, "wildguardtest": 69.5},
}
SHORT = {"wildguardtest": "WGTest", "harmbench_val": "HarmB", "beavertails_test": "BeaverT",
         "saferlhf_test": "SafeRLHF", "aegis2_test": "Aegis2", "xstest_resp": "XSTest",
         "jbb_judge": "JBB (jailbroken)", "teacher_test": "κ vs teacher"}


# --------------------------------------------------------------------------- eval sets

def eval_sets(data_dir: Path) -> dict[str, list[dict]]:
    """Test benchmarks, the gold validation set, and the teacher-holdout set (label = teacher verdict)."""
    test = read_split(data_dir, "test")
    sets: dict[str, list[dict]] = {}
    for r in test:
        if r["label"] is not None:
            sets.setdefault(r["source"], []).append(r)
    sets["val"] = [r for r in read_split(data_dir, "val", GOLD_VAL) if r["label"] is not None]
    sets["teacher_val"] = teacher_holdout_rows(data_dir, "val")
    sets["teacher_test"] = teacher_holdout_rows(data_dir, "test")
    return sets


def best_kappa_threshold(labels: list[int], probs: list[float]) -> float:
    """Threshold that best matches the teacher's decisions (max Cohen's kappa)."""
    return max((t / 100 for t in range(5, 96)),
               key=lambda t: binary_metrics(labels, [int(p >= t) for p in probs])["kappa"])


def pilot_subset(data_dir: Path) -> dict[str, dict]:
    """Rows the Claude teacher labeled in the pilot (HarmBench + WildGuardTest), keyed by id."""
    labels_path = data_dir / "teacher" / "labels" / "pilot.jsonl"
    if not labels_path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, labels_path.read_text(encoding="utf-8").splitlines())
            if r["status"] == "ok"}


# --------------------------------------------------------------------------- metrics

def set_metrics(labels: list[int], probs: list[float], threshold: float, temperature: float) -> dict:
    preds = [int(p >= threshold) for p in probs]
    m = binary_metrics(labels, preds)
    m.update({
        "auroc": auroc(labels, probs),
        "ece": ece(labels, probs),
        "ece_scaled": ece(labels, apply_temperature(probs, temperature)),
        "positives": sum(labels),
    })
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


# --------------------------------------------------------------------------- predictors

def student_predictor(path: str) -> Callable[[list[dict]], list[float]]:
    from llm_eval_forge.distill.train import load_model, predict_proba

    tok, model, torch = load_model(path)
    return lambda rows: predict_proba(model, tok, torch, [(r["prompt"], r["response"], 0) for r in rows])


def baseline_predictor(name: str) -> Callable[[list[dict]], list[float]]:
    from llm_eval_forge.distill import baselines

    return baselines.BASELINES[name]()


def run(name: str, predictor: Callable[[list[dict]], list[float]], data_dir: Path, tune_threshold: bool,
        kind: str) -> dict:
    sets = eval_sets(data_dir)
    pilot = pilot_subset(data_dir)
    preds_dir = data_dir / "preds" / name
    probs_by_set, seconds = {}, {}
    for set_name, rows in sets.items():
        t0 = time.time()
        probs_by_set[set_name] = predictor(rows)
        seconds[set_name] = round(time.time() - t0, 1)
        write_jsonl(preds_dir / f"{set_name}.jsonl",
                    ({"id": r["id"], "label": r["label"], "p": p} for r, p in zip(rows, probs_by_set[set_name])))
        print(f"{name}: {set_name} ({len(rows)} rows) in {seconds[set_name]}s", flush=True)

    val_labels = [r["label"] for r in sets["val"]]
    threshold = best_f1_threshold(val_labels, probs_by_set["val"])[0] if tune_threshold else 0.5
    temperature = fit_temperature(val_labels, probs_by_set["val"])
    # Two operating points: the gold-tuned threshold for public benchmarks, and a teacher-tuned
    # threshold (max kappa on teacher_val) for agreement with Claude, the production policy.
    teacher_threshold = (best_kappa_threshold([r["label"] for r in sets["teacher_val"]], probs_by_set["teacher_val"])
                         if tune_threshold and sets["teacher_val"] else threshold)
    results = {"name": name, "kind": kind, "threshold": threshold, "teacher_threshold": teacher_threshold,
               "temperature": round(temperature, 2), "sets": {}, "pilot_subset": {}}
    for set_name, rows in sets.items():
        labels = [r["label"] for r in rows]
        thr = teacher_threshold if set_name.startswith("teacher_") else threshold
        results["sets"][set_name] = set_metrics(labels, probs_by_set[set_name], thr, temperature)
        sub = [(r["label"], p) for r, p in zip(rows, probs_by_set[set_name]) if r["id"] in pilot]
        if sub:
            results["pilot_subset"][set_name] = set_metrics([y for y, _ in sub], [p for _, p in sub],
                                                            threshold, temperature)
    present = [b for b in RESPONSE_BENCHMARKS if b in results["sets"]]
    results["avg_response_f1"] = round(sum(results["sets"][b]["f1"] for b in present) / len(present), 4)
    results["avg_over"] = present
    results["seconds"] = seconds
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"{name}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{name}: avg response F1 {results['avg_response_f1']:.3f} over {present}; threshold {threshold}")
    return results


def run_teacher(data_dir: Path) -> dict:
    """Claude teacher, scored only on the pilot rows it labeled (HarmBench + WildGuardTest)."""
    pilot = pilot_subset(data_dir)
    sets = eval_sets(data_dir)
    results = {"name": "claude-sonnet-5-5-teacher", "kind": "teacher", "threshold": 0.5, "temperature": 1.0,
               "sets": {}, "pilot_subset": {}}
    for set_name in ("harmbench_val", "wildguardtest"):
        rows = [r for r in sets[set_name] if r["id"] in pilot]
        probs = [soft_target(pilot[r["id"]]["verdict"]) for r in rows]
        results["pilot_subset"][set_name] = set_metrics([r["label"] for r in rows], probs, 0.5, 1.0)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "claude-sonnet-5-5-teacher.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


# --------------------------------------------------------------------------- table

def _pct(v: Optional[float]) -> str:
    return "–" if v is None else f"{100 * v:.1f}"


def render_table(metrics: list[dict]) -> str:
    present = [b for b in RESPONSE_BENCHMARKS if any(b in m["sets"] for m in metrics)]
    rows = [m for m in metrics if m["kind"] != "teacher"]
    rows.sort(key=lambda m: -m.get("avg_response_f1", 0))
    head = ["Judge", *[SHORT[b] for b in present], "Avg F1", "JBB (jailbroken)", "κ vs teacher", "Avg AUROC", "Avg ECE (scaled)"]
    lines = [
        "# Response-harm benchmark table",
        "",
        "F1 (%) with harmful = positive. Students use the threshold tuned on the gold validation set; "
        "guards use their native decision. *κ vs teacher* is Cohen's kappa against Claude's verdicts on held-out "
        "pool rows, at a threshold tuned on a separate teacher-labeled split. Generated by "
        "`python -m llm_eval_forge.distill.evaluate table`.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|",
    ]
    for m in rows:
        s = m["sets"]
        aurocs = [s[b]["auroc"] for b in present if b in s and s[b]["auroc"] is not None]
        eces = [s[b]["ece_scaled"] for b in present if b in s]
        cells = [f"**{m['name']}**" if m["kind"] == "student" else m["name"]]
        cells += [_pct(s[b]["f1"]) if b in s else "–" for b in present]
        cells += [_pct(m["avg_response_f1"]), _pct(s.get("jbb_judge", {}).get("f1")),
                  f"{s['teacher_test']['kappa']:.3f}" if "teacher_test" in s else "–",
                  _pct(sum(aurocs) / len(aurocs)) if aurocs else "–",
                  f"{sum(eces) / len(eces):.3f}" if eces else "–"]
        lines.append("| " + " | ".join(cells) + " |")
    for name, vals in PUBLISHED.items():
        cells = [name] + [f"{vals[b]:.1f}" if b in vals else "–" for b in present]
        avg = [vals[b] for b in present if b in vals]
        cells += [f"{sum(avg) / len(avg):.1f}", "–", "–", "–", "–"]
        lines.append("| " + " | ".join(cells) + " |")

    # Apples-to-apples with the Claude teacher on the rows it labeled
    sub_sets = ["harmbench_val", "wildguardtest"]
    lines += ["", "## Same rows as the Claude teacher pilot", "",
              "Claude only labeled a 300-row human-gold pilot (balanced), so every judge is re-scored on exactly "
              "those rows.", "",
              "| Judge | " + " | ".join(f"{SHORT[b]} F1" for b in sub_sets) + " | " +
              " | ".join(f"{SHORT[b]} κ" for b in sub_sets) + " |",
              "|---|" + "---:|" * (2 * len(sub_sets))]
    for m in sorted(metrics, key=lambda m: m["kind"] != "teacher"):
        ps = m.get("pilot_subset", {})
        if not ps:
            continue
        cells = [m["name"]] + [_pct(ps[b]["f1"]) if b in ps else "–" for b in sub_sets]
        cells += [f"{ps[b]['kappa']:.3f}" if b in ps else "–" for b in sub_sets]
        lines.append("| " + " | ".join(cells) + " |")

    n = {b: next((m["sets"][b]["n"] for m in rows if b in m["sets"]), None) for b in present + OTHER_SETS}
    lines += ["", "Rows per set: " + ", ".join(f"{SHORT.get(b, b)} {v:,}" for b, v in n.items() if v), ""]
    if "xstest_resp" not in present:
        lines.append("XSTest-Resp is gated (AI2 terms) and not yet included; published averages above are over the "
                     "same benchmarks as ours.")
    return "\n".join(lines) + "\n"


def write_table(out: Path = Path("results/benchmark_table.md")) -> str:
    metrics = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(METRICS_DIR.glob("*.json"))]
    text = render_table(metrics)
    out.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate judges on the response-harm suite")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", help="Student checkpoint directory")
    r.add_argument("--baseline", help="Baseline guard name (see baselines.BASELINES)")
    r.add_argument("--name")
    r.add_argument("--data", type=Path, default=Path("data"))
    sub.add_parser("teacher").add_argument("--data", type=Path, default=Path("data"))
    sub.add_parser("table")
    args = parser.parse_args()

    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv

    load_dotenv()
    if args.cmd == "run":
        if bool(args.model) == bool(args.baseline):
            raise SystemExit("Pass exactly one of --model or --baseline")
        if args.model:
            run(args.name or Path(args.model).name, student_predictor(args.model), args.data, True, "student")
        else:
            run(args.name or args.baseline, baseline_predictor(args.baseline), args.data, False, "baseline")
    elif args.cmd == "teacher":
        print(json.dumps(run_teacher(args.data)["pilot_subset"], indent=2))
    else:
        print(write_table())


if __name__ == "__main__":
    main()
