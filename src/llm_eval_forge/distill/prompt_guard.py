"""Input guard: classify the USER PROMPT (before the model runs) as harmful or not.

The response judge (data.py / evaluate.py) answers "is this reply harmful?". This module
builds the prompt-level data, evaluates prompt classifiers, and reports over-blocking: how
often safe prompts that merely sound dangerous ("How do I kill a Python process?") get blocked.

    python -m llm_eval_forge.distill.prompt_guard build
    python -m llm_eval_forge.distill.train --stage A --data data/prompts --out models/prompt-guard-A
    python -m llm_eval_forge.distill.prompt_guard run --model models/prompt-guard-A
    python -m llm_eval_forge.distill.prompt_guard run --baseline qwen3guard-0.6b
    python -m llm_eval_forge.distill.prompt_guard table      # -> results/prompt_guard_table.md

Prompt rows reuse the response pipeline's names (wildguardtrain, aegis2_train, aegis2_val), so
the trainer runs unchanged with `--data data/prompts`; the response field is empty.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from llm_eval_forge.distill.data import (
    Example, SourceSpec, _load, assign_ids, carve_val, data_card, drop_contaminated, has_access, normalize,
    read_split, write_jsonl,
)
from llm_eval_forge.distill.metrics import best_f1_threshold, fit_temperature

PROMPT_DATA = Path("data/prompts")
METRICS_DIR = Path("results/metrics_prompt")
# Averaged prompt-harm benchmarks; XSTest and JBB behaviors measure over-blocking with matched safe prompts.
PROMPT_BENCHMARKS = ["wildguardtest", "aegis2_test", "toxicchat_test"]
OVERBLOCK_SETS = ["xstest_prompts", "jbb_behaviors"]
SHORT = {"wildguardtest": "WGTest", "aegis2_test": "Aegis2", "toxicchat_test": "ToxicChat",
         "xstest_prompts": "XSTest", "jbb_behaviors": "JBB"}


def _prompt_example(source: str, split: str, prompt: str, label: int, label_source: str = "human",
                    meta: dict | None = None) -> Example:
    return Example(id="", source=source, split=split, prompt=prompt, response="", label=label,
                   label_kind="prompt_harmful", label_source=label_source, meta=meta or {})


def dedupe_prompts(rows: list[tuple[str, int, dict]]) -> tuple[list[tuple[str, int, dict]], int]:
    """One row per normalized prompt, majority label; prompts with tied labels are dropped."""
    votes: dict[str, Counter] = defaultdict(Counter)
    first: dict[str, tuple[str, dict]] = {}
    for prompt, label, meta in rows:
        key = normalize(prompt)
        votes[key][label] += 1
        first.setdefault(key, (prompt, meta))
    out, ties = [], 0
    for key, c in votes.items():
        (top, n), *rest = c.most_common()
        if rest and rest[0][1] == n:
            ties += 1
            continue
        prompt, meta = first[key]
        out.append((prompt, top, meta))
    return out, ties


# --------------------------------------------------------------------------- sources

def load_wildguardtrain_prompts() -> list[Example]:
    if has_access("allenai/wildguardmix"):
        ds, origin = _load("allenai/wildguardmix", "wildguardtrain", split="train"), "allenai/wildguardmix"
    else:
        ds, origin = _load("ToxicityPrompts/wildguard-train", "wildguardtrain", split="train"), "ToxicityPrompts/wildguard-train"
    rows = [(r["prompt"], int(r["prompt_harm_label"] == "harmful"), {"adversarial": r["adversarial"], "origin": origin})
            for r in ds if r["prompt_harm_label"] in ("harmful", "unharmful") and r["prompt"]]
    rows, _ = dedupe_prompts(rows)
    return assign_ids([_prompt_example("wildguardtrain", "train", p, y, "gpt4", m) for p, y, m in rows])


def _aegis_prompts(split: str, source: str, split_name: str) -> list[Example]:
    ds = _load("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split=split)
    rows = [(r["prompt"], int(r["prompt_label"] == "unsafe"), {})
            for r in ds if r["prompt_label"] in ("safe", "unsafe") and r["prompt"] and r["prompt"] != "REDACTED"]
    rows, _ = dedupe_prompts(rows)
    return assign_ids([_prompt_example(source, split_name, p, y, "mixed") for p, y, m in rows])


def load_wildguardtest_prompts() -> list[Example]:
    if has_access("allenai/wildguardmix"):
        ds = _load("allenai/wildguardmix", "wildguardtest", split="test")
    else:  # PolyGuardPrompts' English subset is WildGuardTest
        ds = _load("ToxicityPrompts/PolyGuardPrompts", split="test").filter(lambda r: r["language"] == "English")
    return assign_ids([
        _prompt_example("wildguardtest", "test", r["prompt"], int(r["prompt_harm_label"] == "harmful"),
                        meta={"adversarial": r["adversarial"]})
        for r in ds if r["prompt_harm_label"] in ("harmful", "unharmful")
    ])


def load_toxicchat_prompts() -> list[Example]:
    # Standard eval set: the human-annotated test rows (CC-BY-NC, evaluation only)
    ds = _load("lmsys/toxic-chat", "toxicchat0124", split="test")
    return assign_ids([
        _prompt_example("toxicchat_test", "test", r["user_input"], int(r["toxicity"]),
                        meta={"jailbreaking": int(r["jailbreaking"])})
        for r in ds if r["human_annotation"]
    ])


def load_xstest_prompts() -> list[Example]:
    ds = _load("Paul/XSTest", split="train")
    return assign_ids([
        _prompt_example("xstest_prompts", "test", r["prompt"], int(r["label"] == "unsafe"),
                        meta={"type": r["type"], "focus": r["focus"]})
        for r in ds
    ])


def load_jbb_behaviors() -> list[Example]:
    ds = _load("JailbreakBench/JBB-Behaviors", "behaviors")
    rows = [(r["Goal"], 1, r["Category"]) for r in ds["harmful"]] + [(r["Goal"], 0, r["Category"]) for r in ds["benign"]]
    return assign_ids([_prompt_example("jbb_behaviors", "test", g, y, meta={"category": c}) for g, y, c in rows])


PROMPT_SOURCES = [
    SourceSpec("wildguardtrain", "train", "ODC-BY", load_wildguardtrain_prompts, "GPT-4 labels, deduplicated prompts"),
    SourceSpec("aegis2_train", "train", "CC-BY-4.0", lambda: _aegis_prompts("train", "aegis2_train", "train")),
    SourceSpec("aegis2_val", "val", "CC-BY-4.0", lambda: _aegis_prompts("validation", "aegis2_val", "val")),
    SourceSpec("wildguardtest", "test", "ODC-BY", load_wildguardtest_prompts, "human labels"),
    SourceSpec("aegis2_test", "test", "CC-BY-4.0", lambda: _aegis_prompts("test", "aegis2_test", "test")),
    SourceSpec("toxicchat_test", "test", "CC-BY-NC-4.0 (eval only)", load_toxicchat_prompts, "human-annotated rows"),
    SourceSpec("xstest_prompts", "test", "CC-BY-4.0", load_xstest_prompts, "250 safe / 200 unsafe (over-blocking)"),
    SourceSpec("jbb_behaviors", "test", "MIT", load_jbb_behaviors, "100 harmful / 100 matched benign"),
]


def build(out_dir: Path = PROMPT_DATA, card_path: Path = Path("results/data_card_prompts.md")) -> list[Example]:
    examples: list[Example] = []
    for spec in PROMPT_SOURCES:
        rows = spec.loader()
        print(f"{spec.name:16s} {len(rows):>7,} prompts")
        examples.extend(rows)
    carve_val([ex for ex in examples if ex.source == "wildguardtrain"], fraction=0.02, salt="val")
    examples, dropped = drop_contaminated(examples)
    for split in ("train", "val", "test"):
        for source in sorted({ex.source for ex in examples if ex.split == split}):
            write_jsonl(out_dir / split / f"{source}.jsonl",
                        (asdict(ex) for ex in examples if ex.split == split and ex.source == source))
    card = data_card(examples, dropped, PROMPT_SOURCES, []).replace(
        "# Data card: response-safety distillation set", "# Data card: prompt-level (input guard) set").replace(
        "Label 1 = harmful response (`jbb_judge`: 1 = jailbroken)", "Label 1 = harmful prompt").replace(
        "`python -m llm_eval_forge.distill.data build`", "`python -m llm_eval_forge.distill.prompt_guard build`").replace(
        " Pending = to be labeled by the Claude teacher.", "")
    card_path.write_text(card, encoding="utf-8")
    print(f"Wrote {len(examples):,} prompts to {out_dir}/ and {card_path}")
    return examples


# --------------------------------------------------------------------------- evaluation

def prompt_metrics(rows: list[dict], probs: list[float], threshold: float, temperature: float) -> dict:
    from llm_eval_forge.distill.evaluate import set_metrics

    m = set_metrics([r["label"] for r in rows], probs, threshold, temperature)
    safe = [p for r, p in zip(rows, probs) if r["label"] == 0]
    m["safe_block_rate"] = round(sum(p >= threshold for p in safe) / len(safe), 4) if safe else None
    adv = [(r["label"], p) for r, p in zip(rows, probs) if r.get("meta", {}).get("adversarial") is True]
    if adv:
        from llm_eval_forge.distill.teacher import binary_metrics

        m["adversarial_f1"] = round(binary_metrics([y for y, _ in adv], [int(p >= threshold) for _, p in adv])["f1"], 4)
    return m


def run(name: str, predictor: Callable[[list[dict]], list[float]], tune_threshold: bool, kind: str,
        data_dir: Path = PROMPT_DATA) -> dict:
    test = read_split(data_dir, "test")
    sets: dict[str, list[dict]] = defaultdict(list)
    for r in test:
        sets[r["source"]].append(r)
    sets["val"] = read_split(data_dir, "val")
    probs, seconds = {}, {}
    for set_name, rows in sets.items():
        t0 = time.time()
        probs[set_name] = predictor(rows)
        seconds[set_name] = round(time.time() - t0, 1)
        write_jsonl(data_dir / "preds" / name / f"{set_name}.jsonl",
                    ({"id": r["id"], "label": r["label"], "p": p} for r, p in zip(rows, probs[set_name])))
        print(f"{name}: {set_name} ({len(rows)} prompts) in {seconds[set_name]}s", flush=True)
    val_labels = [r["label"] for r in sets["val"]]
    threshold = best_f1_threshold(val_labels, probs["val"])[0] if tune_threshold else 0.5
    temperature = fit_temperature(val_labels, probs["val"])
    results = {"name": name, "kind": kind, "threshold": threshold, "temperature": round(temperature, 2),
               "sets": {s: prompt_metrics(rows, probs[s], threshold, temperature) for s, rows in sets.items()},
               "seconds": seconds}
    present = [b for b in PROMPT_BENCHMARKS if b in results["sets"]]
    results["avg_prompt_f1"] = round(sum(results["sets"][b]["f1"] for b in present) / len(present), 4)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"{name}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{name}: avg prompt F1 {results['avg_prompt_f1']:.3f}; XSTest safe-prompt block rate "
          f"{results['sets'].get('xstest_prompts', {}).get('safe_block_rate')}")
    return results


def render_table(metrics: list[dict]) -> str:
    def pct(v):
        return "–" if v is None else f"{100 * v:.1f}"

    head = ["Input guard", *[f"{SHORT[b]} F1" for b in PROMPT_BENCHMARKS], "Avg F1", "WGTest adversarial F1",
            "XSTest F1", "XSTest safe blocked", "JBB benign blocked", "JBB harmful caught"]
    lines = ["# Input guard: prompt-harm benchmark table", "",
             "F1 (%) with harmful prompt = positive. **Lower is better** for the two *blocked* columns: they are the "
             "share of safe prompts the guard would block (over-blocking). Students use the threshold tuned on the "
             "validation set; guards use their native decision. Generated by "
             "`python -m llm_eval_forge.distill.prompt_guard table`.", "",
             "| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for m in sorted(metrics, key=lambda m: -m["avg_prompt_f1"]):
        s = m["sets"]
        cells = [f"**{m['name']}**" if m["kind"] == "student" else m["name"]]
        cells += [pct(s[b]["f1"]) if b in s else "–" for b in PROMPT_BENCHMARKS]
        cells += [pct(m["avg_prompt_f1"]), pct(s.get("wildguardtest", {}).get("adversarial_f1")),
                  pct(s.get("xstest_prompts", {}).get("f1")), pct(s.get("xstest_prompts", {}).get("safe_block_rate")),
                  pct(s.get("jbb_behaviors", {}).get("safe_block_rate")), pct(s.get("jbb_behaviors", {}).get("recall"))]
        lines.append("| " + " | ".join(cells) + " |")
    n = {b: next((m["sets"][b]["n"] for m in metrics if b in m["sets"]), None) for b in PROMPT_BENCHMARKS + OVERBLOCK_SETS}
    lines += ["", "Prompts per set: " + ", ".join(f"{SHORT[b]} {v:,}" for b, v in n.items() if v), ""]
    return "\n".join(lines) + "\n"


def write_table(out: Path = Path("results/prompt_guard_table.md")) -> str:
    metrics = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(METRICS_DIR.glob("*.json"))]
    text = render_table(metrics)
    out.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    import sys

    parser = argparse.ArgumentParser(description="Input guard: prompt-harm data, evaluation and table")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--model")
    r.add_argument("--baseline")
    r.add_argument("--name")
    sub.add_parser("table")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv

    load_dotenv()
    if args.cmd == "build":
        build()
    elif args.cmd == "run":
        from llm_eval_forge.distill.evaluate import baseline_predictor, student_predictor

        if bool(args.model) == bool(args.baseline):
            raise SystemExit("Pass exactly one of --model or --baseline")
        if args.model:
            run(args.name or Path(args.model).name, student_predictor(args.model), True, "student")
        else:
            run(args.name or args.baseline, baseline_predictor(args.baseline), False, "baseline")
    else:
        print(write_table())


if __name__ == "__main__":
    main()
