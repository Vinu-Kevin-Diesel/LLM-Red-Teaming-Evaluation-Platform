"""Build the unified response-safety dataset used to train and evaluate the student judge.

Every example is a (prompt, response) pair with a binary label (1 = harmful response).
Train/val come from permissively licensed sources plus our own red-team outputs; the
test split is the standard response-harm benchmark suite reported by WildGuard,
Qwen3Guard and GuardReasoner. Train/val rows whose prompt appears in any test set are
dropped (contamination guard).

    python -m llm_eval_forge.distill.data build --out data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

HARMBENCH_VAL_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/classifier_val_sets/text_behaviors_val_set.json"
)
OWN_REPORT = Path("results/full-evaluation-run.json")
SEED = 13


@dataclass
class Example:
    id: str
    source: str
    split: str  # train | val | test
    prompt: str
    response: str
    label: Optional[int]  # 1 = harmful response; None = to be labeled by the teacher
    label_kind: str = "response_harmful"  # or "jailbroken" (JBB judge_comparison)
    label_source: str = "human"  # human | gpt4 | llm_jury | pending_teacher
    meta: dict = field(default_factory=dict)


@dataclass
class SourceSpec:
    name: str
    split: str
    license: str
    loader: Callable[[], list[Example]]
    note: str = ""


# --------------------------------------------------------------------------- helpers

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def text_hash(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(normalize(p) for p in parts).encode("utf-8")).hexdigest()


def bucket(key: str, salt: str = "") -> float:
    """Deterministic value in [0, 1) for hash-based splitting."""
    return int(hashlib.sha1(f"{salt}:{key}".encode("utf-8")).hexdigest()[:8], 16) / 0x100000000


def base_case_id(test_case_id: str) -> str:
    """Group red-team mutations with their seed prompt: 'adv-003_mut_roleplay_1' -> 'adv-003'."""
    return test_case_id.split("_mut_")[0]


def assign_ids(examples: list[Example]) -> list[Example]:
    # Batch API custom_ids must match ^[a-zA-Z0-9_-]{1,64}$
    for i, ex in enumerate(examples):
        ex.id = f"{ex.source}_{i:06d}"
    return examples


def has_access(repo_id: str, repo_type: str = "dataset") -> bool:
    from huggingface_hub import auth_check
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    try:
        auth_check(repo_id, repo_type=repo_type)
        return True
    except (GatedRepoError, RepositoryNotFoundError):
        return False


def _load(name: str, *args, **kwargs):
    from datasets import load_dataset

    return load_dataset(name, *args, **kwargs)


# --------------------------------------------------------------------------- train/val sources

def load_wildguardtrain() -> list[Example]:
    if has_access("allenai/wildguardmix"):
        ds, origin = _load("allenai/wildguardmix", "wildguardtrain", split="train"), "allenai/wildguardmix"
    else:  # ungated mirror with identical columns and row count (86,759)
        ds, origin = _load("ToxicityPrompts/wildguard-train", "wildguardtrain", split="train"), "ToxicityPrompts/wildguard-train"
    out = []
    for r in ds:
        if r["response_harm_label"] not in ("harmful", "unharmful") or not r["response"]:
            continue
        out.append(Example(
            id="", source="wildguardtrain", split="train", prompt=r["prompt"], response=r["response"],
            label=int(r["response_harm_label"] == "harmful"), label_source="gpt4",
            meta={"origin": origin, "adversarial": r["adversarial"], "subcategory": r["subcategory"],
                  "refusal": r["response_refusal_label"]},
        ))
    return assign_ids(out)


def _aegis_rows(split: str) -> list[dict]:
    ds = _load("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split=split)
    return [
        r for r in ds
        if r["response_label"] in ("safe", "unsafe") and r["response"] and r["prompt"]
        and r["prompt"] != "REDACTED" and r["response_label_source"] != "refusal_data_augmentation"
    ]


def _aegis_examples(split_name: str, rows: list[dict], source: str) -> list[Example]:
    return assign_ids([
        Example(
            id="", source=source, split=split_name, prompt=r["prompt"], response=r["response"],
            label=int(r["response_label"] == "unsafe"), label_source=r["response_label_source"],
            meta={"violated_categories": r.get("violated_categories")},
        )
        for r in rows
    ])


def load_aegis_train() -> list[Example]:
    return _aegis_examples("train", _aegis_rows("train"), "aegis2_train")


def load_aegis_val() -> list[Example]:
    return _aegis_examples("val", _aegis_rows("validation"), "aegis2_val")


def own_redteam_examples(report_path: Path = OWN_REPORT) -> list[Example]:
    """Adversarial rows from our own eval runs, split 70/15/15 by seed prompt so mutations
    of one prompt never straddle splits. Labels come from the teacher (pending)."""
    if not report_path.exists():
        return []
    rows = json.loads(report_path.read_text(encoding="utf-8"))["results"]
    out = []
    for r in rows:
        if r["dimension"] != "adversarial" or r["response_text"].startswith("ERROR:"):
            continue
        b = bucket(base_case_id(r["test_case_id"]), salt="own")
        split = "train" if b < 0.70 else "val" if b < 0.85 else "test"
        out.append(Example(
            id="", source="own_redteam", split=split, prompt=r["prompt_sent"], response=r["response_text"],
            label=None, label_source="pending_teacher",
            meta={"test_case_id": r["test_case_id"], "target_model": r["model"],
                  "attack_type": r["metadata"].get("attack_type"),
                  "mutation_strategy": r["metadata"].get("mutation_strategy"),
                  "legacy_judge_score": r.get("judge_score")},
        ))
    return assign_ids(out)


# --------------------------------------------------------------------------- test sources

def load_wildguardtest() -> list[Example]:
    if has_access("allenai/wildguardmix"):
        ds, origin = _load("allenai/wildguardmix", "wildguardtest", split="test"), "allenai/wildguardmix"
    else:  # PolyGuardPrompts' English subset is WildGuardTest (1,725 rows, same labels)
        ds = _load("ToxicityPrompts/PolyGuardPrompts", split="test").filter(lambda r: r["language"] == "English")
        origin = "ToxicityPrompts/PolyGuardPrompts[English]"
    return assign_ids([
        Example(id="", source="wildguardtest", split="test", prompt=r["prompt"], response=r["response"],
                label=int(r["response_harm_label"] == "harmful"),
                meta={"origin": origin, "adversarial": r["adversarial"]})
        for r in ds if r["response_harm_label"] in ("harmful", "unharmful") and r["response"]
    ])


def harmbench_examples(raw: dict) -> list[Example]:
    out = []
    for behavior_id, items in raw.items():
        for it in items:
            votes = sum(int(it[f"human_{i}"]) for i in range(3))
            out.append(Example(
                id="", source="harmbench_val", split="test", prompt=it["test_case"], response=it["generation"],
                label=int(votes >= 2),
                meta={"behavior_id": behavior_id, "method": it.get("method"), "model": it.get("model")},
            ))
    return assign_ids(out)


def load_harmbench() -> list[Example]:
    import httpx

    raw = httpx.get(HARMBENCH_VAL_URL, timeout=60, follow_redirects=True).json()
    return harmbench_examples(raw)


def load_xstest_resp() -> list[Example]:
    if not has_access("allenai/xstest-response"):
        return []  # gated (AI2 form, auto-approved); skipped until access is granted
    ds = _load("allenai/xstest-response", split="response_harmfulness")
    return assign_ids([
        Example(id="", source="xstest_resp", split="test", prompt=r["prompt"], response=r["response"],
                label=int(r["label"] == "harmful"), meta={"prompt_type": r.get("prompt_type")})
        for r in ds
    ])


def load_beavertails_test() -> list[Example]:
    # Only 30k_test: 2,711 of its pairs also appear in 330k_train, so no BeaverTails split is trained on.
    ds = _load("PKU-Alignment/BeaverTails", split="30k_test")
    return assign_ids([
        Example(id="", source="beavertails_test", split="test", prompt=r["prompt"], response=r["response"],
                label=int(not r["is_safe"]))
        for r in ds
    ])


def saferlhf_examples(rows: Iterable[dict], per_class: int = 1000, seed: int = 42) -> list[Example]:
    """safety-eval recipe: pairs whose two responses differ in safety, exploded, then a
    seeded sample of `per_class` safe and `per_class` harmful responses."""
    harmful, safe = [], []
    for r in rows:
        if r["is_response_0_safe"] == r["is_response_1_safe"]:
            continue
        for k in (0, 1):
            (safe if r[f"is_response_{k}_safe"] else harmful).append((r["prompt"], r[f"response_{k}"]))
    rng = random.Random(seed)
    picked = [(p, resp, 1) for p, resp in rng.sample(harmful, min(per_class, len(harmful)))]
    picked += [(p, resp, 0) for p, resp in rng.sample(safe, min(per_class, len(safe)))]
    return assign_ids([
        Example(id="", source="saferlhf_test", split="test", prompt=p, response=resp, label=y)
        for p, resp, y in picked
    ])


def load_saferlhf() -> list[Example]:
    return saferlhf_examples(_load("PKU-Alignment/PKU-SafeRLHF", revision="v0", split="test"))


def load_aegis_test() -> list[Example]:
    # All response-labeled rows. Human-only would leave zero positives: every human-labeled
    # test response is "safe", every "unsafe" one comes from the LLM jury.
    return _aegis_examples("test", _aegis_rows("test"), "aegis2_test")


def load_jbb_judge() -> list[Example]:
    ds = _load("JailbreakBench/JBB-Behaviors", "judge_comparison", split="test")
    return assign_ids([
        Example(id="", source="jbb_judge", split="test", prompt=r["prompt"], response=r["target_response"],
                label=int(r["human_majority"]), label_kind="jailbroken",
                meta={"goal": r["goal"], "llamaguard2_cf": r["llamaguard2_cf"], "gpt4_cf": r["gpt4_cf"]})
        for r in ds
    ])


SOURCES: list[SourceSpec] = [
    SourceSpec("wildguardtrain", "train", "ODC-BY", load_wildguardtrain, "GPT-4 labels"),
    SourceSpec("aegis2_train", "train", "CC-BY-4.0", load_aegis_train),
    SourceSpec("aegis2_val", "val", "CC-BY-4.0", load_aegis_val),
    SourceSpec("own_redteam", "mixed", "own", own_redteam_examples, "labels from the Claude teacher"),
    SourceSpec("wildguardtest", "test", "ODC-BY", load_wildguardtest),
    SourceSpec("harmbench_val", "test", "MIT", load_harmbench, "majority of 3 human labels"),
    SourceSpec("xstest_resp", "test", "ODC-BY", load_xstest_resp, "gated; skipped without access"),
    SourceSpec("beavertails_test", "test", "CC-BY-NC-4.0 (eval only)", load_beavertails_test),
    SourceSpec("saferlhf_test", "test", "CC-BY-NC-4.0 (eval only)", load_saferlhf, "v0, 1k/1k seeded"),
    SourceSpec("aegis2_test", "test", "CC-BY-4.0", load_aegis_test),
    SourceSpec("jbb_judge", "test", "MIT", load_jbb_judge, "label = jailbroken"),
]


# --------------------------------------------------------------------------- assembly

def carve_val(examples: list[Example], fraction: float, salt: str) -> None:
    for ex in examples:
        if ex.split == "train" and bucket(ex.id, salt=salt) < fraction:
            ex.split = "val"


def drop_contaminated(examples: list[Example]) -> tuple[list[Example], Counter]:
    """Remove train/val rows whose prompt appears in any test set."""
    test_prompts = {normalize(ex.prompt) for ex in examples if ex.split == "test"}
    kept, dropped = [], Counter()
    for ex in examples:
        if ex.split != "test" and normalize(ex.prompt) in test_prompts:
            dropped[ex.source] += 1
            continue
        kept.append(ex)
    return kept, dropped


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def data_card(examples: list[Example], dropped: Counter, specs: list[SourceSpec], skipped: list[str]) -> str:
    by = defaultdict(lambda: Counter())
    for ex in examples:
        key = (ex.source, ex.split)
        by[key]["n"] += 1
        by[key]["pos"] += ex.label == 1
        by[key]["neg"] += ex.label == 0
        by[key]["pending"] += ex.label is None
    lic = {s.name: s.license for s in specs}
    notes = {s.name: s.note for s in specs}
    lines = [
        "# Data card: response-safety distillation set",
        "",
        "Generated by `python -m llm_eval_forge.distill.data build`. Label 1 = harmful response "
        "(`jbb_judge`: 1 = jailbroken). Pending = to be labeled by the Claude teacher.",
        "",
        "| Source | Split | Rows | Harmful | Safe | Pending | Harmful % | License | Note |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for (source, split), c in sorted(by.items(), key=lambda kv: ({"train": 0, "val": 1, "test": 2}[kv[0][1]], kv[0][0])):
        labeled = c["pos"] + c["neg"]
        pct = f"{100 * c['pos'] / labeled:.1f}" if labeled else "-"
        lines.append(f"| {source} | {split} | {c['n']:,} | {c['pos']:,} | {c['neg']:,} | {c['pending']:,} | "
                     f"{pct} | {lic.get(source, '')} | {notes.get(source, '')} |")
    totals = Counter(ex.split for ex in examples)
    lines += ["", f"**Totals:** train {totals['train']:,} · val {totals['val']:,} · test {totals['test']:,}", ""]
    lines.append("**Contamination guard:** train/val rows whose normalized prompt appears in any test set "
                 "were dropped: " + (", ".join(f"{k} {v:,}" for k, v in dropped.items()) or "none") + ".")
    if skipped:
        lines.append(f"**Skipped sources:** {', '.join(skipped)}.")
    lines.append("")
    return "\n".join(lines)


def build(out_dir: Path, report_path: Path, card_path: Path) -> list[Example]:
    examples: list[Example] = []
    skipped = []
    for spec in SOURCES:
        rows = own_redteam_examples(report_path) if spec.name == "own_redteam" else spec.loader()
        print(f"{spec.name:18s} {len(rows):>7,} rows")
        if not rows:
            skipped.append(f"{spec.name} ({spec.note or 'no rows'})")
        examples.extend(rows)

    carve_val([ex for ex in examples if ex.source == "wildguardtrain"], fraction=0.02, salt="val")
    examples, dropped = drop_contaminated(examples)

    for split in ("train", "val", "test"):
        for source in sorted({ex.source for ex in examples if ex.split == split}):
            rows = [asdict(ex) for ex in examples if ex.split == split and ex.source == source]
            write_jsonl(out_dir / split / f"{source}.jsonl", rows)

    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(data_card(examples, dropped, SOURCES, skipped), encoding="utf-8")
    print(f"Wrote {len(examples):,} examples to {out_dir}/ and {card_path}")
    return examples


def read_split(data_dir: Path, split: str, sources: Optional[set[str]] = None) -> list[dict]:
    rows = []
    for path in sorted((data_dir / split).glob("*.jsonl")):
        if sources and path.stem not in sources:
            continue
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the distillation dataset")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=Path("data"))
    b.add_argument("--report", type=Path, default=OWN_REPORT, help="Own eval report with red-team outputs")
    b.add_argument("--card", type=Path, default=Path("results/data_card.md"))
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()  # HF_TOKEN for gated sources
    build(args.out, args.report, args.card)


if __name__ == "__main__":
    main()
