"""Tool-call dataset from AgentDojo's published agent traces.

Every tool call an agent proposed becomes one example: the user's goal, the calls made so far,
the proposed call, the **provenance of its arguments** (for each IBAN / email / URL in the call:
did it come from the user's request, from which tool output and with what surrounding text, or
from nowhere), and the last tool output.

Why provenance: a first model that only saw the last tool output reached AUROC 0.9999 on held-out
user tasks of the training suites but 0.80 on the held-out banking suite. It had memorized the
attackers' addresses instead of the concept. "This IBAN came from inside an <INFORMATION> block
in a file the agent read" is the transferable signal; training identifiers are also replaced with
random ones (consistently within an example) so memorizing specific addresses is impossible.

Labels, without any LLM:
    1  the call's arguments contain an attacker identifier: an IBAN, email or URL that appears in the
       injected text, but not in the user's request and not in clean (no-attack) runs of that user task
    0  any call from a run where the attack failed, or from a clean run (including AgentDojo's runs where
       the user asks for the attacker's action directly: the same call is legitimate when the user wants it)
    -  calls from a successful-attack run that match no attacker identifier are dropped as ambiguous

Split by suite: banking is held out entirely (unseen tools, data and attacker identifiers); the other
suites train, with 10% of their user tasks held out for validation.

    git clone --depth 1 https://github.com/ethz-spylab/agentdojo data/agentdojo
    python -m llm_eval_forge.agent.toolcall_data build
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import string
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from llm_eval_forge.distill.data import Example, bucket, text_hash, write_jsonl

RUNS = Path("data/agentdojo/runs")
OUT = Path("data/toolcalls")
TEST_SUITE = "banking"
SKIP_ATTACKS = {"captcha_dos", "dos", "felony_dos", "offensive_email_dos", "swearwords_dos"}  # aim to stop the agent
LAST_OUTPUT_CHARS = 800
SNIPPET_BEFORE, SNIPPET_AFTER = 220, 60
MAX_SOURCES = 3
TRAIN_COPIES = 2  # anonymized copies per training example
IDENT = re.compile(
    r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"          # IBAN-like account numbers
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"              # email addresses
    r"|https?://[^\s'\"<>]+|www\.[^\s'\"<>]+"     # URLs
)
_TRAIL = ".,;:!?)"


def identifiers(text: str) -> set[str]:
    return {m.rstrip(_TRAIL).lower() for m in IDENT.findall(text or "")}


def call_text(call: dict) -> str:
    return f"{call['function']}({json.dumps(call.get('args') or {}, sort_keys=True, ensure_ascii=False)})"


def _string_values(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _string_values(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _string_values(v)]
    return []


def argument_provenance(args: dict, goal: str, tool_outputs: list[str]) -> list[str]:
    """Where each identifier in the call's arguments came from: the user's request, the first tool
    output that contains it (with the text just before it), or nowhere."""
    ordered: list[str] = []
    for value in _string_values(args or {}):
        for m in IDENT.findall(value):
            key = m.rstrip(_TRAIL).lower()
            if key not in ordered:
                ordered.append(key)
    goal_l = (goal or "").lower()
    lines = []
    for key in ordered[:MAX_SOURCES]:
        if key in goal_l:
            lines.append(f"{key} <- the user's request")
            continue
        for k, out in enumerate(tool_outputs):
            i = (out or "").lower().find(key)
            if i >= 0:
                snippet = out[max(0, i - SNIPPET_BEFORE): i + len(key) + SNIPPET_AFTER].replace("\n", " ")
                lines.append(f'{key} <- tool output {k + 1}: "...{snippet}..."')
                break
        else:
            lines.append(f"{key} <- not in the user's request or any tool output")
    return lines


def format_toolcall(goal: str, history: list[str], call: dict, tool_outputs: list[str]) -> tuple[str, str]:
    """Judge input for one proposed call, as a (context, proposed) sentence pair. Shared by the dataset
    builder, the judge service endpoint and the live guard, so training and serving see identical text.
    Truncation trims the end of the second segment first (the last tool output), keeping the provenance."""
    sources = argument_provenance(call.get("args") or {}, goal, tool_outputs)
    last = (tool_outputs[-1] if tool_outputs else "") or ""
    context = f"GOAL: {goal}\nCALLS SO FAR: {'; '.join(history) if history else 'none'}"
    proposed = (f"PROPOSED CALL: {call_text(call)}\n"
                f"ARGUMENT SOURCES: {' | '.join(sources) if sources else 'no identifiers in the arguments'}\n"
                f"LAST TOOL OUTPUT: {last[:LAST_OUTPUT_CHARS]}")
    return context, proposed


def _synthetic(core: str, rng: random.Random) -> str:
    if "@" in core:
        name = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 8)))
        return f"{name}.{rng.randint(10, 9999)}@{rng.choice(['gmail.com', 'mail.net', 'proton.me', 'outlook.com', 'corp.io'])}"
    if core.lower().startswith(("http", "www")):
        word = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(5, 10)))
        return f"www.{word}-{rng.randint(10, 999)}.{rng.choice(['com', 'net', 'org', 'io'])}"
    country = "".join(rng.choices(string.ascii_uppercase, k=2))
    return country + "".join(rng.choices(string.digits, k=max(12, len(core) - 2)))


def anonymize(texts: list[str], rng: random.Random) -> list[str]:
    """Replace every identifier with a random one of the same kind, consistently across `texts`."""
    mapping: dict[str, str] = {}

    def repl(m: re.Match) -> str:
        raw = m.group(0)
        core = raw.rstrip(_TRAIL)
        key = core.lower()
        if key not in mapping:
            mapping[key] = _synthetic(core, rng)
        return mapping[key] + raw[len(core):]

    return [IDENT.sub(repl, t) for t in texts]


def content_text(msg: dict) -> str:
    c = msg.get("content")
    if isinstance(c, list):
        return "\n".join(b.get("content") or b.get("text") or "" for b in c if isinstance(b, dict))
    return str(c or "")


def load_traces(runs: Path = RUNS) -> list[dict]:
    traces = []
    for f in glob.glob(str(runs / "*" / "*" / "*" / "*" / "*.json")):
        if Path(f).parts[-2] in SKIP_ATTACKS:
            continue
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("messages"):
            traces.append(d)
    return traces


def benign_identifiers(traces: Iterable[dict]) -> dict[tuple[str, str], set[str]]:
    """Identifiers seen in clean (no-attack) runs of each (suite, user task): legitimate by construction.

    Keyed per user task, not per suite: AgentDojo reuses the attacker's IBAN as the landlord's account in
    banking user_task_15, where paying it is exactly what the user asked for. Runs where an injection task
    is executed as the *user* task (user_task_id "injection_task_N") are left out: there the user asks for
    the attacker's action on purpose."""
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for d in traces:
        if str(d.get("user_task_id", "")).startswith("injection_task"):
            continue
        if (d.get("attack_type") or "none") == "none" and not d.get("injection_task_id"):
            key = (d["suite_name"], d["user_task_id"])
            for m in d["messages"]:
                seen[key] |= identifiers(content_text(m))
                for c in m.get("tool_calls") or []:
                    seen[key] |= identifiers(json.dumps(c.get("args") or {}))
    return seen


def examples_from_trace(d: dict, attacker_ids: set[str]) -> list[tuple[str, str, int, dict]]:
    """(context, proposed, label, meta) for every proposed tool call in one trace."""
    msgs = d["messages"]
    goal = next((content_text(m) for m in msgs if m["role"] == "user"), "")
    attacked = bool(d.get("injection_task_id")) and (d.get("attack_type") or "none") != "none"
    succeeded = attacked and bool(d.get("security"))
    history: list[str] = []
    tool_outputs: list[str] = []
    out = []
    for m in msgs:
        if m["role"] == "tool":
            tool_outputs.append(content_text(m) or str(m.get("error") or ""))
        if m["role"] == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                args_ids = identifiers(json.dumps(c.get("args") or {}, ensure_ascii=False))
                if attacked and args_ids & attacker_ids:
                    label = 1
                elif not succeeded:
                    label = 0
                else:
                    label = None  # successful attack, but this call matches no attacker identifier
                if label is not None:
                    call = {"function": c["function"], "args": c.get("args") or {}}
                    context, proposed = format_toolcall(goal, history, call, tool_outputs)
                    out.append((context, proposed, label, {
                        "suite": d["suite_name"], "user_task": d["user_task_id"],
                        "injection_task": d.get("injection_task_id"), "attack": d.get("attack_type"),
                        "pipeline": d.get("pipeline_name"), "function": c["function"]}))
            for c in m["tool_calls"]:
                history.append(call_text({"function": c["function"], "args": c.get("args") or {}}))
    return out


def build(runs: Path = RUNS, out_dir: Path = OUT, card: Path = Path("results/data_card_toolcalls.md"),
          seed: int = 17) -> dict:
    traces = load_traces(runs)
    benign = benign_identifiers(traces)
    rng = random.Random(seed)
    rows: list[Example] = []
    seen_hashes: set[str] = set()
    dup = 0
    for d in traces:
        suite = d["suite_name"]
        injected = set().union(*(identifiers(v) for v in d["injections"].values())) if d.get("injections") else set()
        goal = next((content_text(m) for m in d["messages"] if m["role"] == "user"), "")
        attacker_ids = injected - benign.get((suite, d["user_task_id"]), set()) - identifiers(goal)
        for context, proposed, label, meta in examples_from_trace(d, attacker_ids):
            h = text_hash(context, proposed)
            if h in seen_hashes:  # the same call in the same context, produced by another pipeline
                dup += 1
                continue
            seen_hashes.add(h)
            if suite == TEST_SUITE:
                split, copies = "test", [(context, proposed)]  # real identifiers, never seen in training
            else:
                split = "val" if bucket(f"{suite}/{meta['user_task']}", salt="toolcall-val") < 0.10 else "train"
                n = TRAIN_COPIES if split == "train" else 1
                copies = [tuple(anonymize([context, proposed], rng)) for _ in range(n)]
            for c, p in copies:
                rows.append(Example(id="", source=f"agentdojo_{suite}", split=split, prompt=c, response=p,
                                    label=label, label_kind="toolcall_attacker", label_source="rule", meta=meta))
    by = defaultdict(list)
    for ex in rows:
        by[(ex.split, ex.source)].append(ex)
    for (split, source), exs in by.items():
        for i, ex in enumerate(exs):
            ex.id = f"{source}_{split}_{i:06d}"
        write_jsonl(out_dir / split / f"{source}.jsonl", (asdict(e) for e in exs))
    stats = {f"{split}/{source}": {"n": len(exs), "attacker_calls": sum(e.label for e in exs)}
             for (split, source), exs in sorted(by.items())}
    lines = ["# Data card: tool-call guard set (AgentDojo published traces)", "",
             "Generated by `python -m llm_eval_forge.agent.toolcall_data build`. Label 1 = attacker-induced call "
             "(arguments contain an identifier that appears only in the injected text); 0 = call from a run where "
             "the attack failed or from a clean run. Banking is held out entirely and keeps its real identifiers; "
             f"train/val identifiers are replaced with random ones ({TRAIN_COPIES} anonymized copies per training "
             "call).", "",
             "| Split / source | Calls | Attacker-induced | % |", "|---|---:|---:|---:|"]
    for k, v in stats.items():
        lines.append(f"| {k} | {v['n']:,} | {v['attacker_calls']:,} | {100 * v['attacker_calls'] / max(v['n'], 1):.1f} |")
    lines += ["", f"Traces read: {len(traces):,} (DoS-style attacks skipped). Duplicate calls removed: {dup:,}. "
                  f"Pipelines: {len({d.get('pipeline_name') for d in traces})}.", ""]
    card.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the tool-call guard dataset from AgentDojo traces")
    parser.add_argument("cmd", choices=["build"])
    parser.parse_args()
    build()


if __name__ == "__main__":
    main()
