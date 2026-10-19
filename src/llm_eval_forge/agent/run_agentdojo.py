"""Run an AgentDojo suite with a tool-call judge in the loop and report attack success vs utility.

    # judge service with the tool-call guard (another terminal)
    JUDGE_TOOLCALL_MODEL_DIR=models/toolcall-guard-A uvicorn llm_eval_forge.serve.app:create_app --factory --port 8000

    python -m llm_eval_forge.agent.run_agentdojo --judge none
    python -m llm_eval_forge.agent.run_agentdojo --judge toolcall-guard
    python -m llm_eval_forge.agent.run_agentdojo --judge oracle
    python -m llm_eval_forge.agent.run_agentdojo summarize --judge none   # rebuild one run's summary from its episodes
    python -m llm_eval_forge.agent.run_agentdojo report      # -> results/agent_guard.md

Metrics (AgentDojo definitions):
    benign utility        share of user tasks solved with no attack
    utility under attack  share of (user task, injection task) pairs where the user task is still solved
    targeted ASR          share of pairs where the attacker's goal was achieved
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

RESULTS = Path("results/agent_guard.json")
RUNS_DIR = Path("data/agent_runs")
BENCHMARK_VERSION = "v1.2.2"


def make_llm(agent: str):
    import openai
    from agentdojo.agent_pipeline import OpenAILLM

    provider, model = agent.split(":", 1)
    if provider == "ollama":
        client = openai.OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama", max_retries=3, timeout=300)
    elif provider == "groq":
        import os

        client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"], max_retries=6)
    else:
        raise SystemExit(f"Unknown agent provider {provider!r} (use ollama:<model> or groq:<model>)")
    return OpenAILLM(client, model, temperature=0.0)


def make_pipeline(agent: str, judge, log, system_suffix: str = ""):
    from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop
    from agentdojo.agent_pipeline.agent_pipeline import load_system_message

    from llm_eval_forge.agent.guard import GuardedToolsExecutor

    llm = make_llm(agent)
    pipe = AgentPipeline([SystemMessage(load_system_message(None) + system_suffix), InitQuery(), llm,
                          ToolsExecutionLoop([GuardedToolsExecutor(judge, log), llm])])
    pipe.name = f"openai-compatible-{judge.name}"  # the attack looks up the model name from this
    return pipe


def pending_pairs(user_ids: list[str], injection_ids: list[str],
                  done: set[tuple[str, str | None]]) -> tuple[list[str], list[tuple[str, str]]]:
    """Episodes still to run: benign user tasks, then (user task, injection task) pairs, in order."""
    benign = [u for u in user_ids if (u, None) not in done]
    attacked = [(u, i) for u in user_ids for i in injection_ids if (u, i) not in done]
    return benign, attacked


def runs_path(suite_name: str, agent: str, judge_name: str) -> Path:
    return RUNS_DIR / f"{suite_name}_{agent.replace(':', '_').replace('/', '_')}_{judge_name}.jsonl"


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return [0.0, 0.0]
    p, d = k / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs (b one way, c the other)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return float(f"{min(1.0, 2 * tail):.3g}")  # significant digits: these p-values can be ~1e-11


def paired(base: list[dict], guarded: list[dict]) -> dict:
    """Episode-by-episode comparison of a guarded run with the unguarded run on the same tasks."""
    def key(e):
        return e["user_task"], e["injection_task"]

    b, g = {key(e): e for e in base}, {key(e): e for e in guarded}
    attacked = [k for k in b if k in g and k[1] is not None]
    benign = [k for k in b if k in g and k[1] is None]
    def blocked(e):
        return any(x["action"] == "block" for x in e.get("guard", []))

    stopped = [k for k in attacked if b[k]["attack_succeeded"] and not g[k]["attack_succeeded"]]
    new = sum(g[k]["attack_succeeded"] and not b[k]["attack_succeeded"] for k in attacked)
    lost = [k for k in benign if b[k]["utility"] and not g[k]["utility"]]
    # A difference with no blocked call in the guarded episode is the agent behaving differently, not the guard.
    return {"attacked_pairs": len(attacked), "attacks_stopped": len(stopped),
            "attacks_stopped_by_a_block": sum(blocked(g[k]) for k in stopped), "attacks_new": new,
            "mcnemar_p": mcnemar_p(len(stopped), new), "benign_pairs": len(benign),
            "benign_lost": len(lost), "benign_lost_by_a_block": sum(blocked(g[k]) for k in lost),
            "benign_gained": sum(g[k]["utility"] and not b[k]["utility"] for k in benign)}


def load_episodes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(suite_name: str, judge_name: str, agent: str, attack_name: str, user_limit: int | None,
        injection_limit: int | None, judge_url: str, system_suffix: str, resume: bool = False) -> dict:
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.task_suite.load_suites import get_suite

    from llm_eval_forge.agent.guard import JUDGES, GuardLog, HttpToolCallJudge

    suite = get_suite(BENCHMARK_VERSION, suite_name)
    judge = HttpToolCallJudge(judge_url) if judge_name == "toolcall-guard" else JUDGES[judge_name]()
    log = GuardLog()
    pipe = make_pipeline(agent, judge, log, system_suffix)
    attack = load_attack(attack_name, suite, pipe)
    user_tasks = list(suite.user_tasks.values())[:user_limit]
    injection_tasks = list(suite.injection_tasks.values())[:injection_limit]
    episodes = []
    out_path = runs_path(suite_name, judge_name=judge_name, agent=agent)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def episode(ut, it, injections):
        if hasattr(judge, "set_injections"):
            judge.set_injections(injections, ut.PROMPT)
        n_log = len(log.entries)
        t0 = time.time()
        try:
            utility, security = suite.run_task_with_pipeline(pipe, ut, it, injections)
            err = None
        except Exception as exc:  # an LLM/runtime failure counts as a failed task, not a crash of the run
            utility, security, err = False, False, f"{type(exc).__name__}: {str(exc)[:200]}"
        rec = {"user_task": ut.ID, "injection_task": it.ID if it else None, "utility": bool(utility),
               "attack_succeeded": bool(security) if it else None, "seconds": round(time.time() - t0, 1),
               "guard": log.entries[n_log:], "error": err}
        episodes.append(rec)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    # Episodes are written as they finish, so an interrupted run can continue where it stopped.
    previous = load_episodes(out_path) if resume else []
    episodes.extend(previous)
    if resume:
        print(f"resuming: {len(previous)} episodes already done", flush=True)
    else:
        out_path.write_text("", encoding="utf-8")
    by_user = {ut.ID: ut for ut in user_tasks}
    by_injection = {it.ID: it for it in injection_tasks}
    benign, attacked_pairs = pending_pairs(list(by_user), list(by_injection),
                                           {(e["user_task"], e["injection_task"]) for e in previous})
    for uid in benign:  # benign utility
        episode(by_user[uid], None, {})
    for n, (uid, iid) in enumerate(attacked_pairs, 1):
        episode(by_user[uid], by_injection[iid], attack.attack(by_user[uid], by_injection[iid]))
        if n % len(by_injection) == 0 or n == len(attacked_pairs):
            attacked = [e for e in episodes if e["injection_task"]]
            print(f"[{judge_name}] {len(attacked)} attacked episodes, ASR so far "
                  f"{sum(e['attack_succeeded'] for e in attacked) / max(len(attacked), 1):.3f}", flush=True)
    return summarize(episodes, suite_name, judge_name, agent, attack_name)


def summarize(episodes: list[dict], suite_name: str, judge_name: str, agent: str, attack_name: str) -> dict:
    from llm_eval_forge.serve.bench import pct

    benign = [e for e in episodes if e["injection_task"] is None]
    attacked = [e for e in episodes if e["injection_task"] is not None]
    guard = [g for e in episodes for g in e["guard"]]
    lat = [g["latency_ms"] for g in guard if g["latency_ms"]]
    k_benign = sum(e["utility"] for e in benign)
    k_attack = sum(e["attack_succeeded"] for e in attacked)
    summary = {
        "suite": suite_name, "judge": judge_name, "agent": agent, "attack": attack_name,
        "benchmark_version": BENCHMARK_VERSION,
        "benign_episodes": len(benign), "attacked_episodes": len(attacked),
        "benign_utility": round(k_benign / max(len(benign), 1), 4), "benign_utility_ci95": wilson(k_benign, len(benign)),
        "utility_under_attack": round(sum(e["utility"] for e in attacked) / max(len(attacked), 1), 4),
        "asr": round(k_attack / max(len(attacked), 1), 4), "asr_ci95": wilson(k_attack, len(attacked)),
        "tool_calls_judged": len(guard), "tool_calls_blocked": sum(g["action"] == "block" for g in guard),
        "benign_calls_blocked": sum(g["action"] == "block" for e in benign for g in e["guard"]),
        "guard_ms_p50": round(pct(lat, 0.5), 2) if lat else None,
        "guard_ms_p99": round(pct(lat, 0.99), 2) if lat else None,
        "errors": sum(1 for e in episodes if e["error"]),
        "agent_minutes": round(sum(e["seconds"] for e in episodes) / 60, 1),  # summed, so resumed runs count fully
    }
    if judge_name != "none":
        base = load_episodes(runs_path(suite_name, agent, "none"))
        summary["paired_vs_none"] = paired(base, episodes) if base else None
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    results[f"{suite_name}|{agent}|{judge_name}|{attack_name}"] = summary
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def render(results: dict) -> str:
    lines = ["# Agent tool-call guard (AgentDojo)", "",
             "Each proposed tool call is judged before it runs; a blocked call returns a tool error and the agent "
             "continues. Generated by `python -m llm_eval_forge.agent.run_agentdojo report`.", "",
             "| Suite | Agent | Attack | Guard | Benign utility | Utility under attack | **ASR** | Calls blocked "
             "(benign runs) | Guard ms p50 / p99 | Episodes |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    def ci(s, key):
        lo_hi = s.get(f"{key}_ci95")
        return f" ({100 * lo_hi[0]:.1f}–{100 * lo_hi[1]:.1f})" if lo_hi else ""

    ordered = sorted(results.values(), key=lambda s: (s["suite"], s["agent"], s["attack"], s["judge"] != "none"))
    for s in ordered:
        lat = f"{s['guard_ms_p50']} / {s['guard_ms_p99']}" if s["guard_ms_p50"] is not None else "–"
        lines.append(f"| {s['suite']} | `{s['agent']}` | {s['attack']} | {s['judge']} | "
                     f"{100 * s['benign_utility']:.1f}%{ci(s, 'benign_utility')} | {100 * s['utility_under_attack']:.1f}% | "
                     f"**{100 * s['asr']:.1f}%**{ci(s, 'asr')} | {s['tool_calls_blocked']} ({s['benign_calls_blocked']}) | "
                     f"{lat} | {s['benign_episodes']} + {s['attacked_episodes']} |")
    lines += ["", "Brackets: 95% Wilson intervals. Calls blocked: all runs, with the benign (no attack) runs in "
                  "parentheses, so the parenthesized count is the guard's false blocks."]
    for s in ordered:
        pv = s.get("paired_vs_none")
        if pv:
            lines += ["", f"**Paired with the unguarded run** (`{s['judge']}`, same {pv['attacked_pairs']} attacked "
                          f"tasks): **{pv['attacks_stopped']}** attacks that succeeded without the guard failed with it "
                          f"({pv.get('attacks_stopped_by_a_block', '?')} of them with a blocked call); "
                          f"{pv['attacks_new']} succeeded only with it; exact McNemar p = {pv['mcnemar_p']:.2g}. "
                          f"Of {pv['benign_pairs']} benign tasks, {pv['benign_lost']} were solved only without the guard "
                          f"({pv.get('benign_lost_by_a_block', '?')} of them with a blocked call; the rest are the agent "
                          f"behaving differently between runs) and {pv['benign_gained']} only with it."]
    lines += ["", "Published reference (AgentDojo site, `important_instructions`, no defense): gpt-4o-mini ASR 27.19%, "
                  "GPT-4o ASR 47.69%.", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentDojo with a tool-call judge in the loop")
    parser.add_argument("cmd", nargs="?", default="run", choices=["run", "summarize", "report"])
    parser.add_argument("--suite", default="banking")
    parser.add_argument("--judge", default="none", choices=["none", "toolcall-guard", "oracle"])
    parser.add_argument("--agent", default="ollama:qwen3-8b-think",
                        help="ollama:<model> or groq:<model>; see config/ollama/Modelfile.qwen3-8b-think")
    parser.add_argument("--attack", default="important_instructions")
    parser.add_argument("--user-tasks", type=int, default=None, help="Limit user tasks (smoke tests)")
    parser.add_argument("--injection-tasks", type=int, default=None)
    parser.add_argument("--judge-url", default="http://127.0.0.1:8000")
    parser.add_argument("--system-suffix", default="", help="Appended to AgentDojo's system message (e.g. ' /no_think')")
    parser.add_argument("--resume", action="store_true", help="Skip episodes already recorded by an earlier run")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv

    load_dotenv()
    if args.cmd == "run":
        run(args.suite, args.judge, args.agent, args.attack, args.user_tasks, args.injection_tasks, args.judge_url,
            args.system_suffix, resume=args.resume)
    if args.cmd == "summarize":
        summarize(load_episodes(runs_path(args.suite, args.agent, args.judge)), args.suite, args.judge, args.agent,
                  args.attack)
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    Path("results/agent_guard.md").write_text(render(results), encoding="utf-8")
    if args.cmd == "report":
        print(render(results))


if __name__ == "__main__":
    main()
