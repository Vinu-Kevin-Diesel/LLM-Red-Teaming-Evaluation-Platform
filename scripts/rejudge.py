"""Re-judge rows of an existing report that have no valid verdict.

Older runs stored judge parse failures as a 0.0 score with an empty
`judge_scores_detail`; this re-scores those rows (and rows whose verdict is
missing) with the structured-output judge and writes a corrected report.

    python scripts/rejudge.py results/full-evaluation-run.json \
        --out results/full-evaluation-run.rejudged.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from dotenv import load_dotenv

from llm_eval_forge.evaluators.base import EvalResult
from llm_eval_forge.judge.claude_judge import LLMJudge
from llm_eval_forge.judge.rubrics import get_rubric
from llm_eval_forge.reporting.json_report import save_json_report


def needs_rejudge(row: dict) -> bool:
    if row["response_text"].startswith("ERROR:"):
        return False  # provider failed; nothing to judge
    return row.get("judge_score") is None or not row.get("judge_scores_detail")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=None, help="Judge model (default: judge backend default)")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--dry-run", action="store_true", help="Only count rows that would be re-judged")
    parser.add_argument("--allow-api", action="store_true",
                        help="Required to re-judge: each row is a paid Anthropic API call")
    args = parser.parse_args()

    load_dotenv()
    rows = json.loads(args.report.read_text(encoding="utf-8"))["results"]
    todo = [i for i, r in enumerate(rows) if needs_rejudge(r)]
    print(f"{len(todo)}/{len(rows)} rows need re-judging")
    if args.dry_run or not todo:
        return
    if not args.allow_api:
        raise SystemExit("Re-judging calls the paid Anthropic API for every row; pass --allow-api to proceed.")

    judge = LLMJudge(model=args.model, effort=args.effort)
    fields = {f.name for f in dataclasses.fields(EvalResult)}
    results = [EvalResult(**{k: v for k, v in r.items() if k in fields}) for r in rows]
    fixed = failed = 0
    for n, i in enumerate(todo, 1):
        r = results[i]
        verdict = judge.score(r.dimension, r.prompt_sent, r.response_text, get_rubric(r.dimension), r.metadata)
        old = r.judge_score
        r.judge_score = verdict.weighted_score
        r.judge_reasoning = verdict.reasoning
        r.judge_scores_detail = verdict.scores
        r.judge_labels = verdict.labels
        r.judge_model = verdict.judge_model
        r.judge_error = verdict.error
        fixed += verdict.ok
        failed += not verdict.ok
        print(f"[{n}/{len(todo)}] {r.test_case_id}: {old} -> {verdict.weighted_score} {verdict.error or ''}")

    save_json_report(results, args.out)
    print(f"Re-judged {fixed} rows, {failed} still without a verdict. Wrote {args.out}")


if __name__ == "__main__":
    main()
