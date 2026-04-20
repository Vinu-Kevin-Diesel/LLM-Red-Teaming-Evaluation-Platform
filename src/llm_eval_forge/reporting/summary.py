"""Rich terminal summary output."""

from __future__ import annotations

from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_eval_forge.evaluators.base import EvalResult


def print_summary(results: list[EvalResult]) -> None:
    """Print a rich summary table of evaluation results."""
    console = Console()

    if not results:
        console.print("[yellow]No results to display.[/yellow]")
        return

    scored = [r for r in results if r.judge_score is not None]
    errors = [r for r in results if r.response_text.startswith("ERROR:")]

    # Header
    console.print()
    console.print(Panel.fit(
        f"[bold]Evaluation Complete[/bold]\n"
        f"Total: {len(results)} | Scored: {len(scored)} | Errors: {len(errors)}",
        title="LLM Eval Forge",
    ))

    if not scored:
        console.print("[yellow]No scored results available.[/yellow]")
        return

    # Provider x Dimension matrix
    table = Table(title="Scores by Provider x Dimension (0.0 - 1.0)")
    table.add_column("Provider / Model", style="cyan")

    dimensions = sorted({r.dimension for r in scored})
    for dim in dimensions:
        table.add_column(dim.capitalize(), justify="center")
    table.add_column("Average", justify="center", style="bold")

    # Aggregate scores
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in scored:
        key = f"{r.provider}/{r.model}"
        scores[key][r.dimension].append(r.judge_score)

    for provider_model, dim_scores in sorted(scores.items()):
        row = [provider_model]
        all_scores = []
        for dim in dimensions:
            s = dim_scores.get(dim, [])
            if s:
                avg = sum(s) / len(s)
                all_scores.extend(s)
                color = _score_color(avg)
                row.append(f"[{color}]{avg:.3f}[/{color}]")
            else:
                row.append("-")
        overall = sum(all_scores) / len(all_scores) if all_scores else 0
        color = _score_color(overall)
        row.append(f"[{color}]{overall:.3f}[/{color}]")
        table.add_row(*row)

    console.print(table)

    # Latency stats
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    if latencies:
        lat_table = Table(title="Latency Statistics (ms)")
        lat_table.add_column("Metric", style="cyan")
        lat_table.add_column("Value", justify="right")
        lat_table.add_row("Average", f"{sum(latencies) / len(latencies):.0f}")
        lat_table.add_row("Min", f"{min(latencies):.0f}")
        lat_table.add_row("Max", f"{max(latencies):.0f}")
        console.print(lat_table)

    console.print()


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "green"
    if score >= 0.6:
        return "yellow"
    return "red"
