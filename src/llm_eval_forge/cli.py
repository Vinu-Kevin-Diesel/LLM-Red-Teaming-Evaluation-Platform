"""CLI entry point for llm-eval-forge."""

from __future__ import annotations

import asyncio
import logging
import sys

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from llm_eval_forge import __version__
from llm_eval_forge.config import load_config
from llm_eval_forge.runner import EvalRunner


@click.group()
@click.version_option(__version__, prog_name="llm-eval-forge")
def main():
    """LLM Evaluation & Red-Teaming Framework."""
    load_dotenv()


@main.command()
@click.option("--config", "config_path", default="config/default.yaml",
              help="Path to YAML config file")
@click.option("--provider", default=None, help="Run only this provider")
@click.option("--dimension", default=None, help="Run only this dimension")
@click.option("--redteam-only", is_flag=True, help="Only run red-team tests")
@click.option("--output", default=None, help="Output directory override")
@click.option("--verbose", is_flag=True, help="Verbose logging")
@click.option("--dry-run", is_flag=True, help="Show plan without executing")
@click.option("--no-judge", is_flag=True,
              help="Collect responses only — skip judging. Score later via external judge.")
def run(config_path, provider, dimension, redteam_only, output, verbose, dry_run, no_judge):
    """Run an evaluation suite."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    console = Console()
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Config error: {e}[/red]")
        sys.exit(1)

    if output:
        config.output_dir = output

    runner = EvalRunner(config, no_judge=no_judge)

    try:
        results = asyncio.run(runner.run(
            provider_filter=provider,
            dimension_filter=dimension,
            redteam_only=redteam_only,
            dry_run=dry_run,
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    if not dry_run:
        scored = [r for r in results if r.judge_score is not None]
        console.print(f"\n[green]Done. {len(scored)}/{len(results)} results scored.[/green]")
        console.print(f"Report saved to: {config.output_dir}/{config.name}.json")


@main.command("list-providers")
def list_providers_cmd():
    """List available LLM providers."""
    from llm_eval_forge.providers import list_providers

    console = Console()
    table = Table(title="Available Providers")
    table.add_column("Name", style="cyan")
    table.add_column("Default Models")

    from llm_eval_forge.providers.groq import GroqProvider
    from llm_eval_forge.providers.cerebras import CerebrasProvider
    from llm_eval_forge.providers.mistral import MistralProvider
    from llm_eval_forge.providers.huggingface import HuggingFaceProvider

    provider_info = {
        "groq": GroqProvider.DEFAULT_MODELS,
        "cerebras": CerebrasProvider.DEFAULT_MODELS,
        "mistral": MistralProvider.DEFAULT_MODELS,
        "huggingface": HuggingFaceProvider.DEFAULT_MODELS,
    }

    for name in list_providers():
        models = provider_info.get(name, [])
        table.add_row(name, ", ".join(models))

    console.print(table)


@main.command("list-evaluators")
def list_evaluators_cmd():
    """List evaluation dimensions."""
    from llm_eval_forge.evaluators import list_evaluators

    console = Console()
    table = Table(title="Evaluation Dimensions")
    table.add_column("Dimension", style="cyan")

    for dim in list_evaluators():
        table.add_row(dim)

    console.print(table)


@main.command()
@click.argument("report_path")
def report(report_path):
    """Display a summary from an existing JSON report."""
    import json
    from pathlib import Path

    console = Console()
    path = Path(report_path)
    if not path.exists():
        console.print(f"[red]Report not found: {path}[/red]")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    console.print(f"\n[bold]Report: {path.name}[/bold]")
    console.print(f"Total evaluations: {data['metadata']['total_evaluations']}")
    console.print(f"Overall average: {summary.get('overall_average', 'N/A')}")

    if "by_provider" in summary:
        table = Table(title="Scores by Provider")
        table.add_column("Provider", style="cyan")
        table.add_column("Score", justify="center")
        for prov, score in summary["by_provider"].items():
            table.add_row(prov, f"{score:.4f}")
        console.print(table)

    if "by_dimension" in summary:
        table = Table(title="Scores by Dimension")
        table.add_column("Dimension", style="cyan")
        table.add_column("Score", justify="center")
        for dim, score in summary["by_dimension"].items():
            table.add_row(dim, f"{score:.4f}")
        console.print(table)
