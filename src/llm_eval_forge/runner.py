"""Evaluation runner — orchestrates providers, evaluators, judge, and reporting."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from llm_eval_forge.config import EvalConfig, ProviderConfig
from llm_eval_forge.evaluators.base import EvalResult, TestCase
from llm_eval_forge.evaluators.registry import get_evaluator
from llm_eval_forge.judge.claude_judge import LLMJudge
from llm_eval_forge.judge.rubrics import get_rubric
from llm_eval_forge.providers.base import LLMProvider
from llm_eval_forge.providers.registry import get_provider
from llm_eval_forge.redteam.engine import MutationEngine
from llm_eval_forge.reporting.json_report import save_json_report
from llm_eval_forge.reporting.summary import print_summary

logger = logging.getLogger(__name__)


class EvalRunner:
    """Orchestrates the full evaluation pipeline."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.judge = LLMJudge(
            backend=config.judge.provider,
            model=config.judge.model,
        )
        self.results: list[EvalResult] = []

        # Set up rate limit semaphores per provider
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._delays: dict[str, float] = {}
        for name, rl in config.rate_limits.items():
            self._semaphores[name] = asyncio.Semaphore(max(1, rl.requests_per_minute // 10))
            self._delays[name] = rl.delay_between_requests_ms / 1000.0

    async def run(
        self,
        provider_filter: str | None = None,
        dimension_filter: str | None = None,
        redteam_only: bool = False,
        dry_run: bool = False,
    ) -> list[EvalResult]:
        """Run the evaluation suite."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine which dimensions to run
        dimensions = [
            d for d in self.config.dimensions
            if d.enabled and (not dimension_filter or d.name == dimension_filter)
        ]
        if redteam_only:
            dimensions = [d for d in dimensions if d.name == "adversarial"]

        if not dimensions:
            logger.warning("No dimensions to evaluate")
            return []

        # Determine which providers to test
        providers_cfg = [
            p for p in self.config.providers
            if not provider_filter or p.name == provider_filter
        ]

        if not providers_cfg:
            logger.warning("No providers to evaluate")
            return []

        if dry_run:
            self._print_dry_run(providers_cfg, dimensions)
            return []

        # Run evaluations
        for dim_cfg in dimensions:
            evaluator = get_evaluator(dim_cfg.name)
            rubric = get_rubric(dim_cfg.name)

            # Load test cases
            test_cases = evaluator.load_test_cases(dim_cfg.test_cases)
            logger.info(f"Loaded {len(test_cases)} test cases for {dim_cfg.name}")

            # If adversarial + redteam enabled, apply mutations
            if dim_cfg.name == "adversarial" and self.config.redteam.enabled:
                engine = MutationEngine(self.config.redteam.strategies)
                expanded_cases = []
                for tc in test_cases:
                    expanded_cases.append(tc)  # keep original
                    mutations = engine.generate_attacks(
                        tc.prompt,
                        tc.metadata.get("target_behavior", ""),
                        max_per_strategy=self.config.redteam.mutations_per_prompt,
                    )
                    for i, m in enumerate(mutations):
                        expanded_cases.append(TestCase(
                            id=f"{tc.id}_mut_{m.strategy}_{i}",
                            prompt=m.mutated,
                            metadata={
                                **tc.metadata,
                                "mutation_strategy": m.strategy,
                                "mutation_description": m.description,
                                "original_prompt": m.original,
                            },
                        ))
                test_cases = expanded_cases
                logger.info(f"Expanded to {len(test_cases)} cases after red-team mutations")

            # Run each provider x model x test case
            for pcfg in providers_cfg:
                models = pcfg.models or ["default"]
                for model_name in models:
                    provider = get_provider(pcfg.name, model=model_name)
                    logger.info(f"Evaluating {pcfg.name}/{model_name} on {dim_cfg.name}")

                    for tc in test_cases:
                        try:
                            r = await self._eval_single(
                                provider, pcfg, evaluator, rubric, tc, dim_cfg.name
                            )
                            self.results.append(r)
                        except Exception as e:
                            logger.error(f"Evaluation error: {e}")

        # Save and display results
        report_path = output_dir / f"{self.config.name}.json"
        save_json_report(self.results, report_path)
        print_summary(self.results)

        return self.results

    async def _eval_single(
        self,
        provider: LLMProvider,
        pcfg: ProviderConfig,
        evaluator,
        rubric,
        test_case: TestCase,
        dimension: str,
    ) -> EvalResult:
        """Evaluate a single test case against a provider."""
        sem = self._semaphores.get(pcfg.name, asyncio.Semaphore(5))
        delay = self._delays.get(pcfg.name, 1.0)

        async with sem:
            # Get LLM response
            try:
                response = await provider.complete(
                    prompt=test_case.prompt,
                    system=test_case.system,
                    temperature=pcfg.temperature,
                    max_tokens=pcfg.max_tokens,
                )
            except Exception as e:
                logger.error(f"Provider error ({pcfg.name}): {e}")
                return EvalResult(
                    test_case_id=test_case.id,
                    dimension=dimension,
                    provider=pcfg.name,
                    model=provider.model,
                    prompt_sent=test_case.prompt,
                    response_text=f"ERROR: {e}",
                    metadata=test_case.metadata,
                )

            # Delay before judge call to respect rate limits
            await asyncio.sleep(delay)

            # Judge the response
            context = evaluator.build_judge_payload(test_case, response.text)
            try:
                verdict = self.judge.score(
                    dimension=dimension,
                    prompt=test_case.prompt,
                    response=response.text,
                    rubric=rubric,
                    context=context,
                )
            except Exception as e:
                logger.error(f"Judge error: {e}")
                verdict = None

            result = EvalResult(
                test_case_id=test_case.id,
                dimension=dimension,
                provider=pcfg.name,
                model=response.model,
                prompt_sent=test_case.prompt,
                response_text=response.text,
                latency_ms=response.latency_ms,
                judge_score=verdict.weighted_score if verdict else None,
                judge_reasoning=verdict.reasoning if verdict else None,
                judge_scores_detail=verdict.scores if verdict else {},
                metadata=test_case.metadata,
            )

            await asyncio.sleep(delay)
            return result

    def _print_dry_run(self, providers, dimensions):
        """Print what would be run without executing."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Dry Run — Evaluation Plan")
        table.add_column("Dimension")
        table.add_column("Provider")
        table.add_column("Models")
        table.add_column("Test Cases File")

        for dim in dimensions:
            for pcfg in providers:
                models = ", ".join(pcfg.models) if pcfg.models else "default"
                table.add_row(dim.name, pcfg.name, models, dim.test_cases)

        console.print(table)
