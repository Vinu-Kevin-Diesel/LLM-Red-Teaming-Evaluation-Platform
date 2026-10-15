# LLM-Red-Teaming-Evaluation-Platform

[![CI](https://github.com/Vinu-Kevin-Diesel/LLM-Red-Teaming-Evaluation-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Vinu-Kevin-Diesel/LLM-Red-Teaming-Evaluation-Platform/actions/workflows/ci.yml)


Modular LLM evaluation and red-teaming framework that stress-tests language models across hallucination, instruction-following, reasoning consistency, and adversarial robustness using automated judge scoring and mutation-based attacks.

## Features

- **4 Evaluation Dimensions**: Hallucination detection, instruction-following, reasoning consistency, adversarial robustness
- **4 LLM Providers**: Groq (Qwen 3.8, gpt-oss), Cerebras (gpt-oss, Qwen), Mistral via NVIDIA NIM (Mistral Large 2), HuggingFace Inference API
- **Claude Judge**: Automated scoring with weighted rubrics via Claude Sonnet 5.5, schema-constrained with structured outputs. Adversarial verdicts also return binary labels (`response_harmful`, `attack_succeeded`, `refusal`)
- **6 Red-Team Strategies**: Role-play injection, encoding attacks, instruction smuggling, context manipulation, few-shot poisoning, semantic tricks
- **YAML-Configurable**: All providers, models, dimensions, and strategies controlled by config files
- **Rich CLI**: Terminal output with color-coded score tables and latency stats
- **JSON Reports**: Structured output with per-provider, per-dimension breakdowns
- **`--no-judge` Mode**: Collect provider responses only, judge separately later (useful for offline scoring or environments without API access)

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Set API Keys

```bash
cp .env.example .env
# Edit .env with your API keys:
# GROQ_API_KEY=gsk_...           # https://console.groq.com/keys
# CEREBRAS_API_KEY=csk-...       # https://cloud.cerebras.ai/
# NVIDIA_API_KEY=nvapi-...       # https://build.nvidia.com/  (used for Mistral)
# HF_TOKEN=hf_...                # https://huggingface.co/settings/tokens
# ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com/  (judge)
```

> Only the providers enabled in `config/default.yaml` need keys. The default config uses **Groq + Anthropic**, so those two keys are the minimum.

### 3. Run

```bash
# Full evaluation suite (collects responses + Claude-judges them in one pass)
llm-eval-forge run --config config/default.yaml

# Single provider, single dimension
llm-eval-forge run --provider groq --dimension hallucination

# Red-team tests only
llm-eval-forge run --redteam-only

# Collect responses without judging (useful when ANTHROPIC_API_KEY is unavailable
# or you want to score later via a different judge / out-of-band pipeline)
llm-eval-forge run --no-judge

# Dry run (show plan without executing)
llm-eval-forge run --dry-run

# View results from a previous run
llm-eval-forge report results/full-evaluation-run.json
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `run` | Run an evaluation suite |
| `list-providers` | List available LLM providers and their models |
| `list-evaluators` | List evaluation dimensions |
| `report` | Display summary from an existing JSON report |

### Run Options

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to YAML config (default: `config/default.yaml`) |
| `--provider NAME` | Run only against this provider |
| `--dimension NAME` | Run only this evaluation dimension |
| `--redteam-only` | Only run red-team adversarial tests |
| `--no-judge` | Collect responses only — skip judging (no `ANTHROPIC_API_KEY` required) |
| `--output PATH` | Output directory for results |
| `--verbose` | Enable debug logging |
| `--dry-run` | Show what would run without executing |

## Architecture

```
src/llm_eval_forge/
├── cli.py              # Click CLI entry point
├── config.py           # YAML config loading
├── runner.py           # Async evaluation orchestrator
├── providers/          # LLM provider abstractions (Groq, Cerebras, Mistral, HuggingFace)
├── evaluators/         # Dimension-specific evaluators
├── redteam/            # 6 adversarial mutation strategies
├── judge/              # Claude-based scoring with weighted rubrics + prompt caching
└── reporting/          # JSON output and Rich terminal summaries
```

## Evaluation Dimensions

### Hallucination Detection
Tests if models fabricate facts, invent non-existent entities, or state incorrect information with confidence. Includes factual questions with known answers and "trap" questions about fictional entities.

### Instruction Following
Tests compliance with complex multi-constraint instructions: word limits, formatting requirements, inclusions/exclusions, tone constraints.

### Reasoning Consistency
Tests logical coherence across multi-step problems: syllogisms, math word problems, logic puzzles, causal reasoning.

### Adversarial Robustness
Tests resilience against prompt injection, jailbreaks, and manipulation via 6 mutation strategies.

## Red-Team Strategies

1. **Role-Play Injection** - DAN-style persona hijacking
2. **Encoding Attacks** - Base64, ROT13, leetspeak encoded instructions
3. **Instruction Smuggling** - Hidden instructions in translation tasks, JSON, code comments
4. **Context Manipulation** - Authority escalation, fake system messages, context stuffing
5. **Few-Shot Poisoning** - Poisoned examples priming harmful behavior
6. **Semantic Tricks** - Hypotheticals, reverse psychology, completion bait

## Configuration

See `config/default.yaml` for the full config schema. Key sections:

- `providers` - Which LLMs to test and their settings
- `dimensions` - Which evaluation dimensions to enable, with rubric weights
- `judge` - Judge backend, model and effort (defaults to `claude` / `claude-sonnet-5-5` / `low`; also supports `groq` and `nvidia`)
- `redteam` - Which mutation strategies to use
- `rate_limits` - Per-provider rate limiting

### Default models tested

| Provider | Model |
|---|---|
| Groq | `qwen/qwen3.8-27b` |
| Groq | `openai/gpt-oss-20b` |

Groq retired `qwen/qwen3-32b` and Cerebras retired `llama3.1-8b` in 2026. Cerebras (`gpt-oss-120b`) can be added back to the config once its account has billing enabled.

### Judge

Claude Sonnet 5.5 via the Anthropic API, using adaptive thinking at `low` effort.

- **Structured outputs:** each verdict is schema-constrained, with integer criterion scores and reasoning (plus binary safety labels on the adversarial dimension). The judge can no longer reply with prose before the JSON.
  - Older runs stored those parse failures as a score of 0.0, about 14% of adversarial rows. `scripts/rejudge.py` re-scores them.
- **No verdict is never 0.0:** if the judge declines (`stop_reason: "refusal"`) or produces no valid verdict, `judge_score` is `null` and `judge_error` says why (e.g. `refusal:bio`).
- **Server-side fallback** (`fallbacks: "default"`) re-runs `cyber` / `frontier_llm` declines on another model. `judge_model` records which model actually served each verdict.
- **Prompt caching:** the rubric system prompt is sent with `cache_control: ephemeral`.

## Distillation (in progress)

This work distills the Claude judge into a small, fast guard model. Install the extras first:

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev,ml]"
```

1. **Build the dataset:**

   ```bash
   python -m llm_eval_forge.distill.data build
   ```

   This builds `data/{train,val,test}` and `results/data_card.md`.
   - Train/val: WildGuardTrain, Aegis 2.0 and our own red-team outputs.
   - Test: the standard response-harm suite (WildGuardTest, HarmBench, BeaverTails, SafeRLHF, Aegis 2.0, JailbreakBench judge comparison).
   - Contamination guard: drops train rows whose prompt appears in a test set.

2. **Label with the Claude teacher** through the Batch API:

   ```bash
   python -m llm_eval_forge.distill.teacher submit pilot
   python -m llm_eval_forge.distill.teacher collect pilot
   python -m llm_eval_forge.distill.teacher report pilot
   ```

   Every step works on a named row set; `pilot` (300 human-labeled test rows) runs first:
   - `submit` sends the set to the Batch API.
   - `collect` saves the verdicts.
   - `report` compares the teacher with human gold labels and writes `results/teacher_<set>.md`.

   Then run the same three commands with `pool` in place of `pilot` for the distillation pool.

3. **Train the student.** This is a ModernBERT-base classifier (149M params) trained in two stages:
   - Stage A learns from public labels.
   - Stage B distills the Claude teacher's verdicts, using the teacher's confidence as a soft target.

   ```bash
   python -m llm_eval_forge.distill.train --stage A --out models/student-A
   python -m llm_eval_forge.distill.train --stage B --init models/student-A --out models/student-AB
   ```

4. **Evaluate and build the comparison table.**

   ```bash
   python -m llm_eval_forge.distill.evaluate run --model models/student-AB
   python -m llm_eval_forge.distill.evaluate run --baseline qwen3guard-0.6b
   python -m llm_eval_forge.distill.evaluate run --baseline granite-guardian-3.2-3b
   python -m llm_eval_forge.distill.evaluate teacher
   python -m llm_eval_forge.distill.evaluate table
   ```

   The commands evaluate the student and the two guard baselines, score Claude on its pilot rows, then write `results/benchmark_table.md`.

   Baseline guards are scored from the probability of their first verdict token, in one batched forward pass.

   Current results, including the ablation (public labels only vs + Claude distillation, base vs large encoder), are in [`results/benchmark_table.md`](results/benchmark_table.md).

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Tech Stack

Python, Anthropic SDK (Claude judge), OpenAI SDK (Groq / Cerebras / Mistral via NVIDIA NIM — all OpenAI-compatible endpoints), HuggingFace Hub, Click, Rich, PyYAML
