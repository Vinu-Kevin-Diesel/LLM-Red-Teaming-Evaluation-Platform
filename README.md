# LLM Eval Forge

Modular LLM evaluation and red-teaming framework that stress-tests language models across hallucination, instruction-following, reasoning consistency, and adversarial robustness using automated judge scoring and mutation-based attacks.

## Features

- **4 Evaluation Dimensions**: Hallucination detection, instruction-following, reasoning consistency, adversarial robustness
- **3 Free LLM Providers**: Groq (Llama, Mixtral, Gemma), Kimi K2.5 via NVIDIA NIM (Moonshot AI), HuggingFace Inference API
- **Claude Judge**: Automated scoring with weighted rubrics via Anthropic's Claude API
- **6 Red-Team Strategies**: Role-play injection, encoding attacks, instruction smuggling, context manipulation, few-shot poisoning, semantic tricks
- **YAML-Configurable**: All providers, models, dimensions, and strategies controlled by config files
- **Rich CLI**: Terminal output with color-coded score tables and latency stats
- **JSON Reports**: Structured output with per-provider, per-dimension breakdowns

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Set API Keys

```bash
cp .env.example .env
# Edit .env with your API keys:
# GROQ_API_KEY=gsk_...
# NVIDIA_API_KEY=nvapi-...
# HF_TOKEN=hf_...
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run

```bash
# Full evaluation suite
llm-eval-forge run --config config/default.yaml

# Single provider, single dimension
llm-eval-forge run --provider groq --dimension hallucination

# Red-team tests only
llm-eval-forge run --redteam-only

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
| `--output PATH` | Output directory for results |
| `--verbose` | Enable debug logging |
| `--dry-run` | Show what would run without executing |

## Architecture

```
src/llm_eval_forge/
├── cli.py              # Click CLI entry point
├── config.py           # YAML config loading
├── runner.py           # Async evaluation orchestrator
├── providers/          # LLM provider abstractions (Groq, Kimi, HuggingFace)
├── evaluators/         # Dimension-specific evaluators
├── redteam/            # 6 adversarial mutation strategies
├── judge/              # Claude-based scoring with weighted rubrics
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
- `judge` - Claude model and settings for scoring
- `redteam` - Which mutation strategies to use
- `rate_limits` - Per-provider rate limiting

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Tech Stack

Python, Claude API (Anthropic), OpenAI SDK (Groq/Kimi), HuggingFace Hub, Click, Rich, PyYAML
