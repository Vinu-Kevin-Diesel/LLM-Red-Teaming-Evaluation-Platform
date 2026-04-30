# LLM-Red-Teaming-Evaluation-Platform


Modular LLM evaluation and red-teaming framework that stress-tests language models across hallucination, instruction-following, reasoning consistency, and adversarial robustness using automated judge scoring and mutation-based attacks.

## Features

- **4 Evaluation Dimensions**: Hallucination detection, instruction-following, reasoning consistency, adversarial robustness
- **4 Free LLM Providers**: Groq (Qwen, Llama), Cerebras (Llama 3.1), Mistral via NVIDIA NIM (Mistral Medium 3.5), HuggingFace Inference API
- **Claude Judge**: Automated scoring with weighted rubrics via Anthropic's Claude Sonnet 4.6 API (with prompt caching on rubric system prompts)
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

> Only the providers enabled in `config/default.yaml` need keys. The default config uses **Groq + Cerebras + Anthropic**, so those three keys are the minimum.

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
- `judge` - Judge backend and model (defaults to `claude` / `claude-sonnet-4-6`; also supports `groq` and `nvidia`)
- `redteam` - Which mutation strategies to use
- `rate_limits` - Per-provider rate limiting

### Default models tested

| Provider | Model | Free-tier quota |
|---|---|---|
| Groq | `qwen/qwen3-32b` | 500K tokens/day |
| Cerebras | `llama3.1-8b` | 1M tokens/day |

### Judge

Claude Sonnet 4.6 via the Anthropic API. The rubric system prompt is sent with `cache_control: ephemeral` so subsequent calls within the same dimension read from cache (~10% of base input cost).

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Tech Stack

Python, Anthropic SDK (Claude judge), OpenAI SDK (Groq / Cerebras / Mistral via NVIDIA NIM — all OpenAI-compatible endpoints), HuggingFace Hub, Click, Rich, PyYAML
