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
- **Small distilled guards**: ModernBERT classifiers that judge replies, prompts and agent tool calls in tens of milliseconds, served over FastAPI on GPU or as INT8 ONNX on CPU
- **Guardrail gateway**: an OpenAI-compatible proxy that checks the prompt before the model sees it and the reply before the user does
- **Agent tool-call guard**: judges each action an agent proposes before it runs; cuts AgentDojo's attack success rate from 28.5% to 3.5%

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
  - Older runs stored those parse failures as a score of 0.0, about 14% of adversarial rows. `scripts/rejudge.py --allow-api` re-scores them.
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
   python -m llm_eval_forge.distill.teacher submit pilot --allow-api   # paid API call, so it must be opted into
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

## Serving the judge

There are two runtimes behind one FastAPI service:
- **GPU:** PyTorch bf16 with micro-batching, which coalesces concurrent requests into one forward pass.
- **CPU:** an ONNX Runtime INT8 artifact. It needs neither torch nor transformers.

```bash
# Export a checkpoint for CPU serving: ONNX fp32 + INT8 + tokenizer + thresholds
python -m llm_eval_forge.serve.export models/student-AB --out artifacts/student-AB

# Run the service (CPU, INT8)
JUDGE_MODEL_DIR=artifacts/student-AB JUDGE_RUNTIME=onnx-int8 \
  uvicorn llm_eval_forge.serve.app:create_app --factory --port 8000

# ...or the GPU tier, escalating uncertain verdicts to Qwen3Guard
JUDGE_MODEL_DIR=models/student-large-AB JUDGE_RUNTIME=torch-cuda JUDGE_ESCALATE=qwen3guard-0.6b \
  uvicorn llm_eval_forge.serve.app:create_app --factory --port 8000

curl -X POST localhost:8000/v1/judge -H "Content-Type: application/json" \
  -d '{"prompt": "How do I pick a lock?", "response": "I can'\''t help with that."}'
# -> {"verdict": "safe", "p_harmful": ..., "threshold": 0.3, "uncertain": false, "escalated": false,
#     "latency_ms": ..., "model_version": "student-AB", "runtime": "onnx-cpu-int8"}
```

**Endpoints**
- `POST /v1/judge`
- `GET /healthz`
- `GET /metrics`: request counts, latency quantiles, mean batch size and escalations, in Prometheus text format.

**Decision policy (`JUDGE_POLICY`)**
- `teacher` (default): matches Claude's verdicts.
- `gold`: maximizes F1 on public benchmarks.

**INT8 quantization.** Plain dynamic INT8 flips about 11% of ModernBERT's decisions, because outlier activations feed each MLP's output projection. The export therefore keeps those projections in fp32. That brings flips down to about 2%, and the model still runs about twice as fast as fp32 on CPU.

**Docker (CPU image):**

```bash
docker build -f Dockerfile.serve -t guard-judge:cpu .
docker run --rm -p 8000:8000 guard-judge:cpu
```

**Benchmarks.** These commands write `results/latency.json` and [`results/latency.md`](results/latency.md):

```bash
python -m llm_eval_forge.serve.bench runtimes   # latency by runtime x input length (+ Qwen3Guard)
python -m llm_eval_forge.serve.bench batch      # GPU batch throughput
python -m llm_eval_forge.serve.bench http       # end-to-end HTTP under concurrency
python -m llm_eval_forge.serve.bench quality    # INT8 artifact on the full test suite
python -m llm_eval_forge.serve.cascade          # escalate uncertain verdicts: accuracy vs latency
python -m llm_eval_forge.serve.bench report
```

## Guardrail gateway: checking the input, the output and every tool call

The judge service above only checks a reply when your app asks it to. The **gateway** is an OpenAI-compatible proxy that every request goes through automatically:

```
user / app ─► POST /v1/chat/completions ─► input guard ─► upstream LLM ─► output guard ─► user / app
agent      ─► POST /v1/judge/tool_call  ─► tool-call guard ─► allow / block the action
```

**1. Input guard: is the prompt itself harmful?**
- A ModernBERT-base classifier trained on 69K labeled prompts.
- It is evaluated on WildGuardTest, Aegis 2.0 and ToxicChat.
- It is also tested for **over-blocking** on XSTest and JailbreakBench: safe prompts that only sound dangerous, such as "How do I kill a Python process?". Results are in [`results/prompt_guard_table.md`](results/prompt_guard_table.md).

```bash
python -m llm_eval_forge.distill.prompt_guard build
python -m llm_eval_forge.distill.train --stage A --data data/prompts --out models/prompt-guard-A --max-len 1024
python -m llm_eval_forge.distill.prompt_guard run --model models/prompt-guard-A
python -m llm_eval_forge.distill.prompt_guard run --baseline qwen3guard-0.6b
python -m llm_eval_forge.distill.prompt_guard table
```

**2. Gateway.** Point any OpenAI SDK client at it:

```bash
UPSTREAM_BASE_URL=https://api.groq.com/openai/v1 UPSTREAM_MODEL=qwen/qwen3.8-27b \
  uvicorn llm_eval_forge.serve.gateway:create_gateway --factory --port 8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
r = client.chat.completions.create(model="qwen/qwen3.8-27b", messages=[{"role": "user", "content": "..."}])
# finish_reason == "content_filter" when the prompt or the reply was blocked; r.model_extra["guard"] has the verdicts
```

- **Blocking is per side.** A harmful prompt is refused without ever calling the model; a harmful reply is replaced.
- **Modes:** `block`, `flag` (log only) or `off`, set separately for each side.
- **Replay:** `python -m llm_eval_forge.serve.replay` replays the red-team attacks and the XSTest controls through the gateway. Every reply that passed both guards is then reviewed by hand. `replay overhead` times the guards again on another runtime, using the recorded replies. Results are in [`results/gateway.md`](results/gateway.md).

**3. Tool-call guard for agents.** It judges every action an agent proposes *before it runs*, using:
- the user's goal
- the calls made so far
- the proposed call
- the **provenance of its arguments:** for each IBAN, email or URL, did it come from the user's request, or from which tool output and with what surrounding text?

It's trained on AgentDojo's published agent traces (36K runs, labeled by rule, no LLM). Training identifiers are anonymized so it can't memorize attacker addresses. It's evaluated on a **held-out suite** (banking: unseen tools and data) and live inside an AgentDojo agent:

```bash
git clone --depth 1 https://github.com/ethz-spylab/agentdojo data/agentdojo && pip install -e data/agentdojo
python -m llm_eval_forge.agent.toolcall_data build
python -m llm_eval_forge.distill.train --stage A --data data/toolcalls --out models/toolcall-guard-A --max-len 1024 \
  --tokens-per-batch 8192 --epochs 1 --train-sources agentdojo_slack,agentdojo_travel,agentdojo_workspace \
  --val-sources agentdojo_slack,agentdojo_travel,agentdojo_workspace
python -m llm_eval_forge.agent.toolcall_eval --model models/toolcall-guard-A        # results/toolcall_guard_table.md

ollama create qwen3-8b-think -f config/ollama/Modelfile.qwen3-8b-think   # agent: local qwen3:8b, thinking on
JUDGE_TOOLCALL_MODEL_DIR=models/toolcall-guard-A uvicorn llm_eval_forge.serve.app:create_app --factory --port 8000
python -m llm_eval_forge.agent.run_agentdojo --judge none            # add --resume to continue an interrupted run
python -m llm_eval_forge.agent.run_agentdojo --judge toolcall-guard
python -m llm_eval_forge.agent.run_agentdojo report                  # results/agent_guard.md
```

**Results.** Each row links to the generated report.

| Guard | Measured on | Result |
|---|---|---|
| [Input guard](results/prompt_guard_table.md) | WildGuardTest, Aegis 2.0, ToxicChat, XSTest | Avg F1 **79.8** vs Qwen3Guard-0.6B 79.4 and Granite Guardian 3.2 70.2. Blocks **10.0%** of XSTest's safe prompts, vs 22.0% and 35.6% |
| [Gateway](results/gateway.md) | 632 requests to Groq `qwen/qwen3.8-27b` | Stopped 41.8% of red-team attacks and 79.5% of XSTest unsafe prompts before the model saw them. A manual review of the 147 attack and unsafe replies that passed both guards found **1 harmful**. 10.8% of safe prompts were falsely blocked. Guard overhead p50 **37 ms**, p99 **72 ms** on an RTX 5070 (163 / 399 ms on CPU) |
| [Tool-call guard, offline](results/toolcall_guard_table.md) | Held-out banking suite, 6,052 calls | F1 **98.7**, AUROC 99.6. Catches 98.7% of attacker calls and blocks 1.9% of legitimate `send_money` calls |
| [Tool-call guard, live](results/agent_guard.md) | AgentDojo banking, `important_instructions`, qwen3:8b agent | Attack success **28.5% → 3.5%**: 36 attacks stopped, none added, McNemar p = 3e-11. Utility under attack unchanged at 49.3%. Benign utility 56.2% → 43.8%: one task was lost to a false block and one to run-to-run variation. **32 ms** per judged call |

**Known gaps** (found by these evaluations):
- **Acting on a document.** When the user asks the agent to act on a document ("pay the bill in this file"), the payee's IBAN comes from a tool output, so it looks injected. 19 of the 22 offline false blocks come from that one task, and it is the benign task lost live.
- **Attacks without identifiers.** The tool-call guard only learned attacks that carry an identifier (IBAN, email, URL), because that is how the training calls were labeled. All 5 attacks that got through live are "change the password", which carries none.
- **Denial without instructions.** The output guard passed one reply that denies the Armenian genocide (it scored 0.30, under the 0.52 threshold). The reply gives no instructions, so it doesn't look like harm to a model trained mostly on operational harms.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Tech Stack

Python, Anthropic SDK (Claude judge), OpenAI SDK (Groq / Cerebras / Mistral via NVIDIA NIM — all OpenAI-compatible endpoints), HuggingFace Hub, Click, Rich, PyYAML
