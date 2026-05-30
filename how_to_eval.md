# How to Evaluate — LatentSig Medical Triage Router

Complete guide to evaluating the fine-tuned SLM against baselines.

---

## Table of Contents

1. [Overview](#overview)
2. [Eval Dataset](#eval-dataset)
3. [Engines](#engines)
4. [Quick Start](#quick-start)
5. [Eval Modes](#eval-modes)
6. [Metrics Explained](#metrics-explained)
7. [CLI Reference](#cli-reference)
8. [Interactive Testing](#interactive-testing)
9. [Understanding the Report](#understanding-the-report)
10. [Running on Colab (Full Comparison)](#running-on-colab)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The eval system measures how well the fine-tuned SLM routes medical queries to the correct tool, compared against a Mistral API baseline. Both engines use the **identical system prompt** — the only difference is the model weights.

```
┌─────────────────┐         ┌─────────────────┐
│  SLM (Fine-tuned)│         │ Mistral (Baseline)│
│  Qwen3-4B+QLoRA │         │ mistral-small     │
└────────┬────────┘         └────────┬────────┘
         │                           │
         ▼                           ▼
    Same system prompt          Same system prompt
    Same eval dataset           Same eval dataset
         │                           │
         ▼                           ▼
    Compare metrics ◀─────────── Compare metrics
```

**What we measure:**
- Did the model pick the right tool? (tool accuracy)
- Did it assign the right urgency? (category accuracy)
- Did it output valid JSON? (parse success rate)
- How many retries before success? (hallucination recovery)
- How fast is each inference? (latency)

---

## Eval Dataset

**Location:** `synth-ds-framework/eval_dataset.jsonl` (local) or `fhai50032/latentsig-med-triage-router` split=`eval` (HuggingFace)

| Split | Samples | Languages |
|-------|---------|-----------|
| Train | 2,000 | 1,000 EN + 1,000 Hinglish |
| Eval  | 40    | 20 EN + 20 Hinglish |

**Eval samples are NEVER seen during training.** They were generated with the same pipeline but held out specifically for evaluation.

### Dataset Schema

Each eval sample contains:

```json
{
  "user_query": "68-year-old male, sudden facial droop, cannot speak...",
  "response": "{\"reasoning\": \"...\", \"category\": \"emergency\", \"tool\": \"emergency_dispatch\", ...}",
  "tool_called": "emergency_dispatch",
  "category": "emergency",
  "language": "en",
  "generation_model_id": "mistral-large-latest",
  "hash": "a1b2c3d4e5f6..."
}
```

**Ground truth fields:**
- `tool_called` — the correct tool (primary accuracy metric)
- `category` — the correct urgency level (emergency/urgent/semi_urgent/routine)
- `response` — full JSON that the generation model produced (reference)

### Using Local vs HuggingFace

```bash
# Local file (default in repo)
python -m src.eval --mode mistral --eval-file synth-ds-framework/eval_dataset.jsonl

# HuggingFace (downloads automatically)
python -m src.eval --mode mistral --eval-file hf
```

---

## Engines

### 1. Placeholder (testing only)

Keyword-based routing. No API, no model. Good for testing the eval pipeline itself.

```bash
python -m src.eval --mode placeholder
```

**When to use:** Verifying the eval script works, testing new eval features, CI smoke tests.

### 2. Mistral API (baseline)

Calls Mistral's API with the same system prompt the SLM was trained on. This is the baseline the SLM needs to beat.

```bash
python -m src.eval --mode mistral --mistral-model mistral-small-latest
```

**Available models:**

| Model | Speed | Quality | Cost | Use Case |
|-------|-------|---------|------|----------|
| `mistral-small-latest` | Fast | Good | Low | Default baseline |
| `mistral-medium-latest` | Medium | Better | Medium | Stronger baseline |
| `mistral-large-latest` | Slow | Best | High | Strongest baseline |
| `magistral-medium-latest` | Medium | Good reasoning | Medium | Reasoning tasks |

### 3. SLM Fine-tuned (our model)

The fine-tuned Qwen3-4B with QLoRA adapter. Requires GPU (T4 16GB minimum).

```bash
python -m src.eval --mode slm --adapter-path fhai50032/latentsig-med-router-qwen3-4b
```

**When to use:** After fine-tuning, to measure how well the SLM learned.

### 4. Full Comparison (both)

Runs BOTH engines on the same eval set and generates a side-by-side comparison report.

```bash
python -m src.eval --mode full
```

---

## Quick Start

### No API, no GPU (pipeline test)

```bash
cd latentsig-slm-router-med
python -m src.eval --mode placeholder --limit 5
```

### Mistral baseline (needs API keys)

```bash
# Full 40-sample eval
python -m src.eval --mode mistral

# Quick 10-sample test
python -m src.eval --mode mistral --limit 10

# Use a stronger model
python -m src.eval --mode mistral --mistral-model mistral-large-latest
```

### Full comparison (Colab)

```bash
python -m src.eval --mode full \
    --adapter-path fhai50032/latentsig-med-router-qwen3-4b \
    --mistral-model mistral-small-latest \
    --output eval_results_full.jsonl
```

---

## Eval Modes

### Single-Stage (default)

Runs only Stage 1: the tool call. Measures routing accuracy without the response generation stage.

```bash
python -m src.eval --mode mistral
```

**Flow:**
```
Query → System Prompt + Query → Model → JSON → Parse → Compare vs GT
```

This is faster and isolates the core routing skill.

### Full Agent Loop (`--use-agent`)

Runs the complete two-stage loop: tool call → execute → generate response. Measures end-to-end behavior including retries and fallback.

```bash
python -m src.eval --mode mistral --use-agent
```

**Flow:**
```
Query → Tool Call → Parse → (retry if fail) → Execute Tool → Generate Response
```

This is slower but measures real-world agent behavior.

**Key differences:**
| Metric | Single-Stage | Full Agent |
|--------|-------------|------------|
| Speed | Fast (1 API call/sample) | Slow (2+ API calls/sample) |
| Retry accuracy | Simulated | Real (actual retry prompts) |
| Response quality | Not measured | Measured |
| Tool execution | Not measured | Measured |

---

## Metrics Explained

### Tool Accuracy

```
tool_accuracy = correct_tool_predictions / total_samples
```

The primary metric. Did the model pick the same tool as the ground truth?

- **100%** = perfect routing
- **80%** = 8 out of 10 queries routed correctly
- **Baseline (placeholder)** = ~57% (keyword matching)

### Category Accuracy

```
category_accuracy = correct_category_predictions / total_samples
```

Did the model assign the right urgency level?

Categories: `emergency` > `urgent` > `semi_urgent` > `routine`

Note: Category accuracy is often higher than tool accuracy because multiple tools can share the same category (e.g., both `emergency_dispatch` and `mental_health_triage` are `emergency`).

### Parse Success Rate

```
parse_rate = valid_json_outputs / total_samples
```

What percentage of model outputs were valid JSON with all required fields?

- A "parse failure" means the model output garbage, truncated JSON, or missing fields
- Parse failures trigger hallucination recovery (retries)
- If all retries fail, a safety fallback is applied (emergency category)

### Fallback Rate

```
fallback_rate = fallback_count / total_samples
```

How often did the model fail ALL retries and get the safety fallback?

- **0%** = model always produces valid output (good)
- **High fallback rate** = model struggles with the output format

### Latency

Three percentiles reported:

| Metric | What it measures |
|--------|-----------------|
| **Avg** | Mean latency across all samples |
| **P50** | Median (50th percentile) — typical case |
| **P95** | 95th percentile — worst-case typical |

For Mistral API: ~1-2s per call (network + inference)
For SLM on T4: ~200-500ms per call (local GPU)
For placeholder: ~100ms (simulated)

### Retries

```
avg_retries = total_retries / total_samples
```

How many hallucination recovery attempts per sample?

- **0.00** = model always gets it right on the first try
- **0.50** = half the samples needed one retry
- **3.00** = max retries hit frequently (bad)

### Per-Tool Breakdown

Accuracy for each of the 7 tools individually. Reveals which tools the model struggles with.

Common pattern: `emergency_dispatch` has highest accuracy (clear symptoms), `specialist_referral` has lowest (ambiguous routing).

### Per-Language Breakdown

EN vs Hinglish accuracy. Reveals if the model handles code-mixed input as well as pure English.

### Confusion Matrix

Shows which tools get confused for which. Example:
```
triage_assessment → specialist_referral (4x)
```
This means the model chose `triage_assessment` when the correct answer was `specialist_referral` — 4 times.

---

## CLI Reference

### `src.eval` — Batch Evaluation

```
python -m src.eval [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `placeholder` | Engine mode: `placeholder`, `mistral`, `slm`, `full` |
| `--eval-file` | `hf` | Path to eval JSONL or `hf` for HuggingFace |
| `--api-keys` | auto | Comma-separated Mistral API keys |
| `--mistral-model` | `mistral-small-latest` | Mistral model name |
| `--adapter-path` | `fhai50032/latentsig-med-router-qwen3-4b` | SLM LoRA adapter path |
| `--base-model` | `unsloth/Qwen3-4B-Instruct` | SLM base model |
| `--max-retries` | `3` | Max parse retries per sample |
| `--use-agent` | `false` | Use full two-stage agent loop |
| `--output` | auto | Output path for results JSONL |
| `--limit` | `None` | Limit number of eval samples |

### `src.test_engine` — Interactive Testing

```
python -m src.test_engine [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--engine` | `mistral` | Engine: `placeholder`, `mistral`, `slm` |
| `--model` | `mistral-small-latest` | Mistral model name |
| `--api-keys` | auto | Comma-separated API keys |
| `--adapter-path` | `fhai50032/latentsig-med-router-qwen3-4b` | SLM adapter path |
| `--query` | `None` | Single query (non-interactive) |
| `--max-retries` | `3` | Max parse retries |

---

## Interactive Testing

### Single Query

```bash
python -m src.test_engine --engine mistral --query "chest pain, 55yo male"
```

### Interactive Mode

```bash
python -m src.test_engine --engine mistral
```

Then type queries at the prompt:
```
  Query #1> 68yo male, sudden facial droop, cannot speak
  Query #2> 23yo bacha, bhot tej bukhar 102F, 1 week se
  Query #3> quit
```

### Output

```
======================================================================
  QUERY: 68yo male, sudden facial droop, cannot speak
======================================================================

  [INPUT] (attempt 0)
    Query: 68yo male, sudden facial droop, cannot speak

  [THOUGHT] (attempt 1)
    Reasoning: {"reasoning": "Classic stroke presentation...", ...}
    Latency: 1405ms

  [ACTION] (attempt 1)
    Tool: emergency_dispatch
    Args: {"condition": "acute stroke", "symptoms": ["facial droop", "aphasia"], ...}

  [OBSERVATION] (attempt 1)
    Result: {"status": "dispatched", "eta_minutes": 8, "dispatch_id": "EMD-CEF6639F"}

  [FINAL] (attempt 1)
    Answer: [EMERGENCY] Department: Stroke Unit
    Reasoning: Classic stroke presentation requiring immediate emergency dispatch.
    Latency: 1421ms

──────────────────────────────────────────────────────────────────────
  RESULT SUMMARY
──────────────────────────────────────────────────────────────────────
  Success:    True
  Fallback:   False
  Attempts:   1
  Tool:       emergency_dispatch
  Category:   emergency
  Latency:    1421ms
  Call ID:    EMD-CEF6639F
──────────────────────────────────────────────────────────────────────
```

### Testing Different Models

```bash
# Default (mistral-small-latest)
python -m src.test_engine --engine mistral --query "chest pain"

# Stronger model
python -m src.test_engine --engine mistral --model mistral-large-latest --query "chest pain"

# Placeholder (no API)
python -m src.test_engine --engine placeholder --query "chest pain"

# SLM (needs GPU)
python -m src.test_engine --engine slm --query "chest pain"
```

---

## Understanding the Report

### Report Structure

```
========================================================================
  LatentSig Medical Triage Router — Eval Report
========================================================================

Metric                                     mistral    slm-finetuned
──────────────────────────────────────────────────────────────────────
Tool Accuracy                                80.0%          92.5%
Category Accuracy                            87.5%          95.0%
Parse Success Rate                          100.0%         100.0%
Fallback Rate                                 0.0%           0.0%
──────────────────────────────────────────────────────────────────────
Avg Latency (ms)                              1450            320
P50 Latency (ms)                              1491            310
P95 Latency (ms)                              1781            450
──────────────────────────────────────────────────────────────────────
Avg Retries                                   0.00           0.00
Total Retries                                    0              0
──────────────────────────────────────────────────────────────────────
Total Samples                                   40             40

──────────────────────────────────────────────────────────────────────
  WINNER SUMMARY
──────────────────────────────────────────────────────────────────────
  Tool accuracy:  slm-finetuned (92.5%)
  Category acc:   slm-finetuned (95.0%)
  Parse rate:     slm-finetuned (100.0%)
  Latency:        slm-finetuned (320ms)
```

### Reading the Per-Tool Table

```
  Tool                          Correct    Total      Acc
  ──────────────────────────── ──────── ──────── ────────
  emergency_dispatch                 16       16  100.0%    ← perfect
  medication_check                    5        6   83.3%    ← 1 mistake
  specialist_referral                 1        5   20.0%    ← struggles here
```

- `specialist_referral` at 20% means the model rarely routes to this tool correctly
- This is expected — specialist routing is the hardest tool to learn

### Reading the Confusion Matrix

```
  triage_assessment → specialist_referral (4x)
```

This means:
- Ground truth was `specialist_referral`
- Model predicted `triage_assessment`
- Happened 4 times

**Why?** Both tools deal with non-emergency patient routing. The model defaults to `triage_assessment` (the "front desk" tool) when unsure.

---

## Running on Colab

### Full Comparison (SLM + Mistral)

```python
# In a Colab cell:
!pip install unsloth datasets

# Clone repo
!git clone https://github.com/IsNoobgrammer/latentsig-slm-router-med.git
%cd latentsig-slm-router-med

# Run full comparison
!python -m src.eval --mode full \
    --adapter-path fhai50032/latentsig-med-router-qwen3-4b \
    --mistral-model mistral-small-latest \
    --eval-file synth-ds-framework/eval_dataset.jsonl \
    --output eval_results_full.jsonl
```

### SLM Only (after fine-tuning)

```python
!python -m src.eval --mode slm \
    --adapter-path ./adapter \
    --eval-file synth-ds-framework/eval_dataset.jsonl \
    --output eval_results_slm.jsonl
```

### With Agent Loop (end-to-end)

```python
!python -m src.eval --mode full \
    --adapter-path fhai50032/latentsig-med-router-qwen3-4b \
    --use-agent \
    --output eval_results_agent.jsonl
```

---

## Troubleshooting

### "No Mistral API keys found"

The eval script auto-loads keys from `~\AppData\Local\hermes\api_keys.md`. If that doesn't work:

```bash
# Pass keys directly
python -m src.eval --mode mistral --api-keys KEY1,KEY2,KEY3

# Or set env variable
export MISTRAL_API_KEYS="KEY1,KEY2,KEY3"
python -m src.eval --mode mistral
```

### "CUDA out of memory" (SLM mode)

The SLM needs ~5-6GB VRAM. T4 (16GB) works fine. If OOM:

```bash
# Use 4-bit quantization (default)
python -m src.eval --mode slm --adapter-path ./adapter
```

### Eval hangs during API calls

Mistral API has rate limits. The engine rotates across 8 keys automatically. If still rate-limited:

```bash
# Use fewer samples
python -m src.eval --mode mistral --limit 10

# Or use a faster model
python -m src.eval --mode mistral --mistral-model mistral-small-latest
```

### "Parse error" on every sample

The model isn't producing valid JSON. Check:
1. System prompt is correct (should auto-load from `src/prompts.py`)
2. Model is actually loaded (not returning garbage)
3. Temperature is low enough (0.1 default)

### Results look wrong

Check the raw output JSONL:
```bash
head -5 eval_results_mistral.jsonl
```

Each line has: `engine`, `query`, `gt_tool`, `pred_tool`, `tool_correct`, `latency_ms`, `error`.

---

## Output Files

| File | Description |
|------|-------------|
| `eval_results_<mode>.jsonl` | Raw per-sample results (one JSON per line) |
| `eval_results_<mode>_report.txt` | Human-readable report with all tables |
| `tool_logs/*.csv` | Per-tool CSV logs from tool execution (when using `--use-agent`) |

### JSONL Schema

```json
{
  "engine": "mistral",
  "query": "68yo male, sudden facial droop...",
  "language": "en",
  "gt_tool": "emergency_dispatch",
  "gt_category": "emergency",
  "pred_tool": "emergency_dispatch",
  "pred_category": "emergency",
  "tool_correct": true,
  "category_correct": true,
  "parse_success": true,
  "is_fallback": false,
  "retry_count": 0,
  "latency_ms": 1450.2,
  "error": ""
}
```

Use this for custom analysis, plotting, or CI integration.

---

## File Reference

| File | Purpose |
|------|---------|
| `src/eval.py` | Main eval script — batch evaluation, metrics, report generation |
| `src/test_engine.py` | Interactive query tester — single queries, verbose output |
| `src/inference.py` | Engine implementations (Placeholder, MistralAPI, Unsloth) |
| `src/prompts.py` | System prompt (shared by all engines) |
| `src/parser.py` | JSON extraction + Pydantic validation |
| `src/agent.py` | ReAct agent loop (used with `--use-agent`) |
| `synth-ds-framework/eval_dataset.jsonl` | Local eval dataset (40 samples) |
| `synth-ds-framework/gen_eval.py` | Eval dataset generator |
