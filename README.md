# LatentSig Medical Triage Router

> A local-first agentic workflow where a fine-tuned Small Language Model (SLM) acts as a reliable structured router for clinical medical triage.

[![Training](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fakehai/training-latentsig-slm-router)
[![Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fakehai/latentsig-slm-router-agent)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-Report-FFBE00?logo=weightsandbiases)](https://wandb.ai/ablations-tinycompany-ai/latentsig-med-triage-router/reports/LatentSig-SLM-Router--VmlldzoxNzA2MzQ3OA)
[![HF Dataset](https://img.shields.io/badge/🤗_HuggingFace-Dataset-FFD21E?logo=huggingface)](https://huggingface.co/datasets/fhai50032/latentsig-med-triage-router)
[![HF Model](https://img.shields.io/badge/🤗_HuggingFace-GGUF-FFD21E?logo=huggingface)](https://huggingface.co/fhai50032/latentsig-med-router-qwen3-4b-gguf)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/IsNoobgrammer/latentsig-slm-router-med)
[![Eval](https://img.shields.io/badge/📊_Eval-Results-blue)](eval_result.md)

**Author:** LatentSig  
**Model:** Qwen3-4B-Instruct (QLoRA fine-tuned)

---

## Architecture Overview

![Architecture Overview](visuals/architecture_overview.png)

### Agent Pipeline

![Agent Pipeline](visuals/agent_pipeline.png)

The SLM runs per query in a 5-step ReAct loop:
1. **Input** — receive patient query
2. **Thought/Action** — SLM reasons and outputs tool call JSON
3. **Execution** — deterministic mock tool runs
4. **Observation** — tool result captured
5. **Final Answer** — structured summary with category, reasoning, tool, result

---

## Features

### Inference Engines

| Engine | Mode | Speed | Accuracy | Use Case |
|--------|------|-------|----------|----------|
| **Placeholder** | `--mode placeholder` | Instant | ~60% | Pipeline testing, CI |
| **Mistral API** | `--mode mistral` | ~1.5s | 80-87% | Baseline comparison |
| **Unsloth** | `--mode slm` | ~12.5s | Best | Full accuracy, GPU required |
| **llama.cpp (GGUF)** | `--mode gguf` | ~12.5s | Good | Alternative to Unsloth |

### Agent Loop

- **5-step ReAct loop**: Input → Thought → Action → Observation → Final Answer
- **Verbose mode**: `verbose=True` shows each step with reasoning, latency, tool call
- **Hallucination recovery**: up to 3 retries with error context on parse failure
- **Safety fallback**: if all retries fail, defaults to `emergency_dispatch` (over-triage)
- **Deterministic tools**: 7 mock tools with fixed responses, all calls logged to CSV

### Eval System

- **Single-stage**: tool call accuracy only (fast)
- **Full agent loop**: `--use-agent` for end-to-end measurement
- **Metrics**: tool accuracy, category accuracy, parse rate, latency (avg/p50/p95), retry count, confusion matrix
- **Per-breakdown**: per-tool, per-language (EN vs Hinglish), per-category
- **Auto-download**: adapter downloads from Hub if not found locally

### Interactive Testing

```bash
# Single query
python -m src.test_engine --engine mistral --query "chest pain, 55yo male"

# Interactive loop (type queries, 'quit' to exit)
python -m src.test_engine --engine mistral
```

### Eval CLI

```bash
# Placeholder (no API)
python -m src.eval --mode placeholder

# Mistral baseline
python -m src.eval --mode mistral --mistral-model mistral-large-latest

# GGUF (fast)
python -m src.eval --mode gguf --gguf-path ./model.gguf

# Unsloth (best accuracy)
python -m src.eval --mode slm --adapter-path fhai50032/latentsig-med-router-qwen3-4b

# Full comparison
python -m src.eval --mode full --adapter-path fhai50032/latentsig-med-router-qwen3-4b
```

---

## Project Structure

```
latentsig-slm-router-med/
│
├── src/                              ← Core agent + inference
│   ├── agent.py                      ← ReAct loop (5-step, verbose, retry, fallback)
│   ├── agent_colab.py                ← Two-stage SLM loop (Colab)
│   ├── config.py                     ← Model paths, hyperparams, constants
│   ├── eval.py                       ← Eval: SLM vs baseline, metrics, report
│   ├── inference.py                  ← Engines: Placeholder, Unsloth, Mistral, llama.cpp
│   ├── parser.py                     ← JSON extraction + Pydantic validation
│   ├── prompts.py                    ← System prompts (tool-call + assistant)
│   ├── test_agent.py                 ← Smoke test (10 queries, placeholder)
│   ├── test_engine.py                ← Interactive query tester (any engine)
│   ├── tools.py                      ← 7 deterministic mock tools + CSV logging
│   └── verify.py                     ← Post-training sanity check (10 queries)
│
├── synth-ds-framework/               ← Dataset generation pipeline
│   ├── orchestrator_parallel.py      ← Parallel datagen (32 workers, 8 keys)
│   ├── orchestrator.py               ← Serial datagen (fallback)
│   ├── verifier.py                   ← 3-layer: Pydantic + tool enforce + LLM judge
│   ├── tool_schemas.py               ← 7 tool definitions (flat params)
│   ├── prompts.py                    ← Query generation prompts
│   ├── models.py                     ← Pydantic models (TriageCall, DatasetRecord)
│   ├── monitor.py                    ← Flask dashboard (localhost:5000)
│   ├── gen_eval.py                   ← Eval dataset generator
│   ├── gen_visuals.py                ← Seaborn/matplotlib visualizations
│   └── gen_flowcharts.py             ← Flowchart image generator
│
├── docs/
│   └── setup.md                      ← Cell-by-cell Colab setup guide
│
├── tool_logs/                        ← Per-tool CSV logs (generated at runtime)
│   ├── triage_assessment.csv
│   ├── vital_signs_analysis.csv
│   ├── medication_check.csv
│   ├── specialist_referral.csv
│   ├── emergency_dispatch.csv
│   ├── mental_health_triage.csv
│   └── lab_order_suggestion.csv
│
├── visuals/                          ← Architecture + eval charts
│   ├── architecture_overview.png
│   ├── agent_pipeline.png
│   ├── react_loop_flow.png
│   ├── hallucination_recovery.png
│   ├── verification_pipeline.png
│   ├── generation_pipeline.png
│   ├── eval_accuracy_comparison.png
│   ├── eval_latency_comparison.png
│   ├── eval_tool_heatmap.png
│   └── eval_slm_distribution.png
│
├── eval_result.md                    ← Full eval results + fix plan
├── gen_eval_charts.py                ← Eval chart generator
├── setup.py                          ← Package setup (pip install -e)
└── README.md                         ← This file
```

---

## Tool Definitions

The SLM must select from 7 tools. Each tool has flat parameters (no nesting) for easier SLM learning.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `triage_assessment` | Initial symptom triage + urgency | chief_complaint, symptoms, severity, urgency_level |
| `vital_signs_analysis` | Analyze vitals for clinical decisions | bp_systolic, bp_diastolic, heart_rate, spo2_percent |
| `medication_check` | Drug interaction / dosage / overdose | medications, check_type, patient_condition |
| `specialist_referral` | Route to appropriate specialty | specialty, reason, urgency |
| `emergency_dispatch` | Life-threatening conditions ONLY | condition, symptoms, transport_type |
| `mental_health_triage` | Crisis assessment + routing | concern_type, risk_level, immediate_intervention |
| `lab_order_suggestion` | Suggest diagnostic tests | tests, urgency, suspected_condition |

### Output Format

```json
{
  "reasoning": "1-2 sentence clinical reasoning",
  "category": "emergency|urgent|semi_urgent|routine",
  "department": "<target department>",
  "tool": "<tool_name>",
  "args": { /* tool-specific parameters */ }
}
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| SLM | Qwen3-4B-Instruct | #1 fine-tuning benchmark, strong multilingual, Apache 2.0 |
| Fine-tuning | QLoRA 4-bit | Fits T4 16GB, ~5-6GB VRAM |
| LoRA rank | r=16 | Good capacity/speed balance for 4B |
| System prompt | Tool defs in prompt | Model learns to READ tools, not memorize |
| Identity | LatentSig branded | Only tool-calls when given this specific prompt |
| Tools | 7 flat-param tools | Easier for 1B-3B models to learn |
| Verification | 3-layer (Pydantic+Tool+Judge) | Fast structural + unbiased accuracy |
| Generation | 3 Mistral models rotated | Diversity in training data |
| Dedup | Hash on save only | Failed samples can retry same query |
| Output | Per-tool CSV logs | Verifiable out-of-loop |

---

## License

MIT — LatentSig, 2026
