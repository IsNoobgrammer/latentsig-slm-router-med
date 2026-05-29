# ARCH.md — LatentSig SLM Router Architecture
> Version: 1.0 | References: PRD.md, DESIGN.md | Last Updated: 2026-05-29

---

## System Diagram

```mermaid
graph TD
    subgraph "Agentic Loop (agent.py)"
        A[User Query] --> B[Prompt Formatter]
        B --> C[SLM Inference]
        C --> D{JSON Valid?}
        D -->|Yes| E[Extract Fields]
        D -->|No| F[Error Context Builder]
        F -->|retry < 3| C
        F -->|retry >= 3| G[Safety Fallback]
        E --> H[Tool Executor]
        H --> I[Response Synthesizer]
        G --> I
    end

    subgraph "Inference Engine (inference.py)"
        C --> J[vLLM / HF Pipeline]
        J --> K[Tokenizer]
        K --> L[Model Forward Pass]
        L --> M[Decode Output]
        M --> C
    end

    subgraph "Mock Tools (tools.py)"
        H --> N[order_lab_work]
        H --> O[request_imaging]
        H --> P[refer_specialist]
        H --> Q[prescribe_medication]
        H --> R[escalate_to_emergency]
    end

    subgraph "Eval Framework (eval.py)"
        S[Test Suite] --> T[Eval Runner]
        T --> A
        I --> U[Metrics Collector]
        U --> V[Gemini Judge]
        V --> W[Report Generator]
    end

    style A fill:#4CAF50,color:#fff
    style G fill:#F44336,color:#fff
    style W fill:#2196F3,color:#fff
```

---

## Agentic Loop Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User/Eval
    participant A as Agent Loop
    participant I as Inference Engine
    participant P as JSON Parser
    participant T as Mock Tools
    participant G as Gemini Judge (eval only)

    U->>A: symptom_query

    rect rgb(200, 230, 200)
        Note over A,I: Attempt 1
        A->>I: formatted_prompt
        I->>I: tokenize + generate
        I-->>A: raw_output
        A->>P: parse(raw_output)
        P-->>A: error: malformed JSON
    end

    rect rgb(255, 230, 200)
        Note over A,I: Attempt 2 (retry with error context)
        A->>I: prompt + error_message
        I->>I: tokenize + generate
        I-->>A: raw_output
        A->>P: parse(raw_output)
        P-->>A: {category, department, tool, args}
    end

    A->>T: execute(tool, args)
    T-->>A: observation

    A->>A: synthesize_response
    A-->>U: final_answer + latency_ms

    Note over G: Only during eval phase
    G->>G: judge(expected, actual)
    G-->>U: score + reasoning
```

---

## Triage Data Model

```mermaid
erDiagram
    TRIAGE_REQUEST {
        string query PK
        timestamp created_at
    }

    TRIAGE_DECISION {
        string category "emergency | urgent | routine"
        string department "target care unit"
        string tool "tool to invoke"
        json args "tool arguments"
        int retry_count "attempts before success"
        string source "model | fallback"
    }

    TOOL_OBSERVATION {
        string tool_name
        json result "tool output"
        int latency_ms
    }

    FINAL_ANSWER {
        string summary "human-readable triage decision"
        int total_latency_ms
        boolean is_fallback
    }

    EVAL_RESULT {
        string test_case_id
        string expected_category
        string actual_category
        boolean json_valid
        boolean field_correct
        int latency_ms
        string judge_reasoning
    }

    TRIAGE_REQUEST ||--|| TRIAGE_DECISION : "produces"
    TRIAGE_DECISION ||--o| TOOL_OBSERVATION : "triggers"
    TRIAGE_DECISION ||--|| FINAL_ANSWER : "synthesizes"
    TRIAGE_REQUEST ||--o| EVAL_RESULT : "evaluated_as"
```

---

## Component Responsibilities

| Component | Owns | Does NOT own |
|-----------|------|-------------|
| Agent Loop (agent.py) | Orchestration, retry logic, fallback policy, response synthesis | Model inference, JSON parsing, tool logic |
| Inference Engine (inference.py) | Model loading, tokenization, generation, KV-cache | Prompt formatting, output validation |
| JSON Parser (parser.py) | JSON extraction, schema validation, field normalization | Model inference, retry decisions |
| Mock Tools (tools.py) | Tool registration, execution, deterministic outputs | Real tool integrations, external APIs |
| Eval Harness (eval.py) | Test cases, metrics computation, report generation | Model inference, tool execution |
| Prompt Templates (prompts.py) | System prompts, retry prompts, fallback messages | Business logic |

---

## Triage Category Taxonomy

```mermaid
graph LR
    subgraph "Category → Department Mapping"
        E1[Stroke symptoms] --> D1[Stroke Unit]
        E2[Cardiac symptoms] --> D2[Cardiac Unit]
        E3[Respiratory distress] --> D3[Respiratory Unit]
        E4[Pediatric emergency] --> D4[Pediatric ER]
        E5[General emergency] --> D5[Emergency Bay]

        U1[Orthopedic injury] --> D6[Orthopedics]
        U2[Fever + stiffness] --> D7[Pediatrics]
        U3[Fracture] --> D8[Orthopedics]

        R1[Medication refill] --> D9[Refill Clinic]
        R2[Chronic symptoms] --> D10[General Medicine]
        R3[Routine checkup] --> D11[General Medicine]
    end

    style E1 fill:#F44336,color:#fff
    style E2 fill:#F44336,color:#fff
    style E3 fill:#F44336,color:#fff
    style E4 fill:#F44336,color:#fff
    style E5 fill:#F44336,color:#fff
    style U1 fill:#FF9800,color:#fff
    style U2 fill:#FF9800,color:#fff
    style U3 fill:#FF9800,color:#fff
    style R1 fill:#4CAF50,color:#fff
    style R2 fill:#4CAF50,color:#fff
    style R3 fill:#4CAF50,color:#fff
```

---

## Retry & Recovery Flow

```mermaid
flowchart TD
    A[Generate Output] --> B{Valid JSON?}
    B -->|Yes| C{All Required Fields?}
    B -->|No| D[Build Error Context]
    D --> E{Retry Count < 3?}
    E -->|Yes| F[Append Error to Prompt]
    F --> A
    E -->|No| G[Safety Fallback: Emergency]
    C -->|Yes| H{Valid Category?}
    C -->|No| I[Report Missing Fields]
    I --> E
    H -->|Yes| J[Execute Tool]
    H -->|No| K[Report Invalid Enum]
    K --> E
    J --> L[Synthesize Response]
    G --> L

    style G fill:#F44336,color:#fff
    style L fill:#4CAF50,color:#fff
```

---

## Deployment Topology

| Layer | Service | Config |
|-------|---------|--------|
| Compute | Google Colab T4 GPU | 16GB VRAM, Python 3.10 |
| Inference | vLLM or HF Pipeline | Single GPU, fp16 or int4 (from Phase 1) |
| Model | Fine-tuned SLM + LoRA adapter | Loaded from Phase 1 checkpoint |
| Eval Judge | Gemini Flash (API) | Used only in Phase 3 for fuzzy matching |
| Storage | Google Drive / local | Model checkpoints, eval results |

---

## External Dependencies

| Dependency | Purpose | Fallback if unavailable |
|-----------|---------|------------------------|
| vLLM | Fast inference with KV-cache | HF Transformers pipeline (slower but works) |
| HF Transformers | Model loading, tokenization | — (required) |
| Gemini API | Eval judge for fuzzy matching | Exact string matching (less accurate) |
| Google Colab T4 | GPU compute | CPU inference (10-50x slower) |

---

## Failure Modes & Mitigations

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| SLM outputs malformed JSON | Parse failure | Retry with error context (max 3), then emergency fallback |
| SLM outputs wrong category | Incorrect triage | Eval catches this; over-triage rule reduces risk |
| Model OOM on T4 | Inference fails | Use quantized model (int4), reduce max_length |
| Gemini API rate limit | Eval slows down | Exponential backoff + batch eval with delays |
| vLLM not available | Slower inference | Auto-fallback to HF pipeline |
| Tool execution timeout | Loop stalls | 5s timeout per tool, skip on timeout |
| All retries exhausted | No valid output | Emergency fallback (over-triage safety) |

---

## File Structure

```
latentsig-slm-router-med/
├── AGENTS.md                  ← AI agent config
├── docs/
│   ├── PRD.md                 ← Requirements
│   ├── DESIGN.md              ← Technical design
│   ├── ARCH.md                ← This file
│   ├── IMPL-PLAN.md           ← Build plan
│   └── FLOW.md                ← User journeys
├── src/
│   ├── __init__.py
│   ├── agent.py               ← Agentic loop (core)
│   ├── inference.py           ← Model loading + generation
│   ├── parser.py              ← JSON parsing + validation
│   ├── tools.py               ← Mock tool registry
│   ├── prompts.py             ← System + retry prompts
│   ├── config.py              ← Model paths, hyperparams
│   └── utils.py               ← Timing, logging helpers
├── eval/
│   ├── __init__.py
│   ├── harness.py             ← Eval runner
│   ├── test_cases.py          ← 100+ test cases (hard_cases)
│   ├── metrics.py             ← Accuracy, latency, recovery metrics
│   ├── judge.py               ← Gemini Flash judge
│   └── report.py              ← Report generation
├── notebooks/
│   └── eval_demo.ipynb        ← Colab notebook for running evals
├── requirements.txt
└── README.md
```
