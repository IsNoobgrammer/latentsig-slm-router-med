# DESIGN.md — LatentSig SLM Router Technical Design
> Version: 1.0 | References: PRD.md | Last Updated: 2026-05-29

---

## System Overview

The LatentSig SLM Router is a Python-based agentic workflow that wraps a fine-tuned Small Language Model (SLM) in a structured tool-use loop for medical triage routing. The system takes a natural language symptom description and outputs a validated JSON triage decision.

The architecture follows the ReAct (Reasoning + Acting) pattern but optimized for speed: instead of multi-step reasoning, the SLM directly maps input → structured output in a single inference call, with a self-healing retry loop for malformed outputs. This is not a conversational agent — it's a structured extractor with error recovery.

**Three subsystems:**
1. **Inference Engine** — Loads the fine-tuned SLM, generates structured JSON output
2. **Agentic Loop** — Orchestrates predict → parse → execute → observe → respond cycle with retry logic
3. **Eval Framework** — Measures accuracy, latency, recovery rate, and compares against frontier models

**Data flow:** User query → SLM inference → JSON parse attempt → (retry if malformed) → mock tool execution → observation synthesis → final answer

---

## Component Breakdown

### Component 1: Inference Engine (`inference.py`)
- **Technology:** HF Transformers + vLLM (or llama.cpp for quantized models)
- **Responsibility:** Load model, tokenize input, generate structured output, return raw text
- **Key interactions:** Receives prompt string from Agentic Loop, returns raw model output string
- **Optimization:** KV-cache reuse for retry prompts (don't re-tokenize the full context)

### Component 2: Agentic Loop (`agent.py`)
- **Technology:** Pure Python, no framework dependency
- **Responsibility:** Orchestrate the predict → parse → execute → observe → respond cycle
- **Key interactions:** Calls Inference Engine for predictions, JSON parser for validation, Mock Tools for execution
- **Critical feature:** Hallucination recovery via structured re-prompting

### Component 3: JSON Parser & Validator (`parser.py`)
- **Technology:** Python `json` module + custom schema validator
- **Responsibility:** Parse raw model output into structured dict, validate against triage schema
- **Key interactions:** Receives raw string from Agentic Loop, returns (parsed_dict, error_msg) tuple
- **Edge cases:** Partial JSON, extra text before/after JSON, missing required fields, invalid enum values

### Component 4: Mock Tool Registry (`tools.py`)
- **Technology:** Python functions returning deterministic dummy data
- **Responsibility:** Simulate medical tool execution (lab orders, imaging requests, specialist referrals, etc.)
- **Key interactions:** Receives tool name + args from parsed JSON, returns structured observation
- **Tools:** `order_lab_work`, `request_imaging`, `refer_specialist`, `prescribe_medication`, `escalate_to_emergency`

### Component 5: Eval Harness (`eval.py`)
- **Technology:** Python + custom metrics + Gemini API for judge
- **Responsibility:** Run test suite, compute metrics, generate reports
- **Key interactions:** Calls Agentic Loop with test cases, measures accuracy/latency/recovery
- **Metrics:** JSON validity rate, field accuracy, category confusion matrix, latency distribution, recovery rate

---

## Data Flows

### Flow 1: Happy Path (Single Inference)
1. User provides symptom query: "Sudden facial droop, cannot speak, 68-year-old male"
2. Agentic Loop formats prompt with system instructions + user query
3. Inference Engine generates: `{"category": "emergency", "department": "Stroke Unit", "tool": "escalate_to_emergency", "args": {"reason": "suspected stroke"}}`
4. JSON Parser validates → success
5. Mock Tool executes `escalate_to_emergency` → returns observation
6. Agentic Loop synthesizes final answer: "EMERGENCY: Stroke Unit — suspected stroke. Immediate evaluation required."
7. Total latency: ~800ms (single inference)

### Flow 2: Hallucination Recovery (Retry Loop)
1. User provides symptom query
2. Inference Engine generates malformed output: `{"category": "emergency", "department":` (truncated)
3. JSON Parser fails → error: "JSON incomplete, missing closing brace"
4. Agentic Loop constructs retry prompt: original query + error message + "Fix the JSON output"
5. Inference Engine generates valid JSON on retry
6. Continue from happy path step 4
7. Total latency: ~1600ms (two inferences)

### Flow 3: Exhausted Retries (Fallback)
1. User provides symptom query
2. Three consecutive inference attempts produce malformed output
3. Agentic Loop applies safety fallback: `{"category": "emergency", "department": "Emergency Bay", "tool": "escalate_to_emergency", "args": {"reason": "system fallback — over-triage applied"}}`
4. Final answer includes warning: "FALLBACK: Routed to Emergency Bay due to system error. Manual review recommended."
5. This is the "over-triage" safety principle: when in doubt, escalate

### Flow 4: Evaluation Pipeline
1. Eval Harness loads test suite (100+ cases)
2. For each test case: run Agentic Loop, measure latency, compare output to expected
3. Compute aggregate metrics: accuracy, JSON validity, recovery rate, latency distribution
4. Optionally call Gemini Flash as judge for fuzzy matching (e.g., "Stroke Center" vs "Stroke Unit")
5. Generate report with per-category breakdown

---

## Key Design Decisions

| Decision | Option Chosen | Alternatives Considered | Rationale |
|---------|--------------|------------------------|-----------|
| Inference engine | vLLM (primary), HF pipeline (fallback) | llama.cpp, ONNX Runtime | vLLM has best throughput for batch eval, HF pipeline for simple single-query use |
| Retry strategy | Re-prompt with error context (max 3) | Schema-constrained decoding, grammar-based sampling | Most portable; works with any SLM without special inference modifications |
| Fallback policy | Always escalate to emergency (over-triage) | Return "unknown", raise exception | Medical domain: false negatives kill, false positives waste time. Over-triage is always safer |
| JSON validation | Schema-based with fuzzy field matching | Strict JSON Schema validation | SLMs may output "Stroke Unit" vs "Stroke Center" — fuzzy matching prevents unnecessary retries |
| Eval judge | Gemini Flash for fuzzy matching + exact match | Human annotation only, exact match only | Gemini catches semantic equivalence ("Stroke Unit" ≈ "Stroke Center"); exact match would penalize valid variants |
| Agentic framework | Custom Python loop (no LangChain/CrewAI) | LangChain, CrewAI, AutoGen | Zero dependency overhead, full control over retry logic, <100 lines of core code |
| Mock tools | Deterministic functions with fixed outputs | LLM-generated responses, random data | Deterministic = reproducible evals. Random data would make eval non-deterministic |
| Prompt format | System instruction + user query (single turn) | Multi-turn chat format, few-shot examples | Single turn is fastest. Few-shot adds latency. The fine-tuned model shouldn't need examples |

---

## Error Handling Strategy

**Inference Errors:**
- Model load failure → clear error message with GPU memory diagnostics
- Token OOM → automatic input truncation to max_length
- Generation timeout (30s) → abort and return fallback

**Parse Errors (Hallucination Recovery):**
- Invalid JSON syntax → re-prompt with error + "Output valid JSON only"
- Missing required fields → re-prompt with error + list of missing fields
- Invalid enum value → re-prompt with error + valid enum options
- Max 3 retries → fallback to emergency category (over-triage safety)

**Tool Execution Errors:**
- Unknown tool name → log warning, skip tool execution, continue to final answer
- Tool timeout → return observation "Tool execution timed out"
- Tool returns unexpected format → use raw tool output as observation

**Eval Errors:**
- Test case timeout (10s) → mark as "timeout" failure, continue to next case
- Gemini API rate limit → exponential backoff, max 5 retries
- Gemini API failure → fall back to exact string matching

---

## Security Considerations

- [ ] **No real patient data** — All inputs are synthetic. No HIPAA concerns in this prototype.
- [ ] **API key safety** — Gemini API key in environment variable, never in source code
- [ ] **Model isolation** — SLM runs locally, no data sent to external services during inference
- [ ] **Gemini API usage** — Only used for eval judging (Phase 3), not for triage decisions
- [ ] **Output sanitization** — Model output is parsed, not exec()'d. No code injection risk.
- [ ] **Rate limiting** — Eval harness includes delays between Gemini API calls to respect quotas

---

## Scalability Notes

**Current design targets:** Single T4 GPU, <100 queries/minute, research prototype

**Bottlenecks at scale:**
- Single GPU inference → would need batching or model parallelism for >100 QPS
- Gemini API for eval → would need self-hosted judge model for continuous eval
- Mock tools → would need real tool integrations for production

**Intentionally simple:**
- No async/concurrent query handling (sequential is fine for research)
- No caching (repeated queries re-infer; caching would add complexity)
- No model quantization in agentic loop (Phase 1 handles quantization)
