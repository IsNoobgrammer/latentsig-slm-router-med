# IMPL-PLAN.md — LatentSig SLM Router Implementation Plan
> Version: 1.0 | References: PRD.md, ARCH.md | Last Updated: 2026-05-29

## Build Strategy: Vertical Slices

Each phase ships end-to-end working functionality. Phase 1 (Data + Fine-tuning) is handled by user — we start from Phase 2 with the assumption that a fine-tuned model/adapter exists.

**Prerequisites from Phase 1:**
- Fine-tuned SLM checkpoint (or LoRA adapter) that outputs valid JSON triage calls
- Model card / config specifying model name, tokenizer, adapter path
- Training logs showing convergence

---

## Phase 2: Agentic Loop (Core Agent)

**Goal:** A working agent that takes a symptom query, runs the SLM, handles errors, and returns a validated triage decision.

### 2.1: Project Scaffolding [S]
- [ ] Create project directory structure (src/, eval/, docs/, notebooks/)
- [ ] Write `requirements.txt` (transformers, vllm, torch, google-generativeai)
- [ ] Write `src/config.py` with model paths, retry limits, timeout configs
- [ ] Write `src/__init__.py`, `eval/__init__.py`
- [ ] Verify environment: `python -c "import torch; print(torch.cuda.is_available())"`

**Done when:** `python -c "from src.config import Config; print(Config)"` works.

### 2.2: Prompt Templates [S]
- [ ] Write `src/prompts.py` with system prompt for triage routing
- [ ] System prompt must include: valid categories, valid departments, valid tools, output format
- [ ] Write retry prompt template: original query + error message + "Fix the JSON"
- [ ] Write fallback message template
- [ ] Test: print all prompts, verify they're well-formed

**Done when:** All prompt templates produce readable, structured strings.

```python
# Example system prompt structure
SYSTEM_PROMPT = """You are a medical triage router. Given a patient symptom description, output a JSON object with exactly these fields:
{
  "category": "emergency" | "urgent" | "routine",
  "department": "<target care unit>",
  "tool": "<tool_name>",
  "args": {<tool arguments>}
}

Valid tools: order_lab_work, request_imaging, refer_specialist, prescribe_medication, escalate_to_emergency

Rules:
- Life-threatening symptoms (stroke, heart attack, breathing failure) → category: "emergency"
- Severe but stable symptoms → category: "urgent"
- Chronic/routine symptoms → category: "routine"
- When uncertain, default to higher severity (over-triage)
- Output ONLY valid JSON, no other text
"""
```

### 2.3: JSON Parser & Validator [M]
- [ ] Write `src/parser.py` with `parse_triage_output(raw: str) -> (dict, str)`
- [ ] Handle: clean JSON, JSON with surrounding text, partial JSON, completely invalid
- [ ] Extract JSON from markdown code blocks (` ```json ... ``` `)
- [ ] Validate required fields: `category`, `department`, `tool`, `args`
- [ ] Validate enum values: `category` in [emergency, urgent, routine]
- [ ] Normalize field values: strip whitespace, lowercase category
- [ ] Fuzzy department matching: "Stroke Center" ≈ "Stroke Unit" (Levenshtein or containment)
- [ ] Write unit tests: 15+ test cases covering all parse edge cases

**Done when:** `pytest tests/test_parser.py` passes with 15+ cases.

```python
# Key function signature
def parse_triage_output(raw: str) -> tuple[dict | None, str | None]:
    """
    Returns (parsed_dict, error_message).
    If parsed_dict is None, error_message describes what went wrong.
    If error_message is None, parsed_dict is valid.
    """
```

### 2.4: Mock Tool Registry [S]
- [ ] Write `src/tools.py` with tool registration and execution
- [ ] Implement 5 mock tools:
  - `order_lab_work(tests: list) -> dict` — returns mock lab results
  - `request_imaging(type: str, region: str) -> dict` — returns mock imaging order
  - `refer_specialist(department: str, urgency: str) -> dict` — returns mock referral
  - `prescribe_medication(name: str, dosage: str) -> dict` — returns mock prescription
  - `escalate_to_emergency(reason: str) -> dict` — returns mock escalation confirmation
- [ ] Each tool returns: `{"tool": name, "result": {...}, "status": "success"}`
- [ ] Unknown tool handler: return `{"status": "error", "message": "Unknown tool: {name}"}`
- [ ] Write unit tests: each tool returns expected structure

**Done when:** All 5 tools execute and return structured responses.

### 2.5: Inference Engine [M]
- [ ] Write `src/inference.py` with `InferenceEngine` class
- [ ] Implement `load(model_path, adapter_path=None)` — load model + optional LoRA adapter
- [ ] Implement `generate(prompt: str, max_tokens=256) -> str` — raw text generation
- [ ] Support vLLM backend (preferred) with auto-fallback to HF pipeline
- [ ] Implement tokenizer truncation for inputs exceeding model max_length
- [ ] Add timing instrumentation: `load_time_ms`, `inference_time_ms`
- [ ] Handle OOM: catch CUDA OOM, reduce max_tokens, retry once
- [ ] Write smoke test: load model, generate from a sample prompt

**Done when:** `InferenceEngine.load()` + `generate()` returns non-empty string.

```python
class InferenceEngine:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.backend = None  # "vllm" | "hf"

    def load(self, model_path: str, adapter_path: str = None):
        """Load model. Try vLLM first, fall back to HF pipeline."""
        ...

    def generate(self, prompt: str, max_tokens: int = 256) -> tuple[str, float]:
        """Generate text. Returns (output_text, latency_ms)."""
        ...
```

### 2.6: Agent Loop (Core) [L]
- [ ] Write `src/agent.py` with `TriageAgent` class
- [ ] Implement the full loop: prompt → infer → parse → (retry or execute) → respond
- [ ] Retry logic: max 3 attempts, error context appended on each retry
- [ ] Fallback policy: after 3 failures, return emergency category with warning
- [ ] Tool execution: call mock tool from parsed JSON, get observation
- [ ] Response synthesis: combine triage decision + tool observation into human-readable answer
- [ ] Timing: total latency from query to response
- [ ] Logging: each attempt logged with (attempt_num, raw_output, parse_result, latency)
- [ ] Write integration test: 5 sample queries, verify all return valid responses

**Done when:** `TriageAgent().run("chest pain, 45yo male")` returns a structured response.

```python
class TriageAgent:
    def __init__(self, inference_engine, max_retries=3):
        self.engine = inference_engine
        self.max_retries = max_retries
        self.tool_registry = ToolRegistry()

    def run(self, query: str) -> dict:
        """
        Returns {
            "query": str,
            "decision": dict,        # triage JSON
            "observation": dict,     # tool output
            "response": str,         # human-readable
            "attempts": int,         # inference count
            "latency_ms": float,     # total time
            "is_fallback": bool      # used safety fallback?
        }
        """
        ...
```

### 2.7: End-to-End Smoke Test [S]
- [ ] Write `run_smoke_test.py` — runs 10 sample queries through full pipeline
- [ ] Include: 3 emergency, 3 urgent, 3 routine, 1 adversarial (malformed-inducing)
- [ ] Print: query, decision, attempts, latency, is_fallback
- [ ] Verify: all 10 return valid responses, no crashes

**Done when:** `python run_smoke_test.py` completes with 10/10 valid responses.

---

## Phase 3: Evaluation Framework

**Goal:** A comprehensive eval suite that objectively measures SLM routing accuracy, structured output quality, latency, and recovery rate.

### 3.1: Test Case Library [L]
- [ ] Write `eval/test_cases.py` with 100+ test cases
- [ ] Structure each case: `{id, instruction, expected_output, category, difficulty, note}`
- [ ] Categories to cover:
  - **Emergency (30+ cases):** stroke, cardiac, respiratory, pediatric, anaphylaxis, sepsis, trauma
  - **Urgent (25+ cases):** fractures, high fever, severe pain, lacerations, infections
  - **Routine (20+ cases):** refills, checkups, chronic management, mild symptoms
  - **Ambiguous (15+ cases):** symptoms that could be urgent OR routine (tests over-triage rule)
  - **Adversarial (15+ cases):** empty input, gibberish, non-medical, injection attempts, extremely long
- [ ] Each case includes expected: category, department, tool (at minimum)
- [ ] Use the hard_cases provided by user as seed, expand from there
- [ ] Write test: `len(load_test_cases()) >= 100`

**Done when:** `eval/test_cases.py` has ≥100 cases with expected outputs.

```python
# Test case structure
@dataclass
class TestCase:
    id: str
    instruction: str
    expected: dict          # {"category": ..., "department": ..., "tool": ...}
    difficulty: str         # "easy" | "medium" | "hard" | "adversarial"
    note: str               # human-readable explanation
    is_emergency: bool      # for computing emergency recall

# Example
TestCase(
    id="emg_001",
    instruction="Sudden facial droop, cannot speak, 68-year-old male",
    expected={"category": "emergency", "department": "Stroke Unit", "tool": "escalate_to_emergency"},
    difficulty="easy",
    note="Classic stroke presentation — must route to Stroke Unit",
    is_emergency=True,
)
```

### 3.2: Metrics Module [M]
- [ ] Write `eval/metrics.py` with metric computation functions
- [ ] **JSON Validity Rate:** % of outputs that are valid JSON
- [ ] **Field Accuracy:** per-field exact match (category, department, tool)
- [ ] **Category Accuracy:** % correct on emergency/urgent/routine classification
- [ ] **Emergency Recall:** % of emergency cases correctly identified (CRITICAL — must be 100%)
- [ ] **Over-triage Rate:** % of non-emergency cases incorrectly classified as emergency
- [ ] **Retry Rate:** % of queries requiring >1 attempt
- [ ] **Recovery Rate:** % of malformed outputs successfully recovered via retry
- [ ] **Latency Distribution:** p50, p95, p99, max
- [ ] **Fallback Rate:** % of queries that hit safety fallback
- [ ] **Confusion Matrix:** 3x3 matrix for emergency/urgent/routine

**Done when:** `compute_all_metrics(results)` returns a dict with all metrics.

### 3.3: Gemini Judge (Fuzzy Matching) [M]
- [ ] Write `eval/judge.py` with `GeminiJudge` class
- [ ] Uses Gemini Flash to compare expected vs actual when exact match fails
- [ ] Prompt: "Are these two triage decisions equivalent? Expected: {expected}. Actual: {actual}. Answer: YES/NO + reasoning."
- [ ] Caching: cache judge decisions to avoid re-calling for same (expected, actual) pairs
- [ ] Rate limiting: 1 second delay between API calls
- [ ] Fallback: if Gemini API fails, use exact string matching
- [ ] Write test: mock Gemini response, verify caching works

**Done when:** `GeminiJudge.judge(expected, actual)` returns (bool, str_reasoning).

```python
class GeminiJudge:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.cache = {}  # (expected_json, actual_json) -> (bool, str)

    def judge(self, expected: dict, actual: dict) -> tuple[bool, str]:
        """Returns (is_match, reasoning). Uses cache to avoid duplicate API calls."""
        ...
```

### 3.4: Eval Runner [M]
- [ ] Write `eval/harness.py` with `EvalHarness` class
- [ ] Runs all test cases through TriageAgent, collects results
- [ ] Applies Gemini Judge for fuzzy matching on failed cases
- [ ] Computes all metrics via metrics module
- [ ] Progress bar for long eval runs
- [ ] Timeout per case: 10 seconds (skip on timeout)
- [ ] Write report to `eval_results/` directory
- [ ] Support for: `--quick` (10 cases), `--full` (all cases), `--category emergency` (filter)

**Done when:** `python -m eval.harness --quick` runs and produces a report.

### 3.5: Report Generator [S]
- [ ] Write `eval/report.py` with report generation
- [ ] Console output: summary table + per-category breakdown
- [ ] JSON export: full results for programmatic analysis
- [ ] Include: confusion matrix, latency histogram data, failure case details
- [ ] Flag critical failures: any missed emergency case highlighted in red

**Done when:** Report clearly shows accuracy, latency, and any emergency misses.

### 3.6: Colab Notebook [M]
- [ ] Write `notebooks/eval_demo.ipynb` for running full eval on Colab T4
- [ ] Cells: install deps → load model → run eval → display report
- [ ] Include: `%%time` magic for overall timing
- [ ] Include: confusion matrix visualization (matplotlib/seaborn)
- [ ] Include: latency distribution plot
- [ ] Include: failure case analysis (show top-5 worst failures)

**Done when:** Notebook runs end-to-end on Colab T4 without errors.

### 3.7: Baseline Comparison [S]
- [ ] Write `eval/baseline.py` that runs same test cases through Gemini Flash directly
- [ ] Compare: SLM accuracy vs Gemini Flash accuracy
- [ ] Compute: accuracy gap per category
- [ ] This validates that the fine-tuned SLM matches frontier model on structured output

**Done when:** Comparison report shows accuracy gap <5%.

---

## Complexity Summary

| Task | Complexity | Phase |
|------|-----------|-------|
| 2.1: Project scaffolding | S | 2 |
| 2.2: Prompt templates | S | 2 |
| 2.3: JSON parser & validator | M | 2 |
| 2.4: Mock tool registry | S | 2 |
| 2.5: Inference engine | M | 2 |
| 2.6: Agent loop (core) | L | 2 |
| 2.7: End-to-end smoke test | S | 2 |
| 3.1: Test case library | L | 3 |
| 3.2: Metrics module | M | 3 |
| 3.3: Gemini judge | M | 3 |
| 3.4: Eval runner | M | 3 |
| 3.5: Report generator | S | 3 |
| 3.6: Colab notebook | M | 3 |
| 3.7: Baseline comparison | S | 3 |

**Total: ~5 days estimated** (Phase 2: ~2.5 days, Phase 3: ~2.5 days)
