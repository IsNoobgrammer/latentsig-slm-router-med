# AGENTS.md — LatentSig SLM Router
> AI coding agent context file. Read this before writing any code.
> See also: docs/PRD.md (requirements), docs/ARCH.md (architecture)

## Project Context
**What we're building:** A local-first agentic workflow where a fine-tuned SLM (1-4B params) acts as a structured router for clinical medical triage. Takes symptom queries, outputs validated JSON triage decisions, with self-healing retry logic for hallucinated/malformed outputs.
**Stack:** Python 3.10+ + PyTorch + HF Transformers + vLLM (or HF pipeline fallback) + Gemini API (eval only)
**Current phase:** Phase 2 — Agentic Loop + Phase 3 — Eval Framework

## Project Structure
```
src/
├── agent.py          ← Core agentic loop (predict → parse → retry → execute → respond)
├── inference.py      ← Model loading + generation (vLLM or HF pipeline)
├── parser.py         ← JSON extraction + schema validation + fuzzy matching
├── tools.py          ← Mock tool registry (5 medical tools with deterministic outputs)
├── prompts.py        ← System prompt, retry prompt, fallback message templates
├── config.py         ← Model paths, retry limits, timeout configs
└── utils.py          ← Timing helpers, logging utilities

eval/
├── harness.py        ← Eval runner: runs test cases through agent, collects metrics
├── test_cases.py     ← 100+ test cases (emergency/urgent/routine/ambiguous/adversarial)
├── metrics.py        ← Accuracy, JSON validity, recovery rate, latency distribution
├── judge.py          ← Gemini Flash judge for fuzzy department matching
└── report.py         ← Console + JSON report generation
```

## Code Conventions
- **Language:** Python 3.10+ with type hints everywhere. Use `dict`, `list`, `tuple` (not `Dict`, `List`, `Tuple`).
- **Naming:** snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE for constants
- **Docstrings:** Google style. Every public function gets one.
- **Type hints:** All function signatures must have full type hints including return type
- **Error handling:** No bare `except`. Always catch specific exceptions. Log errors with context.
- **Imports:** Absolute imports from project root. `from src.parser import parse_triage_output`

## Medical Domain Rules
- **Over-triage principle:** When uncertain, ALWAYS default to higher severity. Missing an emergency is worse than unnecessary escalation.
- **No medical reasoning:** The SLM routes, it doesn't diagnose. Tool outputs are mock data. Never claim medical accuracy.
- **Structured output only:** The model must output valid JSON. Non-JSON output is a failure, not a feature.
- **Deterministic tools:** Mock tools return fixed outputs for given inputs. No randomness. Reproducible evals.

## Agentic Loop Rules
- **Max 3 retries:** After 3 malformed outputs, apply safety fallback (emergency category)
- **Error context injection:** On retry, append the parse error to the prompt so the model can self-correct
- **No infinite loops:** Every code path must terminate. Timeout after 30s per inference call.
- **Timing instrumentation:** Every `run()` call must report total latency, per-attempt latency, retry count

## Eval Rules
- **Emergency recall is sacred:** 100% of emergency cases MUST route correctly. Any miss is a critical failure.
- **Test cases are data, not code:** Define test cases as data structures, not as test functions. The harness iterates over them.
- **Gemini judge is for fuzzy matching only:** Don't use Gemini for the triage decision itself. Only for comparing expected vs actual during eval.
- **Deterministic eval:** Same model + same test cases = same results. No randomness in mock tools or test ordering.

## Git Workflow
- Branch: `feature/[description]`, `fix/[description]`
- Commits: conventional commits (`feat:`, `fix:`, `test:`, `docs:`)
- Co-author: `Co-Authored-By: Bauna Intern <bauna-intern@shaurya.dev>`
- Never commit API keys or model weights

## Explicit Boundaries
- ❌ Don't use the SLM for medical reasoning — it's a router, not a doctor
- ❌ Don't skip retry logic — hallucination recovery is a core feature
- ❌ Don't use async/concurrent — sequential is fine for research prototype
- ❌ Don't add frontend code — this is CLI/library only
- ❌ Don't use LangChain/CrewAI — custom Python loop, zero framework dependency
- ❌ Don't hardcode API keys — use environment variables
- ❌ Don't make mock tools non-deterministic — evals must be reproducible
- ❌ Don't skip the over-triage safety fallback — it's the last line of defense

## Key Data Structures

```python
# Triage output schema (what the SLM must produce)
TriageOutput = {
    "category": "emergency" | "urgent" | "routine",
    "department": str,      # target care unit
    "tool": str,            # one of 5 valid tool names
    "args": dict            # tool-specific arguments
}

# Agent run result
AgentResult = {
    "query": str,
    "decision": TriageOutput,
    "observation": dict,    # mock tool output
    "response": str,        # human-readable summary
    "attempts": int,        # inference count (1 = first try success)
    "latency_ms": float,
    "is_fallback": bool     # used safety fallback?
}
```
