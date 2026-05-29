# PRD.md — LatentSig SLM Router (Medical Triage)
> Version: 1.0 | Status: Draft | Last Updated: 2026-05-29

---

## Product Vision

**One-liner:** A local-first agentic workflow where a fine-tuned SLM acts as a reliable structured router for clinical medical triage, replacing frontier model dependency.

**Problem Statement:**
Medical triage requires fast, accurate routing of patient symptoms to the correct care unit. Frontier LLMs (GPT-4, Claude) can do this but introduce latency, cost, privacy concerns (patient data leaves the device), and hallucination risk in life-critical scenarios. A small, fine-tuned model running locally can match structured output reliability while eliminating cloud dependency.

**Solution:**
Fine-tune an SLM (1-4B params) to output strict JSON tool calls for triage routing. Wrap it in a self-healing agentic loop that catches malformed outputs and retries. Evaluate with a comprehensive medical benchmark that stress-tests edge cases, ambiguity, and life-critical scenarios.

**Key Insight:** The SLM doesn't need to "know medicine" — it needs to reliably map symptom patterns to structured triage decisions. The fine-tuning teaches format compliance and pattern matching, not medical reasoning.

---

## Target Users & Personas

### Persona 1: Triage Nurse (Primary)
- **Who:** Hospital ER staff, 24-55 years old, moderate tech literacy
- **Goal:** Quickly route patients to the correct care unit based on symptoms
- **Pain:** Manual triage is slow, error-prone under pressure, inconsistent across shifts
- **Success looks like:** "I type symptoms, get a structured triage decision in <500ms"

### Persona 2: Clinical AI Engineer (Builder)
- **Who:** ML engineer building healthcare AI, 25-40, strong Python/ML skills
- **Goal:** Build a reliable local-first medical routing system
- **Pain:** Frontier models are expensive, slow, and can't run on-device; medical domain needs strict accuracy
- **Success looks like:** "I have a 1-4B model that outputs correct JSON triage calls 99.99% of the time"

### Persona 3: Hospital IT Admin (Deployer)
- **Who:** IT staff managing hospital infrastructure, concerned about data privacy
- **Goal:** Deploy AI triage without sending patient data to external APIs
- **Pain:** HIPAA/GDPR compliance, network reliability, cost of cloud AI at scale
- **Success looks like:** "Everything runs on-premise, no data leaves the hospital network"

---

## Feature Requirements

| # | Feature | Description | Priority | Acceptance Criteria | Phase |
|---|---------|-------------|----------|---------------------|-------|
| F1 | Agentic Tool-Use Loop | SLM predicts → execute mock tool → observe → respond | P0 | Loop handles happy path and error recovery | 2 |
| F2 | Hallucination Recovery | Catch malformed JSON, re-prompt with error context | P0 | ≥95% recovery rate on malformed outputs | 2 |
| F3 | Retry with Escalation | Max 3 retries before fallback to safe default | P0 | Never hangs, always returns a triage decision | 2 |
| F4 | Mock Tool Execution | Simulated medical tools returning dummy data | P0 | All 5 tool types return structured responses | 2 |
| F5 | Streaming Latency | End-to-end query → decision in <2s | P1 | p95 latency <2s on T4 GPU | 2 |
| F6 | Comprehensive Eval Suite | 100+ test cases across all triage categories | P0 | All emergency cases route correctly (100%) | 3 |
| F7 | Structured Output Metrics | JSON validity, field completeness, category accuracy | P0 | Report per-field and per-category accuracy | 3 |
| F8 | Latency Benchmarking | Per-stage timing (inference, parsing, retry) | P1 | Identify bottleneck stage | 3 |
| F9 | Stress Testing | Edge cases: empty input, adversarial, ambiguous | P0 | Graceful handling of all edge cases | 3 |
| F10 | Comparative Baseline | Compare SLM vs frontier model (Gemini Flash) | P2 | Accuracy gap <5% on structured output | 3 |

---

## User Stories

### US1: Emergency Triage Routing
As a triage nurse, I want to enter "sudden facial droop, cannot speak, 68-year-old male" and receive `{"category": "emergency", "department": "Stroke Unit", "severity": "critical"}` so that the patient is immediately routed to stroke care.

**Acceptance Criteria:**
- GIVEN a classic stroke presentation WHEN the SLM processes it THEN output is valid JSON with category="emergency" and department="Stroke Unit"
- GIVEN the response time WHEN measured end-to-end THEN it is <2 seconds

### US2: Self-Healing on Malformed Output
As the system, when the SLM outputs `{broken json...}`, I want to automatically re-prompt with the error so that the final output is always valid structured JSON.

**Acceptance Criteria:**
- GIVEN malformed JSON output WHEN the parser fails THEN the system re-prompts with error context
- GIVEN 3 consecutive failures WHEN all retries exhausted THEN return safe fallback (emergency category)

### US3: Ambiguous Symptom Handling
As the system, when symptoms are ambiguous (e.g., "severe abdominal pain"), I want to apply the over-triage rule so that uncertain cases default to higher urgency.

**Acceptance Criteria:**
- GIVEN ambiguous symptoms WHEN severity is uncertain THEN category defaults to "emergency" (over-triage)
- GIVEN the rationale WHEN logged THEN it references the ambiguity rule

### US4: Routine Case Efficiency
As the system, when processing routine cases (medication refill, annual checkup), I want fast inference without retries so that common queries don't waste compute.

**Acceptance Criteria:**
- GIVEN a routine query WHEN processed THEN output is valid on first attempt (no retries)
- GIVEN latency WHEN measured THEN it is <500ms for routine cases

---

## Success Metrics

| Metric | Target | Measurement Method | Timeframe |
|--------|--------|--------------------|-----------|
| Emergency routing accuracy | 100% (zero misses) | Eval suite: all emergency cases | Phase 3 |
| Overall structured output validity | ≥99.99% | JSON parse success rate across full test set | Phase 3 |
| Category accuracy (emergency/urgent/routine) | ≥98% | Confusion matrix on eval set | Phase 3 |
| Hallucination recovery rate | ≥95% | % of malformed outputs recovered via retry | Phase 3 |
| p95 end-to-end latency | <2s | Timing instrumentation on T4 GPU | Phase 2 |
| p99 end-to-end latency | <3s | Timing instrumentation on T4 GPU | Phase 2 |
| Retry rate | <10% | % of queries requiring >1 attempt | Phase 3 |
| Accuracy gap vs frontier model | <5% | Side-by-side comparison on same test set | Phase 3 |

---

## Non-Goals (Explicit Out-of-Scope)

- [ ] **Real patient data** — All data is synthetic. No actual clinical deployment.
- [ ] **Medical reasoning** — The SLM routes, it doesn't diagnose. Tool outputs are mock data.
- [ ] **Multi-turn conversation** — Single query → single structured response. No dialogue.
- [ ] **Phase 1 (Data + Fine-tuning)** — Handled separately by user. We receive the fine-tuned adapter.
- [ ] **Frontend UI** — This is a CLI/library. No web or mobile interface.
- [ ] **Production deployment** — This is a research prototype. HIPAA compliance, audit logging, etc. are out of scope.
- [ ] **Multiple SLM comparison** — We optimize for one model. Benchmark comparison is Phase 3 eval only.

---

## Technical Constraints

- **Hardware:** Google Colab T4 GPU (16GB VRAM)
- **Models:** SLM from Phase 1 (1-4B params, likely Qwen-2.5-1.5B or Llama-3.2-3B with LoRA)
- **Inference:** vLLM or llama.cpp for fast local inference
- **Framework:** Python 3.10+, PyTorch
- **Evaluation:** Custom eval harness + Gemini Flash as judge
- **API Key:** Gemini API for synthetic data generation (Phase 1) and baseline comparison (Phase 3)
- **Timeline:** Phase 2 + Phase 3 = research prototype

---

## Open Questions

| Question | Owner | Due Date | Status |
|---------|-------|----------|--------|
| Which SLM from Phase 1 will be the primary router? | User | After Phase 1 completes | Pending |
| Should the agentic loop support async/concurrent queries? | User | Phase 2 start | Open |
| What's the target inference engine (vLLM vs llama.cpp vs HF pipeline)? | User | Phase 2 start | Open |

---

## Context for AI Agents

> **Note:** Machine-readable context for Cursor/Claude Code/Codex.

**Project:** LatentSig SLM Router — Local-first medical triage router using fine-tuned SLM
**Stack:** Python + PyTorch + HF Transformers + vLLM/llama.cpp + Gemini API
**MVP Features:** Agentic tool-use loop, hallucination recovery, comprehensive eval suite
**Users:** Clinical AI engineers building healthcare AI systems
**Success:** 99.99% structured output validity, 100% emergency routing accuracy, <2s p95 latency
**Phase 2 focus:** Agentic loop with self-healing JSON parsing
**Phase 3 focus:** Evaluation framework with 100+ test cases
**Not building:** Frontend, production deployment, real patient data handling, Phase 1 (data + fine-tuning)
