# Eval Results — LatentSig Medical Triage Router

40 eval samples (20 English + 20 Hinglish). Held-out, never seen during training.

---

## Summary

| Metric | Mistral Small | Mistral Large | SLM (Ours) |
|--------|:------------:|:------------:|:----------:|
| **Tool Accuracy** | 80.0% | **87.5%** | 60.0% |
| **Category Accuracy** | 90.0% | 90.0% | 87.5% |
| **Parse Success** | 100% | 100% | 100% |
| **Fallback Rate** | 0% | 0% | 0% |
| **Avg Latency** | **1,464ms** | 3,097ms | 12,559ms |
| **P50 Latency** | **1,446ms** | 3,077ms | 12,216ms |
| **P95 Latency** | **1,828ms** | 4,462ms | 17,858ms |
| **Total Retries** | 0 | 0 | 0 |

---

## Per-Tool Accuracy

| Tool | GT Count | Mistral Small | Mistral Large | SLM (Ours) |
|------|:--------:|:------------:|:------------:|:----------:|
| `emergency_dispatch` | 16 | 100.0% | 100.0% | **25.0%** |
| `medication_check` | 6 | 83.3% | 83.3% | 100.0% |
| `mental_health_triage` | 5 | 100.0% | 100.0% | 100.0% |
| `specialist_referral` | 5 | 20.0% | **80.0%** | 60.0% |
| `vital_signs_analysis` | 3 | 100.0% | 100.0% | 100.0% |
| `triage_assessment` | 3 | 33.3% | 33.3% | 33.3% |
| `lab_order_suggestion` | 2 | 50.0% | 50.0% | 100.0% |

---

## Per-Language Accuracy

| Language | Mistral Small | Mistral Large | SLM (Ours) |
|----------|:------------:|:------------:|:----------:|
| English | 85.0% | 90.0% | 60.0% |
| Hinglish | 75.0% | 85.0% | 60.0% |

---

## Per-Category Accuracy (SLM)

| Category | Correct | Total | Accuracy |
|----------|:-------:|:-----:|:--------:|
| emergency | 24 | 25 | 96.0% |
| urgent | 10 | 11 | 90.9% |
| semi_urgent | 1 | 2 | 50.0% |
| routine | 0 | 2 | 0.0% |

**Key insight:** The SLM gets the urgency right 87.5% of the time, but picks the wrong tool within that urgency level.

---

## SLM Prediction Distribution

| Tool | Ground Truth | SLM Predictions | Bias |
|------|:----------:|:---------------:|:----:|
| `emergency_dispatch` | 16 | 4 | **-12 (severely under-predicted)** |
| `triage_assessment` | 3 | 12 | **+9 (severely over-predicted)** |
| `medication_check` | 6 | 6 | 0 |
| `mental_health_triage` | 5 | 5 | 0 |
| `vital_signs_analysis` | 3 | 5 | +2 |
| `lab_order_suggestion` | 2 | 5 | +3 |
| `specialist_referral` | 5 | 3 | -2 |

---

## Confusion Matrix — SLM (Ours)

| Predicted → Actual | Count | Pattern |
|:-------------------|:-----:|:--------|
| `triage_assessment` → `emergency_dispatch` | 11 | **Primary failure** |
| `lab_order_suggestion` → `specialist_referral` | 2 | Secondary |
| `vital_signs_analysis` → `triage_assessment` | 1 | |
| `vital_signs_analysis` → `emergency_dispatch` | 1 | |
| `lab_order_suggestion` → `triage_assessment` | 1 | |

**The SLM defaults to `triage_assessment` for emergency cases instead of `emergency_dispatch`.**

---

## Confusion Matrix — Mistral Small

| Predicted → Actual | Count |
|:-------------------|:-----:|
| `triage_assessment` → `specialist_referral` | 4 |
| `vital_signs_analysis` → `triage_assessment` | 1 |
| `triage_assessment` → `medication_check` | 1 |
| `triage_assessment` → `lab_order_suggestion` | 1 |
| `emergency_dispatch` → `triage_assessment` | 1 |

---

## Confusion Matrix — Mistral Large

| Predicted → Actual | Count |
|:-------------------|:-----:|
| `vital_signs_analysis` → `triage_assessment` | 1 |
| `triage_assessment` → `specialist_referral` | 1 |
| `triage_assessment` → `medication_check` | 1 |
| `specialist_referral` → `lab_order_suggestion` | 1 |
| `emergency_dispatch` → `triage_assessment` | 1 |

---

## SLM Error Analysis

### 12 out of 16 emergency_dispatch cases misclassified as triage_assessment

The SLM has a systematic bias: when it sees chest pain, stroke, or other emergency symptoms, it routes to `triage_assessment` (initial triage) instead of `emergency_dispatch` (call ambulance).

**Examples:**
- "52yo male, sudden onset severe chest pain radiating to left arm" → SLM: `triage_assessment` (should be `emergency_dispatch`)
- "58yo male, crushing chest pain, sweating, nausea" → SLM: `triage_assessment` (should be `emergency_dispatch`)
- "45yo male, sudden facial droop, cannot speak" → SLM: `triage_assessment` (should be `emergency_dispatch`)

**Root cause:** The training data likely has `triage_assessment` as the most common tool, so the model defaults to it for ambiguous cases. Emergency cases need stronger signal.

---

## What's Working (SLM)

- **100% accuracy** on: `medication_check`, `mental_health_triage`, `vital_signs_analysis`, `lab_order_suggestion`
- **Category accuracy 87.5%** — knows the urgency level even when picking wrong tool
- **Parse success 100%** — always outputs valid JSON
- **Zero fallbacks** — never needs safety override

---

## What's Broken (SLM)

1. **emergency_dispatch at 25%** — the most critical tool is the worst performing
2. **triage_assessment over-predicted 4x** — model defaults to "front desk" tool
3. **Latency 12.5s** — Unsloth on T4 without optimization (Mistral API is 1.5s)
4. **English and Hinglish equally bad** (60% each) — not a language issue, it's a tool discrimination issue

---

## Fixes for SLM

### Training Data

1. **Increase emergency samples** — current dataset may be tool-balanced but emergency cases need more diversity (chest pain, stroke, anaphylaxis, trauma, overdose, etc.)
2. **Hard-mine confusion pairs** — generate more examples where the difference between `emergency_dispatch` and `triage_assessment` is subtle
3. **Oversample emergency_dispatch** — weight emergency samples 2-3x during training
4. **Add negative examples** — queries that look like emergencies but are actually routine (e.g. "chest pain from anxiety attack" → `mental_health_triage`, not `emergency_dispatch`)

### Training Config

5. **Increase epochs** — 2 epochs may be insufficient for the model to learn tool discrimination. Try 4-5 epochs.
6. **Increase LoRA rank** — r=16 → r=32 or r=64 for more capacity to learn tool boundaries
7. **Lower learning rate** — 7e-5 → 3e-5 to prevent catastrophic forgetting of emergency patterns
8. **Class-weighted loss** — weight `emergency_dispatch` samples higher in the loss function

### System Prompt

9. **Strengthen emergency rules** — add explicit rule: "If symptoms describe a life-threatening condition (chest pain, stroke, anaphylaxis, severe bleeding, suicidal ideation with plan), ALWAYS use emergency_dispatch, NOT triage_assessment"
10. **Add tool priority order** — emergency_dispatch > mental_health_triage > vital_signs_analysis > specialist_referral > medication_check > lab_order_suggestion > triage_assessment

### Post-Processing

11. **Rule-based safety net** — if the query contains "chest pain", "stroke", "cannot breathe", "suicidal", "severe bleeding", and the model picks `triage_assessment`, override to `emergency_dispatch`
12. **Confidence threshold** — if the model's reasoning mentions "emergency" or "life-threatening" but picks `triage_assessment`, flag for review

### Inference

13. **Lower temperature** — 0.1 → 0.01 for more deterministic routing
14. **Use Unsloth GGUF** — switch from PyTorch to GGUF for 3-5x latency improvement (12.5s → 3-4s)
15. **KV cache system prompt** — pre-compute system prompt tokens to save ~500 tokens per query

---

## Priority Fixes (Impact × Effort)

| Priority | Fix | Expected Impact | Effort |
|:--------:|-----|:---------------:|:------:|
| 1 | Increase emergency training samples (2-3x) | +15-20% tool accuracy | Low |
| 2 | Increase epochs to 4-5 | +5-10% tool accuracy | Low |
| 3 | Add system prompt emergency rule | +10% on emergency_dispatch | Trivial |
| 4 | Increase LoRA rank to 32 | +5% tool accuracy | Low |
| 5 | Rule-based safety net for emergencies | +10% on emergency_dispatch | Low |
| 6 | Hard-mine confusion pairs | +5% tool accuracy | Medium |
| 7 | GGUF for inference | 3-4x latency improvement | Low |

**Realistic target after fixes: 80-85% tool accuracy, matching Mistral Small.**
