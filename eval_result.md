# Eval Results — LatentSig Medical Triage Router

40 eval samples (20 English + 20 Hinglish). Held-out, never seen during training.

**Colab Notebooks:**
- [Training](https://colab.research.google.com/drive/1fakehai/training-latentsig-slm-router)
- [Agent Inference](https://colab.research.google.com/drive/1fakehai/latentsig-slm-router-agent)

**W&B Report:** [LatentSig SLM Router](https://wandb.ai/ablations-tinycompany-ai/latentsig-med-triage-router/reports/LatentSig-SLM-Router--VmlldzoxNzA2MzQ3OA)

---

## Summary

![Accuracy Comparison](visuals/eval_accuracy_comparison.png)

| Metric | Qwen3-4B Base | SLM (Fine-tuned) | Mistral Small | Mistral Large |
|--------|:------------:|:----------------:|:------------:|:------------:|
| **Tool Accuracy** | **82.5%** | 60.0% | 80.0% | **87.5%** |
| **Category Accuracy** | 90.0% | 87.5% | 90.0% | 90.0% |
| **Parse Success** | 100% | 100% | 100% | 100% |
| **Fallback Rate** | 0% | 0% | 0% | 0% |
| **Avg Latency** | 9,866ms | 12,559ms | **1,464ms** | 3,097ms |
| **P50 Latency** | 10,007ms | 12,216ms | **1,446ms** | 3,077ms |
| **P95 Latency** | 13,583ms | 17,858ms | **1,828ms** | 4,462ms |

**Critical finding: Fine-tuning HURT the model.** The base Qwen3-4B (82.5%) outperforms the fine-tuned SLM (60.0%) by 22.5 percentage points.

---

## Latency

![Latency Comparison](visuals/eval_latency_comparison.png)

All local models are ~10-12.5s on T4. Mistral API is 7-8x faster (1.5s) because it's a hosted service, not local inference.

---

## Per-Tool Accuracy

![Tool Heatmap](visuals/eval_tool_heatmap.png)

| Tool | GT Count | Qwen3-4B Base | SLM (Fine-tuned) | Mistral Small | Mistral Large |
|------|:--------:|:------------:|:----------------:|:------------:|:------------:|
| `emergency_dispatch` | 16 | **100.0%** | 25.0% | **100.0%** | **100.0%** |
| `medication_check` | 6 | **100.0%** | **100.0%** | 83.3% | 83.3% |
| `mental_health_triage` | 5 | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `specialist_referral` | 5 | 20.0% | 60.0% | 20.0% | **80.0%** |
| `vital_signs_analysis` | 3 | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `triage_assessment` | 3 | 0.0% | 33.3% | 33.3% | 33.3% |
| `lab_order_suggestion` | 2 | **100.0%** | **100.0%** | 50.0% | 50.0% |

---

## Base vs Fine-tuned: What Training Destroyed

![Base vs Fine-tuned](visuals/eval_base_vs_finetuned.png)

| Tool | Base | Fine-tuned | Change |
|------|:----:|:----------:|:------:|
| `emergency_dispatch` | 100% | 25% | **-75%** |
| `medication_check` | 100% | 100% | 0% |
| `mental_health_triage` | 100% | 100% | 0% |
| `specialist_referral` | 20% | 60% | **+40%** |
| `vital_signs_analysis` | 100% | 100% | 0% |
| `triage_assessment` | 0% | 33% | **+33%** |
| `lab_order_suggestion` | 100% | 100% | 0% |

Fine-tuning improved `specialist_referral` (+40%) and `triage_assessment` (+33%), but catastrophically destroyed `emergency_dispatch` (-75%). Net effect: -22.5% overall accuracy.

---

## SLM Prediction Distribution

| Tool | Ground Truth | Base Predictions | Fine-tuned Predictions |
|------|:----------:|:----------------:|:---------------------:|
| `emergency_dispatch` | 16 | 16 | 4 |
| `triage_assessment` | 3 | 0 | 12 |
| `medication_check` | 6 | 6 | 6 |
| `mental_health_triage` | 5 | 5 | 5 |
| `vital_signs_analysis` | 3 | 3 | 5 |
| `lab_order_suggestion` | 2 | 7 | 5 |
| `specialist_referral` | 5 | 3 | 3 |

The base model correctly predicts 16 emergency_dispatch calls. The fine-tuned model predicts only 4 — the training taught it to default to `triage_assessment` instead.

---

## Per-Language Accuracy

| Language | Qwen3-4B Base | SLM (Fine-tuned) | Mistral Small | Mistral Large |
|----------|:------------:|:----------------:|:------------:|:------------:|
| English | 85.0% | 60.0% | 85.0% | 90.0% |
| Hinglish | 80.0% | 60.0% | 75.0% | 85.0% |

---

## Per-Category Accuracy

| Category | Qwen3-4B Base | SLM (Fine-tuned) |
|----------|:------------:|:----------------:|
| emergency | 25/25 = 100% | 24/25 = 96% |
| urgent | 9/11 = 82% | 10/11 = 91% |
| semi_urgent | 0/2 = 0% | 1/2 = 50% |
| routine | 1/2 = 50% | 0/2 = 0% |

---

## Confusion Matrices

### Qwen3-4B Base — 7 errors

| Predicted → Actual | Count |
|:-------------------|:-----:|
| `lab_order_suggestion` → `specialist_referral` | 4 |
| `vital_signs_analysis` → `triage_assessment` | 2 |
| `emergency_dispatch` → `triage_assessment` | 1 |

### SLM (Fine-tuned) — 16 errors

| Predicted → Actual | Count |
|:-------------------|:-----:|
| `triage_assessment` → `emergency_dispatch` | **11** |
| `lab_order_suggestion` → `specialist_referral` | 2 |
| `vital_signs_analysis` → `triage_assessment` | 1 |
| `vital_signs_analysis` → `emergency_dispatch` | 1 |
| `lab_order_suggestion` → `triage_assessment` | 1 |

### Mistral Small — 8 errors

| Predicted → Actual | Count |
|:-------------------|:-----:|
| `triage_assessment` → `specialist_referral` | 4 |
| `vital_signs_analysis` → `triage_assessment` | 1 |
| `triage_assessment` → `medication_check` | 1 |
| `triage_assessment` → `lab_order_suggestion` | 1 |
| `emergency_dispatch` → `triage_assessment` | 1 |

### Mistral Large — 5 errors

| Predicted → Actual | Count |
|:-------------------|:-----:|
| `vital_signs_analysis` → `triage_assessment` | 1 |
| `triage_assessment` → `specialist_referral` | 1 |
| `triage_assessment` → `medication_check` | 1 |
| `specialist_referral` → `lab_order_suggestion` | 1 |
| `emergency_dispatch` → `triage_assessment` | 1 |

---

## Diagnosis: What Went Wrong with Fine-tuning

The fine-tuning catastrophically degraded `emergency_dispatch` accuracy (100% → 25%) while marginally improving `specialist_referral` (20% → 60%) and `triage_assessment` (0% → 33%). Net result: -22.5% overall accuracy.

### Root Causes

1. **Training data quality** — the generated dataset likely has noisy or incorrect tool labels for emergency cases. The model learned to distrust emergency signals.

2. **Overfitting to triage_assessment** — the most common tool in training data. The model defaults to it when uncertain, destroying emergency routing.

3. **Too few epochs** — 2 epochs may not be enough to learn tool boundaries, but enough to corrupt the base model's existing knowledge.

4. **System prompt mismatch** — the training system prompt may differ subtly from inference, causing the model to route differently.

5. **LoRA rank too low** — r=16 may not have enough capacity to learn the tool discrimination task without forgetting base capabilities.

---

## Fixes

### Immediate (Training)

1. **Audit training data** — check emergency_dispatch samples for quality. Are the labels correct?
2. **Increase epochs to 4-5** — more time to learn tool boundaries
3. **Increase LoRA rank to 32** — more capacity
4. **Lower learning rate to 3e-5** — prevent catastrophic forgetting
5. **Oversample emergency_dispatch 2-3x** — counter the triage_assessment bias

### System Prompt

6. **Add emergency rule** — "life-threatening symptoms → ALWAYS emergency_dispatch"
7. **Tool priority order** — emergency > mental_health > vitals > specialist > medication > lab > triage

### Post-Processing

8. **Rule-based safety net** — if query contains "chest pain", "stroke", "suicidal", "cannot breathe" AND model picks triage_assessment, override to emergency_dispatch

### Alternative Approaches

9. **Use base model with better system prompt** — the base model already gets 82.5%. A stronger system prompt might push it to 87%+.
10. **Few-shot prompting** — instead of fine-tuning, use 3-5 examples in the system prompt
11. **Distill from Mistral Large** — use Mistral Large's predictions as training labels instead of the generated dataset

---

## Priority Fixes

| # | Fix | Expected Impact | Effort |
|:-:|-----|:---------------:|:------:|
| 1 | Audit + fix training data labels | +15-25% | Medium |
| 2 | Use base model + stronger system prompt | +5% (to 87%) | Trivial |
| 3 | Increase epochs to 4-5 | +5-10% | Low |
| 4 | Increase LoRA rank to 32 | +5% | Low |
| 5 | Lower learning rate to 3e-5 | +5% | Low |
| 6 | Rule-based emergency safety net | +10% | Low |
| 7 | Distill from Mistral Large | +10-15% | Medium |

**Realistic targets:**
- Base model + better prompt: 85-87% (no training needed)
- Fixed fine-tuning: 80-85% (after data audit + config changes)
- Distillation from Mistral Large: 87-90% (best approach)
