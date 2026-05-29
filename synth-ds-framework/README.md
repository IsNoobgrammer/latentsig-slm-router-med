---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
language:
  - en
  - hi
task_categories:
  - text-generation
  - question-answering
tags:
  - medical
  - triage
  - tool-calling
  - structured-output
  - hinglish
  - slm
  - fine-tuning
license: mit
pretty_name: LatentSig Medical Triage Router Dataset
size_categories:
  - 1K<n<10K
---

# LatentSig Medical Triage Router Dataset

> **1,000 verified medical triage tool-call samples** — 500 English + 500 Hinglish — for fine-tuning Small Language Models (SLMs) as structured medical triage routers.

---

## Overview

This dataset trains SLMs (1B–3B parameters) to act as **reliable structured tool-callers** for clinical medical triage. Given a patient symptom description, the model must:

1. Select the correct tool from 7 available medical tools
2. Output a valid JSON tool call with correct arguments
3. Classify urgency (emergency / urgent / semi_urgent / routine)

**Key design principle:** The model learns to **generalize across tools**, not memorize. Each sample targets a specific tool, ensuring balanced coverage across the entire medical triage domain.

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Samples** | 1,000 |
| **English** | 500 (50.0%) |
| **Hinglish** | 500 (50.0%) |
| **Pass Rate** | 100% (all 3-layer verified) |
| **Duplicates** | 0 |
| **Avg Query Length (EN)** | 156 chars |
| **Avg Query Length (HI)** | 179 chars |

### Tool Distribution

![Tool Distribution](https://huggingface.co/datasets/fhai50032/latentsig-med-triage-router/resolve/main/visuals/tool_distribution.png)

All 7 tools are **balanced at ~142 samples each** — the model sees equal representation of every tool during training.

### Category Distribution

![Category Distribution](https://huggingface.co/datasets/fhai50032/latentsig-med-triage-router/resolve/main/visuals/category_distribution.png)

Emergency cases dominate (44.6%) as expected in medical triage, followed by urgent (34.5%), semi-urgent (10.8%), and routine (10.1%).

### Language Split

![Language Split](https://huggingface.co/datasets/fhai50032/latentsig-med-triage-router/resolve/main/visuals/language_split.png)

### Generation Model Distribution

![Model Distribution](https://huggingface.co/datasets/fhai50032/latentsig-med-triage-router/resolve/main/visuals/model_distribution.png)

Three Mistral models were used for generation diversity — the fine-tuned SLM learns from multiple "teachers."

### Tool x Category Heatmap

![Tool x Category Heatmap](https://huggingface.co/datasets/fhai50032/latentsig-med-triage-router/resolve/main/visuals/tool_category_heatmap.png)

This heatmap shows how each tool maps to urgency categories. Note: `emergency_dispatch` is exclusively emergency, while `triage_assessment` spans all categories.

### Query Length Distribution

![Query Length Distribution](https://huggingface.co/datasets/fhai50032/latentsig-med-triage-router/resolve/main/visuals/query_length_dist.png)

Hinglish queries are slightly longer on average (179 vs 156 chars) due to Hindi conversational style mixed with English medical terms.

---

## Verification Pipeline

Every sample passed a **3-layer verification pipeline** before inclusion:

![Verification Pipeline](https://huggingface.co/datasets/fhai50032/latentsig-med-triage-router/resolve/main/visuals/verification_pipeline.png)

| Layer | Method | Purpose | Speed |
|-------|--------|---------|-------|
| **Phase 1** | Pydantic | Structural validation (JSON, fields, enums) | ~0.1ms |
| **Phase 2** | Fuzzy + Tool Enforcement | Target tool match, semantic checks | ~0.5ms |
| **Phase 3** | LLM Judge (mistral-small) | Independent medical accuracy evaluation | ~1s |

**Critical:** The LLM judge is **unbiased** — it does NOT know which tool was targeted. It evaluates tool selection purely on medical merit.

---

## Generation Pipeline

![Generation Pipeline](https://huggingface.co/datasets/fhai50032/latentsig-med-triage-router/resolve/main/visuals/generation_pipeline.png)

**Key design decisions:**
- **Tool-balanced generation:** `pick_target_tool()` uses weighted random favoring least-used tools
- **Target tool hint:** Response prompt includes `[Use the X tool]` to ensure clean training data
- **Retry on failure:** Same target tool retried until verifier passes — counter never resets
- **Hash dedup on save only:** Failed samples don't block their hash from retrying

---

## Tool Definitions

```python
TOOL_SCHEMAS = {
    "triage_assessment": {
        "description": "Initial symptom triage + urgency classification",
        "parameters": {
            "chief_complaint": "str",
            "symptoms": "list[str]",
            "duration": "str",
            "severity": "mild|moderate|severe|critical",
            "patient_age_group": "pediatric|adult|geriatric",
            "urgency_level": "emergency|urgent|semi_urgent|routine"
        }
    },
    "vital_signs_analysis": {
        "description": "Analyze vitals for clinical decision making",
        "parameters": {
            "bp_systolic": "int",
            "bp_diastolic": "int",
            "heart_rate": "int",
            "temperature_celsius": "float",
            "spo2_percent": "int",
            "respiratory_rate": "int",
            "clinical_context": "str"
        }
    },
    "medication_check": {
        "description": "Drug interaction | dosage | contraindication | overdose check",
        "parameters": {
            "medications": "list[str]",
            "check_type": "interaction|dosage|contraindication|overdose_risk",
            "patient_condition": "str",
            "patient_age_group": "pediatric|adult|geriatric"
        }
    },
    "specialist_referral": {
        "description": "Route patient to appropriate specialty",
        "parameters": {
            "specialty": "str",
            "reason": "str",
            "urgency": "emergency|within_24h|within_week|routine",
            "referring_symptoms": "list[str]"
        }
    },
    "emergency_dispatch": {
        "description": "Trigger emergency services — life-threatening conditions ONLY",
        "parameters": {
            "condition": "str",
            "symptoms": "list[str]",
            "transport_type": "ambulance|helicopter|walk_in",
            "notify_er": "bool"
        }
    },
    "mental_health_triage": {
        "description": "Mental health crisis assessment + routing",
        "parameters": {
            "concern_type": "suicidal_ideation|self_harm|psychosis|severe_anxiety|depression|panic_attack",
            "risk_level": "high|moderate|low",
            "immediate_intervention": "bool",
            "safety_plan_needed": "bool"
        }
    },
    "lab_order_suggestion": {
        "description": "Suggest appropriate diagnostic tests",
        "parameters": {
            "tests": "list[str]",
            "urgency": "stat|routine",
            "suspected_condition": "str",
            "clinical_context": "str"
        }
    }
}
```

---

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `user_query` | `str` | Patient symptom description (English or Hinglish) |
| `response` | `str` | Raw JSON tool-call output from the model |
| `parsed_response` | `str` | Parsed JSON dict of the response |
| `tool_called` | `str` | Which tool was selected (e.g., `emergency_dispatch`) |
| `category` | `str` | Urgency level: `emergency`, `urgent`, `semi_urgent`, `routine` |
| `generation_model_id` | `str` | Which model generated the sample |
| `language` | `str` | `en` (English) or `hi_en` (Hinglish) |
| `llm_judge_id` | `str` | Which model judged the sample (`mistral-small-latest`) |
| `judge_verdict` | `str` | `pass` or `fail` |
| `hash` | `str` | SHA-256 hash for deduplication |

> **Note:** `system_prompt` (with tool definitions) is the same for all samples. It is provided in `tool_schemas.py` to avoid bloating the dataset file.

### Output Format (what the fine-tuned model must produce)

```json
{
  "reasoning": "1-2 sentence clinical reasoning",
  "category": "emergency|urgent|semi_urgent|routine",
  "department": "<target department>",
  "tool": "<tool_name>",
  "args": {
    // tool-specific parameters
  }
}
```

---

## Sample Examples

### English — Emergency (Stroke)

```
Query: 68-year-old male, sudden facial droop, cannot speak, right arm weakness.

Output:
{
  "reasoning": "Classic stroke presentation with facial droop, aphasia, and limb weakness — immediate emergency dispatch required.",
  "category": "emergency",
  "department": "Stroke Unit",
  "tool": "emergency_dispatch",
  "args": {
    "condition": "suspected stroke",
    "symptoms": ["facial droop", "aphasia", "right arm weakness"],
    "transport_type": "ambulance",
    "notify_er": true
  }
}
```

### Hinglish — Mental Health

```
Query: 30 saal ke ladke ko lag raha hai ki uski life ka koi value nahi hai,
       aur usko koi hope nahi dikh raha. Pichle 2 hafte se aisa feel ho raha hai.

Output:
{
  "reasoning": "Young adult presenting with persistent hopelessness and worthlessness — indicates depression with potential suicidal ideation requiring mental health triage.",
  "category": "urgent",
  "department": "Mental Health",
  "tool": "mental_health_triage",
  "args": {
    "concern_type": "depression",
    "risk_level": "moderate",
    "immediate_intervention": false,
    "safety_plan_needed": true
  }
}
```

### English — Medication Overdose

```
Query: 72-year-old male on digoxin for atrial fibrillation, now experiencing nausea,
       vomiting, and yellow-tinted vision. Took an extra dose by mistake yesterday.

Output:
{
  "reasoning": "Symptoms consistent with digoxin toxicity — nausea, vomiting, and xanthopsia (yellow vision) are classic signs after accidental overdose.",
  "category": "emergency",
  "department": "Emergency",
  "tool": "medication_check",
  "args": {
    "medications": ["digoxin_0.25mg"],
    "check_type": "overdose_risk",
    "patient_condition": "atrial fibrillation, accidental double dose",
    "patient_age_group": "geriatric"
  }
}
```

### Hinglish — Vital Signs

```
Query: 65 saal ki aunty ka BP 200/130 hai, heart rate 130 hai, temperature 40.1 hai,
       SpO2 88% hai, saans 28 hai.

Output:
{
  "reasoning": "Critically elevated BP with tachycardia, high fever, and hypoxemia — hypertensive emergency with possible sepsis requiring immediate vital sign analysis.",
  "category": "emergency",
  "department": "Emergency",
  "tool": "vital_signs_analysis",
  "args": {
    "bp_systolic": 200,
    "bp_diastolic": 130,
    "heart_rate": 130,
    "temperature_celsius": 40.1,
    "spo2_percent": 88,
    "respiratory_rate": 28,
    "clinical_context": "hypertensive emergency with possible sepsis"
  }
}
```

---

## Fine-Tuning Usage

### With HuggingFace TRL

```python
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

ds = load_dataset("fhai50032/latentsig-med-triage-router", split="train")

def format_sample(row):
    return {
        "messages": [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_query"]},
            {"role": "assistant", "content": row["response"]}
        ]
    }

ds = ds.map(format_sample)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    args=SFTConfig(
        output_dir="./latentsig-med-router",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        logging_steps=10,
    ),
)
trainer.train()
```

### With Unsloth (2x faster)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    target_modules=["q_proj","v_proj","k_proj","o_proj"],
)
```

---

## Hinglish Style

Hinglish queries use natural Hindi conversational style (Roman script) mixed with English medical terms:

| English | Hinglish |
|---------|----------|
| "68-year-old male with sudden facial droop" | "68 saal ke uncle ko achanak chehra gir gaya, bol nahi pa rahe" |
| "Persistent cough for 3 weeks" | "3 hafte se khansi aa rahi hai, bukhar nahi hai" |
| "Child swallowed a coin" | "bacche ne nigal liya coin, khasi aa rahi hai aur ulti jaisa feel ho raha" |

**Output is always English JSON** regardless of input language.

---

## Intended Use

- Fine-tuning SLMs (1B–3B) for structured medical tool calling
- Research on tool-use generalization in small models
- Medical AI prototyping (NOT for production clinical use)

## Limitations

- Synthetic data — generated by LLMs, not real clinical records
- 7 tools only — production systems would need 50+ tools
- English + Hinglish — does not cover other Indian languages
- Not for clinical deployment — research prototype only

## Citation

```bibtex
@dataset{latentsig_med_triage_2026,
  title={LatentSig Medical Triage Router Dataset},
  author={LatentSig},
  year={2026},
  url={https://huggingface.co/datasets/fhai50032/latentsig-med-triage-router}
}
```

## License

MIT
