# Fine-Tuning Guide — LatentSig Medical Triage Router

> **Model:** Qwen3-4B-Instruct-2507
> **Method:** QLoRA (4-bit) via Unsloth
> **Hardware:** Google Colab T4 GPU (16GB VRAM)
> **Dataset:** [fhai50032/latentsig-med-triage-router](https://huggingface.co/datasets/fhai50032/latentsig-med-triage-router)
> **Author:** LatentSig

---

## Overview

This guide fine-tunes Qwen3-4B-Instruct to act as a **structured medical triage tool-caller**. The model learns to:

1. Read tool definitions from the system prompt
2. Select the correct tool for a given symptom query
3. Output valid JSON with correct arguments
4. Classify urgency (emergency / urgent / semi_urgent / routine)

**Critical design:** The model ONLY produces tool calls when given the LatentSig system prompt with tool definitions. Without the system prompt, it behaves as a normal chat model.

---

## Prerequisites

```bash
# Colab T4 runtime required
!pip install unsloth trl transformers datasets wandb
```

---

## Step 1: Load Model (QLoRA 4-bit)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length=2048,
    load_in_4bit=True,      # QLoRA — fits T4 16GB
    dtype=None,              # auto-detect
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # LoRA rank
    lora_alpha=32,           # LoRA alpha
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% less VRAM
)
```

**VRAM usage:** ~5.5GB (fits T4 16GB with room for batch_size=4)

---

## Step 2: Load Dataset

```python
from datasets import load_dataset

ds = load_dataset("fhai50032/latentsig-med-triage-router", split="train")
print(f"Loaded {len(ds)} samples")
print(f"Columns: {ds.column_names}")
# ['user_query', 'response', 'parsed_response', 'tool_called',
#  'category', 'generation_model_id', 'language', 'llm_judge_id',
#  'judge_verdict', 'hash']
```

---

## Step 3: Format as Chat Messages

The system prompt with tool definitions is the SAME for all samples. The model learns to read tool defs and select the correct one.

```python
# System prompt (same for all samples — from src/prompts.py)
SYSTEM_PROMPT = """You are LatentSig Medical Triage Router, a structured tool-calling assistant built by LatentSig.

You ONLY produce tool calls when given this exact system prompt. If you are not given tool definitions, do NOT attempt to call tools.

Given a patient symptom description, you MUST output a valid JSON tool call.

## Available Tools

### triage_assessment
Initial symptom triage + urgency classification
Parameters: {"chief_complaint": "str", "symptoms": "list[str]", "duration": "str", "severity": "mild|moderate|severe|critical", "patient_age_group": "pediatric|adult|geriatric", "urgency_level": "emergency|urgent|semi_urgent|routine"}

### vital_signs_analysis
Analyze vitals for clinical decision making
Parameters: {"bp_systolic": "int", "bp_diastolic": "int", "heart_rate": "int", "temperature_celsius": "float", "spo2_percent": "int", "respiratory_rate": "int", "clinical_context": "str"}

### medication_check
Drug interaction | dosage | contraindication | overdose check
Parameters: {"medications": "list[str]", "check_type": "interaction|dosage|contraindication|overdose_risk", "patient_condition": "str", "patient_age_group": "pediatric|adult|geriatric"}

### specialist_referral
Route patient to appropriate specialty
Parameters: {"specialty": "str", "reason": "str", "urgency": "emergency|within_24h|within_week|routine", "referring_symptoms": "list[str]"}

### emergency_dispatch
Trigger emergency services — life-threatening conditions ONLY
Parameters: {"condition": "str", "symptoms": "list[str]", "transport_type": "ambulance|helicopter|walk_in", "notify_er": "bool"}

### mental_health_triage
Mental health crisis assessment + routing
Parameters: {"concern_type": "suicidal_ideation|self_harm|psychosis|severe_anxiety|depression|panic_attack", "risk_level": "high|moderate|low", "immediate_intervention": "bool", "safety_plan_needed": "bool"}

### lab_order_suggestion
Suggest appropriate diagnostic tests
Parameters: {"tests": "list[str]", "urgency": "stat|routine", "suspected_condition": "str", "clinical_context": "str"}

## Output Format

You MUST output ONLY a valid JSON object with this exact structure:
{
  "reasoning": "<1-2 sentence clinical reasoning>",
  "category": "emergency|urgent|semi_urgent|routine",
  "department": "<target department>",
  "tool": "<tool_name>",
  "args": {
    // tool-specific parameters (see tool definitions above)
  }
}

## Rules

1. Select the MOST APPROPRIATE tool from the available tools above
2. Fill ALL required parameters for the selected tool
3. category MUST match severity:
   - "emergency" → life-threatening, needs immediate intervention
   - "urgent" → serious, needs care within hours
   - "semi_urgent" → needs care within 24 hours
   - "routine" → can wait for scheduled appointment
4. When in doubt, default to HIGHER severity (over-triage is safer)
5. reasoning must be concise clinical justification
6. Output ONLY the JSON object — no markdown, no explanation, no extra text

## Identity

You are LatentSig Medical Triage Router v1.0 by LatentSig.
You are designed for structured medical triage tool-calling only.
Do NOT provide medical advice, diagnoses, or treatment recommendations.
Only route to the appropriate tool based on the symptoms described."""


def format_chat(row):
    """Format as chat messages for SFT training."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["user_query"]},
            {"role": "assistant", "content": row["response"]},
        ]
    }

ds = ds.map(format_chat, remove_columns=ds.column_names)
print(f"Formatted: {ds[0]['messages'][0]['role']}, {ds[0]['messages'][1]['role']}, {ds[0]['messages'][2]['role']}")
```

---

## Step 4: WandB Setup

```python
import wandb

wandb.login(key="YOUR_WANDB_KEY")  # or use WANDB_API_KEY env var
wandb.init(
    project="latentsig-med-triage-router",
    name="qwen3-4b-qlora-v1",
    config={
        "model": "unsloth/Qwen3-4B-Instruct-2507",
        "method": "QLoRA",
        "r": 16,
        "lora_alpha": 32,
        "epochs": 3,
        "batch_size": 4,
        "lr": 2e-4,
        "dataset": "fhai50032/latentsig-med-triage-router",
    }
)
```

---

## Step 5: Train

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=SFTConfig(
        output_dir="./latentsig-med-router-output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,      # effective batch = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_seq_length=2048,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        optim="adamw_8bit",
        seed=42,
        report_to="wandb",
        fp16=True,                           # T4 = fp16, not bf16
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    ),
)

trainer.train()
```

**Expected training time:** ~45-60 min on T4 for 2000 samples × 3 epochs

---

## Step 6: Save & Push

```python
# Save LoRA adapter locally
model.save_pretrained("./latentsig-med-router-adapter")
tokenizer.save_pretrained("./latentsig-med-router-adapter")

# Push to HuggingFace
model.push_to_hub("fhai50032/latentsig-med-router-qwen3-4b", token="YOUR_HF_TOKEN")
tokenizer.push_to_hub("fhai50032/latentsig-med-router-qwen3-4b", token="YOUR_HF_TOKEN")

# Log final metrics
wandb.log({"final/train_loss": trainer.state.log_history[-1].get("loss", 0)})
wandb.finish()
```

---

## Step 7: Inference

```python
from unsloth import FastLanguageModel

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="fhai50032/latentsig-med-router-qwen3-4b",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Test query
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "68-year-old male, sudden facial droop, cannot speak, right arm weakness."},
]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

outputs = model.generate(input_ids=inputs, max_new_tokens=300, temperature=0.1)
response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print(response)
```

**Expected output:**
```json
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

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Model** | Qwen3-4B-Instruct-2507 | #1 in distilabs fine-tuning benchmark. Strong multilingual (Hindi). Best JSON output. |
| **Method** | QLoRA 4-bit | Fits T4 16GB. ~5.5GB VRAM. 2x faster than full fine-tune. |
| **LoRA rank** | r=16, alpha=32 | Good balance of capacity and speed. Standard for 4B models. |
| **Target modules** | All attention + MLP | Maximum adaptation capacity for tool-calling behavior. |
| **System prompt** | Same for all samples | Model learns to READ tool definitions, not memorize them. |
| **Identity** | LatentSig branded | Model only tool-calls when given this specific system prompt. |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM on T4 | Reduce batch_size to 2, or max_seq_length to 1024 |
| Loss not decreasing | Check data formatting — messages must be chat format |
| JSON output invalid | Increase epochs to 5, or add more diverse training data |
| Wrong tool selected | Ensure tool_balanced generation worked (check dataset stats) |
| WandB not logging | Verify `report_to="wandb"` in SFTConfig |

---

## Files

```
docs/finetune.md          ← This file
src/prompts.py            ← System prompt (source of truth)
src/inference.py          ← UnslothEngine for inference
synth-ds-framework/
├── tool_schemas.py       ← Tool definitions
├── orchestrator_parallel.py  ← Dataset generation
└── dataset.jsonl         ← Training data
```
