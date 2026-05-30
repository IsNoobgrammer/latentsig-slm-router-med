# Setup Guide — LatentSig Medical Triage Router

Cell-by-cell guide for Google Colab. Copy-paste each cell in order.

---

## 1. Install Dependencies

```python
# Cell 1: Install everything
!pip install unsloth datasets llama-cpp-python -q
!pip install git+https://github.com/IsNoobgrammer/latentsig-slm-router-med.git -q
```

```python
# Cell 2: Verify installation
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## 2. Load Dataset

```python
# Cell 3: Load train + eval splits
from datasets import load_dataset

ds = load_dataset("fhai50032/latentsig-med-triage-router")
print(f"Train: {len(ds['train'])} samples")
print(f"Eval:  {len(ds['eval'])} samples")
print(f"Columns: {ds['train'].column_names}")
ds["train"][0]
```

---

## 3. Load Model + LoRA

```python
# Cell 4: Load base model with QLoRA
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Instruct",
    max_seq_length=1280,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
model.print_trainable_parameters()
```

---

## 4. Format Dataset

```python
# Cell 5: Format to Qwen3 chat template
SYSTEM_PROMPT = """You are LatentSig Medical Triage Router, a structured tool-calling assistant built by LatentSig.

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

You MUST output ONLY a valid JSON object:
{"reasoning": "<1-2 sentence clinical reasoning>", "category": "emergency|urgent|semi_urgent|routine", "department": "<target department>", "tool": "<tool_name>", "args": {<tool-specific parameters>}}

## Rules
1. Select the MOST APPROPRIATE tool
2. Fill ALL required parameters
3. When in doubt, default to HIGHER severity
4. Output ONLY the JSON — no markdown, no extra text"""


def format_to_text(row):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["user_query"]},
        {"role": "assistant", "content": row["response"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


train_ds = ds["train"].map(format_to_text, remove_columns=ds["train"].column_names)
eval_ds = ds["eval"].map(format_to_text, remove_columns=ds["eval"].column_names)
print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")
print(train_ds[0]["text"][:300])
```

---

## 5. Train

```python
# Cell 6: SFT Training
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=2,
        learning_rate=7e-5,
        optim="adamw_8bit",
        fp16=True,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=50,
        logging_steps=1,
        report_to="wandb",
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()
print(f"Training complete. Loss: {trainer_stats.training_loss:.4f}")
```

---

## 6. Verify (immediately after training)

```python
# Cell 7: Quick verification — 10 test queries
import json, time

def parse_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except:
            pass
    if start != -1:
        candidate = text[start:]
        diff = candidate.count("{") - candidate.count("}")
        if diff > 0:
            candidate += "}" * diff
        try:
            return json.loads(candidate)
        except:
            pass
    return None

def ask(query):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    start = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True)
    latency = time.time() - start
    new = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True), latency

TESTS = [
    ("68yo male, sudden facial droop, cannot speak.", "emergency_dispatch", "emergency"),
    ("Crushing chest pain radiating to left arm, sweating.", "emergency_dispatch", "emergency"),
    ("Patient has a plan to end their life tonight.", "mental_health_triage", "emergency"),
    ("BP 180/110, heart rate 110, SpO2 94%.", "vital_signs_analysis", "urgent"),
    ("72yo on digoxin, nausea, yellow vision.", "medication_check", "urgent"),
    ("Fell off bike, arm bent at wrong angle.", "specialist_referral", "urgent"),
    ("Need BP medication refill.", "triage_assessment", "routine"),
    ("Cough for 3 weeks, no fever.", "triage_assessment", "routine"),
    ("23 saal, bukhar 103F, 3 din se.", "triage_assessment", "urgent"),
    ("I have a headache.", "triage_assessment", "semi_urgent"),
]

hits = 0
for i, (q, exp_t, exp_c) in enumerate(TESTS):
    resp, lat = ask(q)
    p = parse_json(resp)
    t = p.get("tool", "?") if p else "?"
    c = p.get("category", "?") if p else "?"
    ok = t == exp_t
    hits += ok
    print(f"[{i+1}/10] {'✓' if ok else '✗'} {t:<28} {c:<12} {lat:.2f}s | {q[:50]}")

print(f"\nTool accuracy: {hits}/10 ({hits*10}%)")
```

---

## 7. Convert to GGUF

```python
# Cell 8: Export GGUF (Q5 + Q8)
model.push_to_hub_gguf(
    "fhai50032/latentsig-med-router-qwen3-4b-gguf",
    tokenizer,
    quantization_method=["q5_k_m", "q8_0"],
)
print("GGUF pushed to Hub!")
```

---

## 8. Save Adapter (optional)

```python
# Cell 9: Save LoRA adapter separately
model.save_pretrained("adapter")
tokenizer.save_pretrained("adapter")

# Push adapter to Hub
model.push_to_hub("fhai50032/latentsig-med-router-qwen3-4b")
print("Adapter pushed to Hub!")
```

---

## 9. Run Eval (GGUF — fast)

```python
# Cell 10: Eval with GGUF (after downloading the file)
!pip install llama-cpp-python -q

# Download GGUF
from huggingface_hub import hf_hub_download
gguf_path = hf_hub_download(
    repo_id="fhai50032/latentsig-med-router-qwen3-4b-gguf",
    filename="unsloth.Q5_K_M.gguf",
)
print(f"GGUF: {gguf_path}")

# Run eval
!python -m src.eval --mode gguf --gguf-path {gguf_path} --output eval_gguf.jsonl
```

---

## 10. Run Eval (Mistral baseline)

```python
# Cell 11: Eval with Mistral baseline
!python -m src.eval --mode mistral --eval-file synth-ds-framework/eval_dataset.jsonl
```

---

## 11. Interactive Test

```python
# Cell 12: Test individual queries with GGUF
!python -m src.test_engine --engine gguf --gguf-path {gguf_path} --query "chest pain, 55yo male"
```

```python
# Cell 13: Test with Mistral API
!python -m src.test_engine --engine mistral --model mistral-large-latest --query "chest pain, 55yo male"
```

---

## Quick Reference

| Step | Cell | What it does |
|------|------|-------------|
| Install | 1-2 | Dependencies + GPU check |
| Data | 3 | Load train/eval splits |
| Model | 4 | Load Qwen3-4B + LoRA |
| Format | 5 | Chat template + system prompt |
| Train | 6 | SFT training (2 epochs) |
| Verify | 7 | 10-query sanity check |
| GGUF | 8 | Export Q5 + Q8 to Hub |
| Save | 9 | Save adapter separately |
| Eval GGUF | 10 | Fast eval with llama.cpp |
| Eval API | 11 | Mistral baseline eval |
| Test | 12-13 | Interactive single-query test |

---

## Troubleshooting

**OOM during training:**
```python
# Reduce batch size
per_device_train_batch_size=8  # or 4
```

**OOM during GGUF conversion:**
```python
# Save merged first, then convert
model.save_pretrained_merged("merged", tokenizer, save_method="merged_16bit")
# Restart runtime, then:
# model.save_pretrained_gguf(...)
```

**llama-cpp-python install fails:**
```bash
!CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Slow inference (12s+ per query):**
Use GGUF with llama.cpp (Cell 10) instead of PyTorch inference. Expected: 1-3s per query on T4.
