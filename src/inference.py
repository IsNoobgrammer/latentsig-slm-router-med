# ─────────────────────────────────────────────────────────────
# INFERENCE — Model loading + generation
#
# Supports:
#   - Unsloth + LoRA adapter (primary)
#   - HF Transformers pipeline (fallback)
#   - Placeholder mode (for testing without model)
# ─────────────────────────────────────────────────────────────

import time
from typing import Protocol


class InferenceEngine(Protocol):
    """Protocol for inference engines."""

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        """Generate response. Returns (text, latency_seconds)."""
        ...


# ── Placeholder Engine (for testing pipeline without model) ──

class PlaceholderEngine:
    """Returns fixed responses for testing the agentic loop."""

    def __init__(self):
        self.call_count = 0

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        """Return a deterministic response based on query keywords."""
        self.call_count += 1
        start = time.time()

        query_lower = user_prompt.lower()

        # Deterministic routing based on keywords
        if any(w in query_lower for w in ["facial droop", "stroke", "slurred speech", "cannot speak"]):
            response = '{"reasoning": "Classic stroke presentation requiring immediate emergency dispatch.", "category": "emergency", "department": "Stroke Unit", "tool": "emergency_dispatch", "args": {"condition": "suspected stroke", "symptoms": ["facial droop", "aphasia"], "transport_type": "ambulance", "notify_er": true}}'
        elif any(w in query_lower for w in ["chest pain", "crushing", "heart attack"]):
            response = '{"reasoning": "Crushing chest pain with radiation suggests acute MI.", "category": "emergency", "department": "Cardiac Unit", "tool": "emergency_dispatch", "args": {"condition": "suspected MI", "symptoms": ["chest pain", "arm radiation"], "transport_type": "ambulance", "notify_er": true}}'
        elif any(w in query_lower for w in ["suicidal", "end their life", "self harm"]):
            response = '{"reasoning": "Active suicidal ideation with plan requires immediate mental health intervention.", "category": "emergency", "department": "Mental Health", "tool": "mental_health_triage", "args": {"concern_type": "suicidal_ideation", "risk_level": "high", "immediate_intervention": true, "safety_plan_needed": true}}'
        elif any(w in query_lower for w in ["bp ", "blood pressure", "heart rate", "spo2"]):
            response = '{"reasoning": "Vital signs require systematic analysis for clinical decision making.", "category": "urgent", "department": "Emergency", "tool": "vital_signs_analysis", "args": {"bp_systolic": 180, "bp_diastolic": 110, "heart_rate": 110, "temperature_celsius": 38.5, "spo2_percent": 94, "respiratory_rate": 22, "clinical_context": "hypertensive urgency"}}'
        elif any(w in query_lower for w in ["medication", "drug", "digoxin", "warfarin", "overdose"]):
            response = '{"reasoning": "Medication interaction/overdose risk requires pharmacological assessment.", "category": "urgent", "department": "Pharmacy", "tool": "medication_check", "args": {"medications": ["digoxin_0.25mg"], "check_type": "overdose_risk", "patient_condition": "atrial fibrillation", "patient_age_group": "geriatric"}}'
        elif any(w in query_lower for w in ["refill", "checkup", "annual", "chronic", "mild"]):
            response = '{"reasoning": "Routine care request — no acute symptoms.", "category": "routine", "department": "General Medicine", "tool": "triage_assessment", "args": {"chief_complaint": "routine follow-up", "symptoms": ["none acute"], "duration": "chronic", "severity": "mild", "patient_age_group": "adult", "urgency_level": "routine"}}'
        elif any(w in query_lower for w in ["fatigue", "weight loss", "lab", "test", "blood work"]):
            response = '{"reasoning": "Unexplained symptoms warrant diagnostic workup.", "category": "semi_urgent", "department": "General Medicine", "tool": "lab_order_suggestion", "args": {"tests": ["CBC", "CMP", "TSH"], "urgency": "routine", "suspected_condition": "fatigue workup", "clinical_context": "chronic fatigue"}}'
        elif any(w in query_lower for w in ["fracture", "broken", "orthopedic", "swollen"]):
            response = '{"reasoning": "Suspected fracture requires orthopedic evaluation.", "category": "urgent", "department": "Orthopedics", "tool": "specialist_referral", "args": {"specialty": "orthopedics", "reason": "suspected fracture", "urgency": "within_24h", "referring_symptoms": ["pain", "swelling"]}}'
        else:
            response = '{"reasoning": "Symptoms require initial triage assessment.", "category": "urgent", "department": "General Medicine", "tool": "triage_assessment", "args": {"chief_complaint": "general symptoms", "symptoms": ["unspecified"], "duration": "acute", "severity": "moderate", "patient_age_group": "adult", "urgency_level": "urgent"}}'

        latency = time.time() - start + 0.1  # simulate ~100ms
        return response, latency


# ── Unsloth Engine (for actual inference) ────────────────────

class UnslothEngine:
    """Load fine-tuned model via Unsloth + LoRA adapter."""

    def __init__(self, base_model: str, adapter_path: str, max_seq_length: int = 2048, load_in_4bit: bool = True):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load model + adapter."""
        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
        )
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        # Load adapter weights
        if self.adapter_path:
            self.model.load_adapter(self.adapter_path)
        FastLanguageModel.for_inference(self.model)

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        """Generate response using the fine-tuned model."""
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=300,
                temperature=0.1,
                do_sample=True,
            )
        latency = time.time() - start

        # Decode only new tokens
        new_tokens = outputs[0][inputs.shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response, latency


# ── Mistral API Engine (baseline) ────────────────────────────

class MistralAPIEngine:
    """Mistral API engine for baseline comparison.

    Uses the same system prompt as the SLM but calls Mistral's API.
    Supports key rotation across multiple keys.
    """

    def __init__(self, api_keys: list[str], model: str = "mistral-small-latest",
                 temperature: float = 0.1, max_tokens: int = 300):
        import os
        self.api_keys = api_keys
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._key_idx = 0
        self.call_count = 0

    def _next_key(self) -> str:
        key = self.api_keys[self._key_idx % len(self.api_keys)]
        self._key_idx += 1
        return key

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        """Call Mistral API. Returns (response_text, latency_seconds)."""
        import json
        import urllib.request
        import time as _time

        key = self._next_key()
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }).encode()

        req = urllib.request.Request(
            "https://api.mistral.ai/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )

        start = _time.time()
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            latency = _time.time() - start
            data = json.loads(resp.read())
            self.call_count += 1
            return data["choices"][0]["message"]["content"], latency
        except Exception as e:
            latency = _time.time() - start
            return json.dumps({"error": str(e)}), latency


# ── Factory ──────────────────────────────────────────────────

def create_engine(mode: str = "placeholder", **kwargs) -> InferenceEngine:
    """Create an inference engine.

    Args:
        mode: "placeholder" (test), "unsloth" (real inference)
        **kwargs: passed to engine constructor
    """
    if mode == "placeholder":
        return PlaceholderEngine()
    elif mode == "unsloth":
        engine = UnslothEngine(**kwargs)
        engine.load()
        return engine
    elif mode == "mistral":
        return MistralAPIEngine(**kwargs)
    else:
        raise ValueError(f"Unknown engine mode: {mode}. Use 'placeholder', 'unsloth', or 'mistral'.")
