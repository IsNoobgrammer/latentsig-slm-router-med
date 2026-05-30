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
    """Load fine-tuned model via Unsloth + LoRA adapter.

    Auto-downloads adapter from Hub if not found locally.
    Supports batch inference for eval workloads.
    """

    def __init__(self, base_model: str, adapter_path: str,
                 max_seq_length: int = 2048, load_in_4bit: bool = True,
                 max_new_tokens: int = 300, temperature: float = 0.1):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self.call_count = 0

    def load(self):
        """Load model + adapter. Auto-downloads from Hub if needed."""
        from unsloth import FastLanguageModel
        import os

        # Auto-download adapter if not local
        if self.adapter_path and not os.path.isdir(self.adapter_path):
            print(f"Adapter not found locally: {self.adapter_path}")
            print(f"Downloading from Hub...")
            from huggingface_hub import snapshot_download
            self.adapter_path = snapshot_download(
                repo_id=self.adapter_path,
                ignore_patterns=["*.gguf", "*.bin", "README.md"],
            )
            print(f"  Downloaded to: {self.adapter_path}")

        print(f"Loading model: {self.base_model}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.adapter_path if os.path.isdir(self.adapter_path) else self.base_model,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
        )

        # Apply LoRA if loading from base model
        if os.path.isdir(self.adapter_path):
            # Adapter already merged in from_pretrained
            pass
        else:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0,
                use_gradient_checkpointing="unsloth",
            )
            if self.adapter_path:
                self.model.load_adapter(self.adapter_path)

        FastLanguageModel.for_inference(self.model)
        print(f"  Model loaded. VRAM: {__import__('torch').cuda.memory_allocated()/1e9:.1f}GB")

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
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                use_cache=True,
            )
        latency = time.time() - start

        new_tokens = outputs[0][inputs.shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        self.call_count += 1
        return response, latency

    def generate_batch(self, system_prompt: str, user_prompts: list[str],
                       batch_size: int = 4) -> list[tuple[str, float]]:
        """Batch generate using PyTorch padded batching.

        Tokenizes all queries, pads to same length, generates in batches.

        Args:
            system_prompt: shared system prompt for all queries
            user_prompts: list of user queries
            batch_size: how many sequences per batch (limited by VRAM)

        Returns:
            list of (response_text, latency_seconds) tuples
        """
        import torch
        import time as _time

        results = [None] * len(user_prompts)

        # Tokenize all queries
        all_input_ids = []
        for prompt in user_prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            all_input_ids.append(ids.squeeze(0))

        # Pad to same length within each batch
        total_start = _time.time()

        for batch_start in range(0, len(user_prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(user_prompts))
            batch_ids = all_input_ids[batch_start:batch_end]

            # Pad to max length in this batch
            max_len = max(ids.shape[0] for ids in batch_ids)
            padded = torch.zeros(len(batch_ids), max_len, dtype=torch.long, device="cuda")
            attention_mask = torch.zeros(len(batch_ids), max_len, dtype=torch.long, device="cuda")

            for i, ids in enumerate(batch_ids):
                seq_len = ids.shape[0]
                padded[i, :seq_len] = ids
                attention_mask[i, :seq_len] = 1

            # Generate
            batch_start_time = _time.time()
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=padded,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    use_cache=True,
                )
            batch_latency = _time.time() - batch_start_time

            # Decode each output
            for i, (ids, out) in enumerate(zip(batch_ids, outputs)):
                new_tokens = out[ids.shape[0]:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                global_idx = batch_start + i
                results[global_idx] = (response, batch_latency / len(batch_ids))

        total = _time.time() - total_start
        self.call_count += len(user_prompts)
        print(f"  Batch: {len(user_prompts)} queries in {total:.1f}s "
              f"({total/len(user_prompts):.2f}s/query, batch_size={batch_size})")

        return results


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


# ── LlamaCpp Engine (GGUF inference) ─────────────────────────

class LlamaCppEngine:
    """llama.cpp engine for fast GGUF inference.

    ~3-5x faster than PyTorch on T4. Uses quantized weights
    with optimized KV cache and matmul kernels.

    Supports batch inference for eval workloads.

    Requires: pip install llama-cpp-python
    """

    def __init__(self, model_path: str, n_ctx: int = 4096,
                 n_gpu_layers: int = -1, temperature: float = 0.1,
                 max_tokens: int = 300, n_batch: int = 512):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_batch = n_batch
        self.llm = None
        self.call_count = 0

    def load(self):
        """Load GGUF model."""
        from llama_cpp import Llama

        print(f"Loading GGUF: {self.model_path}")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            verbose=False,
        )
        print(f"  GGUF loaded. Context: {self.n_ctx}, GPU layers: {self.n_gpu_layers}, batch: {self.n_batch}")

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        """Generate response. Returns (text, latency_seconds)."""
        import time as _time

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        start = _time.time()
        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            latency = _time.time() - start
            self.call_count += 1
            return output["choices"][0]["message"]["content"], latency
        except Exception as e:
            latency = _time.time() - start
            return json.dumps({"error": str(e)}), latency

    def generate_batch(self, system_prompt: str, user_prompts: list[str],
                       batch_size: int = 8) -> list[tuple[str, float]]:
        """Batch generate for multiple queries.

        Shares the same system prompt across all queries.
        Processes in batches of `batch_size` using ThreadPoolExecutor.

        Args:
            system_prompt: shared system prompt for all queries
            user_prompts: list of user queries
            batch_size: how many to process concurrently

        Returns:
            list of (response_text, latency_seconds) tuples
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time as _time

        results = [None] * len(user_prompts)

        def _generate_one(idx, prompt):
            return idx, self.generate(system_prompt, prompt)

        start = _time.time()

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [
                executor.submit(_generate_one, i, prompt)
                for i, prompt in enumerate(user_prompts)
            ]
            for future in as_completed(futures):
                idx, (text, lat) = future.result()
                results[idx] = (text, lat)

        total = _time.time() - start
        self.call_count += len(user_prompts)
        print(f"  Batch: {len(user_prompts)} queries in {total:.1f}s "
              f"({total/len(user_prompts):.2f}s/query, {batch_size} parallel)")

        return results


# ── Factory ──────────────────────────────────────────────────

def create_engine(mode: str = "placeholder", **kwargs) -> InferenceEngine:
    """Create an inference engine.

    Args:
        mode: "placeholder" (test), "unsloth" (real inference),
              "mistral" (API baseline), "gguf" (llama.cpp)
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
    elif mode == "gguf":
        engine = LlamaCppEngine(**kwargs)
        engine.load()
        return engine
    else:
        raise ValueError(f"Unknown engine mode: {mode}. Use 'placeholder', 'unsloth', 'mistral', or 'gguf'.")
