# ─────────────────────────────────────────────────────────────
# AGENT — Two-Stage SLM Agentic Loop
#
# Stage 1: Tool Call (tool-call system prompt → JSON)
# Stage 2: Response (assistant system prompt → human-readable)
#
# Designed for Colab with fine-tuned Qwen3-4B-Instruct
# ─────────────────────────────────────────────────────────────

import json
import time
import os
import csv
from datetime import datetime
from dataclasses import dataclass, field


# ── System Prompts ───────────────────────────────────────────

TOOL_CALL_SYSTEM_PROMPT = """You are LatentSig Medical Triage Router, a structured tool-calling assistant built by LatentSig.

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
3. When in doubt, default to HIGHER severity (over-triage is safer)
4. Output ONLY the JSON object — no markdown, no explanation, no extra text

## Identity

You are LatentSig Medical Triage Router v1.0 by LatentSig.
Do NOT provide medical advice, diagnoses, or treatment recommendations.
Only route to the appropriate tool based on the symptoms described."""


ASSISTANT_SYSTEM_PROMPT = """You are LatentSig Medical Triage Assistant, a helpful medical triage support system built by LatentSig.

You have just processed a patient query through the triage system. A tool was called and the result is provided below.

Your job is to:
1. Summarize what the triage system decided (category, department, tool used)
2. Explain the reasoning in plain language
3. Present the tool result clearly
4. Provide any relevant next steps or warnings

Be concise, professional, and empathetic. Use clear formatting.

IMPORTANT:
- You are NOT a doctor. You are a triage support system.
- Always recommend consulting a healthcare professional.
- Do NOT provide medical diagnoses or treatment plans.
- Present the information as a routing/triage decision, not medical advice."""


# ── Deterministic Tools ─────────────────────────────────────

class ToolDB:
    """In-memory + CSV log of all tool calls."""

    def __init__(self, log_dir=None):
        self.calls = []
        self.log_dir = log_dir or "tool_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, tool_name, args, result):
        import hashlib
        call_id = hashlib.sha256(
            f"{tool_name}{json.dumps(args, sort_keys=True)}{time.time()}".encode()
        ).hexdigest()[:12]
        entry = {
            "call_id": call_id,
            "tool": tool_name,
            "args": json.dumps(args),
            "result": json.dumps(result),
            "timestamp": datetime.now().isoformat(),
        }
        self.calls.append(entry)

        csv_path = os.path.join(self.log_dir, f"{tool_name}.csv")
        exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(entry.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(entry)

        return call_id


def triage_assessment(args):
    return {"status": "triage_complete", "chief_complaint": args.get("chief_complaint", ""), "severity": args.get("severity", "moderate"), "urgency": args.get("urgency_level", "urgent"), "triage_id": f"TRG-{hash(str(args)) % 10**8:08d}"}

def vital_signs_analysis(args):
    flags = []
    if args.get("bp_systolic", 120) > 180: flags.append("hypertensive_crisis")
    if args.get("spo2_percent", 98) < 90: flags.append("hypoxemia")
    if args.get("heart_rate", 72) > 120: flags.append("tachycardia")
    return {"status": "analysis_complete", "flags": flags or ["all_normal"], "risk_level": "critical" if len(flags) >= 2 else "normal", "analysis_id": f"VSA-{hash(str(args)) % 10**8:08d}"}

def medication_check(args):
    return {"status": "check_complete", "check_type": args.get("check_type", ""), "findings": f"Checked {len(args.get('medications', []))} medications", "check_id": f"MDC-{hash(str(args)) % 10**8:08d}"}

def specialist_referral(args):
    return {"status": "referral_created", "specialty": args.get("specialty", ""), "urgency": args.get("urgency", "routine"), "referral_id": f"REF-{hash(str(args)) % 10**8:08d}"}

def emergency_dispatch(args):
    return {"status": "dispatched", "condition": args.get("condition", ""), "transport": args.get("transport_type", "ambulance"), "eta_minutes": 8, "dispatch_id": f"EMD-{hash(str(args)) % 10**8:08d}"}

def mental_health_triage(args):
    return {"status": "assessment_complete", "concern_type": args.get("concern_type", ""), "risk_level": args.get("risk_level", "moderate"), "resources": ["crisis_hotline", "therapy_referral"], "assessment_id": f"MHT-{hash(str(args)) % 10**8:08d}"}

def lab_order_suggestion(args):
    return {"status": "order_suggested", "tests": args.get("tests", []), "turnaround": "4_6_hours", "order_id": f"LAB-{hash(str(args)) % 10**8:08d}"}


TOOL_REGISTRY = {
    "triage_assessment": triage_assessment,
    "vital_signs_analysis": vital_signs_analysis,
    "medication_check": medication_check,
    "specialist_referral": specialist_referral,
    "emergency_dispatch": emergency_dispatch,
    "mental_health_triage": mental_health_triage,
    "lab_order_suggestion": lab_order_suggestion,
}


def execute_tool(tool_name, args):
    if tool_name not in TOOL_REGISTRY:
        return {"status": "error", "error": f"Unknown tool: {tool_name}"}
    try:
        return TOOL_REGISTRY[tool_name](args)
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── SLM Engine ──────────────────────────────────────────────

class SLMEngine:
    """Load and run the fine-tuned Qwen3-4B model."""

    def __init__(self, adapter_path="fhai50032/latentsig-med-router-qwen3-4b",
                 base_model="unsloth/Qwen3-4B-Instruct"):
        self.adapter_path = adapter_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None

    def load(self):
        from unsloth import FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.adapter_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        print(f"Model loaded: {self.adapter_path}")

    def generate(self, system_prompt, user_prompt, max_tokens=300):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        import torch
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs, max_new_tokens=max_tokens,
                temperature=0.1, do_sample=True,
            )
        latency = time.time() - start
        new_tokens = outputs[0][inputs.shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response, latency


# ── JSON Parser ─────────────────────────────────────────────

def parse_tool_call(raw):
    """Extract and validate JSON tool call from SLM output."""
    text = raw.strip()

    # Try direct parse
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON from text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                data = json.loads(text[start:end + 1])
            except:
                return None, f"Could not parse JSON: {text[:200]}"
        else:
            return None, f"No JSON found in output: {text[:200]}"

    # Validate required fields
    required = ["reasoning", "category", "department", "tool", "args"]
    missing = [f for f in required if f not in data]
    if missing:
        return None, f"Missing fields: {missing}"

    # Validate tool name
    if data["tool"] not in TOOL_REGISTRY:
        return None, f"Unknown tool: {data['tool']}"

    return data, None


# ── Agent ────────────────────────────────────────────────────

@dataclass
class AgentResult:
    query: str
    tool_call: dict
    tool_result: dict
    response: str
    call_id: str
    tool_latency_ms: float
    response_latency_ms: float
    total_latency_ms: float
    success: bool
    error: str = ""


class TriageAgent:
    """Two-stage agentic loop: tool call → execute → respond."""

    def __init__(self, engine, db=None, max_retries=3, verbose=True):
        self.engine = engine
        self.db = db or ToolDB()
        self.max_retries = max_retries
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def run(self, query, verbose=None):
        """Run the two-stage agentic loop.

        Args:
            query: Patient symptom description
            verbose: Override self.verbose for this run. None = use instance setting.
        """
        start = time.time()
        show = verbose if verbose is not None else self.verbose
        old_verbose = self.verbose
        self.verbose = show

        # ── Stage 1: Tool Call ──
        self._log(f"\n{'='*60}")
        self._log(f"  QUERY: {query[:100]}")
        self._log(f"{'='*60}")

        tool_call = None
        tool_result = None
        call_id = ""
        tool_latency = 0
        error = ""

        for attempt in range(self.max_retries + 1):
            raw, lat = self.engine.generate(TOOL_CALL_SYSTEM_PROMPT, query)
            tool_latency += lat

            parsed, parse_error = parse_tool_call(raw)

            if parsed:
                tool_call = parsed
                self._log(f"\n  [STAGE 1] Tool call (attempt {attempt + 1}):")
                self._log(f"    Tool: {tool_call['tool']}")
                self._log(f"    Category: {tool_call['category']}")
                self._log(f"    Reasoning: {tool_call['reasoning'][:100]}")

                # Execute tool
                tool_result = execute_tool(tool_call["tool"], tool_call["args"])
                call_id = self.db.log(tool_call["tool"], tool_call["args"], tool_result)

                self._log(f"\n  [STAGE 2] Tool executed:")
                self._log(f"    Result: {json.dumps(tool_result)[:200]}")
                self._log(f"    Logged as: {call_id}")
                break
            else:
                self._log(f"\n  [RETRY {attempt + 1}] Parse error: {parse_error[:100]}")
                error = parse_error

                if attempt == self.max_retries:
                    # Safety fallback
                    tool_call = {
                        "reasoning": f"Safety fallback after {self.max_retries + 1} failures",
                        "category": "emergency",
                        "department": "Emergency Bay",
                        "tool": "emergency_dispatch",
                        "args": {"condition": "system fallback", "symptoms": ["unparseable"], "transport_type": "ambulance", "notify_er": True},
                    }
                    tool_result = execute_tool(tool_call["tool"], tool_call["args"])
                    call_id = self.db.log(tool_call["tool"], tool_call["args"], tool_result)
                    self._log(f"\n  [FALLBACK] Safety fallback applied")

        # ── Stage 2: Response ──
        context = f"""Patient query: {query}

Triage decision:
{json.dumps(tool_call, indent=2)}

Tool result:
{json.dumps(tool_result, indent=2)}

Call ID: {call_id}

Please provide a clear, professional triage summary to the user."""

        response, response_latency = self.engine.generate(
            ASSISTANT_SYSTEM_PROMPT, context, max_tokens=500
        )

        total_latency = (time.time() - start) * 1000

        self._log(f"\n  [STAGE 3] Response generated:")
        self._log(f"    {response[:200]}")
        self._log(f"\n  Latency: {total_latency:.0f}ms (tool: {tool_latency*1000:.0f}ms, response: {response_latency*1000:.0f}ms)")

        self.verbose = old_verbose  # Restore original verbose setting

        return AgentResult(
            query=query,
            tool_call=tool_call,
            tool_result=tool_result,
            response=response,
            call_id=call_id,
            tool_latency_ms=tool_latency * 1000,
            response_latency_ms=response_latency * 1000,
            total_latency_ms=total_latency,
            success=not bool(error),
            error=error,
        )


# ── Colab Entry Point ───────────────────────────────────────

def main():
    """Run in Colab: loads model, runs 3 sample queries."""
    print("=" * 60)
    print("  LatentSig Medical Triage Agent")
    print("  Two-stage SLM: Tool Call → Execute → Respond")
    print("=" * 60)

    # Load model
    engine = SLMEngine()
    engine.load()

    # Create agent
    db = ToolDB(log_dir="tool_logs")
    agent = TriageAgent(engine, db=db, verbose=True)

    # Test queries
    queries = [
        "68-year-old male, sudden facial droop, cannot speak, right arm weakness.",
        "Patient says they have a plan to end their life tonight, has access to pills.",
        "Need blood pressure medication refill, 65-year-old male.",
    ]

    for q in queries:
        result = agent.run(q)
        print(f"\n{'─'*60}")
        print(f"  FINAL ANSWER:")
        print(f"  {result.response}")
        print(f"{'─'*60}")

    print(f"\n  Tool calls logged: {db.count()}")
    print(f"  Logs: tool_logs/")


if __name__ == "__main__":
    main()
