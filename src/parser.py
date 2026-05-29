# ─────────────────────────────────────────────────────────────
# PARSER — JSON extraction + Pydantic validation
#
# Handles:
#   - Clean JSON
#   - JSON in markdown code blocks
#   - JSON with surrounding text
#   - Partial/truncated JSON
#   - Field validation + enum checks
# ─────────────────────────────────────────────────────────────

import json
import re
from pydantic import BaseModel, field_validator, model_validator
from typing import Literal


# ── Pydantic Schema ──────────────────────────────────────────

VALID_TOOLS = [
    "triage_assessment",
    "vital_signs_analysis",
    "medication_check",
    "specialist_referral",
    "emergency_dispatch",
    "mental_health_triage",
    "lab_order_suggestion",
]


class TriageCall(BaseModel):
    """Schema the SLM must produce."""

    reasoning: str
    category: Literal["emergency", "urgent", "semi_urgent", "routine"]
    department: str
    tool: Literal[
        "triage_assessment",
        "vital_signs_analysis",
        "medication_check",
        "specialist_referral",
        "emergency_dispatch",
        "mental_health_triage",
        "lab_order_suggestion",
    ]
    args: dict

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("reasoning must not be empty")
        return v.strip()

    @field_validator("department")
    @classmethod
    def department_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("department must not be empty")
        return v.strip()

    @field_validator("args")
    @classmethod
    def args_not_empty(cls, v: dict) -> dict:
        if not v:
            raise ValueError("args must not be empty")
        return v


# ── Parse Result ─────────────────────────────────────────────

class ParseResult:
    """Result of parsing + validation."""

    def __init__(self, success: bool, data: dict | None, error: str | None, raw: str):
        self.success = success
        self.data = data
        self.error = error
        self.raw = raw

    def __bool__(self):
        return self.success

    def __repr__(self):
        if self.success:
            return f"ParseResult(ok, tool={self.data.get('tool')})"
        return f"ParseResult(fail, error={self.error})"


# ── JSON Extraction ──────────────────────────────────────────

def extract_json(raw: str) -> tuple[dict | None, str | None]:
    """Extract JSON from raw model output.

    Handles: clean JSON, markdown blocks, surrounding text, truncated.

    Returns (parsed_dict, error_message).
    """
    text = raw.strip()
    if not text:
        return None, "Empty output"

    # 1. Try direct parse
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line (```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        inner = "\n".join(lines).strip()
        try:
            return json.loads(inner), None
        except json.JSONDecodeError:
            pass

    # 3. Find JSON object in text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError:
            pass

    # 4. Try fixing common truncation (missing closing braces)
    if start != -1:
        candidate = text[start:]
        # Count open/close braces
        opens = candidate.count("{")
        closes = candidate.count("}")
        if opens > closes:
            candidate += "}" * (opens - closes)
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError:
            pass

    return None, f"Could not extract valid JSON from output: {raw[:200]}"


# ── Validation ───────────────────────────────────────────────

def validate_triage_call(data: dict) -> tuple[dict | None, str | None]:
    """Validate extracted JSON against TriageCall schema.

    Returns (validated_dict, error_message).
    """
    try:
        validated = TriageCall(**data)
        return validated.model_dump(), None
    except Exception as e:
        return None, f"Validation error: {e}"


# ── Main Parse Function ──────────────────────────────────────

def parse_model_output(raw: str) -> ParseResult:
    """Parse and validate raw model output.

    Returns ParseResult with success, data, error, raw.
    """
    # Step 1: Extract JSON
    data, extract_error = extract_json(raw)
    if data is None:
        return ParseResult(False, None, extract_error, raw)

    # Step 2: Validate against schema
    validated, validation_error = validate_triage_call(data)
    if validated is None:
        return ParseResult(False, None, validation_error, raw)

    return ParseResult(True, validated, None, raw)


# ── Error Context Builder ────────────────────────────────────

def build_error_context(raw_output: str, error: str, attempt: int) -> str:
    """Build a retry prompt with error context for the SLM.

    The SLM sees its own failed output + what went wrong,
    so it can self-correct.
    """
    return f"""Your previous output was invalid:

--- YOUR OUTPUT ---
{raw_output[:500]}
--- END OUTPUT ---

ERROR: {error}

Please fix the output. Return ONLY a valid JSON object with exactly these fields:
{{
  "reasoning": "<1-2 sentence clinical reasoning>",
  "category": "emergency|urgent|semi_urgent|routine",
  "department": "<target department>",
  "tool": "<tool_name>",
  "args": {{<tool-specific parameters>}}
}}

Attempt {attempt + 1} of 3. Output ONLY the JSON, no other text."""
