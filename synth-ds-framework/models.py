# ─────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# Phase 1 verifier: structural validation
# ─────────────────────────────────────────────────────────────

from pydantic import BaseModel, field_validator, model_validator
from typing import Literal
import json


VALID_CATEGORIES = ["emergency", "urgent", "semi_urgent", "routine"]
VALID_TOOLS = [
    "triage_assessment",
    "vital_signs_analysis",
    "medication_check",
    "specialist_referral",
    "emergency_dispatch",
    "mental_health_triage",
    "lab_order_suggestion",
]


class TriageResponse(BaseModel):
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
        if len(v) > 500:
            raise ValueError("reasoning too long (>500 chars)")
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

    @model_validator(mode="after")
    def category_tool_consistency(self):
        """Emergency category should use emergency_dispatch or triage_assessment."""
        if self.category == "emergency" and self.tool not in [
            "emergency_dispatch", "triage_assessment", "vital_signs_analysis"
        ]:
            # Not a hard error — but flagged as warning in verifier
            pass
        return self


class DatasetRecord(BaseModel):
    """One row in the final dataset."""

    system_prompt: str
    user_query: str
    response: str  # raw JSON string from model
    parsed_response: dict  # parsed JSON
    generation_model_id: str
    language: Literal["en", "hi_en"]
    llm_judge_id: str
    judge_verdict: Literal["pass", "fail"]
    hash: str

    @field_validator("user_query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("user_query must not be empty")
        return v.strip()


def parse_and_validate_response(raw: str) -> tuple[TriageResponse | None, str | None]:
    """Parse raw model output and validate against TriageResponse schema.

    Returns (parsed_model, None) on success, (None, error_msg) on failure.
    """
    # Try direct parse
    text = raw.strip()

    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Try to extract JSON from surrounding text
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"

    try:
        validated = TriageResponse(**data)
        return validated, None
    except Exception as e:
        return None, f"Validation error: {e}"
