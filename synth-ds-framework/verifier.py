# ─────────────────────────────────────────────────────────────
# VERIFIER — 3-layer verification pipeline
# Phase 1: Pydantic (structural)
# Phase 2: Fuzzy (field matching)
# Phase 3: LLM judge (Mistral-small)
# ─────────────────────────────────────────────────────────────

import json
import urllib.request
import time
from difflib import SequenceMatcher
from models import parse_and_validate_response, TriageResponse, VALID_CATEGORIES


# ── Phase 1: Pydantic Structural Validation ─────────────────

def verify_structural(raw_response: str) -> tuple[bool, TriageResponse | None, list[str]]:
    """Phase 1: Validate JSON structure and field types.

    Returns (passed, parsed_model, errors).
    """
    parsed, error = parse_and_validate_response(raw_response)
    if parsed is None:
        return False, None, [error]
    return True, parsed, []


# ── Phase 2: Fuzzy Field Matching ────────────────────────────

EMERGENCY_KEYWORDS = [
    "stroke", "heart attack", "cardiac arrest", "can't breathe",
    "cannot breathe", "unconscious", "seizure", "anaphylaxis",
    "severe bleeding", "chest pain", "overdose", "suicidal",
    "not breathing", "choking", "drowning", "head trauma",
    "gunshot", "stab wound", "amputation", "paralyzed",
    "facial droop", "slurred speech", "blue lips",
    "crushing pain", "thunderclap headache", "worst headache",
]

URGENT_KEYWORDS = [
    "high fever", "fracture", "broken bone", "deep cut",
    "severe pain", "vomiting blood", "blood in stool",
    "severe headache", "confusion", "dehydration",
    "abdominal pain", "swollen", "infection", "laceration",
]

ROUTINE_KEYWORDS = [
    "refill", "checkup", "annual", "mild", "chronic",
    "follow-up", "prescription renewal", "routine",
    "persistent cough", "low-grade fever", "cold symptoms",
]


def fuzzy_match(expected: str, actual: str, threshold: float = 0.6) -> bool:
    """Check if two strings are semantically similar."""
    if not expected or not actual:
        return False
    expected_lower = expected.lower().strip()
    actual_lower = actual.lower().strip()

    # Exact match
    if expected_lower == actual_lower:
        return True

    # Containment
    if expected_lower in actual_lower or actual_lower in expected_lower:
        return True

    # Sequence matcher
    ratio = SequenceMatcher(None, expected_lower, actual_lower).ratio()
    return ratio >= threshold


def verify_semantic(
    query: str, parsed: TriageResponse, tool_schemas: dict, target_tool: str = None
) -> tuple[bool, list[str]]:
    """Phase 2: Check if response makes sense for the query.

    Args:
        target_tool: If set, the model MUST use this tool. Hard fail if mismatch.

    Returns (passed, warnings).
    """
    warnings = []
    query_lower = query.lower()

    # HARD CHECK: target tool enforcement (fast, deterministic)
    if target_tool and parsed.tool != target_tool:
        warnings.append(
            f"TARGET TOOL MISMATCH: expected '{target_tool}', got '{parsed.tool}'"
        )
        return False, warnings

    # Check category vs query keywords
    has_emergency = any(kw in query_lower for kw in EMERGENCY_KEYWORDS)
    has_urgent = any(kw in query_lower for kw in URGENT_KEYWORDS)
    has_routine = any(kw in query_lower for kw in ROUTINE_KEYWORDS)

    if has_emergency and parsed.category not in ["emergency"]:
        warnings.append(
            f"Emergency keywords in query but category={parsed.category}"
        )

    if has_routine and not has_emergency and parsed.category == "emergency":
        warnings.append(f"Routine keywords but category=emergency (may be over-triage)")

    # Check tool exists in schemas
    if parsed.tool not in tool_schemas:
        warnings.append(f"Tool '{parsed.tool}' not in provided schemas")
        return False, warnings

    # Check tool-category consistency
    if parsed.category == "emergency" and parsed.tool == "lab_order_suggestion":
        warnings.append("Emergency category with lab_order_suggestion — suspicious")

    if parsed.category == "routine" and parsed.tool == "emergency_dispatch":
        warnings.append("Routine category with emergency_dispatch — contradictory")
        return False, warnings

    # Check required args for the tool
    schema = tool_schemas[parsed.tool]
    required_params = set(schema["parameters"].keys())
    provided_params = set(parsed.args.keys())
    missing = required_params - provided_params
    if missing:
        warnings.append(f"Missing args for {parsed.tool}: {missing}")

    # Fail if critical issues, pass with warnings otherwise
    critical = [w for w in warnings if "contradictory" in w.lower() or "not in" in w.lower()]
    passed = len(critical) == 0

    return passed, warnings


# ── Phase 3: LLM Judge (Mistral-small) ──────────────────────

JUDGE_PROMPT = """You are a medical triage judge. Your job is to independently evaluate whether the triage decision is correct for the given patient query.

PATIENT QUERY:
{query}

TRIAGE DECISION (model output):
{response}

AVAILABLE TOOLS (for reference):
{tool_list}

EVALUATION CRITERIA (check each one):
1. TOOL SELECTION: Is the selected tool the MOST appropriate for this query? Consider:
   - emergency_dispatch → ONLY for life-threatening conditions (stroke, MI, anaphylaxis, cardiac arrest, severe trauma, respiratory failure)
   - triage_assessment → Initial symptom evaluation for non-emergency cases
   - vital_signs_analysis → When the query mentions specific vital signs (BP, HR, temp, SpO2, RR) that need interpretation
   - medication_check → When the query is about drug interactions, dosage errors, contraindications, or overdose
   - specialist_referral → When symptoms suggest a condition needing specialist evaluation (not emergency)
   - mental_health_triage → When the query involves suicidal ideation, self-harm, psychosis, severe anxiety/panic
   - lab_order_suggestion → When symptoms suggest need for diagnostic tests (fatigue + pallor, weight loss, recurrent infections)

2. CATEGORY: Is the category correct?
   - emergency → immediate life threat
   - urgent → serious, needs care within hours
   - semi_urgent → needs care within 24 hours
   - routine → can wait for scheduled appointment

3. ARGUMENTS: Are the tool arguments correct and complete for the selected tool?

4. REASONING: Is the clinical reasoning sound and consistent with the tool choice?

IMPORTANT: If ANY of the above criteria fail, the verdict must be "fail". Be strict — this data will be used to train a medical AI.

Respond with ONLY a JSON object:
{{"verdict": "pass" or "fail", "reason": "brief explanation focusing on what passed or failed"}}"""


def verify_llm_judge(
    query: str,
    response: str,
    api_key: str,
    tool_schemas: dict = None,
    model: str = "mistral-small-latest",
    max_retries: int = 2,
) -> tuple[bool, str, str]:
    """Phase 3: LLM-as-judge using Mistral. Does NOT know the target tool.

    Returns (passed, judge_model_id, judge_reasoning).
    """
    # Build tool list for the judge (so it knows what tools exist)
    if tool_schemas:
        tool_lines = []
        for name, schema in tool_schemas.items():
            tool_lines.append(f"- {name}: {schema['description']}")
        tool_list = "\n".join(tool_lines)
    else:
        tool_list = "(tool list not provided)"

    prompt = JUDGE_PROMPT.format(query=query, response=response, tool_list=tool_list)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }).encode()

    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                "https://api.mistral.ai/v1/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            judge_output = json.loads(content)

            verdict = judge_output.get("verdict", "fail").lower().strip()
            reason = judge_output.get("reason", "no reason provided")
            passed = verdict == "pass"
            return passed, model, reason

        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return False, model, f"Judge error after {max_retries+1} attempts: {e}"

    return False, model, "Judge exhausted retries"


# ── Combined Verifier ────────────────────────────────────────

class Verifier:
    """3-layer verification pipeline."""

    def __init__(self, tool_schemas: dict, mistral_keys: list[str], judge_model: str = "mistral-small-latest"):
        self.tool_schemas = tool_schemas
        self.mistral_keys = mistral_keys
        self.judge_model = judge_model
        self._key_idx = 0

    def _next_key(self) -> str:
        key = self.mistral_keys[self._key_idx % len(self.mistral_keys)]
        self._key_idx += 1
        return key

    def verify(self, query: str, raw_response: str, target_tool: str = None) -> dict:
        """Run all 3 verification phases.

        Args:
            target_tool: If set, enforce that the model used this specific tool.

        Returns:
            {
                "passed": bool,
                "parsed": dict | None,
                "structural": {"passed": bool, "errors": [...]},
                "semantic": {"passed": bool, "warnings": [...]},
                "llm_judge": {"passed": bool, "model": str, "reason": str},
                "verdict": "pass" | "fail",
            }
        """
        result = {
            "passed": False,
            "parsed": None,
            "structural": {"passed": False, "errors": []},
            "semantic": {"passed": False, "warnings": []},
            "llm_judge": {"passed": False, "model": "", "reason": ""},
            "verdict": "fail",
        }

        # Phase 1: Structural
        struct_ok, parsed, errors = verify_structural(raw_response)
        result["structural"] = {"passed": struct_ok, "errors": errors}
        if not struct_ok:
            result["verdict"] = "fail"
            return result
        result["parsed"] = parsed.model_dump()

        # Phase 2: Semantic
        sem_ok, warnings = verify_semantic(query, parsed, self.tool_schemas, target_tool)
        result["semantic"] = {"passed": sem_ok, "warnings": warnings}
        if not sem_ok:
            result["verdict"] = "fail"
            return result

        # Phase 3: LLM Judge (does NOT know target tool — unbiased)
        key = self._next_key()
        judge_ok, judge_model, reason = verify_llm_judge(
            query, raw_response, key, self.tool_schemas, self.judge_model
        )
        result["llm_judge"] = {"passed": judge_ok, "model": judge_model, "reason": reason}

        # Final verdict: all 3 must pass
        result["passed"] = struct_ok and sem_ok and judge_ok
        result["verdict"] = "pass" if result["passed"] else "fail"

        return result
