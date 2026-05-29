# ─────────────────────────────────────────────────────────────
# TOOLS — 7 deterministic mock tools for medical triage
#
# Design principles:
#   - Deterministic: same input → same output (reproducible)
#   - No LLM calls: pure functions with fixed response templates
#   - Logged: every tool call is logged with timestamp + args
#   - Simple: responses are mock data, not real medical logic
# ─────────────────────────────────────────────────────────────

import os
import json
import time
import hashlib
from datetime import datetime
from typing import Any


# ── Tool Call Log (in-memory + CSV files) ────────────────────

import csv

class ToolCallLog:
    """In-memory + CSV log of all tool calls."""

    def __init__(self, log_dir: str = None):
        self.calls: list[dict] = []
        self.log_dir = log_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tool_logs"
        )
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, tool_name: str, args: dict, result: dict) -> str:
        """Log a tool call to memory + CSV file."""
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

        # Write to tool-specific CSV
        csv_path = os.path.join(self.log_dir, f"{tool_name}.csv")
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["call_id", "tool", "args", "result", "timestamp"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)

        return call_id

    def get_all(self) -> list[dict]:
        return self.calls

    def get_last_n(self, n: int = 10) -> list[dict]:
        return self.calls[-n:]

    def count(self) -> int:
        return len(self.calls)

    def clear(self):
        self.calls.clear()


# Global log instance
_tool_log = ToolCallLog()


def get_tool_log() -> ToolCallLog:
    return _tool_log


# ── Tool Implementations ─────────────────────────────────────

def triage_assessment(args: dict) -> dict:
    """Initial symptom triage + urgency classification."""
    chief_complaint = args.get("chief_complaint", "unknown")
    symptoms = args.get("symptoms", [])
    severity = args.get("severity", "moderate")
    urgency = args.get("urgency_level", "urgent")

    result = {
        "status": "triage_complete",
        "chief_complaint": chief_complaint,
        "symptoms_cataloged": len(symptoms),
        "severity_assessment": severity,
        "assigned_urgency": urgency,
        "recommended_action": (
            "immediate_evaluation" if urgency == "emergency"
            else "same_day_evaluation" if urgency == "urgent"
            else "next_available_appointment"
        ),
        "triage_id": f"TRG-{hashlib.sha256(chief_complaint.encode()).hexdigest()[:8].upper()}",
    }
    _tool_log.log("triage_assessment", args, result)
    return result


def vital_signs_analysis(args: dict) -> dict:
    """Analyze vitals for clinical decision making."""
    bp_sys = args.get("bp_systolic", 120)
    bp_dia = args.get("bp_diastolic", 80)
    hr = args.get("heart_rate", 72)
    temp = args.get("temperature_celsius", 37.0)
    spo2 = args.get("spo2_percent", 98)
    rr = args.get("respiratory_rate", 16)

    # Deterministic flag logic
    flags = []
    if bp_sys > 180 or bp_dia > 120:
        flags.append("hypertensive_crisis")
    if bp_sys < 90:
        flags.append("hypotension")
    if hr > 120:
        flags.append("tachycardia")
    if hr < 50:
        flags.append("bradycardia")
    if temp > 39.0:
        flags.append("high_fever")
    if spo2 < 90:
        flags.append("hypoxemia")
    if rr > 24:
        flags.append("tachypnea")

    result = {
        "status": "analysis_complete",
        "vitals": {
            "bp": f"{bp_sys}/{bp_dia}",
            "heart_rate": hr,
            "temperature": temp,
            "spo2": spo2,
            "respiratory_rate": rr,
        },
        "flags": flags if flags else ["all_normal"],
        "risk_level": "critical" if len(flags) >= 3 else "elevated" if flags else "normal",
        "analysis_id": f"VSA-{hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:8].upper()}",
    }
    _tool_log.log("vital_signs_analysis", args, result)
    return result


def medication_check(args: dict) -> dict:
    """Drug interaction / dosage / contraindication / overdose check."""
    medications = args.get("medications", [])
    check_type = args.get("check_type", "interaction")
    condition = args.get("patient_condition", "")

    # Deterministic mock responses based on check_type
    responses = {
        "interaction": {
            "findings": f"Checked {len(medications)} medications for interactions",
            "interactions_found": len(medications) > 1,
            "severity": "moderate" if len(medications) > 1 else "none",
            "recommendation": "consult_pharmacist" if len(medications) > 1 else "no_action_needed",
        },
        "dosage": {
            "findings": f"Verified dosage for {', '.join(medications)}",
            "dosage_appropriate": True,
            "recommendation": "continue_current_regimen",
        },
        "contraindication": {
            "findings": f"Checked contraindications for {', '.join(medications)} given {condition}",
            "contraindications_found": False,
            "recommendation": "proceed_with_caution",
        },
        "overdose_risk": {
            "findings": f"Assessed overdose risk for {', '.join(medications)}",
            "overdose_risk": True,
            "severity": "high",
            "recommendation": "immediate_evaluation",
        },
    }

    result = {
        "status": "check_complete",
        "check_type": check_type,
        **responses.get(check_type, {"findings": "unknown check type"}),
        "check_id": f"MDC-{hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:8].upper()}",
    }
    _tool_log.log("medication_check", args, result)
    return result


def specialist_referral(args: dict) -> dict:
    """Route patient to appropriate specialty."""
    specialty = args.get("specialty", "general_medicine")
    reason = args.get("reason", "")
    urgency = args.get("urgency", "routine")

    result = {
        "status": "referral_created",
        "specialty": specialty,
        "urgency": urgency,
        "reason": reason,
        "estimated_wait": (
            "immediate" if urgency == "emergency"
            else "24_hours" if urgency == "within_24h"
            else "7_days" if urgency == "within_week"
            else "2_4_weeks"
        ),
        "referral_id": f"REF-{hashlib.sha256(specialty.encode()).hexdigest()[:8].upper()}",
    }
    _tool_log.log("specialist_referral", args, result)
    return result


def emergency_dispatch(args: dict) -> dict:
    """Trigger emergency services — life-threatening conditions ONLY."""
    condition = args.get("condition", "unknown")
    symptoms = args.get("symptoms", [])
    transport = args.get("transport_type", "ambulance")
    notify_er = args.get("notify_er", True)

    result = {
        "status": "dispatched",
        "condition": condition,
        "transport_type": transport,
        "er_notified": notify_er,
        "symptoms_count": len(symptoms),
        "eta_minutes": 8 if transport == "ambulance" else 15 if transport == "helicopter" else 0,
        "dispatch_id": f"EMD-{hashlib.sha256(condition.encode()).hexdigest()[:8].upper()}",
    }
    _tool_log.log("emergency_dispatch", args, result)
    return result


def mental_health_triage(args: dict) -> dict:
    """Mental health crisis assessment + routing."""
    concern = args.get("concern_type", "depression")
    risk = args.get("risk_level", "moderate")
    immediate = args.get("immediate_intervention", False)
    safety_plan = args.get("safety_plan_needed", False)

    result = {
        "status": "assessment_complete",
        "concern_type": concern,
        "risk_level": risk,
        "immediate_intervention_required": immediate,
        "safety_plan_initiated": safety_plan,
        "recommended_resources": (
            ["crisis_hotline", "emergency_psych"] if risk == "high"
            else ["therapy_referral", "psychiatrist"] if risk == "moderate"
            else ["counseling", "support_group"]
        ),
        "assessment_id": f"MHT-{hashlib.sha256(concern.encode()).hexdigest()[:8].upper()}",
    }
    _tool_log.log("mental_health_triage", args, result)
    return result


def lab_order_suggestion(args: dict) -> dict:
    """Suggest appropriate diagnostic tests."""
    tests = args.get("tests", [])
    urgency = args.get("urgency", "routine")
    suspected = args.get("suspected_condition", "")

    result = {
        "status": "order_suggested",
        "tests": tests,
        "test_count": len(tests),
        "urgency": urgency,
        "suspected_condition": suspected,
        "estimated_turnaround": (
            "1_hour" if urgency == "stat"
            else "4_6_hours" if urgency == "urgent"
            else "24_48_hours"
        ),
        "order_id": f"LAB-{hashlib.sha256(json.dumps(tests).encode()).hexdigest()[:8].upper()}",
    }
    _tool_log.log("lab_order_suggestion", args, result)
    return result


# ── Tool Registry ────────────────────────────────────────────

TOOL_REGISTRY = {
    "triage_assessment": triage_assessment,
    "vital_signs_analysis": vital_signs_analysis,
    "medication_check": medication_check,
    "specialist_referral": specialist_referral,
    "emergency_dispatch": emergency_dispatch,
    "mental_health_triage": mental_health_triage,
    "lab_order_suggestion": lab_order_suggestion,
}


def execute_tool(tool_name: str, args: dict) -> dict:
    """Execute a tool by name. Returns error dict if tool not found."""
    if tool_name not in TOOL_REGISTRY:
        return {
            "status": "error",
            "error": f"Unknown tool: {tool_name}",
            "available_tools": list(TOOL_REGISTRY.keys()),
        }
    try:
        return TOOL_REGISTRY[tool_name](args)
    except Exception as e:
        return {
            "status": "error",
            "error": f"Tool execution failed: {str(e)}",
            "tool": tool_name,
        }
