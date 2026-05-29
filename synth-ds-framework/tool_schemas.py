# ─────────────────────────────────────────────────────────────
# TOOL REGISTRY
# Design principles:
#   - Flat params (no nesting) → easier for 1B-3B models to learn
#   - urgency_level inside params (not top-level) → one JSON shape for all
#   - All enums explicit → model has bounded output space
# ─────────────────────────────────────────────────────────────

TOOL_SCHEMAS = {
    "triage_assessment": {
        "description": "Initial symptom triage + urgency classification",
        "parameters": {
            "chief_complaint": "str",
            "symptoms": "list[str]",        # snake_case clinical terms
            "duration": "str",              # e.g. '2_hours', '3_days', 'acute'
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
            "clinical_context": "str"       # free text, why vitals were taken
        }
    },
    "medication_check": {
        "description": "Drug interaction | dosage | contraindication | overdose check",
        "parameters": {
            "medications": "list[str]",     # e.g. ['warfarin_5mg', 'ibuprofen_400mg']
            "check_type": "interaction|dosage|contraindication|overdose_risk",
            "patient_condition": "str",
            "patient_age_group": "pediatric|adult|geriatric"
        }
    },
    "specialist_referral": {
        "description": "Route patient to appropriate specialty",
        "parameters": {
            "specialty": "str",             # cardiology, neurosurgery, etc.
            "reason": "str",               # clinical reason / suspected dx
            "urgency": "emergency|within_24h|within_week|routine",
            "referring_symptoms": "list[str]"
        }
    },
    "emergency_dispatch": {
        "description": "Trigger emergency services — life-threatening conditions ONLY",
        "parameters": {
            "condition": "str",            # suspected dx
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
