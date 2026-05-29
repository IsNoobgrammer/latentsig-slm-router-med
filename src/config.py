# ─────────────────────────────────────────────────────────────
# CONFIG — Model paths, hyperparams, constants
# ─────────────────────────────────────────────────────────────

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Model config (fine-tuned SLM)
BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507"  # or Phi-4-mini
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "adapter")  # LoRA adapter dir
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True  # QLoRA for T4

# Agent config
MAX_RETRIES = 3
INFERENCE_TIMEOUT = 30  # seconds
TEMPERATURE = 0.1  # low for structured output
MAX_NEW_TOKENS = 300

# Triage categories
CATEGORIES = ["emergency", "urgent", "semi_urgent", "routine"]

# Tool names (must match tool_schemas.py)
TOOLS = [
    "triage_assessment",
    "vital_signs_analysis",
    "medication_check",
    "specialist_referral",
    "emergency_dispatch",
    "mental_health_triage",
    "lab_order_suggestion",
]
