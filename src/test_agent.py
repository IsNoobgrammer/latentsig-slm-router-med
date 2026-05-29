# ─────────────────────────────────────────────────────────────
# TEST — Quick smoke test for the agentic loop
#
# Runs 10 diverse queries through the agent (placeholder mode)
# and verifies the full ReAct loop works.
#
# Usage:
#   cd latentsig-slm-router-med
#   python -m src.test_agent
# ─────────────────────────────────────────────────────────────

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import run_agent
from src.inference import create_engine
from src.tools import get_tool_log


# ── Test Queries ─────────────────────────────────────────────

TEST_QUERIES = [
    # Emergency
    "68-year-old male, sudden facial droop, cannot speak, right arm weakness.",
    "Crushing chest pain radiating to left arm, sweating, nausea, 55-year-old male.",
    "Patient says they have a plan to end their life tonight, has access to pills.",

    # Urgent
    "Patient presents with BP 180/110, heart rate 110, temp 38.5C, SpO2 94%.",
    "72-year-old male on digoxin, now experiencing nausea, vomiting, yellow-tinted vision.",
    "Fell off bike, arm bent at wrong angle, 35-year-old.",

    # Routine
    "Need blood pressure medication refill, 65-year-old male.",
    "Persistent cough for 3 weeks, no fever, 42-year-old female.",
    "Annual diabetes checkup, 58-year-old.",

    # Edge case
    "I have a headache.",  # vague — should still route
]


def main():
    print("=" * 60)
    print("  AGENT SMOKE TEST — 10 queries, placeholder engine")
    print("=" * 60)

    engine = create_engine("placeholder")
    results = []

    for i, query in enumerate(TEST_QUERIES):
        print(f"\n{'─' * 60}")
        print(f"  Query {i+1}: {query[:80]}")
        print(f"{'─' * 60}")

        result = run_agent(query, engine, verbose=True)
        results.append(result)

        print(f"\n  >> Success: {result.success}")
        print(f"  >> Attempts: {result.attempts}")
        print(f"  >> Latency: {result.total_latency_ms:.0f}ms")
        print(f"  >> Fallback: {result.is_fallback}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total queries: {len(results)}")
    print(f"  Successful: {sum(1 for r in results if r.success)}")
    print(f"  Fallbacks: {sum(1 for r in results if r.is_fallback)}")
    print(f"  Avg latency: {sum(r.total_latency_ms for r in results) / len(results):.0f}ms")
    print(f"  Tool calls logged: {get_tool_log().count()}")

    # Check all results have valid triage decisions
    all_valid = all(r.triage_decision is not None for r in results)
    all_have_tool = all(r.triage_decision.get("tool") in [
        "triage_assessment", "vital_signs_analysis", "medication_check",
        "specialist_referral", "emergency_dispatch", "mental_health_triage",
        "lab_order_suggestion"
    ] for r in results if r.triage_decision)

    print(f"  All have decisions: {all_valid}")
    print(f"  All tools valid: {all_have_tool}")

    if all_valid and all_have_tool:
        print(f"\n  ALL TESTS PASSED")
    else:
        print(f"\n  SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
