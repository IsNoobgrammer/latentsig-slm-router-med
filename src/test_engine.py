# ─────────────────────────────────────────────────────────────
# TEST — Interactive query tester for any engine
#
# Usage:
#   # Test Mistral baseline
#   python -m src.test_engine --engine mistral
#
#   # Test with specific model
#   python -m src.test_engine --engine mistral --model mistral-large-latest
#
#   # Test placeholder (no API needed)
#   python -m src.test_engine --engine placeholder
#
#   # Test SLM (Colab — needs GPU)
#   python -m src.test_engine --engine slm --adapter-path fhai50032/latentsig-med-router-qwen3-4b
#
#   # Single query (non-interactive)
#   python -m src.test_engine --engine mistral --query "chest pain, 55yo male"
# ─────────────────────────────────────────────────────────────

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import TriageAgent, run_agent
from src.inference import PlaceholderEngine, MistralAPIEngine, create_engine
from src.prompts import SYSTEM_PROMPT


def load_mistral_keys():
    """Load Mistral API keys from api_keys.md or env."""
    import re

    # Try env first
    env_keys = os.environ.get("MISTRAL_API_KEYS")
    if env_keys:
        return [k.strip() for k in env_keys.split(",")]

    # Try api_keys.md
    keys_path = os.path.join(os.path.expanduser("~"), "AppData", "Local",
                             "hermes", "api_keys.md")
    if os.path.exists(keys_path):
        with open(keys_path) as f:
            content = f.read()
        section = re.search(r"## Mistral.*?(?=##|\Z)", content, re.DOTALL)
        if section:
            keys = re.findall(r"`([A-Za-z0-9]{32})`", section.group())
            if keys:
                return keys

    return None


def run_single_query(engine, query, max_retries=3):
    """Run a single query with full verbose output."""
    print(f"\n{'='*70}")
    print(f"  QUERY: {query}")
    print(f"{'='*70}")

    agent = TriageAgent(engine, max_retries=max_retries, verbose=True)
    result = agent.run(query)

    print(f"\n{'─'*70}")
    print(f"  RESULT SUMMARY")
    print(f"{'─'*70}")
    print(f"  Success:    {result.success}")
    print(f"  Fallback:   {result.is_fallback}")
    print(f"  Attempts:   {result.attempts}")
    print(f"  Tool:       {result.triage_decision.get('tool', 'N/A') if result.triage_decision else 'N/A'}")
    print(f"  Category:   {result.triage_decision.get('category', 'N/A') if result.triage_decision else 'N/A'}")
    print(f"  Latency:    {result.total_latency_ms:.0f}ms")
    print(f"  Call ID:    {result.tool_call_id}")
    print(f"{'─'*70}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Interactive engine tester")

    parser.add_argument("--engine", choices=["placeholder", "mistral", "slm"],
                        default="mistral", help="Engine to test")
    parser.add_argument("--model", default="mistral-small-latest",
                        help="Mistral model name")
    parser.add_argument("--api-keys", default=None,
                        help="Comma-separated API keys")
    parser.add_argument("--adapter-path", default="fhai50032/latentsig-med-router-qwen3-4b",
                        help="SLM adapter path")
    parser.add_argument("--query", default=None,
                        help="Single query to test (non-interactive)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max parse retries")

    args = parser.parse_args()

    # Build engine
    if args.engine == "placeholder":
        engine = PlaceholderEngine()
        print("Engine: placeholder (keyword-based, no API)")

    elif args.engine == "mistral":
        api_keys = None
        if args.api_keys:
            api_keys = [k.strip() for k in args.api_keys.split(",")]
        else:
            api_keys = load_mistral_keys()
        if not api_keys:
            print("ERROR: No Mistral API keys. Use --api-keys or set MISTRAL_API_KEYS")
            sys.exit(1)
        engine = MistralAPIEngine(api_keys=api_keys, model=args.model)
        print(f"Engine: mistral ({args.model}, {len(api_keys)} keys)")

    elif args.engine == "slm":
        from src.inference import UnslothEngine
        engine = UnslothEngine(
            base_model="unsloth/Qwen3-4B-Instruct",
            adapter_path=args.adapter_path,
        )
        engine.load()
        print(f"Engine: SLM ({args.adapter_path})")

    # Single query mode
    if args.query:
        run_single_query(engine, args.query, args.max_retries)
        return

    # Interactive mode
    print(f"\n{'='*70}")
    print(f"  LatentSig Interactive Query Tester")
    print(f"  Type a patient query, or 'quit' to exit")
    print(f"  Try: 'chest pain, 55yo male' or 'patient wants to hurt themselves'")
    print(f"{'='*70}")

    query_count = 0
    while True:
        try:
            query = input(f"\n  Query #{query_count + 1}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("  Bye!")
            break

        run_single_query(engine, query, args.max_retries)
        query_count += 1


if __name__ == "__main__":
    main()
