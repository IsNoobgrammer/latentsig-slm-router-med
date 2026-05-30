# ─────────────────────────────────────────────────────────────
# VERIFY — Quick sanity check after fine-tuning
#
# Loads the fine-tuned model and runs test queries to verify
# training produced sensible outputs. Visual inspection only.
#
# Usage (Colab / GPU machine):
#   python -m src.verify
#   python -m src.verify --adapter-path ./adapter
#   python -m src.verify --adapter-path fhai50032/latentsig-med-router-qwen3-4b
#   python -m src.verify --disable-thinking  # skip thinking tokens
# ─────────────────────────────────────────────────────────────

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompts import SYSTEM_PROMPT


# ── Test Queries ─────────────────────────────────────────────

TEST_QUERIES = [
    # Emergency — stroke
    "68-year-old male, sudden facial droop, cannot speak, right arm weakness.",
    # Emergency — MI
    "Crushing chest pain radiating to left arm, sweating, nausea, 55-year-old male.",
    # Emergency — suicidal
    "Patient says they have a plan to end their life tonight, has access to pills.",
    # Urgent — vitals
    "Patient presents with BP 180/110, heart rate 110, temp 38.5C, SpO2 94%.",
    # Urgent — medication
    "72-year-old male on digoxin, now experiencing nausea, vomiting, yellow-tinted vision.",
    # Semi-urgent — fracture
    "Fell off bike, arm bent at wrong angle, 35-year-old.",
    # Routine — refill
    "Need blood pressure medication refill, 65-year-old male.",
    # Routine — chronic
    "Persistent cough for 3 weeks, no fever, 42-year-old female.",
    # Hinglish
    "23 saal ka ladka, bahut tez bukhar 103F, 3 din se, pet mein dard bhi hai.",
    # Vague — edge case
    "I have a headache.",
]


def load_model(adapter_path, base_model, max_seq_length, enable_thinking):
    """Load fine-tuned model via Unsloth."""
    from unsloth import FastLanguageModel
    import torch

    print(f"Loading model...")
    print(f"  Base: {base_model}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Max seq length: {max_seq_length}")
    print(f"  Thinking: {'enabled' if enable_thinking else 'disabled'}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"  Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, tokenizer


def run_query(model, tokenizer, query, max_tokens=512, enable_thinking=True):
    """Run a single query through the model. Returns (response, latency)."""
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.6 if enable_thinking else 0.1,
            top_p=0.95 if enable_thinking else None,
            top_k=20 if enable_thinking else None,
            do_sample=True,
        )
    latency = time.time() - start

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response, latency


def parse_and_validate(response):
    """Try to parse JSON from response. Returns (dict, error)."""
    text = response.strip()

    # Try direct parse
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    # Try extracting JSON
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1]), None
        except json.JSONDecodeError:
            pass

    return None, "Could not parse JSON"


def main():
    parser = argparse.ArgumentParser(description="Verify fine-tuned model")

    parser.add_argument("--adapter-path", default="fhai50032/latentsig-med-router-qwen3-4b",
                        help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--base-model", default="unsloth/Qwen3-4B-Instruct",
                        help="Base model")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Max sequence length")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max new tokens to generate")
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Disable Qwen3 thinking mode")
    parser.add_argument("--queries", nargs="+", default=None,
                        help="Custom queries to test")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(
        args.adapter_path, args.base_model,
        args.max_seq_length, not args.disable_thinking,
    )

    queries = args.queries or TEST_QUERIES
    enable_thinking = not args.disable_thinking

    # ── Run Tests ──
    print(f"\n{'='*70}")
    print(f"  VERIFY — {len(queries)} test queries")
    print(f"{'='*70}")

    results = []
    for i, query in enumerate(queries):
        print(f"\n{'─'*70}")
        print(f"  [{i+1}/{len(queries)}] {query[:80]}")
        print(f"{'─'*70}")

        response, latency = run_query(model, tokenizer, query,
                                       args.max_tokens, enable_thinking)

        # Parse
        parsed, parse_error = parse_and_validate(response)

        # Display
        if enable_thinking:
            # Separate thinking from answer
            think_marker = "<think>"
            end_marker = "</think>"
            if think_marker in response:
                think_start = response.index(think_marker) + len(think_marker)
                think_end = response.index(end_marker) if end_marker in response else len(response)
                thinking = response[think_start:think_end].strip()
                answer = response[think_end + len(end_marker):].strip() if end_marker in response else ""
                print(f"\n  THINKING:\n    {thinking[:300]}...")
                print(f"\n  ANSWER:\n    {answer[:500]}")
            else:
                print(f"\n  OUTPUT:\n    {response[:500]}")
        else:
            print(f"\n  OUTPUT:\n    {response[:500]}")

        # Validation
        if parsed:
            tool = parsed.get("tool", "?")
            category = parsed.get("category", "?")
            reasoning = parsed.get("reasoning", "?")[:100]
            print(f"\n  PARSED OK:")
            print(f"    Tool:      {tool}")
            print(f"    Category:  {category}")
            print(f"    Reasoning: {reasoning}")
            results.append({"query": query, "tool": tool, "category": category,
                          "parsed": True, "latency": latency})
        else:
            print(f"\n  PARSE FAILED: {parse_error}")
            results.append({"query": query, "tool": "?", "category": "?",
                          "parsed": False, "latency": latency, "error": parse_error})

        print(f"  Latency: {latency:.2f}s | Tokens: {len(tokenizer.encode(response))}")

    # ── Summary ──
    parsed_count = sum(1 for r in results if r["parsed"])
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Total:    {len(results)}")
    print(f"  Parsed:   {parsed_count}/{len(results)} ({parsed_count/len(results)*100:.0f}%)")
    print(f"  Avg time: {sum(r['latency'] for r in results)/len(results):.2f}s")

    # Tool distribution
    from collections import Counter
    tools = Counter(r["tool"] for r in results if r["parsed"])
    print(f"\n  Tool distribution:")
    for tool, count in tools.most_common():
        print(f"    {tool:<28} {count}")

    cats = Counter(r["category"] for r in results if r["parsed"])
    print(f"\n  Category distribution:")
    for cat, count in cats.most_common():
        print(f"    {cat:<16} {count}")

    if parsed_count == len(results):
        print(f"\n  ALL QUERIES PARSED — model looks good!")
    elif parsed_count >= len(results) * 0.8:
        print(f"\n  MOSTLY OK — {len(results) - parsed_count} parse failures")
    else:
        print(f"\n  WARNING — {len(results) - parsed_count}/{len(results)} parse failures")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
