# ─────────────────────────────────────────────────────────────
# DATASET ORCHESTRATOR
# Pipeline: tool_schemas → system_prompt → datagen → verify → store
#
# Output columns:
#   system_prompt, user_query, response, parsed_response,
#   generation_model_id, language, llm_judge_id,
#   judge_verdict, hash
#
# Dedup: hash = sha256(user_query + judge_verdict)
# No 2 rows with same hash allowed
# ─────────────────────────────────────────────────────────────

import json
import hashlib
import time
import csv
import os
import random
import urllib.request
from datetime import datetime

from tool_schemas import TOOL_SCHEMAS
from prompts import build_system_prompt
from verifier import Verifier


# ── Config ───────────────────────────────────────────────────

MISTRAL_KEYS = [
    "PqZ4lWpWWLiHtbQN13eqWYaAhqu9YzW5",
    "cEFMMhUj8vxbm24Tlkp7S8LMnK9kjLLx",
    "VP7hFDdY8eI3jEvRydyMOiJKh9bxYuuK",
    "078lyqOcoR5gfSMCRxpwL8hrQYZjNBYQ",
    "R3LaxIzd8DA01GSSdWWydrLSaXIvt7eh",
    "Uru07zRbL5bCoB9OPrWNqKRT9jFM59sW",
    "W7ZPD7aH8yp3ErdlumJXHQErAf74keDn",
    "0BOLgxpIkkZcczWTJL2eykkA2DcIYjMo",
]

# Generation: rotate across 3 better models for diversity
GENERATION_MODELS = [
    "mistral-large-latest",
    "magistral-medium-latest",
    "mistral-medium-latest",
]

# Judge: cheapest model (mistral-small)
JUDGE_MODEL = "mistral-small-latest"

TARGET_EN = 5000
TARGET_HI_EN = 5000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.jsonl")
HASH_SEEN_PATH = os.path.join(OUTPUT_DIR, ".hashes_seen.json")


# ── Key Rotator ──────────────────────────────────────────────

class KeyRotator:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.idx = 0

    def next(self) -> str:
        key = self.keys[self.idx % len(self.keys)]
        self.idx += 1
        return key


# ── Mistral API Caller ───────────────────────────────────────

def call_mistral(
    key: str,
    messages: list[dict],
    model: str = "mistral-small-latest",
    max_tokens: int = 512,
    temperature: float = 0.7,
    json_mode: bool = True,
) -> tuple[str, float, str]:
    """Call Mistral chat API. Returns (content, latency_seconds)."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        "https://api.mistral.ai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    start = time.time()
    resp = urllib.request.urlopen(req, timeout=60)
    latency = time.time() - start
    data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"], latency, model


# ── Query Generator ──────────────────────────────────────────

QUERY_GEN_SYSTEM = """You are a medical scenario generator. Generate realistic patient symptom descriptions for a medical triage system.

You must output ONLY a JSON object:
{{"query": "<patient symptom description in 1-2 sentences>"}}

Make the query:
- Realistic (like a real patient or nurse would describe)
- Varied in age, gender, symptoms, severity
- Include enough detail for triage (age, duration, key symptoms)
- Cover ALL categories: emergency, urgent, semi_urgent, routine
- Use natural language, not clinical jargon"""

HINGLISH_QUERY_GEN_SYSTEM = """You are a Hinglish medical scenario generator. Convert the given English medical query to natural Hinglish (Hindi written in Roman script, mixed with English medical terms).

Rules:
- Keep medical terms in English (e.g. "chest pain", "fever", "breathing problem")
- Use Hindi conversational style (e.g. "mujhe", "bahut", "lag raha hai", "ho gaya")
- Keep it natural — how an actual Hindi speaker would describe symptoms
- Output ONLY: {{"query": "<hinglish query>"}}

Example:
English: "68-year-old male with sudden facial droop, cannot speak"
Hinglish: "68 saal ke uncle ko achanak chehra gir gaya, bol nahi pa rahe"

English: "Persistent cough for 3 weeks, no fever, 42-year-old female"
Hinglish: "42 saal ki aurat ko 3 hafte se khansi aa rahi hai, bukhar nahi hai"

English: "Child swallowed a coin, coughing and gagging"
Hinglish: "bacche ne coin nigal liya, khasi aa rahi hai aur ulti jaisa feel ho raha"
"""


def generate_query(key: str, model: str, category_hint: str = None) -> tuple[str, float, str]:
    """Generate a random medical query. Returns (query, latency, model_used)."""
    hint = ""
    if category_hint:
        hint = f"\nGenerate a {category_hint} level case."

    messages = [
        {"role": "user", "content": QUERY_GEN_SYSTEM + hint}
    ]
    content, latency, model_used = call_mistral(key, messages, model=model, temperature=0.9)
    try:
        data = json.loads(content)
        return data["query"], latency, model_used
    except:
        # Try extracting from text
        if "query" in content:
            start = content.find('"query"')
            if start != -1:
                rest = content[start:]
                colon = rest.find(":")
                if colon != -1:
                    val = rest[colon + 1:].strip()
                    # Find the string value
                    if '"' in val:
                        val = val.split('"')[1]
                        return val, latency, model_used
        return content.strip()[:200], latency, model_used


def translate_to_hinglish(key: str, english_query: str, model: str) -> tuple[str, float, str]:
    """Translate English query to Hinglish. Returns (hinglish_query, latency, model_used)."""
    messages = [
        {"role": "user", "content": HINGLISH_QUERY_GEN_SYSTEM + f"\n\nEnglish: {english_query}"}
    ]
    content, latency, model_used = call_mistral(key, messages, model=model, temperature=0.7)
    try:
        data = json.loads(content)
        return data["query"], latency, model_used
    except:
        # Fallback: return original
        return english_query, latency, model_used


# ── Response Generator ───────────────────────────────────────

def generate_response(key: str, system_prompt: str, user_query: str, model: str) -> tuple[str, float, str]:
    """Generate triage response from the model. Returns (response_json, latency, model_used)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    content, latency, model_used = call_mistral(key, messages, model=model, max_tokens=300, temperature=0.1)
    return content, latency, model_used


# ── Hash & Dedup ─────────────────────────────────────────────

def compute_hash(user_query: str, verdict: str) -> str:
    """Hash of user_query + judge_verdict for dedup."""
    raw = f"{user_query.strip().lower()}|{verdict.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class HashStore:
    """Persistent set of seen hashes for deduplication."""

    def __init__(self, path: str):
        self.path = path
        self.seen: set[str] = set()
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.seen = set(json.load(f))

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(list(self.seen), f)

    def is_duplicate(self, h: str) -> bool:
        return h in self.seen

    def add(self, h: str):
        self.seen.add(h)
        self._save()

    def __len__(self):
        return len(self.seen)


# ── Dataset Writer ───────────────────────────────────────────

FIELDS = [
    "system_prompt",
    "user_query",
    "response",
    "parsed_response",
    "generation_model_id",
    "language",
    "llm_judge_id",
    "judge_verdict",
    "hash",
]


def write_record(path: str, record: dict):
    """Append one record to JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Orchestrator ─────────────────────────────────────────────

class DatasetOrchestrator:
    """Main orchestrator: generates, verifies, and stores dataset samples."""

    def __init__(self):
        self.system_prompt = build_system_prompt(TOOL_SCHEMAS)
        self.keys = KeyRotator(MISTRAL_KEYS)
        self.verifier = Verifier(TOOL_SCHEMAS, MISTRAL_KEYS, JUDGE_MODEL)
        self.hash_store = HashStore(HASH_SEEN_PATH)
        self.stats = {
            "generated": 0,
            "passed": 0,
            "failed_structural": 0,
            "failed_semantic": 0,
            "failed_judge": 0,
            "duplicates": 0,
            "en_count": 0,
            "hi_en_count": 0,
        }

    def _category_hint(self) -> str:
        """Random category hint for query generation."""
        return random.choice(["emergency", "urgent", "semi_urgent", "routine"])

    def generate_one(self, language: str) -> dict | None:
        """Generate one sample, verify, and return record or None if failed.

        Args:
            language: "en" or "hi_en"
        """
        # Pick a random generation model for this sample
        gen_model = random.choice(GENERATION_MODELS)

        key1 = self.keys.next()

        # Step 1: Generate query
        category_hint = self._category_hint()
        en_query, _, _ = generate_query(key1, gen_model, category_hint)

        if language == "hi_en":
            key2 = self.keys.next()
            user_query, _, _ = translate_to_hinglish(key2, en_query, gen_model)
        else:
            user_query = en_query

        # Step 2: Generate response
        key3 = self.keys.next()
        raw_response, _, _ = generate_response(key3, self.system_prompt, user_query, gen_model)

        # Step 3: Verify
        verification = self.verifier.verify(user_query, raw_response)

        verdict = verification["verdict"]

        # Step 4: Dedup
        h = compute_hash(user_query, verdict)
        if self.hash_store.is_duplicate(h):
            self.stats["duplicates"] += 1
            return None

        self.hash_store.add(h)

        # Step 5: Track stats
        self.stats["generated"] += 1
        if verdict == "pass":
            self.stats["passed"] += 1
        elif not verification["structural"]["passed"]:
            self.stats["failed_structural"] += 1
        elif not verification["semantic"]["passed"]:
            self.stats["failed_semantic"] += 1
        else:
            self.stats["failed_judge"] += 1

        if language == "en":
            self.stats["en_count"] += 1
        else:
            self.stats["hi_en_count"] += 1

        # Step 6: Build record
        record = {
            "system_prompt": self.system_prompt,
            "user_query": user_query,
            "response": raw_response,
            "parsed_response": json.dumps(verification["parsed"]) if verification["parsed"] else None,
            "generation_model_id": gen_model,
            "language": language,
            "llm_judge_id": JUDGE_MODEL,
            "judge_verdict": verdict,
            "hash": h,
        }

        return record

    def run(
        self,
        target_en: int = TARGET_EN,
        target_hi_en: int = TARGET_HI_EN,
        save_only_pass: bool = True,
        max_retries_per_sample: int = 3,
    ):
        """Run the full dataset generation pipeline.

        Args:
            target_en: Number of English samples to generate
            target_hi_en: Number of Hinglish samples to generate
            save_only_pass: If True, only save samples that pass all 3 verifiers
            max_retries_per_sample: Retries per failed sample before moving on
        """
        total_target = target_en + target_hi_en
        print(f"{'='*60}")
        print(f"  DATASET ORCHESTRATOR")
        print(f"  Target: {target_en} EN + {target_hi_en} HI_EN = {total_target} total")
        print(f"  Save only pass: {save_only_pass}")
        print(f"  Output: {DATASET_PATH}")
        print(f"{'='*60}\n")

        start_time = time.time()
        last_print = start_time

        while self.stats["en_count"] < target_en or self.stats["hi_en_count"] < target_hi_en:
            # Decide language
            if self.stats["en_count"] < target_en:
                language = "en"
            else:
                language = "hi_en"

            # Try generating with retries
            record = None
            for attempt in range(max_retries_per_sample):
                record = self.generate_one(language)
                if record is not None:
                    break

            if record is None:
                continue  # All retries failed or duplicate, skip

            # Save
            if save_only_pass and record["judge_verdict"] != "pass":
                continue  # Skip failed samples

            write_record(DATASET_PATH, record)

            # Progress
            now = time.time()
            if now - last_print >= 5:  # Print every 5 seconds
                elapsed = now - start_time
                total_saved = self.stats["passed"] if save_only_pass else self.stats["generated"]
                rate = total_saved / elapsed if elapsed > 0 else 0
                eta = (total_target - total_saved) / rate if rate > 0 else 0
                print(
                    f"  [{elapsed:.0f}s] EN={self.stats['en_count']} HI={self.stats['hi_en_count']} "
                    f"passed={self.stats['passed']} failed_struct={self.stats['failed_structural']} "
                    f"failed_sem={self.stats['failed_semantic']} failed_judge={self.stats['failed_judge']} "
                    f"dupes={self.stats['duplicates']} rate={rate:.1f}/s ETA={eta:.0f}s"
                )
                last_print = now

        # Final report
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  COMPLETE in {elapsed:.0f}s")
        print(f"  EN samples: {self.stats['en_count']}")
        print(f"  HI_EN samples: {self.stats['hi_en_count']}")
        print(f"  Passed: {self.stats['passed']}")
        print(f"  Failed (structural): {self.stats['failed_structural']}")
        print(f"  Failed (semantic): {self.stats['failed_semantic']}")
        print(f"  Failed (judge): {self.stats['failed_judge']}")
        print(f"  Duplicates skipped: {self.stats['duplicates']}")
        print(f"  Output: {DATASET_PATH}")
        print(f"{'='*60}")


# ── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Orchestrator")
    parser.add_argument("--en", type=int, default=TARGET_EN, help="English samples target")
    parser.add_argument("--hi-en", type=int, default=TARGET_HI_EN, help="Hinglish samples target")
    parser.add_argument("--save-all", action="store_true", help="Save failed samples too")
    parser.add_argument("--retries", type=int, default=3, help="Max retries per sample")
    args = parser.parse_args()

    orch = DatasetOrchestrator()
    orch.run(
        target_en=args.en,
        target_hi_en=args.hi_en,
        save_only_pass=not args.save_all,
        max_retries_per_sample=args.retries,
    )
