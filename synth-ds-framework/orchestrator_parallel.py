# ─────────────────────────────────────────────────────────────
# PARALLEL ORCHESTRATOR
# Same pipeline as orchestrator.py but with ThreadPoolExecutor
# for concurrent generation + verification + file writes
#
# Usage:
#   python orchestrator_parallel.py --en 5000 --hi-en 5000 --workers 8
# ─────────────────────────────────────────────────────────────

import json
import hashlib
import time
import os
import random
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from threading import Lock

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

GENERATION_MODELS = [
    "mistral-large-latest",
    "magistral-medium-latest",
    "mistral-medium-latest",
]

JUDGE_MODEL = "mistral-small-latest"
TARGET_EN = 5000
TARGET_HI_EN = 5000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.jsonl")
HASH_SEEN_PATH = os.path.join(OUTPUT_DIR, ".hashes_seen.json")
STATS_PATH = os.path.join(OUTPUT_DIR, ".live_stats.json")


# ── Thread-safe Key Rotator ──────────────────────────────────

class KeyRotator:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.idx = 0
        self._lock = Lock()

    def next(self) -> str:
        with self._lock:
            key = self.keys[self.idx % len(self.keys)]
            self.idx += 1
            return key


# ── Mistral API with Retry ───────────────────────────────────

def call_mistral(
    key: str,
    messages: list[dict],
    model: str = "mistral-small-latest",
    max_tokens: int = 512,
    temperature: float = 0.7,
    json_mode: bool = True,
    max_retries: int = 5,
) -> tuple[str, float, str]:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    payload = json.dumps(body).encode()

    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            "https://api.mistral.ai/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        try:
            start = time.time()
            resp = urllib.request.urlopen(req, timeout=60)
            latency = time.time() - start
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"], latency, model

        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                if attempt < max_retries:
                    wait = min(2 ** attempt + random.uniform(0, 1), 30)
                    time.sleep(wait)
                    if e.code == 429:
                        key = MISTRAL_KEYS[(attempt + 1) % len(MISTRAL_KEYS)]
                    continue
            raise

    raise RuntimeError(f"Mistral API failed after {max_retries + 1} attempts")


# ── Prompt Templates ─────────────────────────────────────────

QUERY_GEN_SYSTEM = """You are a medical scenario generator. Generate realistic patient symptom descriptions for a medical triage system.

You must output ONLY a JSON object:
{{"query": "<patient symptom description in 1-2 sentences>"}}

CRITICAL: You MUST generate a query that would specifically require the "{target_tool}" tool.

## Tool Descriptions
{tool_descriptions}

## Few-Shot Examples (observe which tool each query maps to)

### triage_assessment examples:
- "My 3-year-old has had a runny nose and mild cough for 2 days. No fever, still playing normally."
- "45-year-old male, persistent dull headache for a week, worse in mornings, no vision changes."
- "28-year-old female, sprained ankle playing soccer, can walk with limp, mild swelling."

### vital_signs_analysis examples:
- "Patient presents with BP 180/110, heart rate 110, temp 38.5C, SpO2 94%. Known hypertensive."
- "Elderly male, HR 45 bpm, BP 90/60, SpO2 91%, RR 22. Feeling dizzy and weak."
- "Post-op patient: temp 39.2C, HR 120, BP 100/70, SpO2 96%, RR 24."

### medication_check examples:
- "Patient on warfarin wants to take ibuprofen for back pain. Currently on 5mg daily."
- "68-year-old on metformin, lisinopril, and atorvastatin. New prescription for clarithromycin."
- "Mother says child took two doses of amoxicillin by mistake, double the prescribed amount."

### specialist_referral examples:
- "Persistent hoarseness for 6 weeks, non-smoker, no improvement with voice rest."
- "Recurring episodes of severe abdominal pain after eating, ultrasound shows gallstones."
- "Chronic knee pain worsening over 6 months, X-ray shows moderate osteoarthritis."

### emergency_dispatch examples:
- "68-year-old male, sudden facial droop, cannot speak, right arm weakness."
- "Crushing chest pain radiating to left arm, sweating, nausea, 55-year-old male."
- "3-year-old swallowed a button battery, crying, drooling, refusing to eat."

### mental_health_triage examples:
- "Patient says they have a plan to end their life tonight, has access to pills."
- "22-year-old hearing voices telling them to hurt themselves, not sleeping for 3 days."
- "Severe panic attack, hyperventilating, chest tightness, convinced they're dying."

### lab_order_suggestion examples:
- "Chronic fatigue for 3 months, pale skin, shortness of breath on exertion."
- "Unexplained weight loss of 10kg in 2 months, night sweats, low-grade fever."
- "Recurrent UTIs, now dysuria and urgency again, wants to check if infection cleared."

Now generate a NEW, UNIQUE query for the "{target_tool}" tool. Make it realistic and different from the examples above."""

HINGLISH_QUERY_GEN_SYSTEM = '''You are a Hinglish medical scenario generator. Convert the given English medical query to natural Hinglish (Hindi written in Roman script, mixed with English medical terms).

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
Hinglish: "bacche ne nigal liya coin, khasi aa rahi hai aur ulti jaisa feel ho raha"
'''


# ── Generator Functions ──────────────────────────────────────

def generate_query(key: str, model: str, category_hint: str = None, target_tool: str = None) -> tuple[str, float, str]:
    hint = ""
    if category_hint:
        hint = f"\nGenerate a {category_hint} level case."

    # Build tool descriptions for the prompt
    tool_descs = []
    for name, schema in TOOL_SCHEMAS.items():
        params = ", ".join(f"{k}: {v}" for k, v in schema["parameters"].items())
        tool_descs.append(f"- {name}({params}): {schema['description']}")
    tool_descriptions = "\n".join(tool_descs)

    prompt = QUERY_GEN_SYSTEM.format(
        target_tool=target_tool or "triage_assessment",
        tool_descriptions=tool_descriptions,
    ) + hint

    messages = [{"role": "user", "content": prompt}]
    content, latency, model_used = call_mistral(key, messages, model=model, temperature=0.9)
    try:
        data = json.loads(content)
        return data["query"], latency, model_used
    except:
        if "query" in content:
            start = content.find('"query"')
            if start != -1:
                rest = content[start:]
                colon = rest.find(":")
                if colon != -1:
                    val = rest[colon + 1:].strip()
                    if '"' in val:
                        val = val.split('"')[1]
                        return val, latency, model_used
        return content.strip()[:200], latency, model_used


def translate_to_hinglish(key: str, english_query: str, model: str) -> tuple[str, float, str]:
    messages = [{"role": "user", "content": HINGLISH_QUERY_GEN_SYSTEM + f"\n\nEnglish: {english_query}"}]
    content, latency, model_used = call_mistral(key, messages, model=model, temperature=0.7)
    try:
        data = json.loads(content)
        return data["query"], latency, model_used
    except:
        return english_query, latency, model_used


def generate_response(key: str, system_prompt: str, user_query: str, model: str, target_tool_hint: str = None) -> tuple[str, float, str]:
    """Generate triage response. If target_tool_hint is set, append it as guidance.

    The hint ensures clean training data. During inference, the fine-tuned model
    won't have this hint — it learns to pick the right tool from the query alone.
    """
    user_msg = user_query
    if target_tool_hint:
        user_msg += f"\n\n[Use the {target_tool_hint} tool for this case.]"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    content, latency, model_used = call_mistral(key, messages, model=model, max_tokens=300, temperature=0.1)
    return content, latency, model_used


# ── Hash & Dedup ─────────────────────────────────────────────

def compute_hash(user_query: str, verdict: str) -> str:
    raw = f"{user_query.strip().lower()}|{verdict.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class HashStore:
    def __init__(self, path: str):
        self.path = path
        self.seen: set[str] = set()
        self._lock = Lock()
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.seen = set(json.load(f))

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(list(self.seen), f)

    def is_duplicate(self, h: str) -> bool:
        with self._lock:
            return h in self.seen

    def add(self, h: str):
        with self._lock:
            self.seen.add(h)
            self._save()

    def __len__(self):
        with self._lock:
            return len(self.seen)


# ── Dataset Writer ───────────────────────────────────────────

class DatasetWriter:
    """Thread-safe JSONL writer."""

    def __init__(self, path: str):
        self.path = path
        self._lock = Lock()

    def append(self, record: dict):
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Live Stats (for monitor server) ─────────────────────────

class LiveStats:
    """Thread-safe stats that also writes to a JSON file for the monitor."""

    def __init__(self, path: str):
        self.path = path
        self._lock = Lock()
        self.data = {
            "start_time": time.time(),
            "target_en": 0,
            "target_hi_en": 0,
            "existing_en": 0,
            "existing_hi_en": 0,
            "en_count": 0,
            "hi_en_count": 0,
            "passed": 0,
            "failed_structural": 0,
            "failed_semantic": 0,
            "failed_judge": 0,
            "duplicates": 0,
            "errors": 0,
            "workers": 0,
            "rate": 0.0,
            "eta_seconds": 0,
            "recent_samples": [],  # last 10 records
        }

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if k == "recent_sample":
                    self.data["recent_samples"].insert(0, v)
                    self.data["recent_samples"] = self.data["recent_samples"][:20]
                else:
                    self.data[k] = v
            # Compute rate
            elapsed = time.time() - self.data["start_time"]
            new_total = (self.data["en_count"] - self.data["existing_en"]) + (self.data["hi_en_count"] - self.data["existing_hi_en"])
            self.data["rate"] = round(new_total / elapsed, 2) if elapsed > 0 else 0
            remaining = (self.data["target_en"] - self.data["en_count"]) + (self.data["target_hi_en"] - self.data["hi_en_count"])
            self.data["eta_seconds"] = int(remaining / self.data["rate"]) if self.data["rate"] > 0 else 0
            self.data["elapsed"] = int(elapsed)

            # Write to file
            try:
                with open(self.path, "w") as f:
                    json.dump(self.data, f, default=str)
            except:
                pass

    def increment(self, key: str, amount: int = 1):
        with self._lock:
            self.data[key] = self.data.get(key, 0) + amount

    def get(self) -> dict:
        with self._lock:
            return dict(self.data)


# ── Parallel Orchestrator ────────────────────────────────────

class ParallelOrchestrator:
    def __init__(self, workers: int = 8):
        self.system_prompt = build_system_prompt(TOOL_SCHEMAS)
        self.keys = KeyRotator(MISTRAL_KEYS)
        self.verifier = Verifier(TOOL_SCHEMAS, MISTRAL_KEYS, JUDGE_MODEL)
        self.hash_store = HashStore(HASH_SEEN_PATH)
        self.writer = DatasetWriter(DATASET_PATH)
        self.stats = LiveStats(STATS_PATH)
        self.workers = workers
        self._tool_counts = {t: 0 for t in TOOL_SCHEMAS}
        self._tool_lock = Lock()

    def _pick_target_tool(self) -> str:
        """Pick the least-used tool to ensure balanced coverage.
        Only counts SAVED samples (incremented in _generate_one on success).
        """
        with self._tool_lock:
            tools = list(self._tool_counts.keys())
            counts = [self._tool_counts[t] for t in tools]
            max_count = max(counts) if counts else 0
            # Weight = 1 + (max - count) so least-used tools get highest weight
            weights = [1 + (max_count - c) for c in counts]
            chosen = random.choices(tools, weights=weights, k=1)[0]
            return chosen  # Don't increment here — increment on save

    def _scan_existing(self) -> tuple[int, int, int]:
        en_count = 0
        hi_en_count = 0
        total = 0
        if not os.path.exists(DATASET_PATH):
            return 0, 0, 0
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    total += 1
                    if rec.get("language") == "en":
                        en_count += 1
                    elif rec.get("language") == "hi_en":
                        hi_en_count += 1
                    h = rec.get("hash")
                    if h:
                        self.hash_store.seen.add(h)
                    # Load tool counts for balancing
                    try:
                        parsed = json.loads(rec.get("parsed_response", "{}"))
                        tool = parsed.get("tool")
                        if tool:
                            self._tool_counts[tool] = self._tool_counts.get(tool, 0) + 1
                    except:
                        pass
                except json.JSONDecodeError:
                    continue
        return en_count, hi_en_count, total

    def _generate_one(self, language: str, save_only_pass: bool, target_tool: str = None) -> bool:
        """Generate one sample. Returns True if a record was saved.

        Args:
            target_tool: If set, generate a query that would use this specific tool.
        """
        try:
            gen_model = random.choice(GENERATION_MODELS)
            key1 = self.keys.next()

            # Pick target tool ONCE — retry with same tool on failure
            if target_tool is None:
                target_tool = self._pick_target_tool()

            # Generate query targeted at a specific tool
            category_hint = random.choice(["emergency", "urgent", "semi_urgent", "routine"])
            en_query, _, _ = generate_query(key1, gen_model, category_hint, target_tool)

            if language == "hi_en":
                key2 = self.keys.next()
                user_query, _, _ = translate_to_hinglish(key2, en_query, gen_model)
            else:
                user_query = en_query

            # Generate response (with tool hint for clean training data)
            key3 = self.keys.next()
            raw_response, _, _ = generate_response(key3, self.system_prompt, user_query, gen_model, target_tool)

            # Verify (pass target_tool for enforcement)
            verification = self.verifier.verify(user_query, raw_response, target_tool)
            verdict = verification["verdict"]

            # Dedup
            h = compute_hash(user_query, verdict)
            if self.hash_store.is_duplicate(h):
                self.stats.increment("duplicates")
                return False

            self.hash_store.add(h)

            # Track stats
            self.stats.increment("passed" if verdict == "pass" else "failed_" + (
                "structural" if not verification["structural"]["passed"] else
                "semantic" if not verification["semantic"]["passed"] else "judge"
            ))

            if language == "en":
                self.stats.increment("en_count")
            else:
                self.stats.increment("hi_en_count")

            # Save
            if save_only_pass and verdict != "pass":
                return False

            parsed = verification["parsed"] or {}
            record = {
                "system_prompt": self.system_prompt,
                "user_query": user_query,
                "response": raw_response,
                "parsed_response": json.dumps(verification["parsed"]) if verification["parsed"] else None,
                "tool_called": parsed.get("tool", ""),
                "category": parsed.get("category", ""),
                "generation_model_id": gen_model,
                "language": language,
                "llm_judge_id": JUDGE_MODEL,
                "judge_verdict": verdict,
                "hash": h,
            }

            self.writer.append(record)

            # Track tool usage for balancing (only on successful save)
            with self._tool_lock:
                self._tool_counts[target_tool] = self._tool_counts.get(target_tool, 0) + 1

            self.stats.update(recent_sample={
                "query": user_query[:100],
                "tool": verification["parsed"].get("tool") if verification["parsed"] else "?",
                "category": verification["parsed"].get("category") if verification["parsed"] else "?",
                "language": language,
                "verdict": verdict,
                "model": gen_model,
            })
            return True

        except Exception as e:
            self.stats.increment("errors")
            return False

    def run(
        self,
        target_en: int = TARGET_EN,
        target_hi_en: int = TARGET_HI_EN,
        save_only_pass: bool = True,
    ):
        # Resume
        existing_en, existing_hi, existing_total = self._scan_existing()
        remaining_en = max(0, target_en - existing_en)
        remaining_hi = max(0, target_hi_en - existing_hi)

        self.stats.update(
            target_en=target_en,
            target_hi_en=target_hi_en,
            existing_en=existing_en,
            existing_hi_en=existing_hi,
            en_count=existing_en,
            hi_en_count=existing_hi,
            workers=self.workers,
        )

        if existing_total > 0:
            print(f"  RESUME: {existing_total} existing ({existing_en} EN, {existing_hi} HI_EN)")
            print(f"  Need: {remaining_en} EN + {remaining_hi} HI_EN")

        if remaining_en == 0 and remaining_hi == 0:
            print("  Already at target!")
            return

        total_target = remaining_en + remaining_hi
        print(f"{'='*60}")
        print(f"  PARALLEL ORCHESTRATOR — {self.workers} workers")
        print(f"  Target: {remaining_en} EN + {remaining_hi} HI_EN = {total_target} new")
        print(f"  Output: {DATASET_PATH}")
        print(f"{'='*60}\n")

        start_time = time.time()
        last_print = start_time

        # Dynamic loop: keep generating until actual counts hit targets
        # Dupe/error/fail → slot is replaced, counter never resets
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {}
            submitted_en = 0
            submitted_hi = 0

            while True:
                s = self.stats.get()
                done_en = s["en_count"] >= target_en
                done_hi = s["hi_en_count"] >= target_hi_en

                if done_en and done_hi:
                    break

                # Submit new tasks up to worker count
                while len(futures) < self.workers:
                    s = self.stats.get()
                    need_en = s["en_count"] + sum(1 for f, l in futures.items() if l == "en" and not f.done()) < target_en
                    need_hi = s["hi_en_count"] + sum(1 for f, l in futures.items() if l == "hi_en" and not f.done()) < target_hi_en

                    if not need_en and not need_hi:
                        break

                    # Pick language with more remaining
                    lang = None
                    if need_en and need_hi:
                        en_gap = target_en - s["en_count"]
                        hi_gap = target_hi_en - s["hi_en_count"]
                        lang = "en" if en_gap >= hi_gap else "hi_en"
                    elif need_en:
                        lang = "en"
                    elif need_hi:
                        lang = "hi_en"

                    if lang is None:
                        break

                    future = executor.submit(self._generate_one, lang, save_only_pass)
                    futures[future] = lang

                # Wait for at least one to finish
                if not futures:
                    time.sleep(0.5)
                    continue

                done, _ = wait(futures.keys(), timeout=60, return_when=FIRST_COMPLETED)

                if not done:
                    # Timeout — keep waiting
                    continue

                for future in done:
                    futures.pop(future, None)
                    try:
                        future.result()
                    except:
                        pass

                # Progress
                now = time.time()
                if now - last_print >= 5:
                    s = self.stats.get()
                    print(
                        f"  [{s['elapsed']}s] EN={s['en_count']}/{target_en} HI={s['hi_en_count']}/{target_hi_en} "
                        f"passed={s['passed']} struct={s['failed_structural']} sem={s['failed_semantic']} "
                        f"judge={s['failed_judge']} dupes={s['duplicates']} errors={s['errors']} "
                        f"rate={s['rate']}/s ETA={s['eta_seconds']}s"
                    )
                    last_print = now

        # Final
        s = self.stats.get()
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  COMPLETE in {elapsed:.0f}s ({self.workers} workers)")
        print(f"  EN: {s['en_count']} ({existing_en} existing + {s['en_count']-existing_en} new)")
        print(f"  HI_EN: {s['hi_en_count']} ({existing_hi} existing + {s['hi_en_count']-existing_hi} new)")
        print(f"  Passed: {s['passed']} | Failed: struct={s['failed_structural']} sem={s['failed_semantic']} judge={s['failed_judge']}")
        print(f"  Errors: {s['errors']} | Dupes: {s['duplicates']}")
        print(f"  Rate: {s['rate']}/s")
        print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel Dataset Orchestrator")
    parser.add_argument("--en", type=int, default=TARGET_EN)
    parser.add_argument("--hi-en", type=int, default=TARGET_HI_EN)
    parser.add_argument("--workers", type=int, default=32, help="Parallel workers (default: 32)")
    parser.add_argument("--save-all", action="store_true")
    args = parser.parse_args()

    orch = ParallelOrchestrator(workers=args.workers)
    orch.run(
        target_en=args.en,
        target_hi_en=args.hi_en,
        save_only_pass=not args.save_all,
    )
