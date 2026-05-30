# ─────────────────────────────────────────────────────────────
# EVAL — SLM vs Baseline (Mistral API) Comparison
#
# Runs the eval dataset through BOTH engines:
#   1. Fine-tuned SLM (Unsloth / SLMEngine)
#   2. Mistral API baseline (same system prompt)
#
# Measures:
#   - Tool accuracy (exact match vs ground truth)
#   - Category accuracy
#   - JSON parse success rate
#   - Retry count (hallucination recovery)
#   - Fallback rate (safety fallback triggered)
#   - Latency (per-query + aggregate)
#   - Per-tool breakdown
#   - Per-language breakdown (EN vs Hinglish)
#
# Usage:
#   # Local test (placeholder engine — no model needed)
#   python -m src.eval --mode placeholder
#
#   # Mistral baseline only (needs API keys)
#   python -m src.eval --mode mistral --api-keys KEY1,KEY2
#
#   # Full comparison (Colab — needs GPU + fine-tuned adapter)
#   python -m src.eval --mode full --api-keys KEY1,KEY2 --adapter-path fhai50032/latentsig-med-router-qwen3-4b
#
#   # From local eval_dataset.jsonl
#   python -m src.eval --mode mistral --eval-file synth-ds-framework/eval_dataset.jsonl
# ─────────────────────────────────────────────────────────────

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import TriageAgent
from src.inference import (
    InferenceEngine,
    PlaceholderEngine,
    MistralAPIEngine,
    create_engine,
)
from src.parser import parse_model_output
from src.config import TOOLS, CATEGORIES


# ── Eval Sample ─────────────────────────────────────────────

@dataclass
class EvalSample:
    """One eval sample with ground truth."""
    query: str
    gt_tool: str           # ground truth tool
    gt_category: str       # ground truth category
    gt_response: str       # full ground truth JSON string
    language: str          # 'en' or 'hi_en'


@dataclass
class EvalResult:
    """Result of running one sample through one engine."""
    sample: EvalSample
    engine_name: str

    # Prediction
    pred_tool: str = ""
    pred_category: str = ""
    pred_response: str = ""

    # Metrics
    parse_success: bool = False
    tool_correct: bool = False
    category_correct: bool = False
    is_fallback: bool = False
    retry_count: int = 0
    latency_ms: float = 0.0
    error: str = ""


# ── Dataset Loading ─────────────────────────────────────────

def load_eval_dataset(source: str = "hf") -> list[EvalSample]:
    """Load eval dataset from HuggingFace or local file.

    Args:
        source: "hf" to load from HF hub, or path to local .jsonl file
    """
    samples = []

    if source == "hf":
        from datasets import load_dataset
        ds = load_dataset("fhai50032/latentsig-med-triage-router", split="eval")
        for row in ds:
            samples.append(EvalSample(
                query=row["user_query"],
                gt_tool=row["tool_called"],
                gt_category=row["category"],
                gt_response=row["response"],
                language=row.get("language", "en"),
            ))
    else:
        # Local JSONL
        with open(source, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                samples.append(EvalSample(
                    query=row["user_query"],
                    gt_tool=row["tool_called"],
                    gt_category=row["category"],
                    gt_response=row["response"],
                    language=row.get("language", "en"),
                ))

    print(f"Loaded {len(samples)} eval samples")
    return samples


# ── Single-Stage Eval (just tool call, no response stage) ──

def eval_single_sample(engine: InferenceEngine, sample: EvalSample,
                       engine_name: str, max_retries: int = 3) -> EvalResult:
    """Run one sample through an engine and compare against ground truth.

    Uses the tool-call system prompt only (Stage 1 of the agent loop).
    This measures the core routing accuracy without the response stage.
    """
    from src.prompts import SYSTEM_PROMPT, build_user_prompt, build_retry_prompt

    result = EvalResult(sample=sample, engine_name=engine_name)
    start = time.time()

    prompt = build_user_prompt(sample.query)

    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Retry with error context
            prompt = build_retry_prompt(sample.query, result.error, attempt)
            result.retry_count += 1

        raw_output, latency = engine.generate(SYSTEM_PROMPT, prompt)
        result.latency_ms += latency * 1000

        # Parse
        parse_result = parse_model_output(raw_output)

        if parse_result.success:
            result.parse_success = True
            result.pred_tool = parse_result.data["tool"]
            result.pred_category = parse_result.data["category"]
            result.pred_response = raw_output
            break
        else:
            result.error = parse_result.error

    # Check fallback (all retries exhausted)
    if not result.parse_success:
        result.is_fallback = True
        result.pred_tool = "emergency_dispatch"  # safety fallback
        result.pred_category = "emergency"

    # Compare against ground truth
    result.tool_correct = (result.pred_tool == sample.gt_tool)
    result.category_correct = (result.pred_category == sample.gt_category)

    total_latency = (time.time() - start) * 1000
    # Use cumulative latency from generate() calls, not wall time
    # (wall time includes sleep, retries etc.)

    return result


# ── Full Agent Eval (two-stage: tool call + response) ────────

def eval_sample_with_agent(agent: TriageAgent, sample: EvalSample,
                           engine_name: str) -> EvalResult:
    """Run one sample through the full agent loop (both stages).

    This measures end-to-end behavior including retries and fallback.
    """
    result = EvalResult(sample=sample, engine_name=engine_name)

    agent_result = agent.run(sample.query)

    result.parse_success = agent_result.success and not agent_result.is_fallback
    result.is_fallback = agent_result.is_fallback
    result.retry_count = max(0, agent_result.attempts - 1)
    result.latency_ms = agent_result.total_latency_ms

    if agent_result.triage_decision:
        result.pred_tool = agent_result.triage_decision.get("tool", "")
        result.pred_category = agent_result.triage_decision.get("category", "")
    result.pred_response = agent_result.final_answer

    result.tool_correct = (result.pred_tool == sample.gt_tool)
    result.category_correct = (result.pred_category == sample.gt_category)

    if not result.parse_success and agent_result.error:
        result.error = agent_result.error

    return result


# ── Aggregate Metrics ───────────────────────────────────────

@dataclass
class AggregateMetrics:
    """Aggregate metrics for one engine across all eval samples."""
    engine_name: str
    total: int = 0

    # Accuracy
    tool_correct: int = 0
    category_correct: int = 0
    parse_success: int = 0
    fallback_count: int = 0

    # Latency
    total_latency_ms: float = 0.0
    latencies: list = field(default_factory=list)

    # Retries
    total_retries: int = 0

    # Per-tool breakdown
    tool_results: dict = field(default_factory=lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    # Per-language breakdown
    lang_results: dict = field(default_factory=lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    # Per-category breakdown
    cat_results: dict = field(default_factory=lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    # Confusion: pred_tool -> {gt_tool -> count}
    confusion: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    # Errors
    errors: list = field(default_factory=list)

    def add(self, r: EvalResult):
        self.total += 1

        if r.tool_correct:
            self.tool_correct += 1
        if r.category_correct:
            self.category_correct += 1
        if r.parse_success:
            self.parse_success += 1
        if r.is_fallback:
            self.fallback_count += 1

        self.total_latency_ms += r.latency_ms
        self.latencies.append(r.latency_ms)
        self.total_retries += r.retry_count

        # Per-tool
        gt = r.sample.gt_tool
        self.tool_results[gt]["total"] += 1
        if r.tool_correct:
            self.tool_results[gt]["correct"] += 1

        # Per-language
        lang = r.sample.language
        self.lang_results[lang]["total"] += 1
        if r.tool_correct:
            self.lang_results[lang]["correct"] += 1

        # Per-category
        cat = r.sample.gt_category
        self.cat_results[cat]["total"] += 1
        if r.category_correct:
            self.cat_results[cat]["correct"] += 1

        # Confusion
        self.confusion[r.pred_tool][gt] += 1

        if r.error:
            self.errors.append(r.error[:200])

    @property
    def tool_accuracy(self) -> float:
        return self.tool_correct / self.total if self.total else 0

    @property
    def category_accuracy(self) -> float:
        return self.category_correct / self.total if self.total else 0

    @property
    def parse_rate(self) -> float:
        return self.parse_success / self.total if self.total else 0

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / self.total if self.total else 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0

    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies:
            return 0
        s = sorted(self.latencies)
        return s[len(s) // 2]

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies:
            return 0
        s = sorted(self.latencies)
        return s[int(len(s) * 0.95)]

    @property
    def avg_retries(self) -> float:
        return self.total_retries / self.total if self.total else 0


# ── Report Generation ───────────────────────────────────────

def generate_report(metrics_list: list[AggregateMetrics]) -> str:
    """Generate a comparison report from multiple engine metrics."""
    lines = []
    w = 72  # report width

    lines.append("=" * w)
    lines.append("  LatentSig Medical Triage Router — Eval Report")
    lines.append("=" * w)
    lines.append("")

    # ── Summary Table ──
    header = f"{'Metric':<30}"
    for m in metrics_list:
        header += f"  {m.engine_name:>18}"
    lines.append(header)
    lines.append("─" * len(header))

    rows = [
        ("Tool Accuracy", lambda m: f"{m.tool_accuracy:.1%}"),
        ("Category Accuracy", lambda m: f"{m.category_accuracy:.1%}"),
        ("Parse Success Rate", lambda m: f"{m.parse_rate:.1%}"),
        ("Fallback Rate", lambda m: f"{m.fallback_rate:.1%}"),
        ("", None),  # separator
        ("Avg Latency (ms)", lambda m: f"{m.avg_latency_ms:.0f}"),
        ("P50 Latency (ms)", lambda m: f"{m.p50_latency_ms:.0f}"),
        ("P95 Latency (ms)", lambda m: f"{m.p95_latency_ms:.0f}"),
        ("", None),
        ("Avg Retries", lambda m: f"{m.avg_retries:.2f}"),
        ("Total Retries", lambda m: f"{m.total_retries}"),
        ("", None),
        ("Total Samples", lambda m: f"{m.total}"),
    ]

    for label, fn in rows:
        if fn is None:
            lines.append("─" * len(header))
            continue
        row = f"{label:<30}"
        for m in metrics_list:
            row += f"  {fn(m):>18}"
        lines.append(row)

    # ── Winner Summary ──
    if len(metrics_list) >= 2:
        lines.append("")
        lines.append("─" * w)
        lines.append("  WINNER SUMMARY")
        lines.append("─" * w)

        best_tool = max(metrics_list, key=lambda m: m.tool_accuracy)
        best_cat = max(metrics_list, key=lambda m: m.category_accuracy)
        best_parse = max(metrics_list, key=lambda m: m.parse_rate)
        best_latency = min(metrics_list, key=lambda m: m.avg_latency_ms)

        lines.append(f"  Tool accuracy:  {best_tool.engine_name} ({best_tool.tool_accuracy:.1%})")
        lines.append(f"  Category acc:   {best_cat.engine_name} ({best_cat.category_accuracy:.1%})")
        lines.append(f"  Parse rate:     {best_parse.engine_name} ({best_parse.parse_rate:.1%})")
        lines.append(f"  Latency:        {best_latency.engine_name} ({best_latency.avg_latency_ms:.0f}ms)")

    # ── Per-Tool Breakdown ──
    for m in metrics_list:
        lines.append("")
        lines.append("─" * w)
        lines.append(f"  {m.engine_name} — Per-Tool Accuracy")
        lines.append("─" * w)
        lines.append(f"  {'Tool':<28} {'Correct':>8} {'Total':>8} {'Acc':>8}")
        lines.append(f"  {'─'*28} {'─'*8} {'─'*8} {'─'*8}")
        for tool in sorted(m.tool_results.keys()):
            d = m.tool_results[tool]
            acc = d["correct"] / d["total"] if d["total"] else 0
            lines.append(f"  {tool:<28} {d['correct']:>8} {d['total']:>8} {acc:>7.1%}")

    # ── Per-Language Breakdown ──
    for m in metrics_list:
        lines.append("")
        lines.append("─" * w)
        lines.append(f"  {m.engine_name} — Per-Language Accuracy")
        lines.append("─" * w)
        lines.append(f"  {'Language':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}")
        lines.append(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8}")
        for lang in sorted(m.lang_results.keys()):
            d = m.lang_results[lang]
            acc = d["correct"] / d["total"] if d["total"] else 0
            lang_label = "English" if lang == "en" else "Hinglish"
            lines.append(f"  {lang_label:<12} {d['correct']:>8} {d['total']:>8} {acc:>7.1%}")

    # ── Confusion Matrix (worst confusions) ──
    for m in metrics_list:
        if not m.confusion:
            continue
        lines.append("")
        lines.append("─" * w)
        lines.append(f"  {m.engine_name} — Top Confusions (predicted → actual)")
        lines.append("─" * w)

        confusions = []
        for pred, gt_counts in m.confusion.items():
            for gt, count in gt_counts.items():
                if pred != gt and count > 0:
                    confusions.append((count, pred, gt))
        confusions.sort(reverse=True)

        for count, pred, gt in confusions[:10]:
            lines.append(f"    {pred:<28} → {gt:<28} ({count}x)")

    # ── Errors ──
    for m in metrics_list:
        if not m.errors:
            continue
        lines.append("")
        lines.append("─" * w)
        lines.append(f"  {m.engine_name} — Errors ({len(m.errors)} total)")
        lines.append("─" * w)
        for err in m.errors[:5]:
            lines.append(f"    {err}")

    lines.append("")
    lines.append("=" * w)
    return "\n".join(lines)


# ── Save Results ────────────────────────────────────────────

def save_results(results: list[EvalResult], output_path: str):
    """Save raw eval results to JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "engine": r.engine_name,
                "query": r.sample.query[:200],
                "language": r.sample.language,
                "gt_tool": r.sample.gt_tool,
                "gt_category": r.sample.gt_category,
                "pred_tool": r.pred_tool,
                "pred_category": r.pred_category,
                "tool_correct": r.tool_correct,
                "category_correct": r.category_correct,
                "parse_success": r.parse_success,
                "is_fallback": r.is_fallback,
                "retry_count": r.retry_count,
                "latency_ms": round(r.latency_ms, 1),
                "error": r.error[:200] if r.error else "",
            }, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results to {output_path}")


# ── Run Eval ────────────────────────────────────────────────

def run_eval(
    engines: dict[str, InferenceEngine],
    samples: list[EvalSample],
    max_retries: int = 3,
    use_agent: bool = False,
    save_path: str = None,
) -> list[AggregateMetrics]:
    """Run eval across multiple engines.

    Args:
        engines: {name: engine} dict
        samples: list of EvalSample
        max_retries: max parse retries per sample
        use_agent: if True, use full TriageAgent (two-stage); if False, single-stage tool call only
        save_path: path to save raw results JSONL (optional)
    """
    all_metrics = []
    all_results = []

    for engine_name, engine in engines.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {engine_name}")
        print(f"  Samples: {len(samples)}")
        print(f"  Mode: {'full agent' if use_agent else 'single-stage tool call'}")
        print(f"{'='*60}")

        metrics = AggregateMetrics(engine_name=engine_name)
        results = []

        if use_agent:
            agent = TriageAgent(engine, max_retries=max_retries, verbose=False)

        t_start = time.time()
        running_correct = 0

        # Check if engine supports batch inference
        has_batch = hasattr(engine, 'generate_batch') and not use_agent

        if has_batch:
            # Batch mode: process all queries at once
            from src.prompts import SYSTEM_PROMPT, build_user_prompt
            queries = [build_user_prompt(s.query) for s in samples]
            # Auto batch size: smaller for GPU-limited engines
            default_batch = 4 if "unsloth" in engine_name.lower() or "slm" in engine_name.lower() else 8
            batch_size = getattr(args, 'batch_size', None) or default_batch
            print(f"  Batch mode: {len(queries)} queries, {batch_size} parallel...")

            batch_results = engine.generate_batch(SYSTEM_PROMPT, queries, batch_size=batch_size)

            for i, (sample, (raw_output, latency)) in enumerate(zip(samples, batch_results)):
                from src.parser import parse_model_output
                parse_result = parse_model_output(raw_output)

                result = EvalResult(sample=sample, engine_name=engine_name)
                result.latency_ms = latency * 1000

                if parse_result.success:
                    result.parse_success = True
                    result.pred_tool = parse_result.data["tool"]
                    result.pred_category = parse_result.data["category"]
                else:
                    result.error = parse_result.error
                    result.is_fallback = True
                    result.pred_tool = "emergency_dispatch"
                    result.pred_category = "emergency"

                result.tool_correct = (result.pred_tool == sample.gt_tool)
                result.category_correct = (result.pred_category == sample.gt_category)

                results.append(result)
                metrics.add(result)

                if result.tool_correct:
                    running_correct += 1
                running_acc = running_correct / (i + 1) * 100
                status = "✓" if result.tool_correct else "✗"
                print(f"  [{i+1:3d}/{len(samples)}] {status} "
                      f"tool={result.pred_tool:<28} gt={sample.gt_tool:<28} "
                      f"lat={result.latency_ms:.0f}ms  acc={running_acc:.0f}%")
        else:
            # Sequential mode (agent loop or non-batch engine)
            for i, sample in enumerate(samples):
                query_preview = sample.query[:70].replace('\n', ' ')
                lang_tag = "EN" if sample.language == "en" else "HI"
                sys.stdout.write(f"  [{i+1:3d}/{len(samples)}] ({lang_tag}) {query_preview}...")
                sys.stdout.flush()

                if use_agent:
                    result = eval_sample_with_agent(agent, sample, engine_name)
                else:
                    result = eval_single_sample(engine, sample, engine_name, max_retries)

                results.append(result)
                metrics.add(result)

                status = "✓" if result.tool_correct else "✗"
                fallback = " [FALLBACK]" if result.is_fallback else ""
                retries = f" r={result.retry_count}" if result.retry_count > 0 else ""
                if result.tool_correct:
                    running_correct += 1
                running_acc = running_correct / (i + 1) * 100

                print(f"\r  [{i+1:3d}/{len(samples)}] {status} "
                      f"tool={result.pred_tool:<28} gt={sample.gt_tool:<28} "
                      f"lat={result.latency_ms:.0f}ms{retries}{fallback}  "
                      f"acc={running_acc:.0f}%")

        elapsed = time.time() - t_start
        print(f"\n  Done in {elapsed:.1f}s ({elapsed/len(samples):.2f}s/sample)")

        all_metrics.append(metrics)
        all_results.extend(results)

    # Generate report
    report = generate_report(all_metrics)
    print(f"\n{report}")

    # Save
    if save_path:
        save_results(all_results, save_path)

    # Save report
    report_path = save_path.replace(".jsonl", "_report.txt") if save_path else "eval_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    return all_metrics


# ── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LatentSig Eval: SLM vs Baseline")

    parser.add_argument("--mode", choices=["placeholder", "mistral", "slm", "gguf", "full"],
                        default="placeholder",
                        help="placeholder: test pipeline; mistral: baseline only; "
                             "slm: fine-tuned only; gguf: llama.cpp; full: both (Colab)")
    parser.add_argument("--eval-file", default="hf",
                        help="Path to eval JSONL or 'hf' for HuggingFace dataset")
    parser.add_argument("--api-keys", default=None,
                        help="Comma-separated Mistral API keys (or set MISTRAL_API_KEYS env)")
    parser.add_argument("--mistral-model", default="mistral-small-latest",
                        help="Mistral model for baseline")
    parser.add_argument("--adapter-path", default="fhai50032/latentsig-med-router-qwen3-4b",
                        help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--gguf-path", default=None,
                        help="Path to GGUF model file")
    parser.add_argument("--base-model", default="unsloth/Qwen3-4B-Instruct",
                        help="Base model for SLM engine")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max parse retries per sample")
    parser.add_argument("--use-agent", action="store_true",
                        help="Use full two-stage agent loop (slower, measures end-to-end)")
    parser.add_argument("--output", default=None,
                        help="Output path for results JSONL")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of eval samples (for quick testing)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for parallel inference (GGUF only)")

    args = parser.parse_args()

    # Load eval dataset
    samples = load_eval_dataset(args.eval_file)
    if args.limit:
        samples = samples[:args.limit]
        print(f"Limited to {len(samples)} samples")

    # Build engines
    engines: dict[str, InferenceEngine] = {}

    if args.mode == "placeholder":
        engines["placeholder"] = PlaceholderEngine()
        args.output = args.output or "eval_results_placeholder.jsonl"

    elif args.mode == "mistral":
        # Get API keys
        api_keys = None
        if args.api_keys:
            api_keys = [k.strip() for k in args.api_keys.split(",")]
        elif os.environ.get("MISTRAL_API_KEYS"):
            api_keys = [k.strip() for k in os.environ["MISTRAL_API_KEYS"].split(",")]
        else:
            # Try loading from api_keys.md
            keys_path = os.path.join(os.path.expanduser("~"), "AppData", "Local",
                                     "hermes", "api_keys.md")
            if os.path.exists(keys_path):
                with open(keys_path) as f:
                    content = f.read()
                # Extract keys between Mistral section
                import re
                section = re.search(r"## Mistral.*?(?=##|\Z)", content, re.DOTALL)
                if section:
                    api_keys = re.findall(r"`([A-Za-z0-9]{32})`", section.group())
            if not api_keys:
                print("ERROR: No Mistral API keys found. Use --api-keys or set MISTRAL_API_KEYS")
                sys.exit(1)

        engines["mistral"] = MistralAPIEngine(
            api_keys=api_keys,
            model=args.mistral_model,
        )
        args.output = args.output or "eval_results_mistral.jsonl"

    elif args.mode == "slm":
        from src.inference import UnslothEngine
        engine = UnslothEngine(
            base_model=args.base_model,
            adapter_path=args.adapter_path,
        )
        engine.load()
        engines["slm"] = engine
        args.output = args.output or "eval_results_slm.jsonl"

    elif args.mode == "gguf":
        if not args.gguf_path:
            print("ERROR: --gguf-path required for gguf mode")
            sys.exit(1)
        from src.inference import LlamaCppEngine
        engine = LlamaCppEngine(model_path=args.gguf_path)
        engine.load()
        engines["gguf"] = engine
        args.output = args.output or "eval_results_gguf.jsonl"

    elif args.mode == "full":
        # Both engines
        api_keys = None
        if args.api_keys:
            api_keys = [k.strip() for k in args.api_keys.split(",")]
        elif os.environ.get("MISTRAL_API_KEYS"):
            api_keys = [k.strip() for k in os.environ["MISTRAL_API_KEYS"].split(",")]
        else:
            keys_path = os.path.join(os.path.expanduser("~"), "AppData", "Local",
                                     "hermes", "api_keys.md")
            if os.path.exists(keys_path):
                with open(keys_path) as f:
                    content = f.read()
                import re
                section = re.search(r"## Mistral.*?(?=##|\Z)", content, re.DOTALL)
                if section:
                    api_keys = re.findall(r"`([A-Za-z0-9]{32})`", section.group())

        if api_keys:
            engines["mistral-baseline"] = MistralAPIEngine(
                api_keys=api_keys,
                model=args.mistral_model,
            )
        else:
            print("WARNING: No Mistral API keys — skipping baseline")

        if args.gguf_path:
            from src.inference import LlamaCppEngine
            engine = LlamaCppEngine(model_path=args.gguf_path)
            engine.load()
            engines["slm-gguf"] = engine
        else:
            from src.inference import UnslothEngine
            engine = UnslothEngine(
                base_model=args.base_model,
                adapter_path=args.adapter_path,
            )
            engine.load()
            engines["slm-finetuned"] = engine
        args.output = args.output or "eval_results_full.jsonl"

    # Run eval
    metrics = run_eval(
        engines=engines,
        samples=samples,
        max_retries=args.max_retries,
        use_agent=args.use_agent,
        save_path=args.output,
    )

    return metrics


if __name__ == "__main__":
    main()
