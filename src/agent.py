# ─────────────────────────────────────────────────────────────
# AGENT — ReAct Tool-Use Loop
#
# Flow:
#   Input → Thought/Action → Execution → Observation → Final Answer
#
# With hallucination recovery:
#   - Parse error → re-prompt with error context (max 3 retries)
#   - All retries fail → safety fallback (emergency category)
#
# Every step is logged verbosely.
# ─────────────────────────────────────────────────────────────

import time
import json
from dataclasses import dataclass, field

from src.config import MAX_RETRIES
from src.prompts import SYSTEM_PROMPT, build_user_prompt, build_retry_prompt
from src.parser import parse_model_output, ParseResult
from src.tools import execute_tool, get_tool_log
from src.inference import InferenceEngine, create_engine


# ── Step Log ─────────────────────────────────────────────────

@dataclass
class StepLog:
    """One step in the ReAct loop."""
    step_type: str       # "thought", "action", "observation", "final_answer", "error"
    attempt: int
    content: str
    latency_ms: float = 0.0
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: dict = field(default_factory=dict)
    error: str = ""


# ── Agent Result ─────────────────────────────────────────────

@dataclass
class AgentResult:
    """Full result of an agent run."""
    query: str
    success: bool
    steps: list[StepLog]
    final_answer: str
    triage_decision: dict | None
    total_latency_ms: float
    attempts: int
    is_fallback: bool
    tool_call_id: str = ""


# ── Agent ────────────────────────────────────────────────────

class TriageAgent:
    """ReAct agent with tool-use loop and hallucination recovery."""

    def __init__(self, engine: InferenceEngine, max_retries: int = MAX_RETRIES, verbose: bool = True):
        self.engine = engine
        self.max_retries = max_retries
        self.verbose = verbose
        self.steps: list[StepLog] = []

    def _log(self, step: StepLog):
        """Log a step. Print if verbose."""
        self.steps.append(step)
        if self.verbose:
            self._print_step(step)

    def _print_step(self, step: StepLog):
        """Pretty-print a step to console."""
        prefix = {
            "input": "INPUT",
            "thought": "THOUGHT",
            "action": "ACTION",
            "observation": "OBSERVATION",
            "final_answer": "FINAL",
            "error": "ERROR",
            "fallback": "FALLBACK",
            "retry": "RETRY",
        }.get(step.step_type, step.step_type.upper())

        print(f"\n  [{prefix}] (attempt {step.attempt})")
        if step.step_type == "input":
            print(f"    Query: {step.content[:120]}")
        elif step.step_type == "thought":
            print(f"    Reasoning: {step.content[:200]}")
        elif step.step_type == "action":
            print(f"    Tool: {step.tool_name}")
            print(f"    Args: {json.dumps(step.tool_args, indent=6)[:200]}")
        elif step.step_type == "observation":
            print(f"    Result: {json.dumps(step.tool_result, indent=6)[:200]}")
        elif step.step_type == "final_answer":
            print(f"    Answer: {step.content[:300]}")
        elif step.step_type in ("error", "retry"):
            print(f"    Error: {step.error[:200]}")
        elif step.step_type == "fallback":
            print(f"    FALLBACK: {step.content[:200]}")

        if step.latency_ms > 0:
            print(f"    Latency: {step.latency_ms:.0f}ms")

    def run(self, query: str) -> AgentResult:
        """Execute the full ReAct loop.

        Steps:
        1. INPUT: Receive user query
        2. THOUGHT: SLM reasons about the query
        3. ACTION: SLM outputs tool call JSON
        4. Parse + validate JSON (with retry on failure)
        5. OBSERVATION: Execute tool, get deterministic result
        6. FINAL ANSWER: Synthesize human-readable response

        On parse failure:
        - Re-prompt with error context (up to max_retries)
        - After max_retries: safety fallback (emergency)
        """
        self.steps = []
        start_time = time.time()
        triage_decision = None
        is_fallback = False
        tool_call_id = ""

        # ── Step 1: INPUT ──
        self._log(StepLog(
            step_type="input", attempt=0, content=query
        ))

        # ── Step 2-4: THOUGHT → ACTION → PARSE (with retry loop) ──
        user_prompt = build_user_prompt(query)

        for attempt in range(self.max_retries + 1):
            # THOUGHT + ACTION: SLM generates tool call
            if attempt == 0:
                prompt = user_prompt
            else:
                # Retry with error context
                prompt = build_retry_prompt(query, parse_result.error, attempt)
                self._log(StepLog(
                    step_type="retry", attempt=attempt,
                    content="", error=parse_result.error
                ))

            raw_output, inference_latency = self.engine.generate(SYSTEM_PROMPT, prompt)

            # THOUGHT: Log the reasoning (extracted from parsed output later)
            self._log(StepLog(
                step_type="thought", attempt=attempt + 1,
                content=raw_output[:200],
                latency_ms=inference_latency * 1000
            ))

            # Parse + validate
            parse_result = parse_model_output(raw_output)

            if parse_result.success:
                triage_decision = parse_result.data
                # ACTION: Log the tool call
                self._log(StepLog(
                    step_type="action", attempt=attempt + 1,
                    content=f"Tool: {parse_result.data['tool']}",
                    tool_name=parse_result.data["tool"],
                    tool_args=parse_result.data["args"],
                    latency_ms=0
                ))
                break
            else:
                # Parse failed — log error, retry
                self._log(StepLog(
                    step_type="error", attempt=attempt + 1,
                    content="", error=parse_result.error
                ))

                if attempt == self.max_retries:
                    # All retries exhausted — safety fallback
                    is_fallback = True
                    triage_decision = {
                        "reasoning": f"Safety fallback after {self.max_retries + 1} failed attempts. Original error: {parse_result.error}",
                        "category": "emergency",
                        "department": "Emergency Bay",
                        "tool": "emergency_dispatch",
                        "args": {
                            "condition": "system fallback — over-triage applied",
                            "symptoms": ["unparseable_input"],
                            "transport_type": "ambulance",
                            "notify_er": True,
                        },
                    }
                    self._log(StepLog(
                        step_type="fallback", attempt=attempt + 1,
                        content="Safety fallback: emergency category applied"
                    ))

        # ── Step 5: OBSERVATION — Execute tool ──
        if triage_decision:
            tool_name = triage_decision["tool"]
            tool_args = triage_decision["args"]
            tool_result = execute_tool(tool_name, tool_args)
            tool_call_id = tool_result.get("dispatch_id", tool_result.get("triage_id", tool_result.get("check_id", "unknown")))

            self._log(StepLog(
                step_type="observation", attempt=attempt + 1,
                content=json.dumps(tool_result),
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result
            ))

        # ── Step 6: FINAL ANSWER ──
        total_latency = (time.time() - start_time) * 1000
        final_answer = self._synthesize_answer(triage_decision, is_fallback)

        self._log(StepLog(
            step_type="final_answer", attempt=attempt + 1,
            content=final_answer, latency_ms=total_latency
        ))

        return AgentResult(
            query=query,
            success=not is_fallback,
            steps=self.steps,
            final_answer=final_answer,
            triage_decision=triage_decision,
            total_latency_ms=total_latency,
            attempts=attempt + 1,
            is_fallback=is_fallback,
            tool_call_id=tool_call_id,
        )

    def _synthesize_answer(self, decision: dict | None, is_fallback: bool) -> str:
        """Synthesize human-readable final answer from triage decision + tool result."""
        if not decision:
            return "ERROR: Could not process the query. Please try again."

        cat = decision.get("category", "unknown")
        dept = decision.get("department", "unknown")
        tool = decision.get("tool", "unknown")
        reasoning = decision.get("reasoning", "")
        args = decision.get("args", {})

        # Find the observation from steps
        observation = None
        for step in reversed(self.steps):
            if step.step_type == "observation":
                observation = step.tool_result
                break

        # Build answer
        lines = []

        if is_fallback:
            lines.append("FALLBACK: System could not process the input correctly.")
            lines.append("Over-triage safety rule applied.")
            lines.append("")

        urgency_emoji = {"emergency": "EMERGENCY", "urgent": "URGENT", "semi_urgent": "SEMI-URGENT", "routine": "ROUTINE"}
        lines.append(f"[{urgency_emoji.get(cat, cat).upper()}] Department: {dept}")
        lines.append(f"Reasoning: {reasoning}")
        lines.append(f"Tool: {tool}")

        if observation:
            lines.append(f"Result: {json.dumps(observation, indent=2)}")

        return "\n".join(lines)


# ── Convenience Function ─────────────────────────────────────

def run_agent(query: str, engine: InferenceEngine = None, verbose: bool = True) -> AgentResult:
    """Run the agent on a single query. Creates placeholder engine if none provided."""
    if engine is None:
        engine = create_engine("placeholder")
    agent = TriageAgent(engine, verbose=verbose)
    return agent.run(query)
