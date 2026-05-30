import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

bg = '#0d1117'
fg = '#c9d1d9'


def draw_box(ax, x, y, w, h, text, color, text_color='#1e1e1e', fontsize=13, lw=1.5):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.1",
        facecolor=color, edgecolor='#30363d', linewidth=lw
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, wrap=True)


def draw_diamond(ax, cx, cy, w, h, text, color, text_color='#1e1e1e', fontsize=11):
    diamond = plt.Polygon(
        [(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)],
        closed=True, facecolor=color, edgecolor='#30363d', linewidth=1.5
    )
    ax.add_patch(diamond)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color)


def draw_arrow(ax, x1, y1, x2, y2, color='#30363d', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


def draw_dashed_arrow(ax, x1, y1, x2, y2, color='#e03131', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle='dashed'))


# ════════════════════════════════════════════════════
# 1. ARCHITECTURE OVERVIEW — Two-stage SLM system
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 8))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)
ax.set_xlim(-0.5, 18)
ax.set_ylim(-1, 8)
ax.set_aspect('equal')
ax.axis('off')

ax.text(9, 7.5, 'Architecture Overview — LatentSig Medical Triage Router', ha='center', va='center',
        fontsize=20, fontweight='bold', color=fg)

# User
draw_box(ax, 0, 5.5, 2, 1.2, 'User\nQuery', '#c3fae8', fontsize=13)

# Stage 1 box
ax.add_patch(mpatches.FancyBboxPatch((2.8, 4.5), 5.5, 3, boxstyle="round,pad=0.15",
    facecolor='#161b22', edgecolor='#58a6ff', linewidth=2, alpha=0.3))
ax.text(5.5, 7.2, 'Stage 1: Tool Call', ha='center', fontsize=14, fontweight='bold', color='#58a6ff')

draw_box(ax, 3, 5.5, 2, 1.2, 'SLM\n(Qwen3-4B)', '#a5d8ff', fontsize=12)
draw_box(ax, 5.5, 5.5, 2.5, 1.2, 'JSON Parser\n(Pydantic)', '#ffd8a8', fontsize=12)

# Stage 2 box
ax.add_patch(mpatches.FancyBboxPatch((9, 4.5), 5.5, 3, boxstyle="round,pad=0.15",
    facecolor='#161b22', edgecolor='#bc8cff', linewidth=2, alpha=0.3))
ax.text(11.75, 7.2, 'Stage 2: Response', ha='center', fontsize=14, fontweight='bold', color='#bc8cff')

draw_box(ax, 9.2, 5.5, 2.2, 1.2, 'Tool\nExecutor', '#b2f2bb', fontsize=12)
draw_box(ax, 11.9, 5.5, 2.3, 1.2, 'SLM\n(assistant)', '#d0bfff', fontsize=12)

# Bottom: tool logs
draw_box(ax, 9.2, 2.5, 2.2, 1, 'Tool Logs\n(CSV)', '#e7f5ff', fontsize=11)
draw_box(ax, 11.9, 2.5, 2.3, 1, 'Final Answer\n(human-readable)', '#b2f2bb', fontsize=11)

# Retry path
draw_box(ax, 3, 2.5, 2.5, 1, 'Retry\n(max 3)', '#ffc9c9', fontsize=11)
draw_box(ax, 6, 2.5, 2.3, 1, 'Safety\nFallback', '#ffc9c9', fontsize=11)

# Arrows
draw_arrow(ax, 2, 6.1, 3, 6.1)
draw_arrow(ax, 5, 6.1, 5.5, 6.1)
draw_arrow(ax, 8, 6.1, 9.2, 6.1)
draw_arrow(ax, 11.4, 6.1, 11.9, 6.1)

# Parser → Tool executor (down then right)
draw_arrow(ax, 6.75, 5.5, 6.75, 4.3, color='#2f9e44')
draw_arrow(ax, 6.75, 4.3, 10.3, 4.3, color='#2f9e44')
draw_arrow(ax, 10.3, 4.3, 10.3, 5.5, color='#2f9e44')

# Tool executor → logs
draw_arrow(ax, 10.3, 5.5, 10.3, 3.5, color='#1971c2')

# SLM response → final answer
draw_arrow(ax, 13.05, 5.5, 13.05, 3.5, color='#2f9e44')

# Retry path
draw_dashed_arrow(ax, 6.75, 5.5, 6.75, 3.7, color='#e03131')
draw_arrow(ax, 6.75, 3.7, 4.25, 3.5, color='#e03131')
draw_arrow(ax, 4.25, 3.5, 4.25, 5.5, color='#e03131')

# Fallback
draw_arrow(ax, 5.5, 3, 6, 3, color='#e03131')

# Labels
ax.text(6.9, 4.7, 'Valid JSON', fontsize=10, color='#2f9e44', ha='center')
ax.text(7.2, 5.9, 'Invalid', fontsize=10, color='#e03131', ha='center')
ax.text(13.3, 4.5, 'Response', fontsize=10, color='#2f9e44', ha='center')

# Legend
ax.text(0.5, 1, 'SLM = Small Language Model (Qwen3-4B-Instruct, QLoRA fine-tuned)', fontsize=10, color='#868e96')
ax.text(0.5, 0.5, 'Stage 1: tool-call system prompt → JSON tool call | Stage 2: assistant system prompt → human answer', fontsize=10, color='#868e96')

plt.tight_layout()
plt.savefig('visuals/architecture_overview.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: architecture_overview.png')


# ════════════════════════════════════════════════════
# 2. REACT LOOP FLOW — The agentic tool-use loop
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)
ax.set_xlim(-1, 14)
ax.set_ylim(-1, 10)
ax.set_aspect('equal')
ax.axis('off')

ax.text(7, 9.5, 'ReAct Loop — Tool-Use Agent', ha='center', va='center',
        fontsize=20, fontweight='bold', color=fg)

# Step 1: Input
draw_box(ax, 5.5, 8, 3, 0.8, 'INPUT: User Query', '#c3fae8', fontsize=13)

# Step 2: Thought
draw_box(ax, 5.5, 6.5, 3, 0.8, 'THOUGHT: SLM Reasons', '#a5d8ff', fontsize=13)

# Step 3: Action
draw_box(ax, 5.5, 5, 3, 0.8, 'ACTION: Tool Call JSON', '#d0bfff', fontsize=13)

# Step 4: Parse decision
draw_diamond(ax, 7, 3.7, 3, 1.2, 'Valid\nJSON?', '#ffd8a8', fontsize=12)

# Step 5: Execute
draw_box(ax, 0.5, 2, 3, 0.8, 'EXECUTE: Mock Tool', '#b2f2bb', fontsize=13)

# Step 6: Observation
draw_box(ax, 0.5, 0.5, 3, 0.8, 'OBSERVATION: Result', '#e7f5ff', fontsize=13)

# Step 7: Final Answer
draw_box(ax, 5.5, 0.5, 3, 0.8, 'FINAL ANSWER', '#b2f2bb', fontsize=14)

# Retry
draw_box(ax, 10, 3.3, 3, 0.8, 'RETRY + Error', '#ffc9c9', fontsize=13)

# Safety fallback
draw_box(ax, 10, 1.5, 3, 0.8, 'SAFETY FALLBACK', '#ffc9c9', fontsize=13)

# Arrows - main flow
draw_arrow(ax, 7, 8, 7, 7.3)
draw_arrow(ax, 7, 6.5, 7, 5.8)
draw_arrow(ax, 7, 5, 7, 4.3)

# Parse → execute (yes)
draw_arrow(ax, 5.5, 3.7, 2, 2.8, color='#2f9e44')
ax.text(3.5, 3.5, 'Yes', fontsize=11, color='#2f9e44', fontweight='bold')

# Execute → observation
draw_arrow(ax, 2, 2, 2, 1.3)

# Observation → final answer
draw_arrow(ax, 3.5, 0.9, 5.5, 0.9, color='#2f9e44')

# Parse → retry (no)
draw_arrow(ax, 8.5, 3.7, 10, 3.7, color='#e03131')
ax.text(9, 4, 'No', fontsize=11, color='#e03131', fontweight='bold')

# Retry → back to thought
draw_arrow(ax, 11.5, 3.3, 11.5, 2.5, color='#e03131')
draw_arrow(ax, 11.5, 2.5, 7, 2.5, color='#e03131')
draw_arrow(ax, 7, 2.5, 7, 6.5, color='#e03131')

# Retry count check
draw_diamond(ax, 11.5, 2.1, 2, 0.8, '3 fails?', '#ffc9c9', fontsize=10)

# Fallback
draw_arrow(ax, 11.5, 1.7, 11.5, 1.5, color='#e03131')
ax.text(12.2, 1.7, 'Yes', fontsize=10, color='#e03131')

# Fallback → final
draw_arrow(ax, 10, 1.9, 8.5, 0.9, color='#e03131')

# Labels
ax.text(0.5, -0.5, 'Each step is logged. Retry injects error context into prompt for self-healing.',
        fontsize=10, color='#868e96')

plt.tight_layout()
plt.savefig('visuals/react_loop_flow.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: react_loop_flow.png')


# ════════════════════════════════════════════════════
# 3. HALLUCINATION RECOVERY — Self-healing mechanism
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)
ax.set_xlim(-0.5, 16)
ax.set_ylim(-1.5, 5)
ax.set_aspect('equal')
ax.axis('off')

ax.text(8, 4.5, 'Hallucination Recovery — Self-Healing JSON', ha='center', va='center',
        fontsize=20, fontweight='bold', color=fg)

# Attempt 1
draw_box(ax, 0, 2.5, 2.5, 1.2, 'Attempt 1\nSLM Output', '#a5d8ff', fontsize=12)
draw_diamond(ax, 3.75, 3.1, 1.5, 1.2, 'Valid\nJSON?', '#ffd8a8', fontsize=10)

# Attempt 2
draw_box(ax, 6, 2.5, 2.5, 1.2, 'Attempt 2\nSLM + Error', '#a5d8ff', fontsize=12)
draw_diamond(ax, 9.75, 3.1, 1.5, 1.2, 'Valid\nJSON?', '#ffd8a8', fontsize=10)

# Attempt 3
draw_box(ax, 12, 2.5, 2.5, 1.2, 'Attempt 3\nSLM + Error', '#a5d8ff', fontsize=12)

# Success path
draw_box(ax, 3, 0.5, 2, 0.8, 'Parse OK', '#b2f2bb', fontsize=12)

# Fallback
draw_box(ax, 12, 0.5, 2.5, 0.8, 'FALLBACK\nEmergency', '#ffc9c9', fontsize=12)

# Error context box
draw_box(ax, 4, -1, 7, 0.8, 'Error Context: "Your output was invalid: [error]. Fix the JSON."', '#fff3bf', fontsize=10)

# Arrows
draw_arrow(ax, 2.5, 3.1, 3, 3.1)

# Attempt 1 → valid (yes)
draw_arrow(ax, 3.75, 2.5, 4, 1.3, color='#2f9e44')
ax.text(3.5, 1.7, 'Yes', fontsize=10, color='#2f9e44', fontweight='bold')

# Attempt 1 → retry (no)
draw_arrow(ax, 4.5, 3.1, 6, 3.1, color='#e03131')
ax.text(5, 3.4, 'No', fontsize=10, color='#e03131', fontweight='bold')

# Error context injection
draw_dashed_arrow(ax, 3.75, 2.5, 7.5, -0.6, color='#d29922')
ax.text(5, 0, 'Inject error', fontsize=9, color='#d29922')

# Error context → attempt 2
draw_arrow(ax, 7.5, -0.2, 7.5, 2.5, color='#d29922')

# Attempt 2 → valid (yes)
draw_arrow(ax, 9.75, 2.5, 4, 1.3, color='#2f9e44')

# Attempt 2 → retry (no)
draw_arrow(ax, 10.5, 3.1, 12, 3.1, color='#e03131')

# Attempt 3 → valid or fallback
draw_arrow(ax, 13.25, 2.5, 4, 1.3, color='#2f9e44')
draw_dashed_arrow(ax, 13.25, 2.5, 13.25, 1.3, color='#e03131')
ax.text(13.5, 1.7, 'No', fontsize=10, color='#e03131', fontweight='bold')

# Labels
ax.text(1, -1.2, 'Max 3 retries. Each retry injects the parse error into the prompt so the SLM self-corrects.',
        fontsize=10, color='#868e96')

plt.tight_layout()
plt.savefig('visuals/hallucination_recovery.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: hallucination_recovery.png')


# ════════════════════════════════════════════════════
# 4. AGENT PIPELINE — Two-stage SLM flow
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)
ax.set_xlim(-0.5, 16)
ax.set_ylim(-1, 6.5)
ax.set_aspect('equal')
ax.axis('off')

ax.text(8, 6, 'Agent Pipeline — Two-Stage SLM Inference', ha='center', va='center',
        fontsize=20, fontweight='bold', color=fg)

# Stage 1
ax.add_patch(mpatches.FancyBboxPatch((0.5, 3), 6.5, 2.5, boxstyle="round,pad=0.15",
    facecolor='#161b22', edgecolor='#58a6ff', linewidth=2, alpha=0.3))
ax.text(3.75, 5.2, 'Stage 1: Tool Call', ha='center', fontsize=14, fontweight='bold', color='#58a6ff')

draw_box(ax, 1, 3.5, 2, 1, 'SLM + Tool\nSystem Prompt', '#a5d8ff', fontsize=11)
draw_box(ax, 3.5, 3.5, 1.5, 1, 'Parse\nJSON', '#ffd8a8', fontsize=11)
draw_box(ax, 5.3, 3.5, 1.5, 1, 'Execute\nTool', '#b2f2bb', fontsize=11)

# Stage 2
ax.add_patch(mpatches.FancyBboxPatch((7.5, 3), 6.5, 2.5, boxstyle="round,pad=0.15",
    facecolor='#161b22', edgecolor='#bc8cff', linewidth=2, alpha=0.3))
ax.text(10.75, 5.2, 'Stage 2: Response', ha='center', fontsize=14, fontweight='bold', color='#bc8cff')

draw_box(ax, 8, 3.5, 2.5, 1, 'SLM + Assistant\nSystem Prompt', '#d0bfff', fontsize=11)
draw_box(ax, 11, 3.5, 2.5, 1, 'Human-Readable\nAnswer', '#b2f2bb', fontsize=11)

# Bottom: logs
draw_box(ax, 1, 1, 2, 0.8, 'Tool Logs\n(CSV)', '#e7f5ff', fontsize=11)
draw_box(ax, 4, 1, 2.5, 0.8, 'DB/Memory\nUpdate', '#e7f5ff', fontsize=11)
draw_box(ax, 8, 1, 5.5, 0.8, 'AgentResult {query, tool_call, tool_result, response, latency}', '#e7f5ff', fontsize=10)

# Input
draw_box(ax, 0.5, 5.3, 1.5, 0.6, 'Query', '#c3fae8', fontsize=11)

# Arrows
draw_arrow(ax, 2, 5.6, 2, 4.5)
draw_arrow(ax, 3, 4, 3.5, 4)
draw_arrow(ax, 5, 4, 5.3, 4)
draw_arrow(ax, 6.8, 4, 8, 4)
draw_arrow(ax, 10.5, 4, 11, 4)

# Tool → logs
draw_arrow(ax, 6.05, 3.5, 6.05, 2, color='#1971c2')
draw_arrow(ax, 6.05, 1.8, 3, 1.4, color='#1971c2')

# Tool → DB
draw_arrow(ax, 6.05, 2, 5.25, 1.8, color='#1971c2')

# Response → result
draw_arrow(ax, 12.25, 3.5, 12.25, 1.8, color='#2f9e44')

# Labels
ax.text(0.5, 0, 'Stage 1: tool-call prompt → JSON | Stage 2: assistant prompt → readable answer',
        fontsize=10, color='#868e96')

plt.tight_layout()
plt.savefig('visuals/agent_pipeline.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: agent_pipeline.png')

print('\nAll 4 flowcharts created.')
