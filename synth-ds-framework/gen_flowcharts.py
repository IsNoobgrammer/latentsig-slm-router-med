import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

bg = '#0d1117'
fg = '#c9d1d9'

def draw_box(ax, x, y, w, h, text, color, text_color='#1e1e1e', fontsize=14, style='round'):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.1" if style == 'round' else "square,pad=0.05",
        facecolor=color, edgecolor='#30363d', linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, wrap=True)

def draw_diamond(ax, cx, cy, w, h, text, color, text_color='#1e1e1e', fontsize=12):
    diamond = plt.Polygon(
        [(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)],
        closed=True, facecolor=color, edgecolor='#30363d', linewidth=1.5
    )
    ax.add_patch(diamond)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color)

def draw_arrow(ax, x1, y1, x2, y2, color='#30363d', style='->', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def draw_dashed_arrow(ax, x1, y1, x2, y2, color='#e03131', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle='dashed'))


# ════════════════════════════════════════════════════
# 1. VERIFICATION PIPELINE
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)
ax.set_xlim(-0.5, 16.5)
ax.set_ylim(-2, 4)
ax.set_aspect('equal')
ax.axis('off')

ax.text(8, 3.5, 'Verification Pipeline', ha='center', va='center',
        fontsize=22, fontweight='bold', color=fg)

# Input
draw_box(ax, 0, 1, 2, 1.2, 'Generated\nSample', '#c3fae8', fontsize=13)
# Phase 1
draw_diamond(ax, 4, 1.6, 2.2, 1.8, 'Phase 1:\nPydantic', '#a5d8ff', fontsize=11)
# Phase 2
draw_diamond(ax, 7.5, 1.6, 2.2, 1.8, 'Phase 2:\nTool Enforce', '#ffd8a8', fontsize=11)
# Phase 3
draw_diamond(ax, 11, 1.6, 2.2, 1.8, 'Phase 3:\nLLM Judge', '#d0bfff', fontsize=11)
# Save
draw_box(ax, 13.5, 1, 2.5, 1.2, 'Saved to\nDataset', '#b2f2bb', fontsize=13)
# Reject
draw_box(ax, 4.5, -1.5, 7.5, 0.8, 'Rejected  (retry with same target tool)', '#ffc9c9',
         text_color='#c92a2a', fontsize=12)

# Arrows - pass path
draw_arrow(ax, 2, 1.6, 2.9, 1.6, color='#2f9e44')
draw_arrow(ax, 5.1, 1.6, 6.4, 1.6, color='#2f9e44')
draw_arrow(ax, 8.6, 1.6, 9.9, 1.6, color='#2f9e44')
draw_arrow(ax, 12.1, 1.6, 13.5, 1.6, color='#2f9e44')

# Labels on pass arrows
ax.text(2.5, 1.9, 'Valid', fontsize=10, color='#2f9e44', ha='center')
ax.text(5.8, 1.9, 'Match', fontsize=10, color='#2f9e44', ha='center')
ax.text(9.3, 1.9, 'Pass', fontsize=10, color='#2f9e44', ha='center')

# Arrows - fail path (dashed)
draw_dashed_arrow(ax, 4, 0.7, 5.5, -0.7)
draw_dashed_arrow(ax, 7.5, 0.7, 8.6, -0.7)
draw_dashed_arrow(ax, 11, 0.7, 11.5, -0.7)

ax.text(4.3, -0.1, 'Invalid', fontsize=10, color='#e03131', ha='center', style='italic')
ax.text(7.8, -0.1, 'Wrong\nTool', fontsize=10, color='#e03131', ha='center', style='italic')
ax.text(11, -0.1, 'Fail', fontsize=10, color='#e03131', ha='center', style='italic')

# Note
ax.text(8, -2.3, 'LLM Judge is unbiased — does NOT know which tool was targeted',
        fontsize=11, color='#868e96', ha='center', style='italic')

plt.tight_layout()
plt.savefig('visuals/verification_pipeline.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: verification_pipeline.png')


# ════════════════════════════════════════════════════
# 2. GENERATION PIPELINE
# ════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 8))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)
ax.set_xlim(-0.5, 18)
ax.set_ylim(-3, 6)
ax.set_aspect('equal')
ax.axis('off')

ax.text(9, 5.5, 'Generation Pipeline', ha='center', va='center',
        fontsize=22, fontweight='bold', color=fg)

# Row 1: Generation flow
draw_box(ax, 0, 3, 2.5, 1.2, 'Tool Schema\nRegistry (7)', '#c3fae8', fontsize=12)
draw_box(ax, 3, 3, 2.5, 1.2, 'pick_target_tool\n(weighted)', '#a5d8ff', fontsize=11)
draw_box(ax, 6, 3, 2.8, 1.2, 'Query Generator\n(mistral-large/\nmedium/magistral)', '#d0bfff', fontsize=10)
draw_box(ax, 9.3, 3, 2.5, 1.2, 'Hinglish\nTranslator', '#fff3bf', fontsize=12)
draw_box(ax, 12.3, 3, 2.8, 1.2, 'Response Generator\n(with tool hint:\n[Use X tool])', '#d0bfff', fontsize=10)

# Row 2: Verification + Storage
draw_box(ax, 12.3, 0.5, 2.8, 1.2, '3-Layer Verifier\n(Pydantic+Tool+\nLLM Judge)', '#ffd8a8', fontsize=10)
draw_box(ax, 9, 0.5, 2.5, 1.0, 'Hash Dedup', '#a5d8ff', fontsize=12)
draw_box(ax, 6, 0.5, 2.2, 1.0, 'Write\nJSONL', '#b2f2bb', fontsize=12)
draw_box(ax, 3, 0.5, 2.2, 1.0, 'Live Stats\n+ Monitor', '#e7f5ff', fontsize=12)

# Arrows row 1
draw_arrow(ax, 2.5, 3.6, 3, 3.6, color='#30363d')
draw_arrow(ax, 5.5, 3.6, 6, 3.6, color='#30363d')
draw_arrow(ax, 8.8, 3.6, 9.3, 3.6, color='#30363d')
ax.text(9.05, 3.9, 'if HI', fontsize=9, color='#868e96', ha='center')
draw_arrow(ax, 11.8, 3.6, 12.3, 3.6, color='#30363d')

# Arrow down from Response Gen to Verifier
draw_arrow(ax, 13.7, 3, 13.7, 1.7, color='#30363d')

# Arrow from Verifier to Dedup (pass)
draw_arrow(ax, 12.3, 1.1, 11.5, 1, color='#2f9e44')
ax.text(11.9, 1.4, 'Pass', fontsize=10, color='#2f9e44', ha='center')

# Arrow from Dedup to Write
draw_arrow(ax, 9, 1, 8.2, 1, color='#2f9e44')
ax.text(8.6, 1.3, 'New', fontsize=10, color='#2f9e44', ha='center')

# Arrow from Write to Stats
draw_arrow(ax, 6, 1, 5.2, 1, color='#1971c2')

# Fail -> Retry (loop back)
draw_dashed_arrow(ax, 15.1, 0.5, 15.1, -1)
ax.annotate('', xy=(7.4, -1.5), xytext=(15.1, -1),
            arrowprops=dict(arrowstyle='->', color='#e03131', lw=2, linestyle='dashed'))
ax.text(15.3, -0.3, 'Fail', fontsize=10, color='#e03131', ha='center')
ax.text(11, -1.8, 'Retry with same target tool', fontsize=11, color='#e03131',
        ha='center', style='italic')
ax.annotate('', xy=(7.4, 3), xytext=(7.4, -1.5),
            arrowprops=dict(arrowstyle='->', color='#e03131', lw=2, linestyle='dashed'))

# Notes
ax.text(0, -2.3, 'Key: pick_target_tool uses weighted random favoring least-used tools',
        fontsize=11, color='#868e96', style='italic')
ax.text(0, -2.7, 'Tool hint ensures clean training data (stripped at inference)',
        fontsize=11, color='#868e96', style='italic')
ax.text(0, -3.1, '32 parallel workers, 8 Mistral API keys, ~0.7/s throughput',
        fontsize=11, color='#868e96', style='italic')

plt.tight_layout()
plt.savefig('visuals/generation_pipeline.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: generation_pipeline.png')

print('\nBoth flowchart images created.')
