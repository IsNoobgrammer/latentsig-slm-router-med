import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

bg = '#0d1117'
fg = '#c9d1d9'

def draw_box(ax, x, y, w, h, text, color, text_color='#1e1e1e', fontsize=13, style='round'):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1" if style == 'round' else "square,pad=0.05",
        facecolor=color, edgecolor='#30363d', linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='#30363d', style='->', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def draw_dashed_arrow(ax, x1, y1, x2, y2, color='#e03131', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle='dashed'))


fig, ax = plt.subplots(figsize=(18, 10))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)
ax.set_xlim(-0.5, 18)
ax.set_ylim(-1.5, 9.5)
ax.set_aspect('equal')
ax.axis('off')

ax.text(9, 9, 'Eval Pipeline — SLM vs Baseline', ha='center', va='center',
        fontsize=22, fontweight='bold', color=fg)

# ── Input ──
draw_box(ax, 7.5, 7.8, 3, 1, 'Eval Dataset\n(40 samples: 20 EN + 20 HI)', '#c3fae8', fontsize=11)

# ── Split arrow ──
draw_arrow(ax, 8, 7.8, 4, 6.9, color='#30363d')
draw_arrow(ax, 10, 7.8, 14, 6.9, color='#30363d')

# ── SLM Branch ──
draw_box(ax, 1.5, 6.3, 4.5, 1.2, 'SLM (Fine-tuned)\nQwen3-4B + QLoRA', '#d0bfff', fontsize=11)
draw_arrow(ax, 3.75, 6.3, 3.75, 5.5, color='#30363d')

draw_box(ax, 1.5, 4.7, 4.5, 0.8, 'Parse JSON + Validate', '#a5d8ff', fontsize=11)
draw_arrow(ax, 3.75, 4.7, 3.75, 3.9, color='#30363d')

draw_box(ax, 1.5, 3.2, 4.5, 0.7, 'Retry on failure (max 3)', '#fff3bf', fontsize=11)
draw_dashed_arrow(ax, 3.75, 3.2, 3.75, 4.7, color='#e03131', lw=1.2)
ax.text(4.6, 3.95, 'fail', fontsize=9, color='#e03131', style='italic')

draw_arrow(ax, 3.75, 3.2, 3.75, 2.4, color='#2f9e44')

draw_box(ax, 1.5, 1.6, 4.5, 0.8, 'Record: tool, category,\nlatency, retries', '#e7f5ff', fontsize=10)

# ── Baseline Branch ──
draw_box(ax, 12, 6.3, 4.5, 1.2, 'Mistral API (Baseline)\nmistral-small-latest', '#ffd8a8', fontsize=11)
draw_arrow(ax, 14.25, 6.3, 14.25, 5.5, color='#30363d')

draw_box(ax, 12, 4.7, 4.5, 0.8, 'Parse JSON + Validate', '#a5d8ff', fontsize=11)
draw_arrow(ax, 14.25, 4.7, 14.25, 3.9, color='#30363d')

draw_box(ax, 12, 3.2, 4.5, 0.7, 'Retry on failure (max 3)', '#fff3bf', fontsize=11)
draw_dashed_arrow(ax, 14.25, 3.2, 14.25, 4.7, color='#e03131', lw=1.2)
ax.text(15.15, 3.95, 'fail', fontsize=9, color='#e03131', style='italic')

draw_arrow(ax, 14.25, 3.2, 14.25, 2.4, color='#2f9e44')

draw_box(ax, 12, 1.6, 4.5, 0.8, 'Record: tool, category,\nlatency, retries', '#e7f5ff', fontsize=10)

# ── Merge to Compare ──
draw_arrow(ax, 3.75, 1.6, 7, 0.6, color='#1971c2')
draw_arrow(ax, 14.25, 1.6, 11, 0.6, color='#1971c2')

draw_box(ax, 6, -0.2, 6, 1, 'Compare vs Ground Truth\n(tool_accuracy, category, parse_rate,\nlatency, retries, confusion)', '#b2f2bb', fontsize=10)

# ── Output ──
draw_arrow(ax, 9, -0.2, 9, -1, color='#30363d')
draw_box(ax, 5.5, -1.8, 7, 0.8, 'Report: winner summary + per-tool + per-lang + confusion', '#ffc9c9',
         text_color='#c92a2a', fontsize=10)

# Notes
ax.text(0.5, -1.3, 'Same system prompt for both engines — fair comparison',
        fontsize=10, color='#868e96', style='italic')
ax.text(0.5, -1.7, 'Both use Pydantic validation + hallucination recovery loop',
        fontsize=10, color='#868e96', style='italic')

plt.tight_layout()
plt.savefig('visuals/eval_pipeline.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: visuals/eval_pipeline.png')
