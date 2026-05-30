import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

bg = '#0d1117'
fg = '#c9d1d9'
green = '#3fb950'
red = '#f85149'
blue = '#58a6ff'
yellow = '#d29922'
purple = '#bc8cff'

models = ['Qwen3-4B\nBase', 'SLM\n(Fine-tuned)', 'Mistral\nSmall', 'Mistral\nLarge']
colors = [green, yellow, blue, purple]

# ── 1. Overall Accuracy (no triage) ──
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

tool_acc = [89.2, 91.9, 83.8, 91.9]
cat_acc = [89.2, 89.2, 91.9, 89.2]
x = np.arange(len(models))
w = 0.35

bars1 = ax.bar(x - w/2, tool_acc, w, label='Tool Accuracy', color=blue, edgecolor='#30363d')
bars2 = ax.bar(x + w/2, cat_acc, w, label='Category Accuracy', color=green, edgecolor='#30363d')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{bar.get_height():.1f}%',
            ha='center', va='bottom', color=fg, fontweight='bold', fontsize=13)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{bar.get_height():.1f}%',
            ha='center', va='bottom', color=fg, fontweight='bold', fontsize=13)

ax.set_ylim(0, 105)
ax.set_xticks(x)
ax.set_xticklabels(models, color=fg, fontsize=11)
ax.tick_params(axis='y', colors=fg)
ax.set_ylabel('Accuracy (%)', color=fg, fontsize=12)
ax.set_title('Tool vs Category Accuracy — triage_assessment removed, mapped to emergency_dispatch', color=fg, fontsize=14, fontweight='bold')
ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor=fg)
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#21262d', linewidth=0.5)
plt.tight_layout()
plt.savefig('visuals/eval_accuracy_comparison.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: eval_accuracy_comparison.png')

# ── 2. Latency ──
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

avg_lat = [9866, 12559, 1464, 3097]
bars = ax.bar(x, avg_lat, 0.5, color=colors, edgecolor='#30363d')
for bar, val in zip(bars, avg_lat):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{val:,}ms',
            ha='center', va='bottom', color=fg, fontweight='bold', fontsize=13)

ax.set_ylim(0, 16000)
ax.set_xticks(x)
ax.set_xticklabels(models, color=fg, fontsize=11)
ax.tick_params(axis='y', colors=fg)
ax.set_ylabel('Latency (ms)', color=fg, fontsize=12)
ax.set_title('Average Latency per Query — Merge LoRA to match base latency', color=fg, fontsize=14, fontweight='bold')
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#21262d', linewidth=0.5)

# Arrow showing LoRA overhead
ax.annotate('LoRA overhead\n+2.7s', xy=(1, 12559), xytext=(1.8, 14000),
            arrowprops=dict(arrowstyle='->', color=red, lw=2),
            color=red, fontweight='bold', fontsize=11, ha='center')
ax.annotate('Merged model\n≈9.8s', xy=(0, 9866), xytext=(-0.5, 11500),
            arrowprops=dict(arrowstyle='->', color=green, lw=2),
            color=green, fontweight='bold', fontsize=11, ha='center')

plt.tight_layout()
plt.savefig('visuals/eval_latency_comparison.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: eval_latency_comparison.png')

# ── 3. Per-Tool Heatmap (no triage) ──
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

tools = ['emergency\n_dispatch', 'medication\n_check', 'mental_health\n_triage', 'specialist\n_referral', 'vital_signs\n_analysis', 'lab_order\n_suggestion']

base_acc =  [100, 100, 100, 20, 100, 100]
ft_acc =    [94, 100, 100, 60, 100, 100]
ms_acc =    [100, 83, 100, 20, 100, 50]
ml_acc =    [100, 83, 100, 80, 100, 50]

data = np.array([base_acc, ft_acc, ms_acc, ml_acc])
im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

ax.set_xticks(range(len(tools)))
ax.set_xticklabels(tools, color=fg, fontsize=10, rotation=30, ha='right')
ax.set_yticks(range(4))
ax.set_yticklabels(models, color=fg, fontsize=11)

for i in range(4):
    for j in range(len(tools)):
        val = data[i, j]
        color = 'black' if val > 50 else 'white'
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontweight='bold', fontsize=12)

ax.set_title('Per-Tool Accuracy — triage_assessment removed, predictions mapped to emergency_dispatch', color=fg, fontsize=13, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.ax.tick_params(colors=fg)
plt.tight_layout()
plt.savefig('visuals/eval_tool_heatmap.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: eval_tool_heatmap.png')

# ── 4. Base vs Fine-tuned (no triage) ──
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

tool_names = ['emergency\n_dispatch', 'specialist\n_referral', 'medication\n_check', 'mental_health\n_triage', 'vital_signs\n_analysis', 'lab_order\n_suggestion']
base_vals =  [100, 20, 100, 100, 100, 100]
ft_vals =    [94, 60, 100, 100, 100, 100]

x_pos = np.arange(len(tool_names))
w = 0.35

bars1 = ax.bar(x_pos - w/2, base_vals, w, label='Base Model', color=green, edgecolor='#30363d')
bars2 = ax.bar(x_pos + w/2, ft_vals, w, label='Fine-tuned (triage→emergency)', color=yellow, edgecolor='#30363d')

for bar, val in zip(bars1, base_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}%',
            ha='center', va='bottom', color=green, fontweight='bold', fontsize=11)
for bar, val in zip(bars2, ft_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}%',
            ha='center', va='bottom', color=yellow, fontweight='bold', fontsize=11)

ax.set_xticks(x_pos)
ax.set_xticklabels(tool_names, color=fg, fontsize=10)
ax.tick_params(axis='y', colors=fg)
ax.set_ylabel('Accuracy (%)', color=fg, fontsize=12)
ax.set_title('Base vs Fine-tuned — With triage→emergency safety handler', color=fg, fontsize=14, fontweight='bold')
ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor=fg)
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#21262d', linewidth=0.5)

# Show improvement
ax.annotate('+40%', xy=(1, 60), xytext=(1.5, 35),
            arrowprops=dict(arrowstyle='->', color=green, lw=2),
            color=green, fontweight='bold', fontsize=14, ha='center')
ax.text(0, 50, '-6%', color=red, fontweight='bold', fontsize=14, ha='center')

plt.tight_layout()
plt.savefig('visuals/eval_base_vs_finetuned.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: eval_base_vs_finetuned.png')

# ── 5. Accuracy with vs without handler ──
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

models_short = ['Base', 'SLM\n(Fine-tuned)', 'Mistral\nSmall', 'Mistral\nLarge']
before = [82.5, 60.0, 80.0, 87.5]
after = [89.2, 91.9, 83.8, 91.9]

x_pos = np.arange(len(models_short))
w = 0.35

bars1 = ax.bar(x_pos - w/2, before, w, label='With triage_assessment', color=red, edgecolor='#30363d')
bars2 = ax.bar(x_pos + w/2, after, w, label='With safety handler', color=green, edgecolor='#30363d')

for bar, val in zip(bars1, before):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%',
            ha='center', va='bottom', color=red, fontweight='bold', fontsize=12)
for bar, val in zip(bars2, after):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%',
            ha='center', va='bottom', color=green, fontweight='bold', fontsize=12)

# Improvement arrows
for i, (b, a) in enumerate(zip(before, after)):
    diff = a - b
    color = green if diff > 0 else red
    ax.annotate(f'+{diff:.1f}%' if diff > 0 else f'{diff:.1f}%', 
                xy=(i + w/2, a), xytext=(i + w/2, a + 5),
                color=color, fontweight='bold', fontsize=10, ha='center')

ax.set_ylim(0, 105)
ax.set_xticks(x_pos)
ax.set_xticklabels(models_short, color=fg, fontsize=11)
ax.tick_params(axis='y', colors=fg)
ax.set_ylabel('Tool Accuracy (%)', color=fg, fontsize=12)
ax.set_title('Impact of Safety Handler (triage_assessment → emergency_dispatch)', color=fg, fontsize=14, fontweight='bold')
ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor=fg)
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#21262d', linewidth=0.5)
plt.tight_layout()
plt.savefig('visuals/eval_handler_impact.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: eval_handler_impact.png')

print('\nAll 5 charts updated.')
