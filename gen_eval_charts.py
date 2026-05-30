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
orange = '#db6d28'

models = ['Qwen3-4B\nBase', 'SLM\n(Fine-tuned)', 'Mistral\nSmall', 'Mistral\nLarge']
colors = [green, yellow, blue, purple]

# ── 1. Overall Accuracy Comparison ──
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

tool_acc = [82.5, 60.0, 80.0, 87.5]
cat_acc = [90.0, 87.5, 90.0, 90.0]
x = np.arange(len(models))
w = 0.35

bars1 = ax.bar(x - w/2, tool_acc, w, label='Tool Accuracy', color=blue, edgecolor='#30363d')
bars2 = ax.bar(x + w/2, cat_acc, w, label='Category Accuracy', color=green, edgecolor='#30363d')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.0f}%',
            ha='center', va='bottom', color=fg, fontweight='bold', fontsize=13)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.0f}%',
            ha='center', va='bottom', color=fg, fontweight='bold', fontsize=13)

ax.set_ylim(0, 105)
ax.set_xticks(x)
ax.set_xticklabels(models, color=fg, fontsize=11)
ax.tick_params(axis='y', colors=fg)
ax.set_ylabel('Accuracy (%)', color=fg, fontsize=12)
ax.set_title('Tool vs Category Accuracy — All Models', color=fg, fontsize=16, fontweight='bold')
ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor=fg)
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#21262d', linewidth=0.5)

# Highlight fine-tuned is worse
ax.annotate('Fine-tuning\nHURT accuracy', xy=(1, 60), xytext=(1.8, 45),
            arrowprops=dict(arrowstyle='->', color=red, lw=2),
            color=red, fontweight='bold', fontsize=11, ha='center')

plt.tight_layout()
plt.savefig('visuals/eval_accuracy_comparison.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: eval_accuracy_comparison.png')

# ── 2. Latency Comparison ──
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
ax.set_title('Average Latency per Query', color=fg, fontsize=16, fontweight='bold')
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#21262d', linewidth=0.5)

# Add note
ax.text(1.5, 14000, 'Mistral API is 7-8x faster\nthan local 4B inference',
        color=yellow, fontsize=10, ha='center', style='italic')

plt.tight_layout()
plt.savefig('visuals/eval_latency_comparison.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: eval_latency_comparison.png')

# ── 3. Per-Tool Accuracy Heatmap ──
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

tools = ['emergency\n_dispatch', 'medication\n_check', 'mental_health\n_triage', 'specialist\n_referral', 'vital_signs\n_analysis', 'triage\n_assessment', 'lab_order\n_suggestion']

base_acc =  [100, 100, 100, 20, 100, 0, 100]
ft_acc =    [25, 100, 100, 60, 100, 33.3, 100]
ms_acc =    [100, 83.3, 100, 20, 100, 33.3, 50]
ml_acc =    [100, 83.3, 100, 80, 100, 33.3, 50]

data = np.array([base_acc, ft_acc, ms_acc, ml_acc])
im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

ax.set_xticks(range(len(tools)))
ax.set_xticklabels(tools, color=fg, fontsize=9, rotation=30, ha='right')
ax.set_yticks(range(4))
ax.set_yticklabels(models, color=fg, fontsize=11)

for i in range(4):
    for j in range(len(tools)):
        val = data[i, j]
        color = 'black' if val > 50 else 'white'
        weight = 'bold'
        # Highlight fine-tuned regression
        if i == 1 and val < [base_acc[j], ft_acc[j], ms_acc[j], ml_acc[j]][0] - 10:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor=red, linewidth=3))
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontweight=weight, fontsize=11)

ax.set_title('Per-Tool Accuracy (%) — Red border = fine-tuned regression', color=fg, fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.ax.tick_params(colors=fg)
plt.tight_layout()
plt.savefig('visuals/eval_tool_heatmap.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: eval_tool_heatmap.png')

# ── 4. Fine-tuned vs Base: What Changed ──
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

tool_names = ['emergency\n_dispatch', 'specialist\n_referral', 'triage\n_assessment', 'lab_order\n_suggestion', 'medication\n_check', 'mental_health\n_triage', 'vital_signs\n_analysis']
base_vals =  [100, 20, 0, 100, 100, 100, 100]
ft_vals =    [25, 60, 33, 100, 100, 100, 100]

x_pos = np.arange(len(tool_names))
w = 0.35

bars1 = ax.bar(x_pos - w/2, base_vals, w, label='Base Model', color=green, edgecolor='#30363d')
bars2 = ax.bar(x_pos + w/2, ft_vals, w, label='Fine-tuned', color=red, edgecolor='#30363d')

for bar, val in zip(bars1, base_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}%',
            ha='center', va='bottom', color=green, fontweight='bold', fontsize=11)
for bar, val in zip(bars2, ft_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}%',
            ha='center', va='bottom', color=red, fontweight='bold', fontsize=11)

ax.set_xticks(x_pos)
ax.set_xticklabels(tool_names, color=fg, fontsize=9)
ax.tick_params(axis='y', colors=fg)
ax.set_ylabel('Accuracy (%)', color=fg, fontsize=12)
ax.set_title('Base vs Fine-tuned: What the Training Destroyed', color=fg, fontsize=14, fontweight='bold')
ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor=fg)
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#21262d', linewidth=0.5)

# Arrows for regression
ax.annotate('', xy=(0 - w/2, 100), xytext=(0 + w/2, 25),
            arrowprops=dict(arrowstyle='->', color=red, lw=3))
ax.text(0, 55, '-75%', color=red, fontweight='bold', fontsize=14, ha='center')

plt.tight_layout()
plt.savefig('visuals/eval_base_vs_finetuned.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: eval_base_vs_finetuned.png')

print('\nAll 4 charts updated with base model comparison.')
