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

models = ['Mistral\nSmall', 'Mistral\nLarge', 'SLM\n(Ours)']
colors = [blue, purple, yellow]

# ── 1. Overall Accuracy Comparison ──
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

tool_acc = [80.0, 87.5, 60.0]
cat_acc = [90.0, 90.0, 87.5]
x = np.arange(len(models))
w = 0.35

bars1 = ax.bar(x - w/2, tool_acc, w, label='Tool Accuracy', color=blue, edgecolor='#30363d')
bars2 = ax.bar(x + w/2, cat_acc, w, label='Category Accuracy', color=green, edgecolor='#30363d')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.0f}%',
            ha='center', va='bottom', color=fg, fontweight='bold', fontsize=14)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.0f}%',
            ha='center', va='bottom', color=fg, fontweight='bold', fontsize=14)

ax.set_ylim(0, 105)
ax.set_xticks(x)
ax.set_xticklabels(models, color=fg, fontsize=12)
ax.tick_params(axis='y', colors=fg)
ax.set_ylabel('Accuracy (%)', color=fg, fontsize=12)
ax.set_title('Tool vs Category Accuracy', color=fg, fontsize=16, fontweight='bold')
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

# ── 2. Latency Comparison ──
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

avg_lat = [1464, 3097, 12559]
p50_lat = [1446, 3077, 12216]
p95_lat = [1828, 4462, 17858]

bars = ax.bar(x, avg_lat, 0.5, color=colors, edgecolor='#30363d')
for bar, val in zip(bars, avg_lat):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{val:,}ms',
            ha='center', va='bottom', color=fg, fontweight='bold', fontsize=13)

ax.set_ylim(0, 16000)
ax.set_xticks(x)
ax.set_xticklabels(models, color=fg, fontsize=12)
ax.tick_params(axis='y', colors=fg)
ax.set_ylabel('Latency (ms)', color=fg, fontsize=12)
ax.set_title('Average Latency per Query', color=fg, fontsize=16, fontweight='bold')
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#21262d', linewidth=0.5)
plt.tight_layout()
plt.savefig('visuals/eval_latency_comparison.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: eval_latency_comparison.png')

# ── 3. Per-Tool Accuracy Heatmap ──
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

tools = ['emergency\n_dispatch', 'medication\n_check', 'mental_health\n_triage', 'specialist\n_referral', 'vital_signs\n_analysis', 'triage\n_assessment', 'lab_order\n_suggestion']
gt_counts = [16, 6, 5, 5, 3, 3, 2]

ms_acc =  [100, 83.3, 100, 20, 100, 33.3, 50]
ml_acc =  [100, 83.3, 100, 80, 100, 33.3, 50]
slm_acc = [25, 100, 100, 60, 100, 33.3, 100]

data = np.array([ms_acc, ml_acc, slm_acc])
im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

ax.set_xticks(range(len(tools)))
ax.set_xticklabels(tools, color=fg, fontsize=9, rotation=30, ha='right')
ax.set_yticks(range(3))
ax.set_yticklabels(['Mistral Small', 'Mistral Large', 'SLM (Ours)'], color=fg, fontsize=11)

for i in range(3):
    for j in range(len(tools)):
        val = data[i, j]
        color = 'black' if val > 50 else 'white'
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontweight='bold', fontsize=11)

ax.set_title('Per-Tool Accuracy (%) — Green = Good, Red = Bad', color=fg, fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.ax.tick_params(colors=fg)
plt.tight_layout()
plt.savefig('visuals/eval_tool_heatmap.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: eval_tool_heatmap.png')

# ── 4. SLM Prediction Distribution vs Ground Truth ──
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

tool_names = ['emergency\n_dispatch', 'triage\n_assessment', 'medication\n_check', 'mental_health\n_triage', 'vital_signs\n_analysis', 'lab_order\n_suggestion', 'specialist\n_referral']
gt =  [16, 3, 6, 5, 3, 2, 5]
slm = [4, 12, 6, 5, 5, 5, 3]

x_pos = np.arange(len(tool_names))
w = 0.35

bars1 = ax.bar(x_pos - w/2, gt, w, label='Ground Truth', color=blue, edgecolor='#30363d')
bars2 = ax.bar(x_pos + w/2, slm, w, label='SLM Predictions', color=red, edgecolor='#30363d')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(int(bar.get_height())),
            ha='center', va='bottom', color=blue, fontweight='bold', fontsize=11)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(int(bar.get_height())),
            ha='center', va='bottom', color=red, fontweight='bold', fontsize=11)

ax.set_xticks(x_pos)
ax.set_xticklabels(tool_names, color=fg, fontsize=9)
ax.tick_params(axis='y', colors=fg)
ax.set_ylabel('Count', color=fg, fontsize=12)
ax.set_title('SLM: Ground Truth vs Predictions (the imbalance)', color=fg, fontsize=14, fontweight='bold')
ax.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor=fg)
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#21262d', linewidth=0.5)
plt.tight_layout()
plt.savefig('visuals/eval_slm_distribution.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: eval_slm_distribution.png')

print('\nAll 4 eval charts created.')
