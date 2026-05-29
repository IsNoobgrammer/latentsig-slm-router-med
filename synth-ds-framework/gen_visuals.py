import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import os

os.makedirs('visuals', exist_ok=True)

# Load data
records = []
with open('dataset.jsonl') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

tools = Counter(r.get('tool_called', '?') for r in records)
cats = Counter(r.get('category', '?') for r in records)
models = Counter(r.get('generation_model_id', '?') for r in records)
lang = Counter(r.get('language', '?') for r in records)

# Style
sns.set_theme(style='darkgrid', font_scale=1.1)
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3']
bg = '#0d1117'
fg = '#c9d1d9'

# ── 1. Tool Distribution ──
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)
tool_names = [t.replace('_', ' ').title() for t in sorted(tools.keys())]
tool_vals = [tools[k] for k in sorted(tools.keys())]
bars = ax.barh(tool_names, tool_vals, color=colors[:len(tool_names)], edgecolor='#30363d')
for bar, val in zip(bars, tool_vals):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, str(val),
            va='center', color=fg, fontweight='bold', fontsize=11)
ax.set_xlabel('Count', color=fg, fontsize=12)
ax.set_title('Tool Distribution', color=fg, fontsize=16, fontweight='bold', pad=15)
ax.tick_params(colors=fg)
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, max(tool_vals) * 1.15)
plt.tight_layout()
plt.savefig('visuals/tool_distribution.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: tool_distribution.png')

# ── 2. Category Distribution (Donut) ──
fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor(bg)
cat_labels = ['Emergency', 'Urgent', 'Semi-Urgent', 'Routine']
cat_vals = [cats.get('emergency', 0), cats.get('urgent', 0), cats.get('semi_urgent', 0), cats.get('routine', 0)]
cat_colors = ['#f85149', '#d29922', '#58a6ff', '#3fb950']
wedges, texts, autotexts = ax.pie(
    cat_vals, labels=cat_labels, colors=cat_colors, autopct='%1.1f%%',
    startangle=90, pctdistance=0.82, textprops={'color': fg, 'fontsize': 12}
)
for t in autotexts:
    t.set_fontweight('bold')
    t.set_fontsize(11)
centre_circle = plt.Circle((0, 0), 0.60, fc=bg)
ax.add_artist(centre_circle)
ax.text(0, 0, str(sum(cat_vals)), ha='center', va='center', fontsize=28, fontweight='bold', color=fg)
ax.set_title('Category Distribution', color=fg, fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('visuals/category_distribution.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: category_distribution.png')

# ── 3. Language Split ──
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)
lang_labels = ['English', 'Hinglish']
lang_vals = [lang.get('en', 0), lang.get('hi_en', 0)]
bars = ax.bar(lang_labels, lang_vals, color=['#58a6ff', '#bc8cff'], width=0.5, edgecolor='#30363d')
for bar, val in zip(bars, lang_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(val),
            ha='center', va='bottom', color=fg, fontweight='bold', fontsize=14)
ax.set_ylabel('Count', color=fg, fontsize=12)
ax.set_title('Language Split', color=fg, fontsize=16, fontweight='bold', pad=15)
ax.tick_params(colors=fg)
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(lang_vals) * 1.15)
plt.tight_layout()
plt.savefig('visuals/language_split.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: language_split.png')

# ── 4. Model Distribution (Pie) ──
fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor(bg)
model_labels = [m.replace('-latest', '').replace('-', ' ').title() for m in models.keys()]
model_vals = list(models.values())
model_colors = ['#4C72B0', '#DD8452', '#55A868']
wedges, texts, autotexts = ax.pie(
    model_vals, labels=model_labels, colors=model_colors, autopct='%1.1f%%',
    startangle=90, textprops={'color': fg, 'fontsize': 11}
)
for t in autotexts:
    t.set_fontweight('bold')
ax.set_title('Generation Model Distribution', color=fg, fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('visuals/model_distribution.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: model_distribution.png')

# ── 5. Tool x Category Heatmap ──
tool_cat = {}
for r in records:
    t = r.get('tool_called', '?')
    c = r.get('category', '?')
    key = (t, c)
    tool_cat[key] = tool_cat.get(key, 0) + 1

tool_order = sorted(tools.keys())
cat_order = ['emergency', 'urgent', 'semi_urgent', 'routine']
matrix = np.zeros((len(tool_order), len(cat_order)))
for i, t in enumerate(tool_order):
    for j, c in enumerate(cat_order):
        matrix[i, j] = tool_cat.get((t, c), 0)

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)
sns.heatmap(
    matrix, annot=True, fmt='.0f', cmap='YlOrRd',
    xticklabels=[c.replace('_', '-').title() for c in cat_order],
    yticklabels=[t.replace('_', ' ').title() for t in tool_order],
    ax=ax, linewidths=0.5, linecolor='#30363d',
    cbar_kws={'label': 'Count'}
)
ax.set_title('Tool x Category Heatmap', color=fg, fontsize=16, fontweight='bold', pad=15)
ax.tick_params(colors=fg)
ax.set_xlabel('Category', color=fg, fontsize=12)
ax.set_ylabel('Tool', color=fg, fontsize=12)
plt.tight_layout()
plt.savefig('visuals/tool_category_heatmap.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: tool_category_heatmap.png')

# ── 6. Query Length Distribution ──
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)
en_lens = [len(r['user_query']) for r in records if r.get('language') == 'en']
hi_lens = [len(r['user_query']) for r in records if r.get('language') == 'hi_en']
ax.hist(en_lens, bins=30, alpha=0.7, color='#58a6ff', label=f'English (avg {np.mean(en_lens):.0f})', edgecolor='#30363d')
ax.hist(hi_lens, bins=30, alpha=0.7, color='#bc8cff', label=f'Hinglish (avg {np.mean(hi_lens):.0f})', edgecolor='#30363d')
ax.set_xlabel('Query Length (chars)', color=fg, fontsize=12)
ax.set_ylabel('Count', color=fg, fontsize=12)
ax.set_title('Query Length Distribution', color=fg, fontsize=16, fontweight='bold', pad=15)
ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor=fg)
ax.tick_params(colors=fg)
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('visuals/query_length_dist.png', dpi=150, facecolor=bg, bbox_inches='tight')
plt.close()
print('Created: query_length_dist.png')

print('\nAll 6 visualizations created in visuals/')
