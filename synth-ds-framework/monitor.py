# ─────────────────────────────────────────────────────────────
# MONITOR SERVER
# Lightweight Flask dashboard for real-time datagen monitoring
#
# Usage:
#   python monitor.py              # starts on http://localhost:5000
#   python monitor.py --port 8080  # custom port
# ─────────────────────────────────────────────────────────────

import json
import os
import time
from datetime import datetime

try:
    from flask import Flask, jsonify
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "flask", "-q"])
    from flask import Flask, jsonify

app = Flask(__name__)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.jsonl")
STATS_PATH = os.path.join(OUTPUT_DIR, ".live_stats.json")


def read_stats() -> dict:
    """Read live stats from the orchestrator's stats file."""
    if os.path.exists(STATS_PATH):
        try:
            with open(STATS_PATH) as f:
                return json.load(f)
        except:
            pass
    return {}


def read_dataset(limit: int = 50) -> list[dict]:
    """Read last N records from JSONL."""
    records = []
    if not os.path.exists(DATASET_PATH):
        return records
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    pass
    return records[-limit:]


def compute_dataset_stats(records: list[dict]) -> dict:
    """Compute stats from actual JSONL records."""
    total = len(records)
    if total == 0:
        return {"total": 0}

    en = sum(1 for r in records if r.get("language") == "en")
    hi = sum(1 for r in records if r.get("language") == "hi_en")
    passed = sum(1 for r in records if r.get("judge_verdict") == "pass")
    failed = total - passed

    # Model breakdown
    models = {}
    for r in records:
        m = r.get("generation_model_id", "unknown")
        models[m] = models.get(m, 0) + 1

    # Category breakdown
    categories = {}
    for r in records:
        try:
            parsed = json.loads(r.get("parsed_response", "{}"))
            cat = parsed.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        except:
            pass

    # Tool breakdown
    tools = {}
    for r in records:
        try:
            parsed = json.loads(r.get("parsed_response", "{}"))
            tool = parsed.get("tool", "unknown")
            tools[tool] = tools.get(tool, 0) + 1
        except:
            pass

    return {
        "total": total,
        "en": en,
        "hi_en": hi,
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{passed/total*100:.1f}%",
        "models": models,
        "categories": categories,
        "tools": tools,
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>LatentSig Datagen Monitor</title>
<meta http-equiv="refresh" content="5">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }
  h1 { color: #58a6ff; margin-bottom: 20px; font-size: 1.5em; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center; }
  .card .num { font-size: 2em; font-weight: 700; color: #58a6ff; }
  .card .label { font-size: 0.85em; color: #8b949e; margin-top: 4px; }
  .card.green .num { color: #3fb950; }
  .card.red .num { color: #f85149; }
  .card.yellow .num { color: #d29922; }
  .card.purple .num { color: #bc8cff; }
  .section { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .section h2 { color: #58a6ff; font-size: 1.1em; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  th { text-align: left; color: #8b949e; padding: 6px 8px; border-bottom: 1px solid #30363d; }
  td { padding: 6px 8px; border-bottom: 1px solid #21262d; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .tag-emergency { background: #f8514922; color: #f85149; }
  .tag-urgent { background: #d2992222; color: #d29922; }
  .tag-semi_urgent { background: #58a6ff22; color: #58a6ff; }
  .tag-routine { background: #3fb95022; color: #3fb950; }
  .tag-pass { background: #3fb95022; color: #3fb950; }
  .tag-fail { background: #f8514922; color: #f85149; }
  .tag-en { background: #58a6ff22; color: #58a6ff; }
  .tag-hi_en { background: #bc8cff22; color: #bc8cff; }
  .bar { height: 20px; border-radius: 4px; overflow: hidden; display: flex; margin: 4px 0; }
  .bar div { height: 100%; transition: width 0.3s; }
  .progress-bar { background: #21262d; border-radius: 8px; height: 24px; overflow: hidden; margin: 8px 0; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, #238636, #3fb950); border-radius: 8px; transition: width 0.5s; display: flex; align-items: center; justify-content: center; font-size: 0.75em; font-weight: 600; color: #fff; min-width: 40px; }
  .footer { color: #484f58; font-size: 0.8em; text-align: center; margin-top: 20px; }
</style>
</head><body>
<h1>LatentSig Datagen Monitor</h1>

<div class="grid">
  <div class="card"><div class="num">{{stats.total}}</div><div class="label">Total Samples</div></div>
  <div class="card green"><div class="num">{{stats.passed}}</div><div class="label">Passed</div></div>
  <div class="card red"><div class="num">{{stats.failed}}</div><div class="label">Failed</div></div>
  <div class="card"><div class="num">{{stats.pass_rate}}</div><div class="label">Pass Rate</div></div>
  <div class="card"><div class="num">{{live.rate}}/s</div><div class="label">Rate</div></div>
  <div class="card yellow"><div class="num">{{live.eta_seconds}}s</div><div class="label">ETA</div></div>
  <div class="card purple"><div class="num">{{live.workers}}</div><div class="label">Workers</div></div>
  <div class="card"><div class="num">{{live.errors}}</div><div class="label">Errors</div></div>
</div>

<div class="section">
  <h2>Progress</h2>
  <p>EN: {{live.en_count}}/{{live.target_en}} &nbsp;&nbsp; HI_EN: {{live.hi_en_count}}/{{live.target_hi_en}}</p>
  <div class="progress-bar">
    <div class="progress-fill" style="width: {{en_pct}}%">{{en_pct}}% EN</div>
  </div>
  <div class="progress-bar">
    <div class="progress-fill" style="width: {{hi_pct}}%">{{hi_pct}}% HI_EN</div>
  </div>
</div>

<div class="grid" style="grid-template-columns: 1fr 1fr;">
  <div class="section">
    <h2>Language Split</h2>
    <div class="bar">
      <div style="width:{{en_pct_ds}}%; background:#58a6ff;" title="EN: {{stats.en}}"></div>
      <div style="width:{{hi_pct_ds}}%; background:#bc8cff;" title="HI_EN: {{stats.hi_en}}"></div>
    </div>
    <p style="font-size:0.85em; margin-top:4px;">
      <span class="tag tag-en">EN {{stats.en}}</span>
      <span class="tag tag-hi_en">HI_EN {{stats.hi_en}}</span>
    </p>
  </div>
  <div class="section">
    <h2>Categories</h2>
    {% for cat, count in stats.categories.items() %}
    <span class="tag tag-{{cat}}">{{cat}}: {{count}}</span>
    {% endfor %}
  </div>
</div>

<div class="section">
  <h2>Tools Used</h2>
  {% for tool, count in stats.tools.items() %}
  <span class="tag" style="background:#30363d; color:#c9d1d9; margin:2px;">{{tool}}: {{count}}</span>
  {% endfor %}
</div>

<div class="section">
  <h2>Models Used</h2>
  {% for model, count in stats.models.items() %}
  <span class="tag" style="background:#30363d; color:#c9d1d9; margin:2px;">{{model}}: {{count}}</span>
  {% endfor %}
</div>

<div class="section">
  <h2>Recent Samples (last 20)</h2>
  <table>
    <tr><th>Query</th><th>Tool</th><th>Category</th><th>Lang</th><th>Verdict</th><th>Model</th></tr>
    {% for s in recent %}
    <tr>
      <td>{{s.query}}</td>
      <td>{{s.tool}}</td>
      <td><span class="tag tag-{{s.category}}">{{s.category}}</span></td>
      <td><span class="tag tag-{{s.language}}">{{s.language}}</span></td>
      <td><span class="tag tag-{{s.verdict}}">{{s.verdict}}</span></td>
      <td style="font-size:0.8em;">{{s.model}}</td>
    </tr>
    {% endfor %}
  </table>
</div>

<div class="footer">
  Auto-refreshes every 5s &nbsp;|&nbsp; Last updated: {{timestamp}} &nbsp;|&nbsp; Elapsed: {{live.elapsed}}s
</div>
</body></html>"""


@app.route("/")
def dashboard():
    from flask import render_template_string

    live = read_stats()
    records = read_dataset(limit=200)
    ds_stats = compute_dataset_stats(records)

    # Recent from live stats
    recent = live.get("recent_samples", [])

    # Percentages
    target_en = live.get("target_en", 1)
    target_hi = live.get("target_hi_en", 1)
    en_pct = min(100, int(live.get("en_count", 0) / target_en * 100)) if target_en else 0
    hi_pct = min(100, int(live.get("hi_en_count", 0) / target_hi * 100)) if target_hi else 0

    total_ds = ds_stats.get("total", 1) or 1
    en_pct_ds = int(ds_stats.get("en", 0) / total_ds * 100)
    hi_pct_ds = 100 - en_pct_ds

    return render_template_string(
        DASHBOARD_HTML,
        stats=ds_stats,
        live=live,
        recent=recent,
        en_pct=en_pct,
        hi_pct=hi_pct,
        en_pct_ds=en_pct_ds,
        hi_pct_ds=hi_pct_ds,
        timestamp=datetime.now().strftime("%H:%M:%S"),
    )


@app.route("/api/stats")
def api_stats():
    live = read_stats()
    records = read_dataset(limit=200)
    ds_stats = compute_dataset_stats(records)
    return jsonify({"live": live, "dataset": ds_stats})


@app.route("/api/recent")
def api_recent():
    records = read_dataset(limit=20)
    for r in records:
        r.pop("system_prompt", None)  # Too large for API
    return jsonify(records)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Datagen Monitor Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"  Monitor: http://{args.host}:{args.port}")
    print(f"  Stats file: {STATS_PATH}")
    print(f"  Dataset: {DATASET_PATH}")
    app.run(host=args.host, port=args.port, debug=False)
