# ===== src/api.py =====
# Purpose: Local web app. Serves the dashboard and a live single-applicant tester.
# Every decision shown is computed by a real model, never simulated.

import json
import os
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
for sub in ("counterfactuals", "models_under_test", "data"):
    sys.path.insert(0, str(SRC / sub))

from counterfactual_engine import generate_counterfactuals              # noqa: E402
from prompt_renderer import render_prompt, FIELDS                       # noqa: E402
from custom_model_client import approval_probabilities, THRESHOLD       # noqa: E402
import ollama_client                                                    # noqa: E402
from generate_synthetic_profiles import (                               # noqa: E402
    DEMOGRAPHICS, EMPLOYMENT, EDUCATION, AGE_RANGE, INCOME_RANGE, CREDIT_RANGE, LOAN_RANGE)

VALID = ("APPROVE", "REJECT")

# Allowed values come from the same place the test set was generated,
# so the live form can never send something a model has not seen.
CATEGORICAL = {**{k: list(v) for k, v in DEMOGRAPHICS.items()},
               "Employment_Type": list(EMPLOYMENT), "Education_Level": list(EDUCATION)}
NUMERIC = {"Age": AGE_RANGE, "Income": INCOME_RANGE,
           "Credit_Score": CREDIT_RANGE, "Loan_Amount": LOAN_RANGE}

app = Flask(__name__)


def available_models():
    models = ["custom", "ollama"]
    try:
        import openai_client   # noqa: F401
        if os.getenv("OPENAI_API_KEY"):
            models.append("openai")
    except ImportError:
        pass
    return models


def validate_profile(raw):
    if not isinstance(raw, dict):
        return None, "profile must be an object"
    profile = {"profile_id": "LIVE"}
    for key, _ in FIELDS:
        if key not in raw:
            return None, f"missing field: {key}"
        val = raw[key]
        if key in NUMERIC:
            try:
                val = int(val)
            except (TypeError, ValueError):
                return None, f"{key} must be a whole number"
            lo, hi = NUMERIC[key]
            if not lo <= val <= hi:
                return None, f"{key} must be between {lo} and {hi}"
        elif val not in CATEGORICAL[key]:
            return None, f"{key} must be one of {CATEGORICAL[key]}"
        profile[key] = val
    return profile, None


def run_model(model, variants):
    """Returns (list of results, error message)."""
    if model == "custom":
        probs = approval_probabilities([v["profile"] for v in variants])
        return [{"decision": "APPROVE" if p > THRESHOLD else "REJECT",
                 "probability": round(float(p), 3), "reasoning": None} for p in probs], None
    if model == "ollama":
        fn = ollama_client.evaluate_prompt
    else:
        import openai_client
        fn = openai_client.evaluate_prompt
    out = []
    for v in variants:
        r = fn(render_prompt(v["profile"]))
        if r["decision"] == "ERROR":
            return None, r["reasoning"]
        out.append({"decision": r["decision"], "probability": None, "reasoning": r["reasoning"]})
    return out, None


@app.get("/api/models")
def api_models():
    return jsonify(models=available_models())


@app.post("/api/evaluate")
def api_evaluate():
    body = request.get_json(silent=True) or {}
    model = body.get("model")
    if model not in available_models():
        return jsonify(error=f"model must be one of {available_models()}"), 400
    profile, err = validate_profile(body.get("profile"))
    if err:
        return jsonify(error=err), 400

    variants = [{"label": "As entered", "profile": profile}]
    for cf in generate_counterfactuals(profile):
        variants.append({"label": f"{cf['test']}: {cf['from']} → {cf['to']}", "profile": cf["profile"]})

    results, err = run_model(model, variants)
    if err:
        return jsonify(error=err), 502

    base = results[0]["decision"]
    rows = []
    for v, r in zip(variants, results):
        changed = None
        if v["label"] != "As entered" and base in VALID and r["decision"] in VALID:
            changed = r["decision"] != base
        rows.append({"label": v["label"], "changed": changed, **r})
    flips = sum(1 for r in rows if r["changed"])
    return jsonify(model=model, original=base, rows=rows, flips=flips, total=len(rows) - 1)


@app.get("/health")
def health():
    return jsonify(status="ok", models=available_models())


@app.get("/dashboard")
def dashboard():
    path = ROOT / "results" / "dashboard.html"
    if not path.exists():
        abort(404, "Run src/dashboard/generate_dashboard.py first.")
    return send_file(path)


@app.get("/")
def home():
    return ("<body style='font-family:sans-serif;max-width:600px;margin:40px auto'>"
            "<h1>Bias Audit</h1><ul><li><a href='/dashboard'>Dashboard (saved results)</a></li>"
            "<li><a href='/live'>Live tester (one applicant, all models)</a></li></ul></body>")


LIVE_PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Live Tester</title><style>
body{font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:900px;margin:30px auto;padding:0 16px;color:#111827}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}label{font-size:13px;display:flex;flex-direction:column;gap:3px}
input,select{padding:6px;border:1px solid #d1d5db;border-radius:5px}
button{margin-top:14px;padding:10px 20px;background:#2563eb;color:#fff;border:0;border-radius:6px;font-size:15px;cursor:pointer}
button:disabled{background:#9ca3af}.box{border:1px solid #e5e7eb;border-radius:8px;padding:12px 16px;margin-top:16px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}th,td{border:1px solid #e5e7eb;padding:5px 8px;text-align:left}
.chg{background:#fee2e2}.note{color:#4b5563;font-size:13px}.err{color:#b91c1c}
</style></head><body><a href="/">&larr; Home</a><h1>Live tester</h1>
<p class="note">Enter one applicant. Each model decides it as entered, then again with exactly one attribute changed at a time.
A single applicant is an illustration, not statistical evidence. See the dashboard for rates with confidence intervals.
The LLM can take about 30 seconds.</p>
<div class="grid">__FORM__</div><button id="go" onclick="run()">Test applicant</button><div id="out"></div>
<script>
const KEYS = __KEYS__;
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function run() {
  const btn = document.getElementById('go'), out = document.getElementById('out');
  const profile = {}; KEYS.forEach(k => profile[k] = document.getElementById(k).value);
  btn.disabled = true; out.innerHTML = '';
  const models = (await (await fetch('/api/models')).json()).models;
  for (const m of models) {
    const box = document.createElement('div'); box.className = 'box';
    box.innerHTML = '<b>' + esc(m) + '</b> running...'; out.appendChild(box);
    const res = await fetch('/api/evaluate', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                              body: JSON.stringify({model: m, profile})});
    const d = await res.json();
    if (!res.ok) { box.innerHTML = '<b>' + esc(m) + '</b> <span class="err">' + esc(d.error) + '</span>'; continue; }
    const rows = d.rows.map(r => '<tr class="' + (r.changed ? 'chg' : '') + '"><td>' + esc(r.label) + '</td><td>' + esc(r.decision) +
      '</td><td>' + (r.changed ? 'CHANGED' : '') + '</td><td>' + esc(r.probability ?? '') + '</td></tr>').join('');
    box.innerHTML = '<b>' + esc(m) + '</b>: decided ' + esc(d.original) + ' as entered. Decision changed in ' + d.flips + ' of ' +
      d.total + ' counterfactuals.<table><tr><th>Variant</th><th>Decision</th><th></th><th>Approval probability</th></tr>' + rows + '</table>';
  }
  btn.disabled = false;
}
</script></body></html>"""


def form_html():
    rows = []
    for key, label in FIELDS:
        if key in NUMERIC:
            lo, hi = NUMERIC[key]
            rows.append(f'<label>{label}<input id="{key}" type="number" min="{lo}" max="{hi}" value="{(lo + hi) // 2}"></label>')
        else:
            opts = "".join(f"<option>{o}</option>" for o in CATEGORICAL[key])
            rows.append(f'<label>{label}<select id="{key}">{opts}</select></label>')
    return "".join(rows)


@app.get("/live")
def live():
    return LIVE_PAGE.replace("__FORM__", form_html()).replace("__KEYS__", json.dumps([k for k, _ in FIELDS]))


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000")
    print(f"Models available: {available_models()}")
    app.run(host="127.0.0.1", port=5000, debug=False)