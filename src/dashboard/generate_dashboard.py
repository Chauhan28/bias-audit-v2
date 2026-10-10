# ===== src/dashboard/generate_dashboard.py =====
# Purpose: Build results/dashboard.html from the saved raw_*.jsonl files.
# Every raw_<model>.jsonl found is included automatically, so OpenAI and Gemini
# will appear here as soon as their result files exist.

import html
import sys
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "evaluation"))
sys.path.insert(0, str(SRC / "counterfactuals"))
from bias_metrics import read_records, flip_metrics, parity_metrics   # noqa: E402
from counterfactual_engine import load_profiles, PAIRS                  # noqa: E402

RESULTS = SRC.parent / "results"
OUT = RESULTS / "dashboard.html"
SAMPLE = 200   # all models are compared on the same first N profiles
COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]

FROM = {n: "+".join(a.values()) for n, a, b in PAIRS}
TO = {n: "+".join(b.values()) for n, a, b in PAIRS}
GROUPS = [
    ("Protected attributes", ["Gender", "Race", "Disability"]),
    ("Proxy / contested attributes", ["Criminal_Record", "Zip_Code", "Citizenship", "Language"]),
    ("Intersectional", ["Gender+Race"]),
]
esc = html.escape


def collect():
    profiles = load_profiles()
    by_id = {p["profile_id"]: p for p in profiles}
    scopes = {"sample": {p["profile_id"] for p in profiles[:SAMPLE]}, "all": set(by_id)}
    entries = []
    for f in sorted(RESULTS.glob("raw_*.jsonl")):
        name, recs = f.stem.replace("raw_", ""), read_records(f)
        runs = [(name, "sample")]
        if name == "custom":
            runs.append((f"custom (all {len(profiles):,})", "all"))
        for label, scope in runs:
            ids = scopes[scope]
            counts = Counter(r["decision"] for r in recs
                             if r["variant"] == "original" and r["profile_id"] in ids)
            entries.append({
                "label": label, "n": sum(counts.values()), "counts": counts,
                "flips": {(r["test"], r["from"]): r for r in flip_metrics(recs, ids)},
                "parity": {r["test"]: r for r in parity_metrics(recs, {i: by_id[i] for i in ids})},
            })
    return entries


def cards(entries):
    out = []
    for i, e in enumerate(entries):
        n = e["n"] or 1
        out.append(
            f'<div class="card" style="border-top:4px solid {COLORS[i % len(COLORS)]}">'
            f'<b>{esc(e["label"])}</b><div class="big">{e["counts"].get("APPROVE", 0) / n:.0%}</div>'
            f'<div>approval rate</div><div class="sm">{e["n"]} profiles · '
            f'unclear {e["counts"].get("UNCLEAR", 0) / n:.0%}</div></div>')
    return "".join(out)


def chart(entries, tests):
    W, L, RH, MAX = 620, 190, 20, 0.7
    px = lambda v: L + v / MAX * (W - L - 20)
    y, rows = 8, []
    for t in tests:
        rows.append(f'<text x="0" y="{y + 12}" font-weight="600">{esc(t)}: {esc(FROM[t])} → {esc(TO[t])}</text>')
        y += 20
        for i, e in enumerate(entries):
            r = e["flips"].get((t, FROM[t]))
            c = COLORS[i % len(COLORS)]
            rows.append(f'<text x="12" y="{y + 12}" fill="{c}">{esc(e["label"])}</text>')
            if r and r["n"]:
                rows.append(
                    f'<line x1="{px(r["ci_low"]):.1f}" x2="{px(r["ci_high"]):.1f}" y1="{y + 8}" y2="{y + 8}" '
                    f'stroke="{c}" stroke-width="2"/><circle cx="{px(r["flip_rate"]):.1f}" cy="{y + 8}" r="4" fill="{c}"/>')
            y += RH
        y += 12
    grid = "".join(
        f'<line x1="{px(v / 10):.1f}" x2="{px(v / 10):.1f}" y1="0" y2="{y}" stroke="#e5e7eb"/>'
        f'<text x="{px(v / 10) - 8:.1f}" y="{y + 14}" fill="#6b7280">{v * 10}%</text>' for v in range(8))
    return f'<svg viewBox="0 0 {W} {y + 22}" width="100%" font-size="12">{grid}{"".join(rows)}</svg>'


def cell(r):
    if not r or not r["n"]:
        return "–"
    return (f'{r["flip_rate"]:.1%} <small>[{r["ci_low"]:.0%}–{r["ci_high"]:.0%}] n={r["n"]}</small>')


def flip_table(entries):
    head = "".join(f"<th>{esc(e['label'])}</th>" for e in entries)
    body = []
    for title, tests in GROUPS:
        body.append(f'<tr class="grp"><td colspan="{len(entries) + 1}">{title}</td></tr>')
        for t in tests:
            cells = "".join(f"<td>{cell(e['flips'].get((t, FROM[t])))}</td>" for e in entries)
            body.append(f"<tr><td>{esc(t)}: {esc(FROM[t])} → {esc(TO[t])}</td>{cells}</tr>")
    return f"<table><tr><th>Flip (advantaged → disadvantaged)</th>{head}</tr>{''.join(body)}</table>"


def parity_table(entries):
    tests = [t for _, ts in GROUPS for t in ts if t != "Gender+Race"]
    head = "".join(f"<th>{esc(e['label'])}</th>" for e in entries)
    body = []
    for t in tests:
        cells = ""
        for e in entries:
            r = e["parity"].get(t)
            cells += f"<td>{r['gap'] * 100:+.1f} pts</td>" if r and r["gap"] is not None else "<td>–</td>"
        body.append(f"<tr><td>{esc(t)}: {esc(FROM[t])} vs {esc(TO[t])}</td>{cells}</tr>")
    return f"<table><tr><th>Approval-rate gap</th>{head}</tr>{''.join(body)}</table>"


CSS = """
body{font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:900px;margin:30px auto;padding:0 16px;color:#111827}
h1{margin-bottom:4px}h2{margin-top:34px;border-bottom:1px solid #e5e7eb;padding-bottom:6px}
.sub,.note{color:#4b5563;font-size:14px;line-height:1.5}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}
.card{border:1px solid #e5e7eb;border-radius:8px;padding:12px 16px;min-width:150px}
.big{font-size:30px;font-weight:700}.sm{color:#6b7280;font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:10px}
th,td{border:1px solid #e5e7eb;padding:6px 8px;text-align:left}th{background:#f3f4f6}
.grp td{background:#f9fafb;font-weight:600}small{color:#6b7280}li{margin:6px 0;font-size:14px}
"""

LIMITS = [
    "Profiles are synthetic. Demographics are sampled independently of income and credit score, so results isolate the model's behaviour but do not reflect real-world correlations.",
    "The custom model was trained on a dataset built to contain bias, so it is biased by design. The finding is that this framework detects and quantifies it.",
    "LLMs were tested on the first 200 profiles with one prompt wording at temperature 0. Small groups and intersectional tests have wide confidence intervals; treat them as indicative.",
    "Flip rate measures sensitivity to an attribute. It depends on how many applicants sit near a model's approve/reject boundary, so it is reported alongside approval-rate gaps.",
    "Zip code, criminal record, citizenship and language are proxy or contested attributes. An LLM may respond to loaded wording such as 'Historically Redlined', which is not the same as learned discrimination.",
    "Only decisions are analysed. LLM explanations were found to be unreliable and are not used as evidence.",
    "Non-binary applicants and groups with very few profiles are not tested.",
]


def page(entries):
    li = "".join(f"<li>{esc(t)}</li>" for t in LIMITS)
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Bias Audit Dashboard</title><style>{CSS}</style></head><body>"
        #"<h1>Bias Audit Dashboard</h1>"
        "<a href='/'>&larr; Home</a><h1>Bias Audit Dashboard</h1>"
        "<p class='sub'>Each applicant is decided once as-is, then again with exactly one attribute changed (a counterfactual). "
        "The <b>flip rate</b> is the share of applicants whose decision changed. Dots are flip rates; lines are 95% confidence intervals. "
        f"Models are compared on the same first {SAMPLE} profiles.</p>"
        f"<div class='cards'>{cards(entries)}</div>"
        f"<h2>{GROUPS[0][0]}</h2>{chart(entries, GROUPS[0][1])}"
        f"<h2>Proxy, contested and intersectional</h2>{chart(entries, GROUPS[1][1] + GROUPS[2][1])}"
        f"<h2>All flip results</h2>{flip_table(entries)}"
        f"<h2>Approval-rate gaps (demographic parity)</h2><p class='note'>Advantaged group minus disadvantaged group, original profiles only. Positive means the advantaged group is approved more often.</p>{parity_table(entries)}"
        f"<h2>Limitations</h2><ul>{li}</ul></body></html>")


if __name__ == "__main__":
    entries = collect()
    if not entries:
        sys.exit("No raw_*.jsonl files in results/. Run batch_runner.py first.")
    OUT.write_text(page(entries), encoding="utf-8")
    print(f"Models included: {[e['label'] for e in entries]}")
    print(f"Saved: {OUT}")