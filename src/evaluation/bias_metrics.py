# ===== src/evaluation/bias_metrics.py =====
# Purpose: Turn raw decisions into bias metrics, per model:
#   1. Flip rate per test and direction, with a 95% Wilson confidence interval
#   2. Direction of the flips (approve->reject vs reject->approve) + sign test
#   3. Average shift in approval probability (custom model only)
#   4. Demographic parity: approval-rate gap between groups (original profiles only)
#   5. UNCLEAR rate (measurement quality)
# Pairs where either decision is UNCLEAR are excluded and counted separately.

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import binomtest

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "counterfactuals"))
from counterfactual_engine import load_profiles, PAIRS   # noqa: E402

RESULTS_DIR = SRC.parent / "results"
VALID = ("APPROVE", "REJECT")


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def read_records(path):
    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def flip_metrics(records, ids):
    originals = {r["profile_id"]: r for r in records if r["variant"] == "original"}
    groups = defaultdict(lambda: {"n": 0, "excluded": 0, "a2r": 0, "r2a": 0, "shifts": []})

    for r in records:
        if r["variant"] == "original" or r["profile_id"] not in ids:
            continue
        o = originals.get(r["profile_id"])
        if o is None:
            continue
        g = groups[(r["test"], r["from"], r["to"])]
        if o["decision"] not in VALID or r["decision"] not in VALID:
            g["excluded"] += 1
            continue
        g["n"] += 1
        if o["decision"] == "APPROVE" and r["decision"] == "REJECT":
            g["a2r"] += 1
        elif o["decision"] == "REJECT" and r["decision"] == "APPROVE":
            g["r2a"] += 1
        if "approval_probability" in r and "approval_probability" in o:
            g["shifts"].append(r["approval_probability"] - o["approval_probability"])

    rows = []
    for (test, a, b), g in sorted(groups.items()):
        flips = g["a2r"] + g["r2a"]
        lo, hi = wilson_ci(flips, g["n"])
        # If flips were random noise, approve->reject and reject->approve would be
        # about equally common. A small p-value means the flips lean one way.
        p_dir = binomtest(g["a2r"], flips, 0.5).pvalue if flips > 0 else None
        rows.append({
            "test": test, "from": a, "to": b, "n": g["n"], "excluded_unclear": g["excluded"],
            "flips": flips, "flip_rate": flips / g["n"] if g["n"] else None,
            "ci_low": lo, "ci_high": hi,
            "approve_to_reject": g["a2r"], "reject_to_approve": g["r2a"],
            "direction_p_value": p_dir,
            "avg_prob_shift": sum(g["shifts"]) / len(g["shifts"]) if g["shifts"] else None,
        })
    return rows


def parity_metrics(records, profiles_by_id):
    orig = [r for r in records if r["variant"] == "original"
            and r["decision"] in VALID and r["profile_id"] in profiles_by_id]
    rows = []
    for name, a, b in PAIRS:
        if len(a) != 1:
            continue            # skip intersectional tests
        (key, va), = a.items()
        (_, vb), = b.items()
        stats = {}
        for label, value in ((va, va), (vb, vb)):
            grp = [r for r in orig if profiles_by_id[r["profile_id"]][key] == value]
            k = sum(r["decision"] == "APPROVE" for r in grp)
            lo, hi = wilson_ci(k, len(grp))
            stats[label] = {"n": len(grp), "rate": k / len(grp) if grp else None,
                            "ci_low": lo, "ci_high": hi}
        rows.append({"test": name, "group_a": va, "group_b": vb, "a": stats[va], "b": stats[vb],
                     "gap": (stats[va]["rate"] - stats[vb]["rate"])
                     if stats[va]["rate"] is not None and stats[vb]["rate"] is not None else None})
    return rows


def fmt_p(p):
    return "n/a" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def report(model, records, ids, profiles_by_id):
    print("\n" + "=" * 78)
    print(f"MODEL: {model} | profiles: {len(ids)}")
    print("=" * 78)

    counts = Counter(r["decision"] for r in records
                     if r["variant"] == "original" and r["profile_id"] in ids)
    total = sum(counts.values())
    unclear = counts.get("UNCLEAR", 0)
    print(f"\nOriginal-profile decisions: {dict(counts)}")
    if total:
        print(f"Approval rate (originals only): {counts.get('APPROVE', 0) / total:.1%} | "
              f"UNCLEAR rate: {unclear / total:.1%}")

    flips = flip_metrics(records, ids)
    print("\n--- FLIP RATES (decision changed when ONLY the listed field(s) changed) ---")
    print("A->R = approved then rejected after the flip; R->A = the reverse.")
    print("dir p = chance the A->R / R->A split is random (small = flips lean one way)\n")
    for r in flips:
        shift = f" | prob shift {r['avg_prob_shift']:+.3f}" if r["avg_prob_shift"] is not None else ""
        print(f"{r['test']:<16}{r['from'] + ' -> ' + r['to']:<46}n={r['n']:<5}"
              f"flipped {r['flip_rate']:.1%} [{r['ci_low']:.1%}, {r['ci_high']:.1%}]  "
              f"A->R={r['approve_to_reject']} R->A={r['reject_to_approve']}  "
              f"dir p={fmt_p(r['direction_p_value'])}{shift}")
        if r["excluded_unclear"]:
            print(f"{'':<16}({r['excluded_unclear']} pairs excluded because a decision was UNCLEAR)")

    parity = parity_metrics(records, profiles_by_id)
    print("\n--- DEMOGRAPHIC PARITY (approval rate of ORIGINAL profiles by group) ---")
    for r in parity:
        a, b = r["a"], r["b"]
        if a["rate"] is None or b["rate"] is None:
            continue
        print(f"{r['test']:<16}{r['group_a']}: {a['rate']:.1%} (n={a['n']})   "
              f"{r['group_b']}: {b['rate']:.1%} (n={b['n']})   gap {r['gap']:+.1%}")

    return {"model": model, "profiles": len(ids), "originals": dict(counts),
            "flip_metrics": flips, "parity": parity}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None, help="e.g. custom ollama (default: all found)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N profiles")
    args = ap.parse_args()

    profiles = load_profiles(limit=args.limit)
    ids = {p["profile_id"] for p in profiles}
    profiles_by_id = {p["profile_id"]: p for p in profiles}

    files = sorted(RESULTS_DIR.glob("raw_*.jsonl"))
    if args.models:
        files = [f for f in files if f.stem.replace("raw_", "") in args.models]
    if not files:
        sys.exit("No matching raw_*.jsonl files in results/")

    summary = [report(f.stem.replace("raw_", ""), read_records(f), ids, profiles_by_id) for f in files]

    out = RESULTS_DIR / "metrics_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()