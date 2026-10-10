# ===== src/evaluation/batch_runner.py =====
# Purpose: Run every profile and every counterfactual through ONE model and
# save each decision to results/raw_<model>.jsonl (one JSON record per line).
#
#  - Resumable:  rerun the same command and it skips what is already saved.
#  - Crash-safe: each record is written to disk the moment it exists.
#  - Fair:       every model gets exactly the same set of variants.

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "counterfactuals"))
sys.path.insert(0, str(SRC / "models_under_test"))

from counterfactual_engine import load_profiles, generate_counterfactuals  # noqa: E402
from prompt_renderer import render_prompt                                   # noqa: E402

RESULTS_DIR = SRC.parent / "results"
CHUNK = 2000   # the custom model scores this many variants per batch


def build_variants(profiles):
    """Each profile's original + all its counterfactuals, as a flat list."""
    variants = []
    for p in profiles:
        variants.append({"profile_id": p["profile_id"], "variant": "original",
                         "test": None, "from": None, "to": None, "profile": p})
        for cf in generate_counterfactuals(p):
            variants.append({"profile_id": p["profile_id"],
                             "variant": f"{cf['test']}|{cf['from']}->{cf['to']}",
                             "test": cf["test"], "from": cf["from"], "to": cf["to"],
                             "profile": cf["profile"]})
    return variants


def read_records(path):
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass   # half-written last line after a crash
    return records


def remaining(variants, path):
    done = {(r["profile_id"], r["variant"]) for r in read_records(path)}
    return [v for v in variants if (v["profile_id"], v["variant"]) not in done]


def base_record(v, model):
    return {"model": model, "profile_id": v["profile_id"], "variant": v["variant"],
            "test": v["test"], "from": v["from"], "to": v["to"]}


def run_custom(variants, path):
    from custom_model_client import approval_probabilities, THRESHOLD
    todo = remaining(variants, path)
    print(f"To do: {len(todo)} variants")
    with open(path, "a") as f:
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            probs = approval_probabilities([v["profile"] for v in chunk])
            for v, p in zip(chunk, probs):
                rec = base_record(v, "custom")
                rec["decision"] = "APPROVE" if p > THRESHOLD else "REJECT"
                rec["approval_probability"] = round(float(p), 4)
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"  scored {min(i + CHUNK, len(todo))}/{len(todo)}")


def run_llm(model_name, evaluate_fn, variants, path, max_consecutive_errors=5):
    todo = remaining(variants, path)
    print(f"To do: {len(todo)} evaluations")
    start, errors_in_row = time.time(), 0
    with open(path, "a") as f:
        for i, v in enumerate(todo, 1):
            r = evaluate_fn(render_prompt(v["profile"]))
            if r["decision"] == "ERROR":
                # Errors are NOT saved, so they get retried on the next run.
                errors_in_row += 1
                print(f"  ERROR: {r['reasoning']}")
                if errors_in_row >= max_consecutive_errors:
                    print("Too many errors in a row. Stopping. Progress is saved.")
                    return
                continue
            errors_in_row = 0
            rec = base_record(v, model_name)
            rec.update({"decision": r["decision"], "reasoning": r["reasoning"],
                        "seconds": r["seconds"]})
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                elapsed = time.time() - start
                left = elapsed / i * (len(todo) - i)
                print(f"  {i}/{len(todo)} | {elapsed / 60:.1f} min elapsed | ~{left / 60:.0f} min left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["custom", "ollama"])
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N profiles (default: all)")
    args = ap.parse_args()

    profiles = load_profiles(limit=args.limit)
    variants = build_variants(profiles)
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"raw_{args.model}.jsonl"
    print(f"Model: {args.model} | profiles: {len(profiles)} | variants: {len(variants)}")
    print(f"Output: {path}\n")

    if args.model == "custom":
        run_custom(variants, path)
    else:
        from ollama_client import evaluate_prompt
        run_llm("ollama", evaluate_prompt, variants, path)

    counts = Counter(r["decision"] for r in read_records(path))
    print(f"\n=== FILE SUMMARY ({path.name}) ===")
    print(f"Records saved: {sum(counts.values())}")
    print(f"Decisions: {dict(counts)}")
    print(f"Still missing for this run: {len(remaining(variants, path))}")


if __name__ == "__main__":
    main()