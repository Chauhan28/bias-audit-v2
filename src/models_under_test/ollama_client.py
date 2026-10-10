# ===== src/models_under_test/ollama_client.py =====
# Purpose: Send a prompt to a local Ollama model and return a clean decision.
# One call per evaluation. Deterministic settings so that a changed decision
# means the INPUT changed, not that the model rolled different dice.

import json
import sys
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama2"      # change here if you switch models
TIMEOUT = 120
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 150}

VALID = ("APPROVE", "REJECT")


def evaluate_prompt(prompt_text):
    """Returns: decision (APPROVE/REJECT/UNCLEAR/ERROR), confidence, reasoning, raw, seconds."""
    start = time.time()
    payload = {"model": OLLAMA_MODEL, "prompt": prompt_text, "stream": False,
               "format": "json", "options": OPTIONS}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        raw = resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        return _result("ERROR", "Cannot connect to Ollama. Is `ollama serve` running?", "", start)
    except Exception as e:
        return _result("ERROR", str(e), "", start)

    try:
        parsed = json.loads(raw)
        decision = str(parsed.get("decision", "")).strip().upper()
        reason = str(parsed.get("reason", ""))
    except (json.JSONDecodeError, AttributeError):
        return _result("UNCLEAR", "Model did not return valid JSON", raw, start)

    if decision not in VALID:
        return _result("UNCLEAR", f"Unexpected decision value: {decision!r}", raw, start)
    return _result(decision, reason, raw, start)


def _result(decision, reasoning, raw, start):
    return {"decision": decision, "confidence": None, "reasoning": reasoning,
            "raw": raw, "seconds": round(time.time() - start, 2)}


# ===== SELF-TEST =====
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "counterfactuals"))
    from prompt_renderer import render_prompt
    from counterfactual_engine import load_profiles

    profiles = load_profiles(limit=5)
    print(f"Model: {OLLAMA_MODEL}\n")

    print("=== 5 PROFILES ===")
    times = []
    for p in profiles:
        r = evaluate_prompt(render_prompt(p))
        times.append(r["seconds"])
        print(f"{p['profile_id']}: {r['decision']:<8} {r['seconds']:>5}s  {r['reasoning'][:90]}")

    print("\n=== REPEATABILITY: same prompt, 3 times ===")
    prompt = render_prompt(profiles[0])
    repeats = [evaluate_prompt(prompt)["decision"] for _ in range(3)]
    print(repeats, "-> consistent" if len(set(repeats)) == 1 else "-> NOT consistent")

    if all(t > 0 for t in times):
        avg = sum(times) / len(times)
        print(f"\n=== TIME ESTIMATE (avg {avg:.1f}s per call) ===")
        for n_eval in (672, 1369):
            print(f"{n_eval} evaluations: about {n_eval * avg / 3600:.1f} hours")