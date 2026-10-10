# ===== src/models_under_test/custom_model_client.py =====
# Purpose: Load the trained biased model and predict on a profile dict.
# Returns the same shape as the LLM clients so the batch runner can treat
# all four "models" identically.

import json
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "custom_biased_model.pkl"
PROFILES_PATH = PROJECT_ROOT / "data" / "synthetic_profiles.json"
THRESHOLD = 0.5   # approve if P(approved) is above this

_bundle = None    # loaded once, reused for every call


def _load():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. "
                                    "Run src/data/train_custom_model.py first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def approval_probabilities(profiles):
    """P(approved) for a list of profile dicts. Extra keys (like profile_id) are ignored."""
    bundle = _load()
    features = bundle["features"]

    missing = [f for f in features if any(f not in p for p in profiles)]
    if missing:
        raise ValueError(f"Profile is missing required fields: {missing}")

    frame = pd.DataFrame([{f: p[f] for f in features} for p in profiles])
    return bundle["pipeline"].predict_proba(frame)[:, 1]


def evaluate_applicant(profile):
    """Same return shape as the LLM clients: decision, confidence, reasoning."""
    try:
        p = float(approval_probabilities([profile])[0])
    except Exception as e:
        return {"decision": "ERROR", "confidence": 0, "reasoning": str(e),
                "approval_probability": None}

    decision = "APPROVE" if p > THRESHOLD else "REJECT"
    return {
        "decision": decision,
        "confidence": round(max(p, 1 - p), 3),
        "reasoning": f"Model estimated approval probability = {p:.3f} (threshold {THRESHOLD})",
        "approval_probability": round(p, 4),
    }


# ===== SELF-TEST =====
if __name__ == "__main__":
    with open(PROFILES_PATH) as f:
        profiles = json.load(f)["profiles"]
    print(f"Loaded {len(profiles)} synthetic profiles\n")

    print("=== FIRST 5 PROFILES ===")
    for p in profiles[:5]:
        r = evaluate_applicant(p)
        print(f"{p['profile_id']}: {p['Gender']}, {p['Race']}, age {p['Age']}, "
              f"credit {p['Credit_Score']}, income {p['Income']} "
              f"-> {r['decision']} (p={r['approval_probability']})")

    base = approval_probabilities(profiles)
    print(f"\nOverall approval rate on synthetic set: "
          f"{(base > THRESHOLD).mean():.1%}")

    print("\n=== FLIP PREVIEW (change ONLY one attribute) ===")
    for attr, adv, dis in [("Gender", "Male", "Female"),
                           ("Race", "White", "Black"),
                           ("Disability_Status", "No", "Yes")]:
        idx = [i for i, p in enumerate(profiles) if p[attr] == adv]
        original = [profiles[i] for i in idx]
        flipped = [{**p, attr: dis} for p in original]

        before = approval_probabilities(original)
        after = approval_probabilities(flipped)
        changed = int(((before > THRESHOLD) != (after > THRESHOLD)).sum())
        print(f"{attr}: {adv} -> {dis} (n={len(idx)})  "
              f"decisions changed: {changed} ({changed / len(idx):.1%})  "
              f"avg approval probability shift: {(after - before).mean():+.3f}")