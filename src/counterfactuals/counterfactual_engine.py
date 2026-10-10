# ===== src/counterfactuals/counterfactual_engine.py =====
# Purpose: Given a profile, generate counterfactual versions where ONLY the
# targeted attribute(s) change. Used identically for all four models.
#
# Improvements over the old prompt_engine.py:
#  - swaps in BOTH directions (A->B and B->A), not "first different value"
#  - verifies exactly the intended fields changed (no silent failures)
#  - supports intersectional swaps (several attributes at once)

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = PROJECT_ROOT / "data" / "synthetic_profiles.json"

# Each test: (name, group A, group B). A profile in group A is flipped to B,
# and a profile in group B is flipped to A.
# To add a test, add one line. Age is left out for now (numeric; needs a
# separate design decision).
PAIRS = [
    ("Gender",             {"Gender": "Male"},               {"Gender": "Female"}),
    ("Race",               {"Race": "White"},                {"Race": "Black"}),
    ("Disability",         {"Disability_Status": "No"},      {"Disability_Status": "Yes"}),
    ("Criminal_Record",    {"Criminal_Record": "No"},        {"Criminal_Record": "Yes"}),
    ("Zip_Code",           {"Zip_Code_Group": "High-income Suburban"},
                           {"Zip_Code_Group": "Historically Redlined"}),
    ("Citizenship",        {"Citizenship_Status": "Citizen"}, {"Citizenship_Status": "Visa Holder"}),
    ("Language",           {"Language_Proficiency": "Fluent"}, {"Language_Proficiency": "Limited"}),
    # Intersectional: several attributes change together
    ("Gender+Race",        {"Gender": "Male", "Race": "White"},
                           {"Gender": "Female", "Race": "Black"}),
]


def _matches(profile, group):
    return all(profile.get(k) == v for k, v in group.items())


def _label(group):
    return "+".join(group.values())


def _verify(original, flipped, target):
    """Raise if anything other than the targeted fields changed."""
    changed = {k for k in original if original[k] != flipped.get(k)}
    if changed != set(target.keys()):
        raise ValueError(f"Counterfactual changed {changed}, expected {set(target.keys())}")


def generate_counterfactuals(profile):
    """Return a list of counterfactual records for ONE profile."""
    records = []
    for name, a, b in PAIRS:
        for src, dst in ((a, b), (b, a)):
            if _matches(profile, src):
                flipped = {**profile, **dst}
                _verify(profile, flipped, dst)
                records.append({
                    "profile_id": profile["profile_id"],
                    "test": name,
                    "from": _label(src),
                    "to": _label(dst),
                    "profile": flipped,
                })
    return records


def load_profiles(limit=None):
    with open(PROFILES_PATH) as f:
        profiles = json.load(f)["profiles"]
    return profiles[:limit] if limit else profiles


# ===== SELF-TEST =====
if __name__ == "__main__":
    profiles = load_profiles()
    print(f"Loaded {len(profiles)} profiles\n")

    all_cfs = [cf for p in profiles for cf in generate_counterfactuals(p)]
    print(f"Total counterfactuals generated: {len(all_cfs)} "
          f"({len(all_cfs) / len(profiles):.1f} per profile)\n")

    print("=== COUNTERFACTUALS PER TEST AND DIRECTION ===")
    counts = Counter((cf["test"], f"{cf['from']} -> {cf['to']}") for cf in all_cfs)
    for (test, direction), n in sorted(counts.items()):
        print(f"{test:<16} {direction:<40} {n}")

    print("\n=== EXAMPLE: what actually changes ===")
    p = next(p for p in profiles if generate_counterfactuals(p))
    cf = generate_counterfactuals(p)[0]
    print(f"Profile {p['profile_id']}, test '{cf['test']}': {cf['from']} -> {cf['to']}")
    for k in p:
        if p[k] != cf["profile"][k]:
            print(f"   {k}: {p[k]} -> {cf['profile'][k]}")
    print("   (every other field is identical)")

    print("\n=== COST PLANNING FOR THE LLMs ===")
    for n in (100, 200):
        subset = profiles[:n]
        cfs = sum(len(generate_counterfactuals(p)) for p in subset)
        print(f"First {n} profiles: {n} originals + {cfs} counterfactuals "
              f"= {n + cfs} evaluations per LLM")