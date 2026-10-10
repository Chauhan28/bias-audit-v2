# ===== src/data/generate_synthetic_profiles.py =====
# Purpose: Build the shared, label-free test set used by ALL models.
# Protected attributes are sampled INDEPENDENTLY of income / credit score,
# so any group difference in a model's decisions can't be blamed on finances.

import json
import os
import random
from collections import Counter

N_PROFILES = 5000
SEED = 42
OUTPUT_PATH = "data/synthetic_profiles.json"

# Protected / demographic attributes: value -> sampling weight.
# Minority groups are deliberately OVERSAMPLED compared with the Kaggle data
# (e.g. Kaggle is ~60% White) so each group has enough rows for tight
# confidence intervals. This is a testing choice, not a claim about reality.
DEMOGRAPHICS = {
    "Gender": {"Male": 0.45, "Female": 0.45, "Non-binary": 0.10},
    "Race": {"White": 0.30, "Black": 0.20, "Hispanic": 0.20, "Asian": 0.15,
             "Multiracial": 0.075, "Native American": 0.075},
    "Disability_Status": {"No": 0.70, "Yes": 0.30},
    "Criminal_Record": {"No": 0.75, "Yes": 0.25},
    "Zip_Code_Group": {"High-income Suburban": 0.2, "Urban Professional": 0.2,
                       "Rural": 0.2, "Working Class Urban": 0.2,
                       "Historically Redlined": 0.2},
    "Citizenship_Status": {"Citizen": 0.70, "Permanent Resident": 0.15,
                           "Visa Holder": 0.15},
    "Language_Proficiency": {"Fluent": 0.75, "Limited": 0.25},
}

# Employment and education follow the Kaggle proportions (row counts from the
# inspection step). They are also sampled independently of demographics.
EMPLOYMENT = {"Full-time": 4932, "Part-time": 1107, "Self-employed": 711,
              "Gig": 598, "Unemployed": 152}
EDUCATION = {"Bachelor's": 2577, "High School": 2244, "Some College": 1500,
             "Graduate": 1179}

# Numeric ranges taken from the Kaggle dataset (min / max seen in the data)
AGE_RANGE = (18, 82)
INCOME_RANGE = (25000, 175000)
CREDIT_RANGE = (500, 849)
LOAN_RANGE = (50000, 450000)


def weighted_choice(weights):
    return random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]


def make_profile(profile_id):
    profile = {"profile_id": f"P{profile_id:04d}"}

    # Demographics: each drawn on its own, no link to finances
    for attr, weights in DEMOGRAPHICS.items():
        profile[attr] = weighted_choice(weights)

    # Financial block
    profile["Age"] = random.randint(*AGE_RANGE)
    profile["Income"] = random.randint(*INCOME_RANGE)
    profile["Credit_Score"] = random.randint(*CREDIT_RANGE)
    profile["Loan_Amount"] = random.randint(*LOAN_RANGE)
    profile["Employment_Type"] = weighted_choice(EMPLOYMENT)
    profile["Education_Level"] = weighted_choice(EDUCATION)
    return profile


def summarize(profiles):
    print(f"Generated {len(profiles)} profiles (seed={SEED})\n")

    print("=== GROUP SIZES ===")
    for attr in ["Gender", "Race", "Disability_Status", "Criminal_Record"]:
        counts = Counter(p[attr] for p in profiles)
        print(f"{attr}: {dict(counts)}")

    # Independence check: average finances should look the same for every group
    print("\n=== INDEPENDENCE CHECK: average finances by group ===")
    print("(Should be similar across groups. Big differences mean something is wrong.)\n")
    for attr in ["Gender", "Race", "Disability_Status"]:
        print(f"{attr}:")
        groups = {}
        for p in profiles:
            groups.setdefault(p[attr], []).append(p)
        for value, rows in sorted(groups.items()):
            avg_credit = sum(r["Credit_Score"] for r in rows) / len(rows)
            avg_income = sum(r["Income"] for r in rows) / len(rows)
            print(f"   {value:<18} n={len(rows):<4} avg credit={avg_credit:6.1f}   avg income={avg_income:9.0f}")
        print()


def main():
    random.seed(SEED)   # same seed = same test set every time (reproducible)
    profiles = [make_profile(i + 1) for i in range(N_PROFILES)]

    output = {
        "metadata": {
            "n_profiles": N_PROFILES,
            "seed": SEED,
            "schema": "Kaggle Financial Loan Access Dataset (no labels)",
            "note": "Demographics sampled independently of financial attributes; "
                    "minority groups oversampled for statistical power.",
        },
        "profiles": profiles,
    }

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    summarize(profiles)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()