# ===== src/data/train_custom_model.py =====
# Purpose: Train the deliberately-biased custom model on the Kaggle dataset,
# then check whether the bias actually carried into the model.

import glob
import os
import math

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "Loan_Approved"
POSITIVE_LABEL = "Approved"

# Protected attributes are INCLUDED as features on purpose: the model is
# meant to be able to learn the bias so our counterfactual test can detect it.
CATEGORICAL = ["Gender", "Race", "Employment_Type", "Education_Level",
               "Citizenship_Status", "Language_Proficiency",
               "Disability_Status", "Criminal_Record", "Zip_Code_Group"]
NUMERIC = ["Age", "Income", "Credit_Score", "Loan_Amount"]
FEATURES = CATEGORICAL + NUMERIC   # ID and Age_Group are dropped (ID is noise, Age_Group duplicates Age)

MODEL_PATH = "models/custom_biased_model.pkl"

# (attribute, group that data suggests is advantaged, group that appears disadvantaged)
# The "advantaged" label is an assumption taken from the raw approval rates.
FLIP_TESTS = [
    ("Gender", "Male", "Female"),
    ("Race", "White", "Black"),
    ("Disability_Status", "No", "Yes"),
    ("Criminal_Record", "No", "Yes"),
    ("Zip_Code_Group", "High-income Suburban", "Historically Redlined"),
]


def wilson_ci(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def load_data():
    files = glob.glob("data/raw/Financial_Loan_Access_Dataset.csv")
    if not files:
        raise FileNotFoundError("No CSV in data/raw/")
    df = pd.read_csv(files[0])
    df["approved"] = (df[TARGET] == POSITIVE_LABEL).astype(int)
    return df

#/Users/digvijaysinghchauhan/Desktop/BIAS/bias-audit-v2/data/raw/Financial_Loan_Access_Dataset.csv


def confounding_check(df):
    """Do the approval gaps persist WITHIN the same credit-score band?"""
    print("=== CONFOUNDING CHECK: approval rate within credit-score bands ===")
    print("(If a gap survives inside each band, it isn't just credit score.)\n")
    df = df.copy()
    df["credit_band"] = pd.cut(df["Credit_Score"], bins=[499, 600, 700, 850],
                               labels=["500-600", "601-700", "701-850"])
    for attr, a, b in [("Gender", "Male", "Female"),
                       ("Race", "White", "Black"),
                       ("Disability_Status", "No", "Yes")]:
        sub = df[df[attr].isin([a, b])]
        rate = sub.pivot_table(index="credit_band", columns=attr,
                               values="approved", aggfunc="mean", observed=True).round(3)
        cnt = sub.pivot_table(index="credit_band", columns=attr,
                              values="approved", aggfunc="count", observed=True)
        rate["gap"] = (rate[a] - rate[b]).round(3)
        print(f"{attr}: {a} vs {b}")
        print(rate.to_string())
        print(f"  row counts per band:\n{cnt.to_string()}\n")


def build_pipeline():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                 min_samples_leaf=10, random_state=42, n_jobs=-1 , class_weight= "balanced")
    return Pipeline([("prep", pre), ("clf", clf)])


def flip_test(pipe, X_test):
    """Change ONLY one protected attribute, keep everything else identical,
    and count how often the model's decision changes."""
    print("\n=== FLIP TEST ON THE TRAINED MODEL (test set only) ===")
    print("Take applicants from the 'advantaged' group, change ONLY that attribute\n"
          "to the 'disadvantaged' value, and see if the decision changes.\n")
    base_pred = pd.Series(pipe.predict(X_test), index=X_test.index)

    for attr, adv, dis in FLIP_TESTS:
        rows = X_test[X_test[attr] == adv]
        n = len(rows)
        if n == 0:
            continue
        flipped = rows.copy()
        flipped[attr] = dis
        new_pred = pd.Series(pipe.predict(flipped), index=rows.index)
        old_pred = base_pred.loc[rows.index]

        harmful = int(((old_pred == 1) & (new_pred == 0)).sum())   # approve -> deny
        helpful = int(((old_pred == 0) & (new_pred == 1)).sum())   # deny -> approve
        flips = harmful + helpful
        lo, hi = wilson_ci(flips, n)
        print(f"{attr}: {adv} -> {dis}  (n={n})")
        print(f"   decision changed: {flips} ({flips / n:.1%})  "
              f"95% CI [{lo:.1%}, {hi:.1%}]")
        print(f"   approve->deny: {harmful} | deny->approve: {helpful} | "
              f"net against '{dis}': {harmful - helpful}\n")


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows\n")

    confounding_check(df)

    X = df[FEATURES]
    y = df["approved"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    acc = accuracy_score(y_test, pipe.predict(X_test))
    baseline = max(y_test.mean(), 1 - y_test.mean())
    print("=== MODEL PERFORMANCE ===")
    print(f"Test accuracy: {acc:.3f}  (always-predict-majority baseline: {baseline:.3f})\n")

    # Approval rate the MODEL gives each group vs what the data had
    print("=== MODEL APPROVAL RATE BY GROUP (test set) ===")
    test = X_test.copy()
    test["pred"] = pipe.predict(X_test)
    test["actual"] = y_test
    for attr in ["Gender", "Race", "Disability_Status"]:
        t = test.groupby(attr)[["actual", "pred"]].mean().round(3)
        t.columns = ["data_approval_rate", "model_approval_rate"]
        print(f"\n{attr}:\n{t.to_string()}")

    flip_test(pipe, X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": FEATURES,
                 "categorical": CATEGORICAL, "numeric": NUMERIC}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()