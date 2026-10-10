# ===== src/data/inspect_dataset.py =====
# Purpose: Look at the Kaggle dataset BEFORE training anything.
# Answers: what are the columns/values, and is there real bias in the labels?

import glob
import pandas as pd

TARGET = "Loan_Approved"
POSITIVE_LABEL = "Approved"
PROTECTED = ["Gender", "Race", "Age_Group", "Citizenship_Status",
             "Disability_Status", "Criminal_Record", "Zip_Code_Group",
             "Language_Proficiency", "Employment_Type", "Education_Level"]


def find_csv():
    files = glob.glob("data/raw/Financial_Loan_Access_Dataset.csv")
    if not files:
        raise FileNotFoundError("No CSV found in data/raw/. Put the Kaggle file there.")
    return files[0]

#/Users/digvijaysinghchauhan/Desktop/BIAS/bias-audit-v2/data/raw/Financial_Loan_Access_Dataset.csv


def main():
    path = find_csv()
    df = pd.read_csv(path)
    print(f"Loaded: {path}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"Duplicate IDs: {df['ID'].duplicated().sum()}\n")

    print(f"=== TARGET: {TARGET} ===")
    print(df[TARGET].value_counts(dropna=False), "\n")

    # Binary version of the target so we can average it into an approval rate
    df["_approved"] = (df[TARGET] == POSITIVE_LABEL).astype(int)
    print(f"Overall approval rate: {df['_approved'].mean():.3f}\n")

    print("=== APPROVAL RATE BY GROUP (is the bias in the data?) ===")
    for col in PROTECTED:
        print(f"\n{col}:")
        table = (df.groupby(col)["_approved"].agg(["mean", "count"])
                   .rename(columns={"mean": "approval_rate"})
                   .round(3)
                   .sort_values("approval_rate", ascending=False))
        print(table)


if __name__ == "__main__":
    main()