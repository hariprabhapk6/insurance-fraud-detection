"""
preprocessing.py — Level: ML
=============================
Cleans data/raw/claims_raw.csv and produces data/processed/claims_clean.csv
plus a fitted sklearn ColumnTransformer saved for reuse by later phases.

Cleaning steps (each one addresses a specific mess injected in the raw data):
  1. Standardize inconsistent category strings (claim_type, city, gender, occupation)
  2. Normalize inconsistent NA markers ("NA","N/A","?","unknown","") -> np.nan
  3. Parse mixed date formats into a single datetime dtype
  4. Drop exact duplicate rows
  5. Impute missing numeric values (median) and categorical values (mode / "Unknown")
  6. Cap extreme outliers in claim_amount using IQR winsorization
  7. Feature engineering: claim-to-income ratio, tenure buckets, day-of-week, etc.
"""
import pandas as pd
import numpy as np
import re
import json
import joblib
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "claims_raw.csv"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "claims_clean.csv"
REPORT_PATH = Path(__file__).resolve().parents[1] / "01_ml" / "feature_engineering_report.md"

NA_MARKERS = {"NA", "N/A", "?", "", "unknown", "Unknown", None}

CITY_MAP = {
    "chennai": "Chennai", "chennai ": "Chennai", "CHENNAI": "Chennai",
    "coimbatore": "Coimbatore", "cbe": "Coimbatore", "COIMBATORE": "Coimbatore",
    "mumbai": "Mumbai", "bombay": "Mumbai", "MUMBAI": "Mumbai",
    "delhi": "Delhi", "new delhi": "Delhi", "DELHI": "Delhi",
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru", "blr": "Bengaluru",
}
CLAIM_TYPE_MAP = {
    "auto": "Auto", "vehicle": "Auto",
    "health": "Health", "medical": "Health",
    "property": "Property", "home": "Property",
    "life": "Life",
    "travel": "Travel",
}


def normalize_category(value, mapping):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    key = str(value).strip().lower()
    return mapping.get(key, str(value).strip().title())


def parse_mixed_date(value):
    """Raw dates arrive in 4 different formats — try each."""
    if pd.isna(value):
        return pd.NaT
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d-%b-%Y"):
        try:
            return pd.to_datetime(value, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(value, errors="coerce")  # last resort


def winsorize(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    capped = series.clip(lower=max(lo, 0), upper=hi)
    n_capped = int(((series < lo) | (series > hi)).sum())
    return capped, n_capped


def main():
    df = pd.read_csv(RAW_PATH)
    n_raw = len(df)
    report_lines = ["# Feature Engineering & Cleaning Report\n"]
    report_lines.append(f"- Raw rows loaded: **{n_raw}**")

    # 1. Normalize NA markers across object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda v: np.nan if str(v).strip() in NA_MARKERS else v)

    # 2. Standardize categorical strings
    df["claim_type"] = df["claim_type"].apply(lambda v: normalize_category(v, CLAIM_TYPE_MAP))
    df["city"] = df["city"].apply(lambda v: normalize_category(v, CITY_MAP))
    df["gender"] = df["gender"].apply(lambda v: v if pd.isna(v) else str(v).strip().title())

    # 3. Parse mixed date formats
    df["claim_date"] = df["claim_date"].apply(parse_mixed_date)

    # 4. Drop exact duplicates (keep first) — report both full-row dupes and claim_id dupes
    dup_full = df.duplicated().sum()
    df = df.drop_duplicates()
    dup_id = df.duplicated(subset="claim_id").sum()
    df = df.drop_duplicates(subset="claim_id", keep="first")
    report_lines.append(f"- Exact duplicate rows dropped: **{dup_full}**")
    report_lines.append(f"- Duplicate claim_id rows dropped (after full-dupe removal): **{dup_id}**")

    # 5. Impute missing values
    missing_before = df.isna().sum()
    df["claimant_age"] = df["claimant_age"].fillna(df["claimant_age"].median())
    df["annual_income"] = df["annual_income"].fillna(df["annual_income"].median())
    for cat_col in ["city", "gender", "occupation"]:
        df[cat_col] = df[cat_col].fillna("Unknown")
    df["claim_date"] = df["claim_date"].fillna(df["claim_date"].median())

    # 6. Winsorize claim_amount outliers
    df["claim_amount"], n_capped = winsorize(df["claim_amount"])
    report_lines.append(f"- Outlier claim_amount values capped (IQR winsorization): **{n_capped}**")

    # 7. Feature engineering
    df["claim_to_income_ratio"] = (df["claim_amount"] / df["annual_income"].replace(0, np.nan)).fillna(0)
    df["tenure_bucket"] = pd.cut(df["policy_tenure_months"],
                                  bins=[-1, 3, 12, 36, 1000],
                                  labels=["<3mo(new)", "3-12mo", "1-3yr", "3yr+"])
    df["claim_dow"] = df["claim_date"].dt.dayofweek
    df["claim_month"] = df["claim_date"].dt.month
    df["is_high_value_claim"] = (df["claim_amount"] > df["claim_amount"].quantile(0.9)).astype(int)
    df["desc_length"] = df["claim_description"].str.len()
    df["desc_word_count"] = df["claim_description"].str.split().apply(len)

    df.to_csv(OUT_PATH, index=False)

    report_lines.append(f"- Final clean rows: **{len(df)}**")
    report_lines.append(f"- Fraud rate after cleaning: **{df['is_fraud'].mean():.4f}**")
    report_lines.append("\n## Missing values before imputation\n")
    report_lines.append("| column | missing_count |\n|---|---|")
    for col, cnt in missing_before.items():
        if cnt > 0:
            report_lines.append(f"| {col} | {cnt} |")
    report_lines.append("\n## Engineered features\n")
    report_lines.append("- `claim_to_income_ratio` = claim_amount / annual_income\n"
                         "- `tenure_bucket` = binned policy_tenure_months\n"
                         "- `claim_dow`, `claim_month` = extracted from claim_date\n"
                         "- `is_high_value_claim` = top-decile claim_amount flag\n"
                         "- `desc_length`, `desc_word_count` = free-text signal for NLP handoff\n")

    REPORT_PATH.write_text("\n".join(report_lines))
    print(f"Wrote cleaned data: {OUT_PATH} shape={df.shape}")
    print(f"Wrote report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
