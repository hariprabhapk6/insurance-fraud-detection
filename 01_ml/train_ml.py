"""
train_ml.py — Level: ML
=========================
Baseline ML models: Logistic Regression + Random Forest.
Outputs metrics, confusion matrix, ROC-AUC, and feature importances.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve, classification_report)
import joblib

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed" / "claims_clean.csv"
OUTDIR = BASE / "01_ml"

NUMERIC = ["claimant_age", "annual_income", "policy_tenure_months", "claim_amount",
           "prior_claims_count", "claim_to_income_ratio", "claim_dow", "claim_month",
           "is_high_value_claim", "desc_length", "desc_word_count"]
CATEGORICAL = ["claim_type", "city", "gender", "occupation", "tenure_bucket"]
TARGET = "is_fraud"


def build_pipeline(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def evaluate(name, pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    cm = confusion_matrix(y_test, pred).tolist()
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    print("Confusion matrix [ [TN FP] [FN TP] ]:", cm)
    return metrics, cm, proba


def main():
    df = pd.read_csv(DATA)
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    # Logistic Regression baseline (class_weight balanced for imbalance)
    lr_pipe = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    lr_pipe.fit(X_train, y_train)
    lr_metrics, lr_cm, lr_proba = evaluate("Logistic Regression", lr_pipe, X_test, y_test)
    results["logistic_regression"] = {"metrics": lr_metrics, "confusion_matrix": lr_cm}

    # Random Forest
    rf_pipe = build_pipeline(RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1))
    rf_pipe.fit(X_train, y_train)
    rf_metrics, rf_cm, rf_proba = evaluate("Random Forest", rf_pipe, X_test, y_test)
    results["random_forest"] = {"metrics": rf_metrics, "confusion_matrix": rf_cm}

    # Feature importances from RF
    ohe = rf_pipe.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL))
    feat_names = NUMERIC + cat_names
    importances = rf_pipe.named_steps["clf"].feature_importances_
    fi = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:15]

    # Pick best model by ROC-AUC as the deployed ML baseline
    best_name = "random_forest" if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"] else "logistic_regression"
    best_pipe = rf_pipe if best_name == "random_forest" else lr_pipe
    joblib.dump(best_pipe, OUTDIR / "best_ml_model.joblib")

    with open(OUTDIR / "ml_results.json", "w") as f:
        json.dump({
            "results": results,
            "best_model": best_name,
            "top_features": [(n, float(v)) for n, v in fi],
            "test_size": len(X_test),
            "train_size": len(X_train),
        }, f, indent=2)

    print(f"\nBest model by ROC-AUC: {best_name}")
    print("Top 10 features:")
    for n, v in fi[:10]:
        print(f"  {n:35s} {v:.4f}")

    print(f"\nSaved: {OUTDIR/'best_ml_model.joblib'}, {OUTDIR/'ml_results.json'}")


if __name__ == "__main__":
    main()
