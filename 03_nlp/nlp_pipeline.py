"""
nlp_pipeline.py — Level: NLP
===============================
No internet access -> spaCy/transformer models can't be downloaded, so this
pipeline uses:
  - Regex-based tokenization/cleaning (text preprocessing)
  - TF-IDF + Logistic Regression / Linear SVM for text classification
    (does the claim DESCRIPTION alone signal fraud?)
  - A lightweight rule-based Named Entity Recognizer (money amounts, dates,
    locations, organizations/vendor mentions) using regex + gazetteers —
    a legitimate, explainable NER approach used in production systems when
    a heavier statistical NER model isn't available/needed.
  - TF-IDF-weighted keyword extraction per claim (feeds the SLM summarizer).
"""
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
import joblib

BASE = Path(__file__).resolve().parents[1]
CLAIMS = BASE / "data" / "processed" / "claims_clean.csv"
OUT = BASE / "03_nlp"

CITIES = ["Chennai", "Coimbatore", "Mumbai", "Delhi", "Bengaluru", "Hyderabad",
          "Pune", "Kolkata", "Ahmedabad", "Jaipur"]
ORG_KEYWORDS = ["hospital", "police", "fire department", "bank", "vendor",
                "repair shop", "insurance", "agent"]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_entities(text):
    """Rule-based NER: money amounts, locations, organizations, dates."""
    entities = {"MONEY": [], "LOCATION": [], "ORG": [], "DATE": []}
    for m in re.finditer(r"\b\d+(?:,\d{3})*(?:\.\d+)?\s?(?:rs|rupees|inr)?\b", text, re.I):
        if re.search(r"\d", m.group()):
            entities["MONEY"].append(m.group().strip())
    for city in CITIES:
        if city.lower() in text.lower():
            entities["LOCATION"].append(city)
    for org in ORG_KEYWORDS:
        if org in text.lower():
            entities["ORG"].append(org)
    for m in re.finditer(r"\b\d{1,2}\s?(?:hour|hours|day|days|week|weeks)\b", text, re.I):
        entities["DATE"].append(m.group())
    return entities


def extract_keywords(vectorizer, tfidf_row, top_n=5):
    row = tfidf_row.toarray().flatten()
    top_idx = row.argsort()[::-1][:top_n]
    terms = np.array(vectorizer.get_feature_names_out())
    return [terms[i] for i in top_idx if row[i] > 0]


def main():
    df = pd.read_csv(CLAIMS)
    df["clean_desc"] = df["claim_description"].apply(clean_text)

    # ---- Text classification: does description alone predict fraud? ----
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_desc"], df["is_fraud"], test_size=0.2, random_state=42,
        stratify=df["is_fraud"])

    vec = TfidfVectorizer(max_features=800, ngram_range=(1, 2), min_df=2)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, y_train)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = clf.predict(Xte)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    print("=== NLP text classifier (TF-IDF + Logistic Regression) ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    print("Confusion matrix:", confusion_matrix(y_test, pred).tolist())

    joblib.dump({"vectorizer": vec, "clf": clf}, OUT / "text_classifier.joblib")

    # ---- NER + keyword extraction over full dataset (sample for report) ----
    sample = df.sample(8, random_state=1)
    Xsample_tfidf = vec.transform(sample["clean_desc"])
    examples = []
    for (i, row), tfidf_row in zip(sample.iterrows(), Xsample_tfidf):
        ents = extract_entities(row["claim_description"])
        kws = extract_keywords(vec, tfidf_row)
        examples.append({
            "claim_id": row["claim_id"],
            "is_fraud": int(row["is_fraud"]),
            "description": row["claim_description"],
            "entities": ents,
            "keywords": kws,
        })

    with open(OUT / "nlp_results.json", "w") as f:
        json.dump({"classification_metrics": metrics, "ner_keyword_examples": examples}, f, indent=2)

    print(f"\nSaved: {OUT/'text_classifier.joblib'}, {OUT/'nlp_results.json'}")
    print("\nSample NER output:")
    for ex in examples[:3]:
        print(f"- [{'FRAUD' if ex['is_fraud'] else 'legit'}] {ex['description'][:70]}...")
        print(f"    entities: {ex['entities']}")
        print(f"    keywords: {ex['keywords']}")


if __name__ == "__main__":
    main()
