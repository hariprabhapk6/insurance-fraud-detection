"""
app.py — thin Flask API that exposes the project's EXISTING trained models.

Nothing here retrains, edits or replaces any model or dataset. It only:
  * loads  01_ml/best_ml_model.joblib      (Random Forest pipeline)
  * loads  03_nlp/text_classifier.joblib   (TF-IDF + Logistic Regression)
  * reads  data/processed/claims_clean.csv (read-only, for the same
           "top-decile claim amount" threshold that preprocessing.py used,
           and for the value ranges the model was trained on)
  * repeats the feature-derivation steps from 01_ml/preprocessing.py so a raw
    form submission becomes the 16 columns the model expects.

Run:  python3 web_app/backend/app.py     (from the fraud_project folder)
"""


from flask_cors import CORS
import re
import warnings
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE = Path(__file__).resolve().parents[2]           # .../fraud_project
ML_MODEL_PATH = BASE / "01_ml" / "best_ml_model.joblib"
NLP_MODEL_PATH = BASE / "03_nlp" / "text_classifier.joblib"
CLEAN_CSV = BASE / "data" / "processed" / "claims_clean.csv"
DIST_DIR = Path(__file__).resolve().parents[1] / "frontend" / "dist"

# Same column lists as 01_ml/train_ml.py (order matters for the pipeline).
NUMERIC = ["claimant_age", "annual_income", "policy_tenure_months", "claim_amount",
           "prior_claims_count", "claim_to_income_ratio", "claim_dow", "claim_month",
           "is_high_value_claim", "desc_length", "desc_word_count"]
CATEGORICAL = ["claim_type", "city", "gender", "occupation", "tenure_bucket"]

DECISION_THRESHOLD = 0.5          # RandomForest.predict() uses 0.5 on predict_proba
# Display bands for the UI only (NOT part of the original project). The saved
# model's probabilities are compressed (median 0.34 across the dataset because
# of class_weight="balanced"), so "medium" starts at 0.40, just under the 0.5 decision line.
RISK_BANDS = {"medium_from": 0.40, "high_from": DECISION_THRESHOLD}

# Model/dataset artifacts were pickled with scikit-learn 1.9.0 / 1.8.0; harmless
# version warnings are silenced here but the versions are reported at startup.
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ml_pipe = joblib.load(ML_MODEL_PATH)
_nlp = joblib.load(NLP_MODEL_PATH)
nlp_vec, nlp_clf = _nlp["vectorizer"], _nlp["clf"]

_clean = pd.read_csv(CLEAN_CSV)
HIGH_VALUE_THRESHOLD = float(_clean["claim_amount"].quantile(0.9))   # as in preprocessing.py
TRAIN_RANGES = {c: (float(_clean[c].min()), float(_clean[c].max()))
                for c in ["claimant_age", "annual_income", "policy_tenure_months",
                          "claim_amount", "prior_claims_count"]}

_ohe = ml_pipe.named_steps["pre"].named_transformers_["cat"]
CATEGORY_OPTIONS = {name: [str(v) for v in cats]
                    for name, cats in zip(CATEGORICAL, _ohe.categories_)
                    if name != "tenure_bucket"}      # tenure_bucket is derived, not entered

app = Flask(__name__, static_folder=None)

CORS(app, resources={
    r"/api/*": {
        "origins": ["https://fraudshield-ai-delta.vercel.app"]
    }
})

app.config["MAX_CONTENT_LENGTH"] = 32 * 1024


# ---------------------------------------------------------------- validation
def _num(payload, key, label, lo, hi, integer=False, errors=None):
    raw = payload.get(key)
    if raw is None or str(raw).strip() == "":
        errors[key] = f"{label} is required."
        return None
    try:
        val = float(raw)
        if not np.isfinite(val):
            raise ValueError
    except (TypeError, ValueError):
        errors[key] = f"{label} must be a number."
        return None
    if integer and val != int(val):
        errors[key] = f"{label} must be a whole number."
        return None
    if val < lo or val > hi:
        errors[key] = f"{label} must be between {lo:,.0f} and {hi:,.0f}."
        return None
    return int(val) if integer else val


def validate(payload):
    errors, v = {}, {}
    v["claim_type"] = payload.get("claim_type")
    for key, label in [("claim_type", "Claim type"), ("city", "City"),
                       ("gender", "Gender"), ("occupation", "Occupation")]:
        val = payload.get(key)
        if val not in CATEGORY_OPTIONS[key]:
            errors[key] = f"Choose a valid {label.lower()}."
        v[key] = val
    v["claimant_age"] = _num(payload, "claimant_age", "Age", 18, 100, True, errors)
    v["annual_income"] = _num(payload, "annual_income", "Annual income", 1, 100_000_000, False, errors)
    v["policy_tenure_months"] = _num(payload, "policy_tenure_months", "Policy tenure", 0, 600, True, errors)
    v["claim_amount"] = _num(payload, "claim_amount", "Claim amount", 1, 100_000_000, False, errors)
    v["prior_claims_count"] = _num(payload, "prior_claims_count", "Prior claims", 0, 50, True, errors)

    try:
        v["claim_date"] = datetime.strptime(str(payload.get("claim_date", "")), "%Y-%m-%d").date()
        if v["claim_date"] > date.today():
            errors["claim_date"] = "Claim date cannot be in the future."
    except ValueError:
        errors["claim_date"] = "Enter the claim date."

    desc = str(payload.get("claim_description") or "").strip()
    if len(desc) < 10:
        errors["claim_description"] = "Describe the claim in at least 10 characters."
    elif len(desc) > 500:
        errors["claim_description"] = "Keep the description under 500 characters."
    v["claim_description"] = desc
    return v, errors


# ------------------------------------------- feature derivation (preprocessing.py)
def tenure_bucket(months):
    # pd.cut(bins=[-1,3,12,36,1000], labels=["<3mo(new)","3-12mo","1-3yr","3yr+"])
    if months <= 3:
        return "<3mo(new)"
    if months <= 12:
        return "3-12mo"
    if months <= 36:
        return "1-3yr"
    return "3yr+"


def build_model_row(v):
    d = v["claim_date"]
    desc = v["claim_description"]
    income = v["annual_income"]
    return {
        "claimant_age": v["claimant_age"], "annual_income": income,
        "policy_tenure_months": v["policy_tenure_months"], "claim_amount": v["claim_amount"],
        "prior_claims_count": v["prior_claims_count"],
        "claim_to_income_ratio": v["claim_amount"] / income if income else 0,
        "claim_dow": d.weekday(), "claim_month": d.month,
        "is_high_value_claim": int(v["claim_amount"] > HIGH_VALUE_THRESHOLD),
        "desc_length": len(desc), "desc_word_count": len(desc.split()),
        "claim_type": v["claim_type"], "city": v["city"], "gender": v["gender"],
        "occupation": v["occupation"], "tenure_bucket": tenure_bucket(v["policy_tenure_months"]),
    }


def clean_text(text):                    # same cleaning used by 07_agentic_ai/agent.py
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def risk_level(p):
    if p >= RISK_BANDS["high_from"]:
        return "high"
    return "medium" if p >= RISK_BANDS["medium_from"] else "low"


# ------------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    return jsonify(status="ok", models=["random_forest", "tfidf_logreg"])


@app.get("/api/schema")
def schema():
    return jsonify(options=CATEGORY_OPTIONS, ranges=TRAIN_RANGES,
                   decision_threshold=DECISION_THRESHOLD, risk_bands=RISK_BANDS,
                   high_value_threshold=round(HIGH_VALUE_THRESHOLD, 2))


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send the claim as a JSON object."), 400
    v, errors = validate(payload)
    if errors:
        return jsonify(error="Some fields need attention.", field_errors=errors), 422

    row = build_model_row(v)
    X = pd.DataFrame([row])[NUMERIC + CATEGORICAL]
    prob = float(ml_pipe.predict_proba(X)[0, 1])
    pred = int(ml_pipe.predict(X)[0])

    vec = nlp_vec.transform([clean_text(v["claim_description"])])
    text_prob = float(nlp_clf.predict_proba(vec)[0, 1])
    terms = np.array(nlp_vec.get_feature_names_out())
    arr = vec.toarray().ravel()
    keywords = [str(terms[i]) for i in arr.argsort()[::-1][:3] if arr[i] > 0]

    # Rule-based indicators: the same three rules agent.py's explain() uses.
    indicators = []
    if row["is_high_value_claim"]:
        indicators.append(f"Claim amount is in the top 10% of claims in the project dataset (above {HIGH_VALUE_THRESHOLD:,.0f}).")
    if v["policy_tenure_months"] < 3:
        indicators.append(f"Very new policy ({v['policy_tenure_months']} months).")
    if v["prior_claims_count"] >= 2:
        indicators.append(f"{v['prior_claims_count']} prior claims on record.")

    warnings_out = []
    for key, label in [("claimant_age", "Age"), ("annual_income", "Annual income"),
                       ("policy_tenure_months", "Policy tenure"), ("claim_amount", "Claim amount"),
                       ("prior_claims_count", "Prior claims")]:
        lo, hi = TRAIN_RANGES[key]
        if v[key] < lo or v[key] > hi:
            warnings_out.append(f"{label} is outside the range seen in training data ({lo:,.0f} to {hi:,.0f}); treat the score with extra caution.")

    return jsonify(
        predicted_class=pred,
        predicted_label="Potentially suspicious" if pred == 1 else "Not flagged",
        fraud_probability=round(prob, 4),
        decision_threshold=DECISION_THRESHOLD,
        risk_level=risk_level(prob),
        risk_bands=RISK_BANDS,
        text_signal={"probability": round(text_prob, 4), "keywords": keywords,
                     "model": "TF-IDF + Logistic Regression (03_nlp)"},
        indicators=indicators,
        derived_features={k: row[k] for k in ["claim_to_income_ratio", "tenure_bucket", "claim_dow",
                                               "claim_month", "is_high_value_claim", "desc_length",
                                               "desc_word_count"]},
        warnings=warnings_out,
        model="Random Forest pipeline (01_ml/best_ml_model.joblib)",
    )


# Optional: serve the built frontend so one command runs the whole site.
@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def frontend(path):
    if path.startswith("api/"):
        return jsonify(error="Not found."), 404
    if not DIST_DIR.exists():
        return ("Frontend not built yet. Run `npm run build` in web_app/frontend, "
                "or use `npm run dev` for development.", 404)
    target = DIST_DIR / path
    if path and target.is_file():
        return send_from_directory(DIST_DIR, path)
    return send_from_directory(DIST_DIR, "index.html")


if __name__ == "__main__":
    import sklearn
    print(f"scikit-learn {sklearn.__version__} | high-value threshold {HIGH_VALUE_THRESHOLD:,.2f}")
    app.run(host="127.0.0.1", port=5000, debug=False)
