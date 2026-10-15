"""
export_frontend_data.py — READ-ONLY export of verified numbers for the website.

Reads the project's saved result files and writes two JSON files into
web_app/frontend/src/data/. Nothing in the ML project is modified, and no
number is typed by hand: every metric on the Model Insights page comes from
a saved *_results.json / *_summary.json file, or from counting claims_clean.csv.

Run (from fraud_project/):  python3 web_app/scripts/export_frontend_data.py
"""
import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "frontend" / "src" / "data"
OUT.mkdir(parents=True, exist_ok=True)
J = lambda rel: json.loads((BASE / rel).read_text())

ml, nlp = J("01_ml/ml_results.json"), J("03_nlp/nlp_results.json")
mlp, rnn, gnn = J("02_dl/mlp_results.json"), J("02_dl/rnn_results.json"), J("02_dl/gnn_results.json")
gen, agent = J("06_genai/genai_results.json"), J("07_agentic_ai/agent_summary.json")
df = pd.read_csv(BASE / "data/processed/claims_clean.csv")

rnn_cm, gnn_cm = rnn["confusion_matrix"], gnn["confusion_matrix"]
metrics = {
    "dataset": {
        "total_claims": int(len(df)), "fraud": int(df.is_fraud.sum()),
        "not_fraud": int((df.is_fraud == 0).sum()),
        "fraud_rate": float(df.is_fraud.mean()),
        "train_size": ml["train_size"], "test_size": ml["test_size"],
        "distinct_descriptions": int(df.claim_description.nunique()),
        "claim_types": sorted(df.claim_type.unique().tolist()),
        "source": "data/processed/claims_clean.csv, 01_ml/ml_results.json",
    },
    "ml": {
        "source": "01_ml/ml_results.json (20% stratified holdout, random_state=42)",
        "best_model": ml["best_model"],
        "random_forest": ml["results"]["random_forest"],
        "logistic_regression": ml["results"]["logistic_regression"],
        "top_features": ml["top_features"][:10],
    },
    "nlp": {"source": "03_nlp/nlp_results.json", "metrics": nlp["classification_metrics"]},
    "mlp": {"source": "02_dl/mlp_results.json", "metrics": mlp["metrics"]},
    "rnn": {"source": "02_dl/rnn_results.json", "metrics": rnn["test_metrics"], "confusion": rnn_cm},
    "gnn": {"source": "02_dl/gnn_results.json", "metrics": gnn["test_metrics"], "confusion": gnn_cm,
            "test_nodes": gnn["test_nodes"]},
    "genai": {
        "source": "06_genai/genai_results.json",
        "baseline": gen["baseline_metrics_real_data_only"],
        "augmented": gen["augmented_metrics_with_genai_synthetic"],
        "stress_detection_rate": gen["stress_test_detection_rate_on_novel_synth_fraud"],
        "n_synthetic": gen["n_synthetic_generated"],
    },
    "agent": {"source": "07_agentic_ai/agent_summary.json + README", **agent},
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

# Demo claims: real rows (inputs only, no label) so the form can be filled in one click.
cols = ["claim_type", "claim_date", "claimant_age", "annual_income", "policy_tenure_months",
        "claim_amount", "prior_claims_count", "city", "gender", "occupation", "claim_description"]
# In the synthetic dataset claim_type is independent of the description text, so we only
# pick rows where the two happen to agree (e.g. an Auto claim about a vehicle) to keep demos sensible.
# Demo convenience: among those, pick a typical row the saved Random Forest scores clearly
# (>= 0.60 for fraud-labelled rows, <= 0.30 for genuine ones). Real recall is only ~0.58, so
# some fraud-labelled claims score low; the Insights page reports that honestly.
import warnings, joblib
warnings.filterwarnings("ignore")
rf = joblib.load(BASE / "01_ml" / "best_ml_model.joblib")
NUM = ["claimant_age", "annual_income", "policy_tenure_months", "claim_amount", "prior_claims_count",
       "claim_to_income_ratio", "claim_dow", "claim_month", "is_high_value_claim", "desc_length", "desc_word_count"]
CAT = ["claim_type", "city", "gender", "occupation", "tenure_bucket"]
THEMES = [(0, "Auto", "Minor collision"), (0, "Health", "Patient admitted"),
          (1, "Auto", "Vehicle stolen"), (1, "Property", "Property damage claim filed right before")]
picks = []
for label, ctype, phrase in THEMES:
    pool = df[(df.is_fraud == label) & (df.claim_type == ctype) & df.claim_description.str.startswith(phrase)].copy()
    pool["p"] = rf.predict_proba(pool[NUM + CAT])[:, 1]
    pool = pool[pool.p >= 0.60] if label == 1 else pool[pool.p <= 0.30]
    r = pool.sort_values("p").iloc[len(pool) // 2]
    picks.append({c: (int(r[c]) if c in ("claimant_age", "policy_tenure_months", "prior_claims_count")
                      else round(float(r[c]), 2) if c in ("annual_income", "claim_amount") else str(r[c]))
                  for c in cols})
(OUT / "samples.json").write_text(json.dumps(picks, indent=2))
print("wrote", OUT / "metrics.json", "and", OUT / "samples.json")
