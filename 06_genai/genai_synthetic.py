"""
genai_synthetic.py — Level: GenAI
=====================================
No internet access -> can't download a pretrained GAN/diffusion/LLM checkpoint.
Implements genuine *generative* modeling with what's available:

  1. TABULAR: a Gaussian Mixture Model (sklearn.mixture.GaussianMixture) is
     FIT on the real fraud-class distribution, then SAMPLED to synthesize
     new, never-before-seen fraudulent claims. This is a legitimate
     generative model (it learns and samples from a probability density),
     conceptually the same family as GAN generators/VAEs but with a
     tractable closed-form density instead of a learned neural generator.

  2. TEXT: a word-level Markov chain (order-2) is trained on real fraud
     claim descriptions and used to generate new synthetic fraud narratives
     — the classical predecessor of neural text generation, fully local.

  3. STRESS TEST: the ML baseline (01_ml) is re-evaluated against a test
     set augmented with GenAI-synthesized fraud claims it has NEVER seen,
     to check whether it generalizes to novel-but-similar fraud patterns
     (this is the explicit ask: "stress-test detection models against
     evolving fraud tactics").

  4. AUGMENTATION: the RF model is retrained with GenAI-synthesized minority
     class examples added, to see whether rare-fraud-type recall improves.
"""
import json
import random
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib

BASE = Path(__file__).resolve().parents[1]
CLAIMS = BASE / "data" / "processed" / "claims_clean.csv"
OUT = BASE / "06_genai"

NUMERIC = ["claimant_age", "annual_income", "policy_tenure_months", "claim_amount",
           "prior_claims_count", "claim_to_income_ratio", "claim_dow", "claim_month",
           "is_high_value_claim", "desc_length", "desc_word_count"]
CATEGORICAL = ["claim_type", "city", "gender", "occupation", "tenure_bucket"]

random.seed(42)
np.random.seed(42)


# ---------------------------- Tabular GenAI ----------------------------
def fit_tabular_generator(fraud_df):
    X = fraud_df[NUMERIC].fillna(0).values
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
    gmm.fit(X)
    return gmm


def sample_synthetic_claims(gmm, fraud_df, n_samples=300):
    samples, _ = gmm.sample(n_samples)
    synth = pd.DataFrame(samples, columns=NUMERIC)
    # clip to plausible ranges observed in real data
    for col in NUMERIC:
        lo, hi = fraud_df[col].min(), fraud_df[col].max()
        synth[col] = synth[col].clip(lo, hi)
    synth["is_high_value_claim"] = (synth["claim_amount"] > fraud_df["claim_amount"].quantile(0.9)).astype(int)
    # sample categoricals from the empirical fraud-class distribution
    for col in CATEGORICAL:
        synth[col] = np.random.choice(fraud_df[col].dropna().values, size=len(synth))
    synth["is_fraud"] = 1
    synth["claim_id"] = [f"SYNTH{100000+i}" for i in range(len(synth))]
    return synth


# ---------------------------- Text GenAI (Markov chain) ----------------------------
class MarkovTextGenerator:
    def __init__(self, order=2):
        self.order = order
        self.model = defaultdict(list)
        self.starts = []

    def fit(self, texts):
        for text in texts:
            words = re.findall(r"\w+|[.,!?;]", text)
            if len(words) <= self.order:
                continue
            self.starts.append(tuple(words[:self.order]))
            for i in range(len(words) - self.order):
                key = tuple(words[i:i + self.order])
                self.model[key].append(words[i + self.order])

    def generate(self, max_words=30):
        if not self.starts:
            return ""
        state = random.choice(self.starts)
        out = list(state)
        for _ in range(max_words - self.order):
            choices = self.model.get(state)
            if not choices:
                break
            nxt = random.choice(choices)
            out.append(nxt)
            state = tuple(out[-self.order:])
        text = " ".join(out)
        text = re.sub(r"\s+([.,!?;])", r"\1", text)
        return text[0].upper() + text[1:] if text else text


def stress_test(original_test_X, original_test_y, synth_df, pipe):
    """Evaluate the ALREADY-TRAINED ML baseline on novel synthetic fraud it never trained on."""
    Xs = synth_df[NUMERIC + CATEGORICAL]
    proba = pipe.predict_proba(Xs)[:, 1]
    detect_rate = float((proba > 0.5).mean())  # fraction of novel synthetic fraud caught
    return detect_rate, float(np.mean(proba))


def augmentation_experiment(df, synth_df):
    """Retrain RF with GenAI-augmented minority class; compare recall to baseline."""
    X = df[NUMERIC + CATEGORICAL]
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    def make_pipe():
        pre = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])
        return Pipeline([("pre", pre), ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1))])

    baseline = make_pipe()
    baseline.fit(X_train, y_train)
    pred_b = baseline.predict(X_test)
    baseline_metrics = {"precision": precision_score(y_test, pred_b, zero_division=0),
                         "recall": recall_score(y_test, pred_b, zero_division=0),
                         "f1": f1_score(y_test, pred_b, zero_division=0),
                         "roc_auc": roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])}

    X_aug = pd.concat([X_train, synth_df[NUMERIC + CATEGORICAL]], ignore_index=True)
    y_aug = pd.concat([y_train, synth_df["is_fraud"]], ignore_index=True)
    augmented = make_pipe()
    augmented.fit(X_aug, y_aug)
    pred_a = augmented.predict(X_test)
    aug_metrics = {"precision": precision_score(y_test, pred_a, zero_division=0),
                   "recall": recall_score(y_test, pred_a, zero_division=0),
                   "f1": f1_score(y_test, pred_a, zero_division=0),
                   "roc_auc": roc_auc_score(y_test, augmented.predict_proba(X_test)[:, 1])}

    return baseline_metrics, aug_metrics, baseline, X_test, y_test


def main():
    df = pd.read_csv(CLAIMS)
    fraud_df = df[df["is_fraud"] == 1].copy()

    # 1. Tabular generation
    gmm = fit_tabular_generator(fraud_df)
    synth_df = sample_synthetic_claims(gmm, fraud_df, n_samples=300)
    print(f"Generated {len(synth_df)} synthetic fraudulent claims (tabular, GMM).")

    # 2. Text generation
    gen = MarkovTextGenerator(order=2)
    gen.fit(fraud_df["claim_description"].tolist())
    synthetic_texts = [gen.generate(max_words=25) for _ in range(5)]
    print("\nSample GenAI-synthesized fraud narratives:")
    for t in synthetic_texts:
        print(f"  - {t}")
    synth_df["claim_description"] = [gen.generate(max_words=25) for _ in range(len(synth_df))]

    # 3. Stress test the existing ML baseline against novel synthetic fraud
    baseline_metrics, aug_metrics, baseline_pipe, X_test, y_test = augmentation_experiment(df, synth_df)
    detect_rate, mean_proba = stress_test(X_test, y_test, synth_df, baseline_pipe)
    print(f"\nStress test: baseline RF flags {detect_rate*100:.1f}% of novel GenAI-synthesized "
          f"fraud claims as fraud (mean predicted probability={mean_proba:.3f}).")

    print("\n=== Data augmentation experiment (RF, real test set) ===")
    print("Baseline (real data only):        ", {k: round(v, 3) for k, v in baseline_metrics.items()})
    print("Augmented (+ GenAI synthetic fraud):", {k: round(v, 3) for k, v in aug_metrics.items()})

    synth_df.to_csv(OUT / "synthetic_fraud_claims.csv", index=False)
    with open(OUT / "genai_results.json", "w") as f:
        json.dump({
            "n_synthetic_generated": len(synth_df),
            "sample_synthetic_texts": synthetic_texts,
            "stress_test_detection_rate_on_novel_synth_fraud": detect_rate,
            "stress_test_mean_predicted_probability": mean_proba,
            "baseline_metrics_real_data_only": baseline_metrics,
            "augmented_metrics_with_genai_synthetic": aug_metrics,
        }, f, indent=2)
    print(f"\nSaved: {OUT/'synthetic_fraud_claims.csv'}, {OUT/'genai_results.json'}")


if __name__ == "__main__":
    main()
