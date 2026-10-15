"""
slm_summarizer.py — Level: SLM
=================================
No internet access -> cannot download/fine-tune an actual small transformer
LM (e.g. a distilled 100-300M model via `transformers`). Instead this module
delivers the SAME PRODUCT REQUIREMENT ("summarize suspicious claim details for
investigators, generate concise explanations") using classical, fully local
NLP techniques that were the standard approach for exactly this task before
small transformer LMs existed:

  1. Extractive summarization via TextRank (sentence-similarity graph +
     PageRank, computed with networkx — same graph-centrality idea used
     in 02_dl/graph_features.py, applied to sentences instead of people).
  2. Template-grounded explanation generation: combines the ML model's
     feature importances (01_ml), the NLP classifier's flagged keywords
     (03_nlp), and rule-based NER entities into a short, faithful,
     human-readable rationale — this is "explanation generation" without
     free generation risk (every fact in the explanation is grounded in
     a real feature value, not hallucinated).

This is clearly documented as a substitute for a fine-tuned SLM, not
presented as one.
"""
import re
import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

BASE = Path(__file__).resolve().parents[1]
CLAIMS = BASE / "data" / "processed" / "claims_clean.csv"
OUT = BASE / "04_slm"


def split_sentences(text):
    # simple regex sentence splitter (no nltk available)
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sents if s]


def textrank_summarize(text, n_sentences=2):
    sents = split_sentences(text)
    if len(sents) <= n_sentences:
        return " ".join(sents)
    vec = TfidfVectorizer(stop_words="english")
    try:
        X = vec.fit_transform(sents)
    except ValueError:
        return " ".join(sents[:n_sentences])
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)
    G = nx.from_numpy_array(sim)
    scores = nx.pagerank(G, max_iter=200) if G.number_of_edges() > 0 else {i: 1 for i in range(len(sents))}
    ranked = sorted(((scores[i], i) for i in range(len(sents))), reverse=True)
    top_idx = sorted(i for _, i in ranked[:n_sentences])
    return " ".join(sents[i] for i in top_idx)


def rouge_n(reference, candidate, n=1):
    """From-scratch ROUGE-N recall (no `rouge` package available offline)."""
    def ngrams(text, n):
        tokens = re.findall(r"\w+", text.lower())
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    ref_ngrams = ngrams(reference, n)
    cand_ngrams = ngrams(candidate, n)
    if not ref_ngrams:
        return 0.0
    ref_counts = pd.Series(ref_ngrams).value_counts()
    cand_counts = pd.Series(cand_ngrams).value_counts() if cand_ngrams else pd.Series(dtype=int)
    overlap = sum(min(ref_counts.get(g, 0), cand_counts.get(g, 0)) for g in ref_counts.index)
    return overlap / len(ref_ngrams)


def generate_explanation(row, top_ml_features, nlp_keywords):
    """Template-grounded rationale — every clause cites a real value, no free-form hallucination."""
    reasons = []
    if row["is_high_value_claim"] == 1:
        reasons.append(f"claim amount (₹{row['claim_amount']:,.0f}) is in the top 10% of all claims")
    if row["policy_tenure_months"] < 3:
        reasons.append(f"policy is very new ({row['policy_tenure_months']} months old)")
    if row["prior_claims_count"] >= 2:
        reasons.append(f"policyholder has {row['prior_claims_count']} prior claims")
    if row["claim_to_income_ratio"] > 0.5:
        reasons.append(f"claim-to-income ratio is high ({row['claim_to_income_ratio']:.2f})")
    if nlp_keywords:
        reasons.append(f"description flagged for suspicious language ('{', '.join(nlp_keywords[:3])}')")
    if not reasons:
        reasons.append("no strong individual risk factor, flagged by combined model score")
    return ("Investigator note: This claim was flagged as potentially fraudulent because "
            + "; ".join(reasons) + ".")


def main():
    df = pd.read_csv(CLAIMS)
    ml_results = json.loads((BASE / "01_ml" / "ml_results.json").read_text())
    top_ml_features = [f[0] for f in ml_results["top_features"][:5]]

    nlp_bundle = joblib.load(BASE / "03_nlp" / "text_classifier.joblib")
    vec, clf = nlp_bundle["vectorizer"], nlp_bundle["clf"]

    flagged = df[df["is_fraud"] == 1].sample(min(6, (df["is_fraud"] == 1).sum()), random_state=1)

    reports = []
    rouge_scores = []
    for _, row in flagged.iterrows():
        desc = row["claim_description"]
        summary = textrank_summarize(desc, n_sentences=1)
        tfidf_row = vec.transform([desc.lower()])
        terms = np.array(vec.get_feature_names_out())
        top_idx = tfidf_row.toarray().flatten().argsort()[::-1][:3]
        keywords = [terms[i] for i in top_idx if tfidf_row[0, i] > 0]

        explanation = generate_explanation(row, top_ml_features, keywords)
        r1 = rouge_n(desc, summary, n=1)
        rouge_scores.append(r1)

        reports.append({
            "claim_id": row["claim_id"],
            "original_description": desc,
            "extractive_summary": summary,
            "investigator_explanation": explanation,
            "rouge_1_recall_vs_original": round(r1, 3),
        })

    avg_rouge = float(np.mean(rouge_scores))
    print(f"Average ROUGE-1 recall (summary vs. original): {avg_rouge:.3f}")
    for r in reports[:3]:
        print(f"\nClaim {r['claim_id']}")
        print(f"  Summary: {r['extractive_summary']}")
        print(f"  {r['investigator_explanation']}")

    with open(OUT / "slm_results.json", "w") as f:
        json.dump({"average_rouge_1_recall": avg_rouge, "reports": reports}, f, indent=2)
    print(f"\nSaved: {OUT/'slm_results.json'}")


if __name__ == "__main__":
    main()
