"""
train_mlp.py — Level: DL (feed-forward deep neural network component)
=========================================================================
Trains a Multi-Layer Perceptron (a genuine backprop-trained deep neural
network, via sklearn's MLPClassifier) on tabular claim features PLUS
graph-structural features from graph_features.py AND the learned node
embeddings from the from-scratch GCN in gnn_from_scratch.py, and compares
performance against the Level-ML Random Forest baseline.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
import joblib

BASE = Path(__file__).resolve().parents[1]
CLAIMS = BASE / "data" / "processed" / "claims_clean.csv"
GRAPH = BASE / "02_dl" / "policyholder_graph_features.csv"
GNN_EMB = BASE / "02_dl" / "policyholder_gnn_embeddings.csv"
OUT = BASE / "02_dl"

GRAPH_COLS = ["graph_degree", "graph_pagerank", "graph_clustering_coef", "graph_component_size"]
GNN_COLS = [f"gnn_emb_{i}" for i in range(8)]

NUMERIC = ["claimant_age", "annual_income", "policy_tenure_months", "claim_amount",
           "prior_claims_count", "claim_to_income_ratio", "claim_dow", "claim_month",
           "is_high_value_claim", "desc_length", "desc_word_count"] + GRAPH_COLS + GNN_COLS
CATEGORICAL = ["claim_type", "city", "gender", "occupation", "tenure_bucket"]


def main():
    claims = pd.read_csv(CLAIMS)
    graph = pd.read_csv(GRAPH)
    gnn_emb = pd.read_csv(GNN_EMB)
    df = claims.merge(graph, on="policyholder_id", how="left")
    df = df.merge(gnn_emb, on="policyholder_id", how="left")
    for c in GRAPH_COLS + GNN_COLS:
        df[c] = df[c].fillna(0)

    X = df[NUMERIC + CATEGORICAL]
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                         solver="adam", alpha=1e-4, learning_rate_init=5e-4,
                         max_iter=2000, early_stopping=True, n_iter_no_change=30,
                         random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", mlp)])
    # MLPClassifier has no class_weight param -> emulate with sample weights
    class_counts = y_train.value_counts()
    weight_map = {c: len(y_train) / (2 * cnt) for c, cnt in class_counts.items()}
    sample_weight = y_train.map(weight_map).values
    pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)

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
    print("=== MLP (Deep Neural Net, tabular + graph features + GCN embeddings) ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    print("Confusion matrix:", cm)

    ml_results = json.loads((BASE / "01_ml" / "ml_results.json").read_text())
    rf_auc = ml_results["results"]["random_forest"]["metrics"]["roc_auc"]
    print(f"\nComparison — ML Random Forest ROC-AUC: {rf_auc:.4f} | DL MLP ROC-AUC: {metrics['roc_auc']:.4f}")

    joblib.dump(pipe, OUT / "best_dl_model.joblib")
    with open(OUT / "mlp_results.json", "w") as f:
        json.dump({"metrics": metrics, "confusion_matrix": cm,
                    "comparison_ml_rf_auc": rf_auc}, f, indent=2)
    print(f"Saved: {OUT/'best_dl_model.joblib'}, {OUT/'mlp_results.json'}")


if __name__ == "__main__":
    main()
