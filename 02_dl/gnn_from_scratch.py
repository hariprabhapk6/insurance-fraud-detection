"""
gnn_from_scratch.py — Level: DL (Graph Neural Network component)
========================================================================
No deep-learning / graph-learning framework (torch, torch-geometric, dgl)
is installed and this environment has no internet access to fetch one,
so we implement a REAL Graph Convolutional Network (Kipf & Welling style,
2-layer spectral GCN) from scratch in NumPy, trained end-to-end with
backpropagation (full-batch gradient descent) on the policyholder
network.

This replaces the earlier `graph_features.py` substitute (hand-computed
degree/PageRank/clustering) with an actual LEARNED message-passing model:
node representations are built by repeatedly aggregating neighbor
features through the normalized adjacency matrix and a trainable weight
matrix, exactly the mechanism a GNN layer uses (just without a framework
doing the autodiff for us — we derive and code the gradients by hand).

Task: node classification on the policyholder graph.
  Node   = policyholder (only policyholders who appear in the
           shared-agent / shared-address / shared-bank-account network).
  Label  = 1 if ANY of that policyholder's claims was fraudulent, else 0.
  Features = aggregated (non-leaking) claim-level stats per policyholder:
           claim count, avg claim amount, avg income, avg tenure, avg age,
           avg prior_claims_count, high-value-claim ratio.

Architecture:
  Z1 = A_norm @ X  @ W0 + b0 ;  H1 = ReLU(Z1)
  Z2 = A_norm @ H1 @ W1 + b1 ;  H2 = tanh(Z2)   <- node embeddings
  logits = H2 @ Wout + bout ;   y_hat = sigmoid(logits)

A_norm = D^-1/2 (A + I) D^-1/2 is the symmetric-normalized adjacency with
self-loops (the standard GCN propagation rule). Because A_norm is a fixed
(non-trainable) linear operator, backprop through it is just a matrix
multiply by A_norm again (it's symmetric), which is what makes "message
passing" differentiable without any special-case code.

Output: gnn_results.json (metrics) and policyholder_gnn_embeddings.csv
(the learned H2 embeddings for every node), which train_mlp.py consumes
as additional DL-derived features alongside the RNN/tabular ones.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json

BASE = Path(__file__).resolve().parents[1]
EDGES = BASE / "data" / "raw" / "policyholder_network.csv"
CLAIMS = BASE / "data" / "processed" / "claims_clean.csv"
OUT = BASE / "02_dl"

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def build_graph():
    edges = pd.read_csv(EDGES)
    nodes = sorted(set(edges["source"]) | set(edges["target"]))
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n))
    for _, r in edges.iterrows():
        i, j = idx[r["source"]], idx[r["target"]]
        A[i, j] = 1.0
        A[j, i] = 1.0
    return nodes, idx, A


def normalize_adj(A):
    A_hat = A + np.eye(A.shape[0])
    deg = A_hat.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = np.power(deg[nz], -0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def build_node_features(nodes, claims):
    agg = claims.groupby("policyholder_id").agg(
        claim_count=("claim_id", "count"),
        avg_claim_amount=("claim_amount", "mean"),
        avg_income=("annual_income", "mean"),
        avg_tenure=("policy_tenure_months", "mean"),
        avg_age=("claimant_age", "mean"),
        avg_prior_claims=("prior_claims_count", "mean"),
        high_value_ratio=("is_high_value_claim", "mean"),
        any_fraud=("is_fraud", "max"),
    )
    feat_cols = ["claim_count", "avg_claim_amount", "avg_income", "avg_tenure",
                 "avg_age", "avg_prior_claims", "high_value_ratio"]
    row_list = []   
    label_list = []
    mask_list = []
    for node in nodes:
        if node in agg.index:
            row = agg.loc[node, feat_cols].values.astype(float)
            label = float(agg.loc[node, "any_fraud"])
            row_list.append(row)
            label_list.append(label)
            mask_list.append(True)
        else:
            row_list.append(np.zeros(len(feat_cols)))
            label_list.append(0.0)
            mask_list.append(False)
    X = np.array(row_list)
    y = np.array(label_list)
    mask = np.array(mask_list)
    # standardize using only labeled (real) rows
    mu = X[mask].mean(axis=0)
    sigma = X[mask].std(axis=0)
    sigma[sigma == 0] = 1.0
    X = (X - mu) / sigma
    return X, y, mask, feat_cols


class GCN:
    """2-layer Graph Convolutional Network, node classification, from scratch."""

    def __init__(self, in_dim, hid_dim, emb_dim, lr=0.05, l2=1e-3):
        s0 = 1 / np.sqrt(in_dim)
        s1 = 1 / np.sqrt(hid_dim)
        s2 = 1 / np.sqrt(emb_dim)
        self.W0 = np.random.randn(in_dim, hid_dim) * s0
        self.b0 = np.zeros(hid_dim)
        self.W1 = np.random.randn(hid_dim, emb_dim) * s1
        self.b1 = np.zeros(emb_dim)
        self.Wout = np.random.randn(emb_dim, 1) * s2
        self.bout = np.zeros(1)
        self.lr = lr
        self.l2 = l2

    def forward(self, A_norm, X):
        Z1 = A_norm @ X @ self.W0 + self.b0
        H1 = np.maximum(Z1, 0)          # ReLU
        AH1 = A_norm @ H1
        Z2 = AH1 @ self.W1 + self.b1
        H2 = np.tanh(Z2)                # node embeddings
        logits = (H2 @ self.Wout + self.bout).flatten()
        y_hat = sigmoid(logits)
        cache = (X, Z1, H1, AH1, Z2, H2, logits)
        return y_hat, cache

    def train_step(self, A_norm, X, y, mask):
        y_hat, (X_, Z1, H1, AH1, Z2, H2, logits) = self.forward(A_norm, X)
        n_labeled = mask.sum()

        # BCE loss, masked to labeled nodes only
        eps = 1e-9
        yh = np.clip(y_hat, eps, 1 - eps)
        loss = -np.mean(y[mask] * np.log(yh[mask]) + (1 - y[mask]) * np.log(1 - yh[mask]))

        # dL/dlogits, zero for unlabeled nodes (they don't contribute to loss)
        dlogits = np.zeros_like(y_hat)
        dlogits[mask] = (y_hat[mask] - y[mask]) / n_labeled

        dWout = H2.T @ dlogits.reshape(-1, 1)
        dbout = dlogits.sum(keepdims=True)
        dH2 = np.outer(dlogits, self.Wout.flatten())

        dZ2 = dH2 * (1 - H2 ** 2)               # tanh'
        dW1 = AH1.T @ dZ2
        db1 = dZ2.sum(axis=0)
        dAH1 = dZ2 @ self.W1.T
        dH1 = A_norm.T @ dAH1                   # A_norm symmetric

        dZ1 = dH1 * (Z1 > 0)                    # relu'
        dW0 = (A_norm @ X_).T @ dZ1
        db0 = dZ1.sum(axis=0)

        for param, grad in [(self.W0, dW0), (self.b0, db0), (self.W1, dW1),
                             (self.b1, db1), (self.Wout, dWout), (self.bout, dbout)]:
            param -= self.lr * (grad + self.l2 * param)

        return loss

    def predict(self, A_norm, X):
        y_hat, cache = self.forward(A_norm, X)
        return y_hat, cache[5]  # probs, embeddings H2


def main():
    claims = pd.read_csv(CLAIMS)
    nodes, idx, A = build_graph()
    A_norm = normalize_adj(A)
    X, y, mask, feat_cols = build_node_features(nodes, claims)

    labeled_idx = np.where(mask)[0]
    rng = np.random.RandomState(42)
    rng.shuffle(labeled_idx)
    split = int(len(labeled_idx) * 0.8)
    train_nodes, test_nodes = labeled_idx[:split], labeled_idx[split:]

    train_mask = np.zeros_like(mask)
    train_mask[train_nodes] = True
    test_mask = np.zeros_like(mask)
    test_mask[test_nodes] = True

    gcn = GCN(in_dim=X.shape[1], hid_dim=16, emb_dim=8, lr=0.1, l2=1e-3)

    epochs = 300
    history = []
    for epoch in range(epochs):
        loss = gcn.train_step(A_norm, X, y, train_mask)
        history.append(loss)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}  train_loss={loss:.4f}")

    probs, embeddings = gcn.predict(A_norm, X)
    y_test = y[test_mask]
    p_test = probs[test_mask]
    pred_labels = (p_test > 0.5).astype(int)

    tp = int(((pred_labels == 1) & (y_test == 1)).sum())
    tn = int(((pred_labels == 0) & (y_test == 0)).sum())
    fp = int(((pred_labels == 1) & (y_test == 0)).sum())
    fn = int(((pred_labels == 0) & (y_test == 1)).sum())
    accuracy = (tp + tn) / len(y_test) if len(y_test) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    pos = p_test[y_test == 1]
    neg = p_test[y_test == 0]
    if len(pos) and len(neg):
        auc = float(np.mean([1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg]))
    else:
        auc = float("nan")

    print(f"\nGNN test set (node classification): n={len(y_test)} "
          f"acc={accuracy:.3f} prec={precision:.3f} recall={recall:.3f} f1={f1:.3f} auc={auc:.3f}")

    with open(OUT / "gnn_results.json", "w") as f:
        json.dump({
            "architecture": "2-layer GCN (spectral, symmetric-normalized adjacency), "
                             "NumPy from scratch, manual backprop, full-batch GD",
            "graph_nodes": len(nodes),
            "graph_edges": int(A.sum() / 2),
            "labeled_nodes": int(mask.sum()),
            "train_nodes": int(train_mask.sum()),
            "test_nodes": int(test_mask.sum()),
            "epochs": epochs,
            "loss_history": history,
            "test_metrics": {"accuracy": accuracy, "precision": precision,
                              "recall": recall, "f1": f1, "auc": auc},
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }, f, indent=2)

    emb_df = pd.DataFrame(embeddings, columns=[f"gnn_emb_{i}" for i in range(embeddings.shape[1])])
    emb_df.insert(0, "policyholder_id", nodes)
    emb_df.to_csv(OUT / "policyholder_gnn_embeddings.csv", index=False)

    print(f"Saved: {OUT/'gnn_results.json'}, {OUT/'policyholder_gnn_embeddings.csv'}")


if __name__ == "__main__":
    main()
