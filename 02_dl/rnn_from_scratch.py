"""
rnn_from_scratch.py — Level: DL (Recurrent Neural Network component)
========================================================================
No deep-learning framework (torch/tensorflow) is installed and this
environment has no internet access to fetch one, so we implement a real
Elman RNN (many-to-one) from scratch in NumPy, trained with
backpropagation-through-time (BPTT) and mini-batch gradient descent, on
the sequential claim EVENT LOG (SUBMITTED -> DOC_REQUESTED -> ... -> PAID).

Goal: predict is_fraud from the *shape* of a claim's event sequence
(e.g. claims that get ESCALATED then AMOUNT_REVISED quickly, or that skip
ADJUSTER_ASSIGNED, look different from routine claims).
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json

BASE = Path(__file__).resolve().parents[1]
EVENTS = BASE / "data" / "raw" / "claim_events_raw.csv"
CLAIMS = BASE / "data" / "processed" / "claims_clean.csv"
OUT = BASE / "02_dl"

np.random.seed(42)

EVENT_TYPES = ["SUBMITTED", "DOC_REQUESTED", "DOC_RECEIVED", "ADJUSTER_ASSIGNED",
               "SITE_VISIT", "AMOUNT_REVISED", "ESCALATED", "APPROVED", "REJECTED", "PAID"]
EV2IDX = {e: i for i, e in enumerate(EVENT_TYPES)}
MAX_LEN = 7


def one_hot_sequence(event_list):
    seq = np.zeros((MAX_LEN, len(EVENT_TYPES)))
    for t, ev in enumerate(event_list[:MAX_LEN]):
        seq[t, EV2IDX[ev]] = 1.0
    return seq


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


class SimpleRNN:
    """Elman RNN, many-to-one, single hidden layer, sigmoid output for binary classification."""

    def __init__(self, input_dim, hidden_dim, lr=0.05):
        self.h_dim = hidden_dim
        self.lr = lr
        scale = 1 / np.sqrt(hidden_dim)
        self.Wxh = np.random.randn(input_dim, hidden_dim) * scale
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * scale
        self.bh = np.zeros(hidden_dim)
        self.Why = np.random.randn(hidden_dim, 1) * scale
        self.by = np.zeros(1)

    def forward(self, X):  # X: (T, input_dim)
        T = X.shape[0]
        h = np.zeros((T + 1, self.h_dim))
        for t in range(T):
            h[t + 1] = np.tanh(X[t] @ self.Wxh + h[t] @ self.Whh + self.bh)
        y_logit = h[T] @ self.Why + self.by
        y = sigmoid(y_logit)
        return y, h

    def train_step(self, X, target):
        y, h = self.forward(X)
        T = X.shape[0]
        dy = (y - target)  # d(loss)/d(logit) for BCE+sigmoid
        dWhy = np.outer(h[T], dy)
        dby = dy
        dh_next = (dy @ self.Why.T).flatten()

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        for t in reversed(range(T)):
            dh = dh_next * (1 - h[t + 1] ** 2)
            dWxh += np.outer(X[t], dh)
            dWhh += np.outer(h[t], dh)
            dbh += dh
            dh_next = dh @ self.Whh.T

        for param, grad in [(self.Wxh, dWxh), (self.Whh, dWhh), (self.bh, dbh),
                             (self.Why, dWhy), (self.by, dby)]:
            param -= self.lr * grad
        loss = -(target * np.log(y + 1e-9) + (1 - target) * np.log(1 - y + 1e-9)).item()
        return loss

    def predict(self, X):
        y, _ = self.forward(X)
        return y.item()


def main():
    events = pd.read_csv(EVENTS)
    claims = pd.read_csv(CLAIMS)[["claim_id", "is_fraud"]]

    seq_by_claim = events.sort_values("step").groupby("claim_id")["event_type"].apply(list)
    merged = claims.set_index("claim_id").join(seq_by_claim, how="inner")
    merged = merged.dropna(subset=["event_type"])

    X_all = np.stack([one_hot_sequence(seq) for seq in merged["event_type"]])
    y_all = merged["is_fraud"].values.astype(float)

    n = len(y_all)
    idx = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    rnn = SimpleRNN(input_dim=len(EVENT_TYPES), hidden_dim=16, lr=0.08)

    epochs = 6
    history = []
    for epoch in range(epochs):
        np.random.shuffle(train_idx)
        losses = []
        for i in train_idx:
            loss = rnn.train_step(X_all[i], y_all[i])
            losses.append(loss)
        avg_loss = float(np.mean(losses))
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}  avg_loss={avg_loss:.4f}")

    # Evaluate
    preds = np.array([rnn.predict(X_all[i]) for i in test_idx])
    y_test = y_all[test_idx]
    pred_labels = (preds > 0.5).astype(int)

    tp = int(((pred_labels == 1) & (y_test == 1)).sum())
    tn = int(((pred_labels == 0) & (y_test == 0)).sum())
    fp = int(((pred_labels == 1) & (y_test == 0)).sum())
    fn = int(((pred_labels == 0) & (y_test == 1)).sum())
    accuracy = (tp + tn) / len(y_test)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # simple AUC via rank-based formula (Mann-Whitney U)
    pos = preds[y_test == 1]
    neg = preds[y_test == 0]
    if len(pos) and len(neg):
        auc = np.mean([1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg])
    else:
        auc = float("nan")

    print(f"\nRNN test set: n={len(y_test)}  acc={accuracy:.3f} prec={precision:.3f} "
          f"recall={recall:.3f} f1={f1:.3f} auc={auc:.3f}")

    with open(OUT / "rnn_results.json", "w") as f:
        json.dump({
            "epochs": epochs,
            "loss_history": history,
            "test_metrics": {"accuracy": accuracy, "precision": precision,
                              "recall": recall, "f1": f1, "auc": auc},
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }, f, indent=2)
    print(f"Saved: {OUT/'rnn_results.json'}")


if __name__ == "__main__":
    main()
