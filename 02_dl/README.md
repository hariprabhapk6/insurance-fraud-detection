# Level: DL — Deep Learning Enhancement

## Environment constraint & how it was handled
This sandbox has **no internet access**, so `torch` / `tensorflow` / `torch-geometric`
cannot be installed. Rather than skip Deep Learning, four genuine neural
techniques were implemented with what's available — including a real,
end-to-end-trained **RNN** and a real, end-to-end-trained **GNN**, both
built from scratch in NumPy with hand-derived backpropagation (no autodiff
library used):

| Requirement | Implementation |
|---|---|
| Feed-forward / tabular DL | `sklearn.neural_network.MLPClassifier` — a real backprop-trained multi-layer perceptron (128→64→32, ReLU, Adam) |
| RNN for sequential claim events | `rnn_from_scratch.py` — an Elman RNN implemented from scratch in NumPy, trained via manual backpropagation-through-time (BPTT) |
| GNN for policyholder networks | `gnn_from_scratch.py` — a genuine **2-layer Graph Convolutional Network** (Kipf & Welling-style spectral GCN) implemented from scratch in NumPy: node features are propagated through the symmetric-normalized adjacency matrix, passed through two trainable weight layers (ReLU then tanh), and trained end-to-end with manually-derived backprop (full-batch gradient descent) on a node-classification task. This is a real learned message-passing model, not hand-computed graph statistics. |
| Hand-engineered graph features (baseline/complement) | `graph_features.py` — networkx-derived structural features (degree, PageRank, clustering coefficient, component size). Kept as an additional, non-learned feature source alongside the learned GCN embeddings — not a substitute for the GNN anymore. |

## Results

### GNN on the policyholder network (`gnn_results.json`)
Node-classification task: predict whether a policyholder has **any** fraudulent
claim, using only aggregated (non-leaking) tabular stats as node features and
the shared-agent / shared-address / shared-bank-account graph as structure.

| Metric | Value |
|---|---|
| Graph | 128 nodes, 157 edges (123 nodes have claims data / labels) |
| Train / test nodes | 98 / 25 |
| Accuracy | 0.720 |
| Precision | 0.625 |
| Recall | 0.556 |
| F1 | 0.588 |
| **AUC** | **0.774** |

The GCN's 2-layer message passing lets a policyholder's prediction be informed
by its *neighbors'* claim behavior (shared agents / addresses / bank accounts) —
a relational signal the row-by-row ML and MLP models structurally cannot see.
The learned 8-dimensional node embeddings (`policyholder_gnn_embeddings.csv`)
are fed into the tabular MLP below as additional features.

### RNN on claim event sequences (`rnn_results.json`)
Predicts fraud purely from the *shape* of a claim's lifecycle (event order).
| Metric | Value |
|---|---|
| Accuracy | 0.932 |
| Precision | 0.833 |
| Recall | 0.059 (at 0.5 threshold — imbalanced target, threshold tunable) |
| **AUC** | **0.861** |

The RNN clearly learned the intended fraud-correlated event pattern (fraud claims
skip `ADJUSTER_ASSIGNED`, get `ESCALATED`/`AMOUNT_REVISED`, and are more often
`REJECTED`) — a signal the ML-level tabular model cannot see at all, since it
never looks at event sequences.

### MLP on tabular + graph features + GCN embeddings (`mlp_results.json`)
| Metric | MLP (DL) | Random Forest (ML baseline) |
|---|---|---|
| ROC-AUC | 0.648 | **0.864** |

**Honest finding**: on this small/medium structured tabular dataset, the tree-based
ML baseline still outperforms the vanilla MLP, even after adding both the
hand-engineered graph features and the learned GCN embeddings as inputs. This
matches well-documented behavior in the ML literature — gradient-boosted trees /
random forests routinely beat vanilla feed-forward nets on small-to-medium
tabular data, where DL's advantage (automatic representation learning) matters
less than on sequence/graph/image data. The DL level's real payoff is in the
**RNN** (sequence) and **GNN** (relational structure) — signal categories the
ML level structurally cannot use — rather than in the MLP beating the RF on
raw AUC.

## Common mistakes avoided
- Vanishing gradients: RNN uses `tanh` + small hidden size (16) + gradient scale checked empirically. GCN uses ReLU in the first layer and tanh at the embedding layer, with a small hidden/embedding size (16/8) to keep gradients well-scaled over only 2 layers.
- Overfitting: MLP uses `early_stopping=True`, `alpha` L2 regularization. GCN uses explicit L2 weight decay (`l2=1e-3`) since the labeled graph is small (123 nodes).
- Class imbalance: sample weights applied to MLP; balanced RNN and GCN eval reported via AUC (not just accuracy) since both targets are skewed toward the majority class.
- Train/test leakage on the graph: GCN node features are aggregated non-fraud claim stats only (claim count, average amount, income, tenure, age, prior claims, high-value ratio) — the `is_fraud` label itself is never used as an input feature, only as the training/eval target.

## How to reproduce
```bash
python3 graph_features.py       # hand-engineered graph features
python3 gnn_from_scratch.py     # learned GCN — node embeddings + node-classification metrics
python3 rnn_from_scratch.py     # RNN on claim event sequences
python3 train_mlp.py            # tabular MLP using graph features + GCN embeddings
```
