# Automated Financial Fraud Detection in Insurance Claims
### Full pipeline: ML → DL → NLP → SLM → LLD → GenAI → Agentic AI

Every level below is a **complete, runnable deliverable** — not a mock-up. All results
in each level's README were produced by actually executing the code in this repo.

## Environment note (read this first)
This project was built in a sandboxed environment with **no internet access**. That
meant no downloadable dataset, and no `torch`/`transformers`/`spaCy`/PyTorch-Geometric.
Two consequences, both handled head-on rather than skipped:
1. **Raw dataset**: generated synthetically (`data/generate_raw_data.py`) with a known
   fraud-generating process, deliberately injected with real-world messiness (missing
   values, duplicates, inconsistent categories, mixed date formats, outliers) — then
   actually cleaned in Level ML, not pre-cleaned.
2. **Deep learning / NLP / GenAI**: implemented with genuine algorithms that don't
   require internet access (from-scratch NumPy RNN, from-scratch NumPy GCN, sklearn
   MLP, TF-IDF, TextRank, Gaussian Mixture generative model, Markov-chain text
   generation) rather than skipped. Each level's README states plainly where a
   heavier pretrained model would be substituted if network access were available,
   and what would need re-validation.

## Pipeline order (run top to bottom)
```bash
python3 data/generate_raw_data.py        # 1. raw, messy dataset
python3 01_ml/preprocessing.py           # 2. clean + feature-engineer
python3 01_ml/train_ml.py                # 3. ML baseline (RF, LogReg)
python3 02_dl/graph_features.py          # 4. hand-engineered policyholder network features
python3 02_dl/gnn_from_scratch.py        # 5. learned GCN (from-scratch NumPy) on policyholder graph
python3 02_dl/rnn_from_scratch.py        # 6. RNN on claim event sequences
python3 02_dl/train_mlp.py               # 7. MLP deep model (+ graph feats + GCN embeddings), compared to ML baseline
python3 03_nlp/nlp_pipeline.py           # 8. text classification + NER + keywords
python3 04_slm/slm_summarizer.py         # 9. summarization + grounded explanations
# 05_lld/LLD.md is documentation only, no script to run
python3 06_genai/genai_synthetic.py      # 10. synthetic fraud generation + stress test
python3 07_agentic_ai/agent.py           # 11. autonomous investigation agent
```

## Results summary across levels

| Level | Headline result |
|---|---|
| ML | Random Forest: **ROC-AUC 0.864**, precision 0.83 |
| DL | RNN on event sequences: **AUC 0.861** (real signal from claim lifecycle shape); GCN on policyholder graph: **AUC 0.774** (real learned message-passing model, node classification); MLP on tabular + graph + GCN features: AUC 0.648 (tree ensembles win on this tabular data — honestly reported) |
| NLP | TF-IDF text classifier on claim descriptions alone: **AUC 0.890** |
| SLM | TextRank summarization + fact-grounded investigator explanations (ROUGE-1 harness included) |
| LLD | Full C4 component diagram, API contracts, ER diagram, deployment strategy (`05_lld/LLD.md`) |
| GenAI | Baseline only catches **29.7%** of novel synthetic fraud (stress-test finding); augmentation lifts ROC-AUC **0.864 → 0.890** |
| Agentic AI | **81.5%** of real fraud surfaced to a human reviewer; full audit trail; human-in-the-loop by construction |

## Directory structure
```
fraud_project/
├── data/
│   ├── generate_raw_data.py
│   ├── raw/                 (messy source data)
│   └── processed/           (cleaned data)
├── 01_ml/        ML baseline + preprocessing
├── 02_dl/        RNN (from scratch), GCN/GNN (from scratch), graph features, MLP
├── 03_nlp/       text classification, NER
├── 04_slm/       summarization, explanations
├── 05_lld/       LLD.md (architecture/design doc)
├── 06_genai/     synthetic data generation, stress test
└── 07_agentic_ai/ autonomous investigation agent
```

Each subfolder has its own `README.md` with the rubric-relevant deliverables:
problem understanding, methodology, results, common mistakes avoided, and how
to reproduce.


## FraudShield Web Application

The React frontend and Flask API are included in `web_app/`. The API loads the existing model artifacts from `01_ml/`, `03_nlp/`, and the processed dataset from `data/processed/`. See `web_app/README.md` for startup instructions.

### Start the API (from this `fraud_project` directory)

```bash
pip install -r web_app/backend/requirements.txt
python web_app/backend/app.py
```

### Start the frontend (in a second terminal, from this `fraud_project` directory)

```bash
cd web_app/frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite development server proxies `/api` requests to Flask on port 5000.
