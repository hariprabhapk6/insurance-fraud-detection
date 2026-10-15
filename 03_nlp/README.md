# Level: NLP — Natural Language Processing

## Environment constraint & how it was handled
No internet access → spaCy/transformer models unavailable for download. Built with:
- **Preprocessing**: regex-based lowercase/clean tokenization
- **Text classification**: TF-IDF (unigram+bigram, 800 features) + Logistic Regression
- **NER**: rule-based/gazetteer NER (money amounts, locations, organizations, durations) —
  a legitimate, explainable, production-viable approach when a statistical NER model isn't available
- **Keyword extraction**: per-document TF-IDF top-term ranking (feeds Level SLM)

## Results
| Metric | Value |
|---|---|
| Accuracy | 0.981 |
| Precision | 1.000 |
| Recall | 0.753 |
| F1 | 0.859 |
| **ROC-AUC** | **0.890** |

The claim **description text alone** is highly predictive of fraud (AUC 0.89) — almost
as strong as the full tabular Random Forest (0.86) and stronger than the tabular MLP
(0.64). This makes sense: fraudulent claims in this dataset use language patterns
(vague locations, missing documentation, urgency, mismatched details) that a bag-of-words
model picks up cleanly.

## NER limitations (documented, not hidden)
The rule-based MONEY extractor is a simple digit regex and will over-match plain
numbers (e.g. "9 hours" partially matches "9"). In a production system this would be
replaced by a constrained regex requiring a currency symbol/unit, or a proper
statistical NER model once network access to `spaCy`/`transformers` is available.

## How to reproduce
```bash
python3 nlp_pipeline.py
```
