# Level: ML — Machine Learning Baseline

## Pipeline
1. `../data/generate_raw_data.py` → produces the messy raw dataset (run once, already executed).
2. `preprocessing.py` → cleans `data/raw/claims_raw.csv` into `data/processed/claims_clean.csv`.
   See `feature_engineering_report.md` for exact cleaning actions taken.
3. `train_ml.py` → trains Logistic Regression + Random Forest, evaluates, saves the best model.

## Results (test set, 20% holdout, stratified)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.647 | 0.103 | 0.462 | 0.169 | 0.596 |
| **Random Forest**   | **0.948** | **0.830** | 0.419 | 0.557 | **0.864** |

Random Forest is selected as the deployed baseline (`best_ml_model.joblib`) — much higher
precision and ROC-AUC, which matters for fraud triage (false positives waste investigator time).

## Top predictive features
`desc_length` / `desc_word_count` (proxy signal that gets formally handled in the NLP level),
`policy_tenure_months` (new policies are riskier), `annual_income`, `claimant_age`,
`claim_to_income_ratio`, `prior_claims_count`, `claim_amount`.

## Common mistakes avoided
- **Data leakage**: `is_fraud` never used to build features; target encoding not used.
- **Class imbalance**: handled via `class_weight="balanced"` rather than naive accuracy optimization.
- **Overfitting**: Random Forest depth capped (`max_depth=10`, `min_samples_leaf=3`).

## How to reproduce
```bash
python3 ../data/generate_raw_data.py
python3 preprocessing.py
python3 train_ml.py
```
