# Feature Engineering & Cleaning Report

- Raw rows loaded: **6120**
- Exact duplicate rows dropped: **118**
- Duplicate claim_id rows dropped (after full-dupe removal): **2**
- Outlier claim_amount values capped (IQR winsorization): **443**
- Final clean rows: **6000**
- Fraud rate after cleaning: **0.0772**

## Missing values before imputation

| column | missing_count |
|---|---|
| claimant_age | 133 |
| annual_income | 181 |
| city | 564 |
| gender | 186 |
| occupation | 121 |

## Engineered features

- `claim_to_income_ratio` = claim_amount / annual_income
- `tenure_bucket` = binned policy_tenure_months
- `claim_dow`, `claim_month` = extracted from claim_date
- `is_high_value_claim` = top-decile claim_amount flag
- `desc_length`, `desc_word_count` = free-text signal for NLP handoff
