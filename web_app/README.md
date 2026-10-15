# FraudShield AI website

A React + Vite website and a small Flask API that sit **next to** your ML project.
Nothing in `01_ml` … `07_agentic_ai` or `data/` was changed.

```
web_app/
├── backend/app.py                    Flask API: loads your saved .joblib models (read-only)
├── frontend/                         React + Vite site (plain CSS)
│   └── src/{components,pages,services,styles,data}
└── scripts/export_frontend_data.py   Copies verified metrics from your *_results.json files
```

## Run it (two terminals, both start in the `fraud_project` folder)

**1. Backend** (http://127.0.0.1:5000)
```bash
pip install -r web_app/backend/requirements.txt
python3 web_app/backend/app.py
```

**2. Frontend** (http://localhost:5173)
```bash
cd web_app/frontend
npm install
npm run dev
```
Open http://localhost:5173. Vite forwards `/api` requests to the backend.

**One-command alternative:** `npm run build` in `web_app/frontend`, then run only the backend and open
http://127.0.0.1:5000 (Flask serves the built site).

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Is the service up |
| GET | `/api/schema` | Dropdown options and training ranges, read from the trained model |
| POST | `/api/predict` | Score one claim |

`POST /api/predict` body: `claim_type, claim_date (YYYY-MM-DD), claim_amount, claim_description, claimant_age, gender, occupation, city, annual_income, policy_tenure_months, prior_claims_count`.
Response: `predicted_class, fraud_probability, risk_level, indicators, text_signal, derived_features, warnings`.

## Refreshing the Insights numbers
If you retrain, run `python3 web_app/scripts/export_frontend_data.py` from `fraud_project/`, then rebuild.
