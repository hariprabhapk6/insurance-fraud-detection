# FraudShield AI — Automated Financial Fraud Detection in Insurance Claims

An end-to-end insurance fraud detection project combining Machine Learning, Deep Learning, NLP, Small Language Model techniques, Low-Level Design, Generative AI, and Agentic AI.

## 🚀 Live Application

**Live Demo:** https://fraudshield-ai-delta.vercel.app

**Backend API:** https://fraudshield-ai-backend-5me5.onrender.com

FraudShield AI allows users to submit insurance claim details and receive fraud-risk prediction results through a deployed web application.

- **Frontend:** React + Vite
- **Backend:** Flask REST API
- **Machine Learning:** Scikit-learn
- **Deployment:** Vercel (frontend) and Render (backend)
- **Source Code:** https://github.com/hariprabhapk6/insurance-fraud-detection

> Note: The backend runs on Render's free tier. If it has been idle, its first request may take some time while the service starts.

---

## 📌 Project Overview

Insurance fraud increases financial losses and makes claim investigation more challenging. This project explores multiple computational approaches to identify suspicious insurance claims, analyze claim descriptions, generate explanations, and support human investigation.

The repository contains seven learning and implementation levels:

**ML → DL → NLP → SLM → LLD → GenAI → Agentic AI**

Each level has its own implementation or design deliverable and README documentation.

## 🎯 Objectives

- Detect potentially fraudulent insurance claims using machine learning.
- Explore deep learning techniques for claim sequences and policyholder relationships.
- Analyze claim descriptions using natural language processing.
- Generate summaries and explanations to support investigators.
- Explore synthetic fraud generation and model stress testing.
- Demonstrate an investigation workflow with human oversight.
- Provide a web interface for submitting claims and viewing prediction results.

## 🧰 Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, Random Forest, Logistic Regression |
| Deep learning experiments | NumPy-based RNN, NumPy-based GCN, MLP |
| Natural language processing | TF-IDF, text classification, named entity recognition |
| Summarization | TextRank and grounded explanations |
| Generative AI experiments | Synthetic data generation, Gaussian Mixture Model, Markov-chain text generation |
| Backend | Flask |
| Frontend | React, Vite |
| Deployment | Vercel, Render |
| Version control | Git, GitHub |

## 🧠 Project Pipeline

The project is organized into the following levels:

1. **Machine Learning (ML):** Data cleaning, feature engineering, model training, and evaluation.
2. **Deep Learning (DL):** Claim-event sequence modelling, policyholder graph learning, and MLP experiments.
3. **Natural Language Processing (NLP):** Claim-description classification, entity extraction, and keyword analysis.
4. **SLM techniques:** Text summarization and fact-grounded investigator explanations.
5. **Low-Level Design (LLD):** Architecture, component diagrams, API contracts, and data modelling.
6. **Generative AI (GenAI):** Synthetic fraud generation and stress testing.
7. **Agentic AI:** An investigation workflow with an audit trail and human review.

## 📊 Reported Experimental Results

The following results are documented by the project's existing experiments. They are experimental results on the project's synthetic dataset, not guarantees of real-world fraud detection performance.

| Level | Reported result |
|---|---|
| ML | Random Forest ROC-AUC: **0.864**, precision: **0.83** |
| DL | RNN ROC-AUC: **0.861** |
| DL | GCN AUC: **0.774** |
| DL | MLP AUC: **0.648** |
| NLP | TF-IDF text classifier AUC: **0.890** |
| GenAI | Baseline detected 29.7% of the novel synthetic fraud cases in the documented stress test |
| GenAI | Documented augmentation experiment improved ROC-AUC from **0.864 to 0.890** |
| Agentic AI | Documented experiment surfaced **81.5%** of fraud cases to a human reviewer |

The results above refer to different experimental setups and should not be interpreted as directly comparable without considering their datasets, splits, and evaluation methods.

## 🌐 FraudShield Web Application

The web application provides a frontend for submitting insurance claim information and displaying the backend's prediction response.

### Application architecture

```text
User
  |
  v
React + Vite Frontend
  |
  v
Flask REST API
  |
  v
Loaded Model Artifacts
  |
  v
Fraud-Risk Prediction
  |
  v
Result Displayed to User
```

### Main components

- **Frontend:** `web_app/frontend/`
- **Backend:** `web_app/backend/`
- **ML artifacts:** `01_ml/`
- **NLP artifacts:** `03_nlp/`
- **Processed data:** `data/processed/`

The backend loads the existing model artifacts and processed dataset from the repository. The web application's prediction output depends on the implemented backend logic and loaded artifacts.

## 📁 Repository Structure

```text
fraud_project/
├── data/
│   ├── generate_raw_data.py
│   ├── raw/
│   └── processed/
├── 01_ml/
│   └── ML preprocessing and model training
├── 02_dl/
│   └── RNN, GCN/GNN, graph features, and MLP
├── 03_nlp/
│   └── Text classification and entity extraction
├── 04_slm/
│   └── Summarization and explanations
├── 05_lld/
│   └── LLD.md
├── 06_genai/
│   └── Synthetic generation and stress testing
├── 07_agentic_ai/
│   └── Investigation agent
└── web_app/
    ├── frontend/
    ├── backend/
    └── README.md
```

Each level's directory contains its own implementation or documentation. Consult the relevant subfolder README for exact execution steps and requirements.

## ⚙️ Run Locally

### Prerequisites

- Python installed
- Node.js and npm installed
- Git installed

### 1. Clone the repository

```bash
git clone https://github.com/hariprabhapk6/insurance-fraud-detection.git
cd insurance-fraud-detection
```

### 2. Install backend dependencies

```bash
pip install -r web_app/backend/requirements.txt
```

### 3. Start the backend

From the repository root:

```bash
python web_app/backend/app.py
```

The Flask API runs locally on port `5000` when started with the provided application configuration.

### 4. Start the frontend

Open a second terminal from the repository root:

```bash
cd web_app/frontend
npm install
npm run dev
```

Open the local frontend URL printed by Vite, usually:

http://localhost:5173

The Vite development server is configured to proxy API requests to the local Flask backend.

For additional setup details, see `web_app/README.md`.

## 🔌 API Endpoints

The Flask backend includes these application endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Check backend health |
| `GET /api/schema` | Retrieve claim input schema and available options |
| `POST /api/predict` | Submit claim data for prediction |

The prediction endpoint expects JSON in the format supported by the application's schema. Refer to `web_app/backend/app.py` for the exact request fields and response structure.

### Deployed API

- Health: https://fraudshield-ai-backend-5me5.onrender.com/api/health
- Schema: https://fraudshield-ai-backend-5me5.onrender.com/api/schema

## 🧪 Dataset and Limitations

The repository uses synthetically generated insurance-claim data. The generator deliberately introduces data-quality issues such as missing values, duplicates, inconsistent categories, mixed date formats, and outliers so that preprocessing can be demonstrated.

Important limitations:

- Synthetic data may not represent actual insurance claims or real-world fraud patterns.
- Model performance on synthetic data does not establish performance on real claims.
- Prediction results are decision-support outputs, not proof that a claim is fraudulent.
- Real-world use would require representative, legally obtained data, leakage checks, external validation, calibration, privacy safeguards, fairness evaluation, and human review.
- The documented experimental metrics belong to their respective experiments and should be interpreted in context.

## 🔮 Future Improvements

- Evaluate the models using representative real-world data where legally and ethically available.
- Add more systematic unit tests and API integration tests.
- Improve error handling and monitoring for the deployed application.
- Evaluate prediction calibration and model performance across different claim groups.
- Improve model explanations and investigator feedback workflows.
- Add automated testing and deployment checks.

## 👩‍💻 Author

**Hariprabha**

GitHub: https://github.com/hariprabhapk6

---

*FraudShield AI is an educational and experimental project intended to demonstrate fraud-detection methods and a deployed web application. It is not a replacement for professional claim investigation or an independently validated production fraud-detection system.*
