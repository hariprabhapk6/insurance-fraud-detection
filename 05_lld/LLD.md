# Level: LLD — Low-Level Design
## Automated Financial Fraud Detection in Insurance Claims

## 1. System Architecture (C4 — Component level)

```mermaid
flowchart TB
    subgraph Ingestion
        A[Claim Submission API] --> B[Kafka: claims.raw topic]
    end

    subgraph Feature_Layer[Feature Engineering Layer]
        B --> C[Stream Processor: cleaning + feature engineering]
        C --> D[(Feature Store: Redis - online / Parquet - offline)]
    end

    subgraph Scoring_Layer[Real-Time Scoring Layer]
        D --> E[ML Service: RandomForest baseline]
        D --> F[DL Service: RNN sequence model + Graph features]
        C --> G[NLP Service: TF-IDF text classifier + NER]
        E & F & G --> H[Score Aggregator: weighted ensemble]
    end

    subgraph Decisioning
        H --> I{Fraud Score Threshold}
        I -- low --> J[Auto-approve]
        I -- medium --> K[SLM Service: summary + explanation]
        I -- high --> L[Agentic AI: investigation workflow]
        K --> M[Investigator Queue]
        L --> M
    end

    subgraph GenAI_Offline[Offline: GenAI]
        N[Synthetic Data Generator] --> O[(Training Data Augmentation Store)]
        O --> E
        O --> F
    end

    M --> P[(Case Management DB)]
    L --> Q[External Systems: Police DB / Bank verification API]
```

## 2. Component Design Specifications

| Component | Responsibility | Tech |
|---|---|---|
| Claim Submission API | Validates & accepts new claims | REST, sync, p99 < 200ms |
| Stream Processor | Cleaning (see `01_ml/preprocessing.py`), feature engineering | Kafka Streams / Spark Structured Streaming |
| Feature Store | Low-latency feature lookup for scoring | Redis (online), Parquet/Delta (offline/training) |
| ML Service | Tabular baseline scoring | Random Forest (`01_ml/best_ml_model.joblib`), served via a lightweight model server |
| DL Service | Sequence (RNN) + graph-feature scoring | NumPy RNN weights (`02_dl/rnn_from_scratch.py`), graph features precomputed nightly |
| NLP Service | Text classification + NER | TF-IDF + LogisticRegression (`03_nlp/text_classifier.joblib`) |
| Score Aggregator | Weighted ensemble of ML/DL/NLP scores into one fraud probability | Simple weighted average, versioned weights |
| SLM Service | Summarizes claim + generates grounded explanation for medium-risk claims | `04_slm/slm_summarizer.py` logic, cache-backed |
| Agentic AI | Autonomous multi-step investigation for high-risk claims | `07_agentic_ai/agent.py` — perceive → reason → plan → act loop |
| GenAI Service | Generates synthetic fraud patterns to augment rare-class training data | `06_genai/genai_synthetic.py`, offline batch job |

## 3. API Contracts (OpenAPI-style summary)

### POST /v1/claims
```json
Request:
{
  "policyholder_id": "PH11413",
  "claim_type": "Auto",
  "claim_amount": 47710.50,
  "claim_description": "text...",
  "claim_date": "2025-06-01"
}
Response 202 Accepted:
{ "claim_id": "CLM503687", "status": "SCORING_IN_PROGRESS" }
```

### GET /v1/claims/{claim_id}/score
```json
Response 200:
{
  "claim_id": "CLM503687",
  "fraud_probability": 0.87,
  "component_scores": { "ml": 0.79, "dl_rnn": 0.91, "nlp_text": 0.88 },
  "decision": "ESCALATE_TO_INVESTIGATION",
  "explanation": "Investigator note: ...grounded rationale..."
}
```

### POST /v1/agent/investigate/{claim_id}
Triggers the Agentic AI workflow (see Level: Agentic AI). Returns a case report
with recommended action (`APPROVE` / `REJECT` / `ESCALATE_HUMAN`) and an audit trail
of every tool call the agent made.

## 4. Data Models

```mermaid
erDiagram
    POLICYHOLDER ||--o{ CLAIM : files
    POLICYHOLDER ||--o{ NETWORK_EDGE : "connected via agent/address/bank"
    CLAIM ||--o{ CLAIM_EVENT : "lifecycle events"
    CLAIM ||--|| FRAUD_SCORE : "scored by"
    CLAIM ||--o| INVESTIGATION_CASE : "escalated to"

    POLICYHOLDER {
        string policyholder_id PK
        int age
        float annual_income
        string occupation
        string agent_id
        string bank_account
    }
    CLAIM {
        string claim_id PK
        string policyholder_id FK
        string claim_type
        date claim_date
        float claim_amount
        string claim_description
        int is_fraud
    }
    CLAIM_EVENT {
        string claim_id FK
        int step
        string event_type
        datetime event_time
    }
    FRAUD_SCORE {
        string claim_id FK
        float ml_score
        float dl_score
        float nlp_score
        float final_score
    }
    INVESTIGATION_CASE {
        string case_id PK
        string claim_id FK
        string agent_decision
        string audit_trail_json
    }
```

## 5. Non-Functional Requirements
- **Throughput**: system must sustain 500 claims/sec at peak (auto-renewal season)
- **Latency**: p99 scoring latency < 300ms for ML/NLP path; async for Agentic path (SLA: case report within 5 min)
- **Scalability**: Stream Processor and ML/NLP services horizontally scalable (stateless, feature store is the shared state)
- **Reliability**: at-least-once claim processing via Kafka; idempotent scoring keyed by `claim_id`
- **Security**: PII (bank_account, income) encrypted at rest; access to raw claim text audit-logged
- **Observability**: per-component score distributions monitored for drift; alert if fraud rate deviates > 2σ from rolling baseline

## 6. Deployment Strategy
- Containerized services (ML/DL/NLP/SLM/Agent) on Kubernetes, each independently scalable
- Model artifacts versioned in a model registry; canary rollout for new model versions (5% → 25% → 100% traffic)
- Feature store: Redis cluster (online), Delta Lake on object storage (offline/training)
- Nightly batch job recomputes graph features (`02_dl/graph_features.py`) and retrains on GenAI-augmented data

## 7. Common Mistakes Avoided
- Non-functional requirements (throughput/latency/security) specified up front, not as an afterthought
- Idempotency key (`claim_id`) prevents duplicate scoring under Kafka at-least-once delivery
- Explanation service is grounded-template based specifically to avoid unauditable free-form generation reaching investigators
