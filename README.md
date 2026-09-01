# Customer Churn Prediction — End-to-End MLOps Pipeline

A production-ready, beginner-friendly Machine Learning project demonstrating the complete MLOps lifecycle: from raw data preprocessing and threshold-tuned Logistic Regression to automated testing, MLflow tracking, DVC data pipelines, secure FastAPI inference, Docker containerization, GitHub Actions CI/CD, and Kubernetes orchestration.

---

## Architecture Overview

```text
               ┌────────────────────────────────────────────────────────┐
               │                  END-TO-END WORKFLOW                   │
               └────────────────────────────────────────────────────────┘

    Raw Data (data/raw/)
            │
            ▼
    [DVC Pipeline / src/preprocessing.py]
            │  • Target encoding (handles whitespace, casing, missing targets)
            │  • SimpleImputer (median for numeric, mode for categorical)
            │  • StandardScaler & OneHotEncoder(handle_unknown="ignore")
            │  • Outputs: preprocessor.pkl & processed CSVs
            ▼
    [Model Training / src/train.py]
            │  • 80/20 Stratified validation split
            │  • Threshold sweep optimizing Class-1 F1 Score
            │  • MLflow parameter, metric, and artifact logging
            │  • Outputs: logistic_regression_model.pkl & model_config.json
            ▼
    [Evaluation & Tests / src/predict.py & pytest]
            │  • Test set evaluation using tuned threshold
            │  • 15 automated Pytest unit/integration tests
            │  • Flake8 linting
            ▼
    [FastAPI Inference API / src/api.py]
            │  • Preprocesses raw JSON payload on the fly
            │  • Security: X-API-Key auth (hmac.compare_digest), CORS, Pydantic validation
            │  • Endpoints: GET /, GET /health, POST /predict
            ▼
    [Docker Containerization / Dockerfile]
            │  • Non-root user (appuser), python:3.11-slim
            │  • Listens on 0.0.0.0:8000
            ▼
    [GitHub Actions CI/CD / .github/workflows/]
            │  • CI (ci.yml): Lint (flake8) + Automated Tests (pytest) on PRs
            │  • CD (cd.yml): Docker build + Container smoke test + K8s dry-run on main
            ▼
    [Kubernetes Orchestration / k8s/]
               • ConfigMap (CORS_ORIGINS) & Secret (API_KEY)
               • Deployment (2 replicas, health probes, resource limits)
               • Service (NodePort 30080 -> 8000)
```

---

## Project Structure

```text
C:\cc\
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Continuous Integration (Lint + Pytest)
│       └── cd.yml                 # Continuous Deployment (Docker build + smoke test)
│
├── data/
│   ├── raw/                       # Raw datasets (tracked by DVC)
│   │   ├── train.csv.dvc
│   │   └── test.csv.dvc
│   └── preprocessed/              # Preprocessed CSVs (ignored by git)
│
├── k8s/                           # Kubernetes Manifests
│   ├── configmap.yaml             # Non-sensitive configuration (CORS_ORIGINS)
│   ├── secret.yaml                # Sensitive credentials (API_KEY)
│   ├── deployment.yaml            # Deployment with 2 replicas & health probes
│   └── service.yaml               # NodePort service on port 30080
│
├── models/                        # Saved artifacts
│   ├── preprocessor.pkl           # Fitted ColumnTransformer pipeline
│   ├── logistic_regression_model.pkl
│   └── model_config.json          # Selected decision threshold
│
├── src/                           # Core source code
│   ├── preprocessing.py           # Preprocessing & feature engineering pipeline
│   ├── train.py                   # Logistic Regression training + MLflow tracking
│   ├── predict.py                 # Evaluation & metrics report
│   └── api.py                     # Secured FastAPI application
│
├── tests/                         # Pytest test suite (15 tests)
│   ├── test_data.py               # Data validation & target encoding tests
│   ├── test_model.py              # Model artifact & inference tests
│   └── test_api.py                # FastAPI endpoints & security tests
│
├── .dockerignore                  # Files excluded from Docker image
├── .env.example                   # Example environment variable template
├── .flake8                        # Flake8 linting rules
├── .gitignore                     # Git ignore rules
├── Dockerfile                     # Lightweight container definition
├── dvc.yaml                       # Reproducible DVC pipeline stages
├── dvc.lock                       # DVC pipeline lockfile
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11 / 3.12 / 3.13)
- Git & Docker (optional for containerization)

### 2. Clone & Install Dependencies
```powershell
# Clone the repository
git clone https://github.com/your-username/customer-churn-mlops.git
cd customer-churn-mlops

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy the example environment file:
```powershell
Copy-Item .env.example .env
```
Edit `.env` if desired:
```ini
API_KEY=my-secret-key-123
CORS_ORIGINS=http://localhost:3000
```

---

## How to Run the Project

### Option 1: Run the Full Pipeline Locally

#### Step 1: Preprocess Data
```powershell
python src/preprocessing.py
```
*Handles missing values, encodes categories, standardizes features, and saves `models/preprocessor.pkl`.*

#### Step 2: Train Model & Track with MLflow
```powershell
python src/train.py
```
*Sweeps classification thresholds (0.05–0.95) on an 80/20 validation split to maximize Class-1 F1 score, retrains on full data, and logs parameters, metrics, and models to MLflow.*

#### Step 3: Evaluate on Test Set
```powershell
python src/predict.py
```
*Evaluates model using the tuned threshold from `models/model_config.json` and prints the classification report and ROC-AUC.*

#### Step 4: Run Automated Tests
```powershell
# Lint check
flake8 src/ tests/

# Run all 15 Pytest tests
pytest tests/ -v
```

---

### Option 2: Run via DVC Pipeline
Reproduce the entire DAG pipeline (`preprocess` ➔ `train` ➔ `evaluate`) with a single command:
```powershell
dvc repro
```
Check pipeline status:
```powershell
dvc status
```

---

## Running & Testing the FastAPI Server

### 1. Start the API Server
```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Test Endpoints

#### Health Check (`GET /health`)
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health -Method Get | ConvertTo-Json
```
**Response:**
```json
{
    "status": "healthy"
}
```

#### Make a Prediction (`POST /predict`)
```powershell
$headers = @{
    "X-API-Key"    = "my-secret-key-123"
    "Content-Type" = "application/json"
}

$body = '{
    "Age": 45,
    "Gender": "Male",
    "Tenure": 24,
    "Usage Frequency": 10,
    "Support Calls": 7,
    "Payment Delay": 20,
    "Subscription Type": "Basic",
    "Contract Length": "Monthly",
    "Total Spend": 500,
    "Last Interaction": 5
}'

Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -Headers $headers -Body $body | ConvertTo-Json
```

**Response:**
```json
{
    "prediction": 1,
    "churn": true,
    "probability": 1.0
}
```

---

## Running with Docker

### 1. Build the Docker Image
```bash
docker build -t churn-api:latest .
```

### 2. Run the Container
```bash
docker run -d -p 8000:8000 \
  -e API_KEY="my-secret-key-123" \
  -e CORS_ORIGINS="http://localhost:3000" \
  --name churn-api-app \
  churn-api:latest
```

### 3. Verify Container Health
```bash
curl http://localhost:8000/health
```

---

## Deploying on Kubernetes

Deploy to any local or remote Kubernetes cluster (Minikube, Docker Desktop K8s, Kind, K3s):

```bash
# Apply all Kubernetes manifests
kubectl apply -f k8s/

# Verify Pods and Services
kubectl get pods
kubectl get svc

# Port-forward to access locally
kubectl port-forward service/churn-api-service 8000:8000
```

---

## Experiment Tracking with MLflow

Launch the MLflow interactive web dashboard:
```powershell
mlflow ui
```
Open **`http://127.0.0.1:5000`** in your browser to view:
- Training runs, parameters (`max_iter`, `val_split_ratio`), and validation metrics (`val_f1_score`, `val_roc_auc`).
- Saved model packages and artifact files.

---

## Security Best Practices Implemented

| Security Layer | Implementation Details |
| :--- | :--- |
| **API Key Authentication** | Requires `X-API-Key` header verified against `API_KEY` environment variable. |
| **Timing Attack Resistance** | Uses constant-time comparison `hmac.compare_digest()` to prevent latency-based key guessing. |
| **No Hardcoded Secrets** | Credentials loaded strictly via `.env` or system environment variables. |
| **Pydantic Input Validation** | Validates and sanitizes input ranges (e.g. `Age >= 0`, `Total Spend >= 0`) before hitting the ML pipeline. |
| **Information Disclosure Prevention** | Custom global exception handler catches errors and returns sanitized `{"detail": "Internal server error"}` without exposing internal tracebacks. |
| **CORS Protection** | Origins restricted via `CORS_ORIGINS` environment variable. |
| **Non-Root Docker Execution** | Runs under a dedicated non-root user `appuser` inside the container. |
| **Attack Surface Minimization** | `.dockerignore` excludes git history, raw datasets, virtualenvs, and test caches. |

---

## CI/CD Workflow (Gitflow)

- **`develop` Branch**: Active development branch. Pull Requests trigger the **CI Pipeline** ([`.github/workflows/ci.yml`](file:///c:/cc/.github/workflows/ci.yml)) which runs Flake8 linting and the Pytest test suite.
- **`main` Branch**: Production branch. Merges to `main` trigger the **CD Pipeline** ([`.github/workflows/cd.yml`](file:///c:/cc/.github/workflows/cd.yml)) which builds the Docker image, performs container smoke tests, and dry-run validates Kubernetes manifests.
