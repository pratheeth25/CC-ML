# Customer Churn ML Pipeline 🚀

> **Status:** Under Active Development 🛠️

An end-to-end Machine Learning pipeline for customer churn prediction using Scikit-Learn, FastAPI, MLflow, DVC, Docker, GitHub Actions, and Kubernetes.

---

## Tech Stack

| Layer | Tool |
| :--- | :--- |
| **Model** | Scikit-learn (Logistic Regression + Threshold Tuning) |
| **Data & Versioning** | Pandas, DVC |
| **Experiment Tracking** | MLflow |
| **API Serving** | FastAPI, Uvicorn, Pydantic |
| **Testing & Quality** | Pytest (15 tests), Flake8 |
| **Container & Orchestration** | Docker, Kubernetes |
| **CI/CD** | GitHub Actions (`ci.yml`, `cd.yml`), Gitflow |

---

## Project Structure

```text
├── data/              # Raw (.dvc) and preprocessed datasets
├── models/            # Saved preprocessor.pkl, model.pkl, model_config.json
├── src/               # preprocessing.py, train.py, predict.py, api.py
├── tests/             # test_data.py, test_model.py, test_api.py
├── k8s/               # Kubernetes manifests (deployment, service, configmap, secret)
├── .github/workflows/ # GitHub Actions CI/CD pipelines
├── Dockerfile         # API container definition
├── dvc.yaml           # Pipeline DAG stages
└── requirements.txt   # Dependencies
```

---

## Quick Start

### 1. Installation
```powershell
# Create & activate environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Setup env variables
Copy-Item .env.example .env
```

### 2. Run Pipeline
```powershell
# Option A: Python scripts
python src/preprocessing.py
python src/train.py
python src/predict.py

# Option B: DVC pipeline
dvc repro
```

### 3. Run Tests
```powershell
flake8 src/ tests/
pytest tests/ -v
```

### 4. Start API Server
```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

---

## API Usage

### Health Check (`GET /health`)
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health -Method Get | ConvertTo-Json
```

### Predict (`POST /predict`)
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

---

## Docker & Kubernetes

### Docker
```powershell
# Build image
docker build -t churn-api:latest .

# Run container
docker run -d -p 8000:8000 -e API_KEY="my-secret-key-123" -e CORS_ORIGINS="http://localhost:3000" --name churn-app churn-api:latest
```

### Kubernetes
```powershell
# Deploy manifests
kubectl apply -f k8s/

# Port forward
kubectl port-forward service/churn-api-service 8000:8000
```

---

## MLflow Tracking
```powershell
mlflow ui
# Open http://127.0.0.1:5000
```
