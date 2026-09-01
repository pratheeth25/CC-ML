import os
from fastapi.testclient import TestClient

os.environ.setdefault("API_KEY", "test-api-key-456")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000")

from src.api import app  # noqa: E402

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "Age": 45,
    "Gender": "Male",
    "Tenure": 24,
    "Usage Frequency": 10,
    "Support Calls": 7,
    "Payment Delay": 20,
    "Subscription Type": "Basic",
    "Contract Length": "Monthly",
    "Total Spend": 500.0,
    "Last Interaction": 5
}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Customer Churn API"}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_without_api_key():
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_predict_with_invalid_api_key():
    headers = {"X-API-Key": "wrong-key"}
    response = client.post("/predict", json=SAMPLE_CUSTOMER, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_predict_with_valid_api_key():
    headers = {"X-API-Key": "test-api-key-456"}
    response = client.post("/predict", json=SAMPLE_CUSTOMER, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in (0, 1)
    assert "churn" in data
    assert isinstance(data["churn"], bool)
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_validation_error():
    headers = {"X-API-Key": "test-api-key-456"}
    invalid_payload = SAMPLE_CUSTOMER.copy()
    invalid_payload["Age"] = -5
    response = client.post("/predict", json=invalid_payload, headers=headers)
    assert response.status_code == 422
