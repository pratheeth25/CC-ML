import os
import json
import hmac
import pickle

import pandas as pd
from fastapi import FastAPI, Request, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression_model.pkl")
CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
    THRESHOLD = config["threshold"]

app = FastAPI(title="Customer Churn API", version="1.0.0")

env_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(env_file_path):
    with open(env_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v

cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000")
cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    server_api_key = os.getenv("API_KEY", "")
    if not server_api_key:
        raise HTTPException(status_code=500, detail="API_KEY not configured on server")
    if api_key is None or not hmac.compare_digest(api_key, server_api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return api_key


class CustomerData(BaseModel):
    Age: int = Field(..., ge=0, le=120)
    Gender: str = Field(..., min_length=1, max_length=20)
    Tenure: int = Field(..., ge=0)
    Usage_Frequency: int = Field(..., ge=0, alias="Usage Frequency")
    Support_Calls: int = Field(..., ge=0, alias="Support Calls")
    Payment_Delay: int = Field(..., ge=0, alias="Payment Delay")
    Subscription_Type: str = Field(..., min_length=1, max_length=50, alias="Subscription Type")
    Contract_Length: str = Field(..., min_length=1, max_length=50, alias="Contract Length")
    Total_Spend: float = Field(..., ge=0, alias="Total Spend")
    Last_Interaction: int = Field(..., ge=0, alias="Last Interaction")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    prediction: int
    churn: bool
    probability: float


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


@app.get("/")
async def root():
    return {"message": "Customer Churn API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData, api_key: str = Depends(verify_api_key)):
    row = {
        "Age": customer.Age,
        "Gender": customer.Gender,
        "Tenure": customer.Tenure,
        "Usage Frequency": customer.Usage_Frequency,
        "Support Calls": customer.Support_Calls,
        "Payment Delay": customer.Payment_Delay,
        "Subscription Type": customer.Subscription_Type,
        "Contract Length": customer.Contract_Length,
        "Total Spend": customer.Total_Spend,
        "Last Interaction": customer.Last_Interaction,
    }
    df = pd.DataFrame([row])

    try:
        X_transformed = preprocessor.transform(df)
        feature_names = preprocessor.get_feature_names_out()
        X = pd.DataFrame(X_transformed, columns=feature_names)
        probability = float(model.predict_proba(X)[0][1])
    except Exception:
        raise HTTPException(status_code=422, detail="Could not process input data")

    prediction = int(probability >= THRESHOLD)

    return PredictionResponse(
        prediction=prediction,
        churn=bool(prediction),
        probability=round(probability, 4)
    )
