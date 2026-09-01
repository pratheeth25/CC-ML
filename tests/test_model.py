import os
import json
import pickle
import pytest
import pandas as pd
from sklearn.metrics import roc_auc_score


def test_model_artifacts_exist():
    assert os.path.exists("./models/preprocessor.pkl"), "models/preprocessor.pkl missing"
    assert os.path.exists("./models/logistic_regression_model.pkl"), "models/logistic_regression_model.pkl missing"
    assert os.path.exists("./models/model_config.json"), "models/model_config.json missing"


def test_model_config_threshold():
    with open("./models/model_config.json", "r") as f:
        config = json.load(f)
    assert "threshold" in config, "'threshold' key missing from model_config.json"
    threshold = config["threshold"]
    assert isinstance(threshold, (int, float)), "threshold must be numeric"
    assert 0.0 < threshold < 1.0, f"threshold must be between 0 and 1, got {threshold}"


def test_model_inference_pipeline():
    with open("./models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("./models/logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)

    sample_raw = pd.DataFrame([{
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
    }])

    transformed = preprocessor.transform(sample_raw)
    feature_names = preprocessor.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=feature_names)

    proba = model.predict_proba(transformed_df)[:, 1]
    assert len(proba) == 1
    assert 0.0 <= proba[0] <= 1.0


@pytest.mark.skipif(
    not os.path.exists("./data/preprocessed/test_processed.csv"),
    reason="Preprocessed data tracked by DVC"
)
def test_model_performance_baseline():
    with open("./models/logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)

    test_df = pd.read_csv("./data/preprocessed/test_processed.csv")
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    assert auc >= 0.65, f"ROC-AUC score {auc:.4f} is below baseline of 0.65"
