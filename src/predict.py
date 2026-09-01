import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(
    test_data_path: str = "./data/preprocessed/test_processed.csv",
    model_path: str = "./models/logistic_regression_model.pkl",
    config_path: str = "./models/model_config.json"
):
    print("=" * 60)
    print("Starting Model Evaluation & Prediction")
    print("=" * 60)

    print(f"Loading trained model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    threshold = config["threshold"]
    print(f"Using classification threshold: {threshold}")

    print(f"Loading preprocessed test data from: {test_data_path}")
    test_df = pd.read_csv(test_data_path)

    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    print(f"Test dataset shape: X_test={X_test.shape}, y_test={y_test.shape}")

    print("\nGenerating predictions on test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 60)
    print(f"MODEL EVALUATION RESULTS (threshold={threshold})")
    print("=" * 60)
    print(f"Accuracy:        {acc:.4f}")
    print(f"Precision:       {prec:.4f}")
    print(f"Recall:          {rec:.4f}")
    print(f"F1-Score:        {f1:.4f}")
    print(f"ROC-AUC Score:   {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()
