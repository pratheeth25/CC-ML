import os
import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score


def find_best_threshold(model, X_val, y_val):
    probas = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.05, 0.96, 0.05)

    print("\n--- Threshold Sweep on Validation Set ---")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)

    best_threshold = 0.5
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0

    for t in thresholds:
        preds = (probas >= t).astype(int)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        print(f"{t:>10.2f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_prec = prec
            best_rec = rec
            best_threshold = round(float(t), 2)

    print("-" * 45)
    print(f"Best threshold: {best_threshold} (class-1 F1 = {best_f1:.4f})")
    return best_threshold, best_f1, best_prec, best_rec


def train_model(
    train_data_path: str = "./data/preprocessed/train_processed.csv",
    model_output_path: str = "./models/logistic_regression_model.pkl",
    config_output_path: str = "./models/model_config.json",
    experiment_name: str = "customer_churn_prediction"
):
    print("=" * 60)
    print("Starting Model Training with MLflow Tracking")
    print("=" * 60)

    mlflow.set_experiment(experiment_name)

    print(f"Loading preprocessed training data from: {train_data_path}")
    train_df = pd.read_csv(train_data_path)

    X = train_df.drop("Churn", axis=1)
    y = train_df["Churn"]

    print(f"Full training dataset shape: X={X.shape}, y={y.shape}")
    print(f"Target missing values: {y.isna().sum()}")

    val_split_ratio = 0.2
    max_iter = 1000
    random_state = 42

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split_ratio, random_state=random_state, stratify=y
    )
    print(f"Train split: {X_train.shape[0]} rows | Validation split: {X_val.shape[0]} rows")

    with mlflow.start_run(run_name="logistic_regression_run") as run:
        run_id = run.info.run_id
        print(f"\n[MLflow] Started Run ID: {run_id}")

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("val_split_ratio", val_split_ratio)
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("train_samples", X.shape[0])

        print("\nTraining Logistic Regression on train split...")
        val_model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        val_model.fit(X_train, y_train)

        best_threshold, best_f1, best_prec, best_rec = find_best_threshold(val_model, X_val, y_val)

        val_probs = val_model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
        val_acc = accuracy_score(y_val, (val_probs >= best_threshold).astype(int))

        mlflow.log_metric("val_best_threshold", best_threshold)
        mlflow.log_metric("val_f1_score", best_f1)
        mlflow.log_metric("val_precision", best_prec)
        mlflow.log_metric("val_recall", best_rec)
        mlflow.log_metric("val_roc_auc", val_auc)
        mlflow.log_metric("val_accuracy", val_acc)

        print(f"\nRetraining on FULL training set ({X.shape[0]} rows)...")
        model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        model.fit(X, y)

        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

        with open(model_output_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to: {model_output_path}")

        config = {"threshold": best_threshold}
        with open(config_output_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Config saved to: {config_output_path} (threshold={best_threshold})")

        mlflow.log_artifact(model_output_path, artifact_path="model_files")
        mlflow.log_artifact(config_output_path, artifact_path="model_files")
        if os.path.exists("./models/preprocessor.pkl"):
            mlflow.log_artifact("./models/preprocessor.pkl", artifact_path="model_files")

        mlflow.sklearn.log_model(model, artifact_path="sklearn_model")

        print(f"\n[MLflow] Run successfully logged to experiment '{experiment_name}'.")
        print(f"[MLflow] Run ID: {run_id}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    train_model()
