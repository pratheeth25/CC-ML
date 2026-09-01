import os
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def clean_and_encode_target(series: pd.Series, name: str = "Churn") -> pd.Series:
    valid_map = {
        "1": 1, "1.0": 1, "yes": 1, "y": 1, "true": 1, 1: 1, 1.0: 1, True: 1,
        "0": 0, "0.0": 0, "no": 0, "n": 0, "false": 0, 0: 0, 0.0: 0, False: 0
    }

    def map_val(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, str):
            cleaned = val.strip().lower()
            if cleaned in valid_map:
                return valid_map[cleaned]
            try:
                num = float(cleaned)
                if num in (0.0, 1.0):
                    return int(num)
            except ValueError:
                pass
            return "UNEXPECTED"
        elif isinstance(val, (int, float, np.integer, np.floating, bool)):
            if val in (1, 1.0, True):
                return 1
            elif val in (0, 0.0, False):
                return 0
            return "UNEXPECTED"
        return "UNEXPECTED"

    mapped = series.apply(map_val)
    unexpected = series[mapped == "UNEXPECTED"].unique()
    if len(unexpected) > 0:
        error_msg = f"ERROR: Found unexpected target value(s) in '{name}' column: {unexpected.tolist()}"
        print(error_msg)
        raise ValueError(error_msg)

    return mapped


def preprocess_data(
    train_path: str = "./data/raw/train.csv",
    test_path: str = "./data/raw/test.csv",
    output_train_path: str = "./data/preprocessed/train_processed.csv",
    output_test_path: str = "./data/preprocessed/test_processed.csv",
    preprocessor_path: str = "./models/preprocessor.pkl"
):
    print("=" * 60)
    print("Starting Preprocessing Pipeline")
    print("=" * 60)

    print(f"Loading raw training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading raw testing data from: {test_path}")
    test_df = pd.read_csv(test_path)

    print("\n--- Target Diagnostics (Before Encoding) ---")
    print(f"Train 'Churn' unique values: {train_df['Churn'].unique().tolist()}")
    print(f"Test 'Churn' unique values: {test_df['Churn'].unique().tolist()}")

    missing_target_train = train_df["Churn"].isna().sum()
    missing_target_test = test_df["Churn"].isna().sum()
    print(f"Number of missing target values in train: {missing_target_train}")
    print(f"Number of missing target values in test: {missing_target_test}")

    if missing_target_train > 0:
        print(f"\nDropping {missing_target_train} row(s) with missing target values from training set...")
        train_df = train_df.dropna(subset=["Churn"]).reset_index(drop=True)

    if missing_target_test > 0:
        print(f"\nDropping {missing_target_test} row(s) with missing target values from testing set...")
        test_df = test_df.dropna(subset=["Churn"]).reset_index(drop=True)

    drop_cols = ["Churn"]
    if "CustomerID" in train_df.columns:
        drop_cols.append("CustomerID")

    X_train = train_df.drop(columns=drop_cols)
    y_train = clean_and_encode_target(train_df["Churn"], name="Churn").astype(int)

    test_drop_cols = ["Churn"]
    if "CustomerID" in test_df.columns:
        test_drop_cols.append("CustomerID")

    X_test = test_df.drop(columns=test_drop_cols)
    y_test = clean_and_encode_target(test_df["Churn"], name="Churn").astype(int)

    print("\n--- Feature Diagnostics (Missing Values) ---")
    print("Train missing feature values per column:")
    train_feat_missing = X_train.isna().sum()
    if (train_feat_missing > 0).any():
        print(train_feat_missing[train_feat_missing > 0])
    else:
        print("No missing feature values in train.")

    print("\nTest missing feature values per column:")
    test_feat_missing = X_test.isna().sum()
    if (test_feat_missing > 0).any():
        print(test_feat_missing[test_feat_missing > 0])
    else:
        print("No missing feature values in test.")

    numeric_cols = X_train.select_dtypes(include=["int64", "float64", "number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"\nIdentified numerical columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Identified categorical columns ({len(categorical_cols)}): {categorical_cols}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                numeric_cols
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]),
                categorical_cols
            )
        ]
    )

    print("\nFitting preprocessing transformations on training set and transforming datasets...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    X_train_df["Churn"] = y_train.values
    X_test_df["Churn"] = y_test.values

    print("\n--- Final Dataset Diagnostics ---")
    print(f"Final shape of processed train dataset: {X_train_df.shape}")
    print(f"Final shape of processed test dataset: {X_test_df.shape}")
    print(f"Train target distribution:\n{X_train_df['Churn'].value_counts().to_dict()}")
    print(f"Test target distribution:\n{X_test_df['Churn'].value_counts().to_dict()}")

    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

    X_train_df.to_csv(output_train_path, index=False)
    X_test_df.to_csv(output_test_path, index=False)

    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)

    print(f"\nSaved processed training data to: {output_train_path}")
    print(f"Saved processed testing data to: {output_test_path}")
    print(f"Saved fitted preprocessor to: {preprocessor_path}")
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess_data()
