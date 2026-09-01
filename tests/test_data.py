import os
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import clean_and_encode_target

EXPECTED_RAW_COLUMNS = [
    "CustomerID", "Age", "Gender", "Tenure", "Usage Frequency",
    "Support Calls", "Payment Delay", "Subscription Type",
    "Contract Length", "Total Spend", "Last Interaction", "Churn"
]


def test_raw_files_exist():
    assert os.path.exists("./data/raw/train.csv"), "data/raw/train.csv missing"
    assert os.path.exists("./data/raw/test.csv"), "data/raw/test.csv missing"


def test_raw_data_schema():
    train_df = pd.read_csv("./data/raw/train.csv", nrows=5)
    test_df = pd.read_csv("./data/raw/test.csv", nrows=5)

    for col in EXPECTED_RAW_COLUMNS:
        assert col in train_df.columns, f"Column '{col}' missing from train.csv"
        assert col in test_df.columns, f"Column '{col}' missing from test.csv"


def test_clean_and_encode_target_valid_inputs():
    series = pd.Series([1.0, 0.0, "yes", "NO", "1", "0", True, False, " Yes "])
    result = clean_and_encode_target(series)
    expected = [1, 0, 1, 0, 1, 0, 1, 0, 1]
    assert result.tolist() == expected


def test_clean_and_encode_target_unexpected_input():
    series = pd.Series(["yes", "invalid_value", "no"])
    with pytest.raises(ValueError) as excinfo:
        clean_and_encode_target(series)
    assert "unexpected target value" in str(excinfo.value).lower()


def test_preprocessed_data_validity():
    assert os.path.exists("./data/preprocessed/train_processed.csv")
    assert os.path.exists("./data/preprocessed/test_processed.csv")

    train_df = pd.read_csv("./data/preprocessed/train_processed.csv")
    test_df = pd.read_csv("./data/preprocessed/test_processed.csv")

    assert "Churn" in train_df.columns
    assert "Churn" in test_df.columns
    assert train_df.isna().sum().sum() == 0, "train_processed.csv contains NaN values"
    assert test_df.isna().sum().sum() == 0, "test_processed.csv contains NaN values"
    assert set(train_df["Churn"].unique()).issubset({0, 1})
    assert set(test_df["Churn"].unique()).issubset({0, 1})
