import pandas as pd
import pytest

# Set the path to your preprocessed file
DATA_PATH = "preprocess.csv"

@pytest.fixture
def data():
    """Fixture to load data once for all tests."""
    return pd.read_csv(DATA_PATH)

def test_data_loaded(data):
    """Ensure the file loads correctly and is not empty."""
    assert not data.empty, "Preprocessed data is empty"

def test_expected_columns_exist(data):
    """Check if essential expected columns exist."""
    required_columns = [
        "log_amount", "log_value", "TransactionHour", "TransactionDay",
        "TransactionMonth", "TransactionYear", "amount_sum", "amount_mean",
        "amount_std", "amount_count"
    ]
    for col in required_columns:
        assert col in data.columns, f"Missing column: {col}"

def test_no_null_values(data):
    """Ensure there are no missing/null values in the data."""
    assert data.isnull().sum().sum() == 0, "Data contains null values"

def test_customer_id_column_format(data):
    """Ensure CustomerId column exists and looks correctly formatted."""
    assert "CustomerId" in data.columns, "CustomerId column is missing"
    sample = data["CustomerId"].iloc[0]
    assert sample.startswith("CustomerId_"), f"Unexpected format: {sample}"
