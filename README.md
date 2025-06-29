# Task 3: Data Preprocessing

## Overview

This module prepares transaction data for machine learning by performing feature engineering, aggregation, and transformation using a scikit-learn pipeline.

## Structure

### 🔧 1. Feature Engineering (`FeatureEngineer`)
- Converts `TransactionStartTime` to datetime.
- Extracts:
  - `TransactionHour`
  - `TransactionDay`
  - `TransactionMonth`
  - `TransactionYear`
- Caps outliers in `Amount` and `Value` at the 1st and 99th percentiles.
- Applies log transformation:
  - `log_amount = log1p(abs(Amount_capped))`
  - `log_value = log1p(Value_capped)`

### 📊 2. Customer-Level Aggregation (`CustomerAggregator`)
- Aggregates transaction data by `CustomerId`:
  - Sum, mean, std, and count for `Amount` and `Value`.
- Merges aggregates back to main dataframe.

### 🔁 3. Preprocessing Pipeline
- Defines:
  - `categorical_features`: `ProductCategory`, `ChannelId`, `ProviderId`, `PricingStrategy`
  - `numerical_features`: log-transformed, time, and aggregate columns

- **Numerical pipeline**:
  - Impute with mean
  - Standard scaling

- **Categorical pipeline**:
  - Impute with most frequent
  - One-hot encoding

- Uses `ColumnTransformer` to combine pipelines.
- Entire preprocessing is encapsulated in `build_preprocessing_pipeline()` function.

## Usage

```python
from src.data_processing import build_preprocessing_pipeline

pipeline = build_preprocessing_pipeline()
X_processed = pipeline.fit_transform(df)
