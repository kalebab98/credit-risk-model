# Task 2: Exploratory Data Analysis (EDA)

## Overview

This task involves understanding the structure, distribution, and relationships in the transaction dataset to uncover patterns and potential data quality issues. Key steps included data loading, summary statistics, handling outliers, and visualization.

## Steps Performed

### 📥 1. Data Loading
- Loaded `data.csv` using `pandas`.
- Verified no missing values using `.isnull().sum()`.

### 🧾 2. Dataset Summary
- Total records: **95,662**
- Columns: 16 (object, int, and float types)
- Used `.info()` and `.describe()` for an overview.

### 🔍 3. Unique Values
- Checked unique categories in `ProductCategory`, `ProviderId`, and `ChannelId`.

### 📊 4. Outlier Handling
- Used IQR method to cap outliers in `Amount` and `Value`.
- Created new columns: `Amount_capped`, `Value_capped`.

### 📉 5. Distributions & Boxplots
- Visualized:
  - Original and capped distributions of `Amount` and `Value`.
  - Boxplots for detecting extreme values.

### 📈 6. Correlation Analysis
- Plotted heatmap for `Amount_capped`, `Value_capped`, `PricingStrategy`, and `FraudResult`.

### 📊 7. Categorical Distributions
- Count plots for:
  - `CurrencyCode`
  - `ProductCategory`
  - `ProviderId`
  - `ChannelId`

## Key Insights

- No missing values detected.
- Some negative values in `Amount`, likely due to reversals/refunds.
- All transactions occur in country code 256 (likely Uganda).
- `FraudResult` is highly imbalanced.
- Visual patterns suggest correlation between value-based features and fraud flag.

## Output
- Saved visualizations and insights for modeling.
