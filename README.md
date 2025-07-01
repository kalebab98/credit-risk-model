# Task 5 – Model Training and Tracking

## Objective

To build, tune, and evaluate models that predict customer credit risk (`is_high_risk`) using the engineered features. Additionally, to implement MLflow for tracking experiments and metrics.

---

## Workflow Overview

1. Load labeled dataset (`df_final_task4.csv`)
2. Split into training and testing sets
3. Train multiple models:
   - Logistic Regression
   - Random Forest (baseline and tuned)
4. Evaluate performance
5. Track experiments using MLflow
6. Save the best model locally (`best_random_forest_model.pkl`)

---

## Model Training Scripts

**File:** `src/train.py`

### Data Loading

```python
X, y = load_data("df_final_task4.csv")
