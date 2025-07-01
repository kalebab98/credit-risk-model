# Credit Risk Probability Model (B5W5 - Bati Bank)

An end-to-end implementation of a credit scoring system that predicts the risk probability of customers using behavioral transaction data. This project was developed as part of Bati Bank's collaboration with an eCommerce platform to enable a Buy-Now-Pay-Later (BNPL) feature.

---

## 🚀 Overview

This project involves building a credit risk scoring model using alternative data, particularly transaction histories. Since traditional default labels are unavailable, we create a **proxy target** using **RFM segmentation** and build interpretable ML pipelines with full CI/CD automation.

---

## 📁 Project Structure

```bash
credit-risk-model/
│
├── .github/workflows/ci.yml        # GitHub Actions CI/CD
├── data/                           # Raw & processed data
│   ├── raw/
│   └── processed/
├── notebooks/                      # EDA & analysis notebooks
│   └── 1.0-eda.ipynb
├── src/                            # All source code
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/                          # Unit tests
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord requires banks to measure credit risk in a consistent, transparent, and explainable way. It emphasizes that models used to estimate risk (like the probability of default) must be well-understood by the institution and regulators. This means that black-box models without explanation are risky in regulated environments. Therefore, it is important to choose models that are both accurate and interpretable, and to document each step clearly—from data processing to feature selection to model assumptions.

In our case, since this credit scoring system may be used for financial decision-making, regulatory expectations mean we must prefer models that are explainable (like logistic regression or decision trees), or at least provide post-hoc explanations for complex models (like using SHAP or LIME for gradient boosting models).

---

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In our dataset, there is no explicit "default" label showing whether a customer failed to repay a loan. Instead, we must create a **proxy variable** that estimates who might be a high-risk (bad) customer. This can be done by using behavior patterns like:
- Long inactivity after a purchase
- Sudden drop in transaction frequency
- High number of reversed or small value transactions

However, creating a proxy introduces risks:
- **Label leakage**: If our proxy doesn’t truly reflect default behavior, we may misclassify good customers as bad, or vice versa.
- **Bias**: If the proxy is based on limited or biased patterns (e.g., only based on one product type), the model might unfairly penalize certain customer segments.
- **Regulatory and ethical risk**: Since this proxy influences financial decisions, poor assumptions could lead to unfair denial of credit.

That’s why it’s critical to validate the proxy against business understanding, consult domain experts, and test performance carefully.

---

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

| Factor | Simple Model (Logistic Regression with WoE) | Complex Model (e.g., Gradient Boosting) |
|--------|---------------------------------------------|------------------------------------------|
| **Interpretability** | Very high – easy to explain to regulators and stakeholders | Low – requires tools like SHAP/LIME to explain |
| **Regulatory Approval** | More likely to be approved and trusted | May face scrutiny due to black-box nature |
| **Training and Debugging** | Simple to train, test, and debug | Harder to understand and debug |
| **Performance** | May be less accurate on complex patterns | Typically higher accuracy and better handling of nonlinearities |
| **Maintenance** | Easier to update and maintain | Requires more resources and monitoring |

In financial services, a **hybrid approach** is common: start with interpretable models for baseline and regulatory reporting, and optionally use complex models for internal decision support if explainability tools are in place.

---
### 📊 Task 4: Model Training and Evaluation
- Implemented train.py to:
  - Load and split data
  - Train Logistic Regression and Random Forest classifiers
  - Tune hyperparameters with RandomizedSearchCV
  - Evaluate models using accuracy, precision, recall, F1, and ROC AUC
  - Log experiments and metrics to MLflow
  - Save the best model as best_random_forest_model.pkl

### 🤖 Prediction Script
- Created predict.py to:
  - Load the saved model
  - Accept preprocessed CSV (preprocessed.csv)
  - Generate predictions and probabilities
  - Output results to predict_out.csv

### 🧪 Testing
- Created test_data_processing.py using pytest to:
  - Validate data loading
  - Check column consistency
  - Ensure data shape and null handling
- Ran all 4 tests successfully ✅

### 🐳 Dockerization
- Wrote Dockerfile to containerize the FastAPI service
- Created docker-compose.yml to manage build and port exposure
- Verified:
  - Docker version: 28.2.2
  - Docker Compose version: v2.37.1-desktop.1

### ⚙️ FastAPI Web Service
- Built a FastAPI app in main.py to:
  - Load model
  - Receive JSON input via POST
  - Return prediction and probability

