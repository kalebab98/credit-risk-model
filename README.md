# Task 6 – Model Deployment and CI/CD

## Objective

To deploy the best-performing credit risk model as a REST API using **FastAPI**, containerize it with **Docker**, and set up **CI/CD pipelines** using GitHub Actions to ensure testability and code quality.

---

## API Overview

**Framework**: FastAPI  
**Endpoint**: `/predict`  
**Model**: `best_random_forest_model.pkl` (trained in Task 5)

### ✅ Features
- Accepts input features via POST request
- Returns `is_high_risk` prediction and risk probability
- Input schema enforced with **Pydantic**

---

## API Usage

### Endpoint: `POST /predict`

**Sample Request Body:**
```json
{
  "log_amount": 7.2,
  "log_value": 7.2,
  "TransactionHour": 15,
  "TransactionDay": 17,
  "TransactionMonth": 6,
  "TransactionYear": 2025,
  "amount_sum": 25000,
  "amount_mean": 6000,
  "amount_std": 1000,
  "amount_count": 4,
  "ProviderId_ProviderId_2": 0,
  "ProviderId_ProviderId_3": 1,
  "ProviderId_ProviderId_4": 0,
  "ProviderId_ProviderId_5": 0,
  "ProviderId_ProviderId_6": 0,
  "PricingStrategy_0": 1,
  "PricingStrategy_1": 0,
  "PricingStrategy_2": 0,
  "PricingStrategy_4": 0
}
src/
├── api/
│   ├── main.py              # FastAPI app
│   └── pydantic_models.py   # Input schema
best_random_forest_model.pkl # Trained model
Dockerfile
docker-compose.yml
```
