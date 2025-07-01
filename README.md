# Task 4 – Proxy Target Variable Engineering

## Objective

To create a supervised learning target (`is_high_risk`) from customer behavior data. Since no explicit label for default or credit risk exists, we generate a **proxy target** using Recency, Frequency, and Monetary (RFM) analysis combined with **K-Means clustering**.

---

## Methodology

### 1. RFM Metric Calculation

We computed the following metrics per `CustomerId`:

- **Recency**: Days since last transaction (lower = more recent)
- **Frequency**: Total number of transactions
- **Monetary**: Total transaction value (positive or negative)

Snapshot date was chosen to ensure Recency is consistently calculated across all customers.

| Cluster | Recency | Frequency | Monetary        |
|---------|---------|-----------|-----------------|
| 0       | 61.86   | 7.73      | 81,724          |
| 1       | 29.00   | 4091.00   | -104,900,000    |
| 2       | 12.71   | 34.81     | 272,655         |

> 📌 **Insight:** Cluster 0 was identified as **least engaged**, indicating potential risk behavior (low frequency and monetary).

---

### 2. Clustering with K-Means

- Used **K-Means** (k=3) on scaled RFM data.
- Features were scaled with `StandardScaler`.
- Used `random_state=42` for reproducibility.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(rfm_scaled)
