from fastapi import FastAPI, HTTPException
from pydantic_models import PredictionRequest
import joblib
import numpy as np

app = FastAPI(title="Credit Risk API")
MODEL = joblib.load("best_random_forest_model.pkl")  # adjust path as needed

@app.post("/predict")
def predict(req: PredictionRequest):
    data = np.array([list(req.dict().values())])
    try:
        pred = MODEL.predict(data)[0]
        proba = MODEL.predict_proba(data)[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "predicted_is_high_risk": int(pred),
        "risk_proba": proba
    }
