"""
Gold price prediction API.

Wraps the RandomForestRegressor trained in gold.ipynb behind a small
HTTP API so the frontend (index.html) can call it directly.

Run locally:
    uvicorn app:app --reload

Then test:
    curl -X POST http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"SPX": 1654.32, "USO": 31.84, "SLV": 20.08, "EUR_USD": 1.2837}'
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "gold_model.pkl"

# Must match the column order the model was trained on:
# X = gold_data.drop(['Date', 'GLD'], axis=1)  ->  SPX, USO, SLV, EUR/USD
FEATURE_ORDER = ["SPX", "USO", "SLV", "EUR/USD"]

app = FastAPI(title="Gold Price Predictor API")

# Allow the frontend (running on a different domain) to call this API.
# Replace "*" with your actual frontend URL once you know it, e.g.
# ["https://your-site.vercel.app"], to lock this down for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Load the model once at startup, not on every request.
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


class PredictionRequest(BaseModel):
    SPX: float = Field(..., gt=0, description="S&P 500 close")
    USO: float = Field(..., gt=0, description="Oil fund price")
    SLV: float = Field(..., gt=0, description="Silver ETF price")
    EUR_USD: float = Field(..., gt=0, description="EUR/USD exchange rate")


class PredictionResponse(BaseModel):
    prediction: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model file not found on server. Make sure gold_model.pkl "
            "is deployed alongside app.py.",
        )

    row = pd.DataFrame(
        [[request.SPX, request.USO, request.SLV, request.EUR_USD]],
        columns=FEATURE_ORDER,
    )

    try:
        prediction = model.predict(row)[0]
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(prediction=float(prediction))