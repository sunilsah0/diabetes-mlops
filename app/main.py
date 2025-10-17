from fastapi import FastAPI, HTTPException
from app.schemas import Features
from app.model import load_model, load_scaler
import numpy as np

app = FastAPI()
model = load_model()
scaler = load_scaler()
MODEL_VERSION = "0.1"

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict")
def predict(features: Features):
    try:
        arr = np.array([[getattr(features, f) for f in features.__fields__]])
        arr_scaled = scaler.transform(arr)
        prediction = model.predict(arr_scaled)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
