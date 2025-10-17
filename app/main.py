from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

class Features(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

scaler = joblib.load("app/scaler.joblib")
model = joblib.load("app/model.joblib")
MODEL_VERSION = "0.1"

app = FastAPI()

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
        raise HTTPException(status_code=400, detail=f"Input error: {e}")
