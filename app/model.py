import joblib

MODEL_PATH = "app/model.joblib"
SCALER_PATH = "app/scaler.joblib"

def load_model():
    return joblib.load(MODEL_PATH)

def load_scaler():
    return joblib.load(SCALER_PATH)
