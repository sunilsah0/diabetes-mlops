import joblib
from sklearn.datasets import load_diabetes

def test_model_loads():
    model = joblib.load("app/model.joblib")
    scaler = joblib.load("app/scaler.joblib")
    assert model is not None
    assert scaler is not None

def test_model_predicts():
    model = joblib.load("app/model.joblib")
    scaler = joblib.load("app/scaler.joblib")
    data = load_diabetes(as_frame=True).data.head(1)
    preds = model.predict(scaler.transform(data))
    assert len(preds) == 1
