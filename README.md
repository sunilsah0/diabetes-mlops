text
# Virtual Diabetes Clinic MLOps System

This repo builds an automated ML pipeline for a diabetes progression prediction service.

## How to Run
### Locally
pip install -r requirements.txt
python scripts/train.py
uvicorn app.main:app --reload

text

### Docker
docker build -t diabetes-mlops .
docker run -p 8000:8000 diabetes-mlops

text

## Endpoints
- `GET /health` → `{"status":"ok","model_version":"0.1"}`
- `POST /predict` → numeric prediction for JSON input

Example request:
{
"age": 0.02,
"sex": -0.044,
"bmi": 0.06,
"bp": -0.03,
"s1": -0.02,
"s2": 0.03,
"s3": -0.02,
"s4": 0.02,
"s5": 0.02,
"s6": -0.001
}

text
Response:
{"prediction": 142.7}

text
undefined
