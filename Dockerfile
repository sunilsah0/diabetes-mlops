FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
WORKDIR /app
COPY app/ app/
COPY scaler.joblib model.joblib .
RUN pip install --no-cache-dir fastapi[all] gunicorn
EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "app.main:app"]
