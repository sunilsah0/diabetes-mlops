import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

SEED = 42

def main():
    diabetes = load_diabetes(as_frame=True)
    X, y = diabetes.frame.drop(columns=["target"]), diabetes.frame["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"RMSE: {rmse}")

    joblib.dump(scaler, "app/scaler.joblib")
    joblib.dump(model, "app/model.joblib")
    with open("model_metrics.txt", "w") as f:
        f.write(f"RMSE: {rmse}\n")

if __name__ == "__main__":
    main()
