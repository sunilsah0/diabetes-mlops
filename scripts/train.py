import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

SEED = 42

def main():
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=SEED),
        "RandomForestRegressor": RandomForestRegressor(random_state=SEED, n_estimators=100)
    }

    rmse_scores = {}
    for name, m in models.items():
        m.fit(X_train_scaled, y_train)
        preds = m.predict(X_test_scaled)
        rmse = mean_squared_error(y_test, preds, squared=False)
        rmse_scores[name] = rmse

    best_model_name = min(rmse_scores, key=rmse_scores.get)
    best_model = models[best_model_name]

    joblib.dump(best_model, "app/model.joblib")
    joblib.dump(scaler, "app/scaler.joblib")

    with open("model_metrics.txt", "w") as f:
        for name, score in rmse_scores.items():
            f.write(f"{name} RMSE: {score:.4f}\n")
        f.write(f"Best model: {best_model_name}\n")

if __name__ == "__main__":
    main()
