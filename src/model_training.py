import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_model(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

    # Ensure 'models' directory exists
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/random_forest_model.pkl")
    print("Model saved to models/random_forest_model.pkl")
