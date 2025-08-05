from preprocess import encoding
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import os

X_scaled, y, scaler = encoding()
os.makedirs("models", exist_ok=True)

targets_info = {
    "Addiction_Level": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
        "type": "classification"
    },
    "Affects_Academic_Performance": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "type": "classification"
    },
    "Mental_Health_Score": {
        "model": XGBRegressor(),
        "type": "regression"
    },
    "Conflicts_Over_Social_Media": {
        "model": XGBRegressor(),
        "type": "regression"
    }
}

for name, info in targets_info.items():
    print(f"\n Training for: {name}")
    y_target = y[name]
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_target, test_size=0.2, random_state=42)

    model = info["model"]
    model.fit(X_train, y_train)

    model_path = f"models/{name}_model.pkl"
    joblib.dump(model, model_path)
    print(f" Saved model to {model_path}")

    y_pred = model.predict(X_test)

    if info["type"] == "classification":
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
