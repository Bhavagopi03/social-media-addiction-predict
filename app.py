from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)


models = {
    "Addiction_Level": joblib.load("models/Addiction_Level_model.pkl"),
    "Affects_Academic_Performance": joblib.load("models/Affects_Academic_Performance_model.pkl"),
    "Mental_Health_Score": joblib.load("models/Mental_Health_Score_model.pkl"),
    "Conflicts_Over_Social_Media": joblib.load("models/Conflicts_Over_Social_Media_model.pkl")
}
scaler = joblib.load("models/scaler.pkl")
target_decoder = joblib.load("models/target_encoder_addiction.pkl")

@app.route("/")
def home():
    return render_template("smaapp.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
       
        age = float(request.form.get("age") or 0)
        gender = float(request.form.get("gender") or 0)
        usage = float(request.form.get("daily_usage") or 0)
        sleep = float(request.form.get("sleep_hours") or 0)
        platform = float(request.form.get("platform") or 0)
        relationship = float(request.form.get("relationship") or 0)

        input_data = [age, gender, usage, sleep, platform, relationship]
        input_scaled = scaler.transform([input_data])

       
        addiction_pred = models["Addiction_Level"].predict(input_scaled)[0]
        addiction_text = target_decoder.inverse_transform([addiction_pred])[0]

        predictions = {
            "Addiction Level": addiction_text,
            "Academic Performance Impact": "Yes" if models["Affects_Academic_Performance"].predict(input_scaled)[0] == 1 else "No",
            "Mental Health Score": round(models["Mental_Health_Score"].predict(input_scaled)[0], 2),
            "Conflicts Over SNS": round(models["Conflicts_Over_Social_Media"].predict(input_scaled)[0], 2)
        }

        display = "".join(f"<li><strong>{k}:</strong> {v}</li>" for k, v in predictions.items())
        return render_template("smaapp.html", prediction_text=f"<ul>{display}</ul>")

    except Exception as e:
        return f" Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
