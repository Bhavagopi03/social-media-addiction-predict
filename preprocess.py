import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pickle

def encoding():
    df = pd.read_csv("Students Social Media Addiction.csv")

    if "Student_ID" in df.columns:
        df.drop("Student_ID", axis=1, inplace=True)

    def classify(score):
        if score <= 3:
            return "Not Addicted"
        elif score <= 6:
            return "Mildly Addicted"
        else:
            return "Heavily Addicted"

    if "Addicted_Score" in df.columns:
        df["Addiction_Level_Text"] = df["Addicted_Score"].apply(classify)
        df.drop("Addicted_Score", axis=1, inplace=True)

    df.fillna("Unknown", inplace=True)

    label_encoders = {}
    for col in ["Gender", "Most_Used_Platform", "Relationship_Status", "Affects_Academic_Performance"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    target_encoder = LabelEncoder()
    df["Addiction_Level"] = target_encoder.fit_transform(df["Addiction_Level_Text"])

    feature_cols = ["Age", "Gender", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Most_Used_Platform", "Relationship_Status"]
    X = df[feature_cols]

    y = {
        "Addiction_Level": df["Addiction_Level"],
        "Addiction_Level_Text": df["Addiction_Level_Text"],  # Just for reference
        "Affects_Academic_Performance": df["Affects_Academic_Performance"],
        "Mental_Health_Score": df["Mental_Health_Score"],
        "Conflicts_Over_Social_Media": df["Conflicts_Over_Social_Media"]
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    with open("models/target_encoder_addiction.pkl", "wb") as f:
        pickle.dump(target_encoder, f)

    return X_scaled, y, scaler
