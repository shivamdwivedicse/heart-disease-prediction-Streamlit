import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Predictor")

st.title("❤️ Heart Disease Prediction App")
st.write("Fill details and click Predict")

# Load files
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

age = st.number_input("Age", 1, 120, 40)
resting_bp = st.number_input("Resting BP", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    df = pd.DataFrame([raw])

    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]
    df_scaled = scaler.transform(df)

    pred = model.predict(df_scaled)[0]

    if pred == 1:
        st.error("⚠️ High chance of Heart Disease")
    else:
        st.success("✅ Low chance of Heart Disease")