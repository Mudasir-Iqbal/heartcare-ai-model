import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.title("💓 Heart Disease Prediction App")

st.markdown("""
This app predicts whether a person is likely to have heart disease based on medical information.
**Please enter all the required details accurately.**
""")

# ----------------------------
# Input fields
# ----------------------------
age = st.number_input("Age", min_value=20, max_value=100, value=52)
resting_bp = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=85, max_value=603, value=250)
fasting_bs = st.selectbox("Fasting Blood Sugar (0 = No, 1 = Yes)", [0, 1])
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=140)
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=6.0, step=0.1, value=1.5)

# Categorical encoded fields (one-hot encoded format based on your model)
sex_m = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp_ata = st.selectbox("Chest Pain Type: ATA", [0, 1])
cp_nap = st.selectbox("Chest Pain Type: NAP", [0, 1])
cp_ta = st.selectbox("Chest Pain Type: TA", [0, 1])
ecg_normal = st.selectbox("Resting ECG: Normal", [0, 1])
ecg_st = st.selectbox("Resting ECG: ST", [0, 1])
angina_y = st.selectbox("Exercise-Induced Angina: Yes", [0, 1])
slope_flat = st.selectbox("ST Slope: Flat", [0, 1])
slope_up = st.selectbox("ST Slope: Up", [0, 1])

# ----------------------------
# Prepare Input for Model
# ----------------------------
columns = [
    'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
    'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
    'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y',
    'ST_Slope_Flat', 'ST_Slope_Up'
]

# Create input DataFrame
input_data = pd.DataFrame([[
    age, resting_bp, chol, fasting_bs, max_hr, oldpeak,
    sex_m, cp_ata, cp_nap, cp_ta,
    ecg_normal, ecg_st, angina_y,
    slope_flat, slope_up
]], columns=columns)

# Apply scaling to numerical columns only
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# ----------------------------
# Predict Button
# ----------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ You may have heart disease.")
    else:
        st.success("✅ No heart disease detected.")



