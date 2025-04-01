import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
model = joblib.load("xgboost_model.pkl")
selected_features = joblib.load("selectedff.pkl")
scaler = joblib.load("scalerr.pkl")

st.title("Breast Cancer Classification App")
st.write("Enter feature values to predict if a tumor is Malignant or Benign.")


input_data = []
for feature in selected_features:
    value = st.number_input(f"{feature}", value=0.0)
    input_data.append(value)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)


    if scaler:
        input_array = scaler.transform(input_array)


    prediction = model.predict(input_array)[0]
    result = "Benign (Non-Cancerous)" if prediction == 1 else "Malignant (Cancerous)"

    st.write(f"Prediction: **{result}**")