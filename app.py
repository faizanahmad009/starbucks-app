import streamlit as st
import pandas as pd
import joblib

st.title("Starbucks Drink Category Predictor")
st.write("Enter the nutrition values and the model will guess the drink category.")

model = joblib.load("starbucks_model.joblib")
columns = joblib.load("columns.joblib")
medians = joblib.load("medians.joblib")

values = {}
col1, col2 = st.columns(2)
for i, name in enumerate(columns):
    box = col1 if i % 2 == 0 else col2
    values[name] = box.number_input(name, value=float(medians[name]))

if st.button("Predict"):
    row = pd.DataFrame([values])[columns]
    prediction = model.predict(row)[0]
    st.success(f"Predicted category: {prediction}")

    probs = model.predict_proba(row)[0]
    chart = pd.Series(probs, index=model.classes_).sort_values(ascending=False)
    st.bar_chart(chart)