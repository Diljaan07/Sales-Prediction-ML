
import streamlit as st
import joblib
import pandas as pd
import os

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Sales Prediction",
    page_icon="📈",
    layout="centered"
)

# TITLE
st.title("📈 Sales Prediction System")
st.write(
    "Predict product sales using advertising expenditure "
    "and Machine Learning."
)

st.divider()

# LOAD TRAINED MODEL
model_path = "model/sales_prediction_model.pkl"

if not os.path.exists(model_path):
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

model = joblib.load(model_path)

# USER INPUT
st.subheader("Enter Advertising Budget")

tv = st.number_input(
    "TV Advertising Budget",
    min_value=0.0,
    value=100.0,
    step=1.0
)

radio = st.number_input(
    "Radio Advertising Budget",
    min_value=0.0,
    value=25.0,
    step=1.0
)

newspaper = st.number_input(
    "Newspaper Advertising Budget",
    min_value=0.0,
    value=25.0,
    step=1.0
)

st.divider()

# PREDICTION
if st.button("Predict Sales", type="primary"):

    input_data = pd.DataFrame(
        [[tv, radio, newspaper]],
        columns=["TV", "radio", "newspaper"]
    )

    prediction = model.predict(input_data)[0]

    st.success("Prediction Generated Successfully!")

    st.metric(
        label="Predicted Product Sales",
        value=f"{prediction:.2f} thousand units"
    )

    st.caption(
        "This is an estimate based on historical advertising data, "
        "not a guaranteed sales result."
    )

st.divider()

st.subheader("About This Project")

st.write("""
This application uses Multiple Linear Regression to estimate
product sales based on TV, radio and newspaper advertising budgets.

Machine Learning algorithm: Linear Regression

Dataset: Advertising Dataset
""")
