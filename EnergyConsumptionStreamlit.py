import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Energy Consumption Prediction", page_icon="⚡")

st.title("⚡ Energy Consumption Prediction")

# Load model
model = joblib.load("Energy_Consumption_Model.joblib")
feature_names = joblib.load("feature_names.joblib")


st.header("Enter Building Information")

building_type = st.selectbox(
    "Building Type",
    ["Residential", "Commercial", "Industrial"]
)

square_footage = st.number_input(
    "Area Square Footage",
    min_value=0,
    value=2000
)

occupants = st.number_input(
    "Number of Occupants",
    min_value=1,
    value=10
)

appliances = st.number_input(
    "Appliances Used",
    min_value=0,
    value=15
)

temperature = st.number_input(
    "Average Temperature (°C)",
    value=25.0
)

day = st.selectbox(
    "Day of Week",
    ["Weekday", "Weekend"]
)

if st.button("Predict Energy Consumption"):

    input_data = pd.DataFrame({
        "Building Type": [building_type],
        "Square Footage": [square_footage],
        "Number of Occupants": [occupants],
        "Appliances Used": [appliances],
        "Average Temperature": [temperature],
        "Day of Week": [day]
    })

    # Convert to dummy variables
    input_data = pd.get_dummies(
        input_data,
        columns=["Building Type", "Day of Week"],
        drop_first=True
    )

    # Match training columns
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(input_data)

    st.success(f"Predicted Energy Consumption: {prediction[0]:.2f}")