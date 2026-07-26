import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit page configuration
st.set_page_config(
    page_title="Bengaluru House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title("🏠 Bengaluru House Price Prediction")

st.write("Enter the details below to estimate the house price.")

# User Inputs
location = st.text_input("Location", placeholder="e.g., Whitefield")

total_sqft = st.number_input(
    "Total Square Feet",
    min_value=100.0,
    value=1000.0,
    step=50.0
)

bath = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=20,
    value=2
)

balcony = st.number_input(
    "Number of Balconies",
    min_value=0,
    max_value=10,
    value=1
)

# Predict Button
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "location": [location],
        "total_sqft": [total_sqft],
        "bath": [bath],
        "balcony": [balcony]
    })

    prediction = model.predict(input_data)

    st.success(
        f"🏡 Estimated House Price: ₹ {prediction[0]:.2f} Lakhs"
    )
