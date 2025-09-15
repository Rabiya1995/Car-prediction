import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("🚗 Car Price Prediction App")

# Inputs from user
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1)
present_price = st.number_input("Present Price (Rs)", min_value=0.0, step=0.1)
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
owner = st.number_input("Number of Previous Owners", min_value=0, step=1)
car_age = st.number_input("Car Age (years)", min_value=0, step=1)

fuel_type = st.radio("Fuel Type", ("CNG", "Diesel", "Petrol"))
transmission = st.radio("Transmission", ("Automatic", "Manual"))

# Convert categorical inputs into model format
fuel_cng, fuel_diesel, fuel_petrol = 0, 0, 0
if fuel_type == "CNG":
    fuel_cng = 1
elif fuel_type == "Diesel":
    fuel_diesel = 1
else:
    fuel_petrol = 1

trans_auto, trans_manual = 0, 0
if transmission == "Automatic":
    trans_auto = 1
else:
    trans_manual = 1

# Prediction button
if st.button("Predict Price"):
    features = np.array([[year, present_price, kms_driven, owner, car_age,
                          fuel_cng, fuel_diesel, fuel_petrol,
                          trans_auto, trans_manual]])
    
    prediction = model.predict(features)[0]
    st.success(f"Predicted Selling Price: ₹{prediction:,.2f}")
