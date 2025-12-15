

import streamlit as st
import numpy as np
import pickle
from datetime import datetime

# Load trained ML model
model = pickle.load(open("CPP_1_model.pkl", "rb"))

st.title("🚗 Car Selling Price Prediction")

# User Inputs
present_price = st.number_input("Present Price (Lakhs)", min_value=0.0)
kms_driven = st.number_input("Kms Driven", min_value=0)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", [0, 1, 2, 3])
year = st.selectbox("Year", list(range(1990, datetime.now().year + 1)))

# Encoding (MUST be same as training)
fuel = {"Petrol": 0, "Diesel": 1, "CNG": 2}[fuel_type]
seller = {"Dealer": 0, "Individual": 1}[seller_type]
trans = {"Manual": 0, "Automatic": 1}[transmission]

age = datetime.now().year - year

# Prediction
if st.button("Predict Selling Price"):
    input_data = np.array([[present_price, kms_driven, fuel,
                            seller, trans, owner, age]])

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Selling Price: ₹ {prediction:.2f} Lakhs")


