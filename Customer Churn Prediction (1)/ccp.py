# gender -> 1 Female 0 Male
# Churn -> 1 Yes 0 No
# loaded_scaler = pickle.load(open("scaler.pkl", "rb"))
# loaded_model = pickle.load(open("model.pkl", "rb"))
# order of the X -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import pickle 
import numpy as np

loaded_scaler = pickle.load(open("Machine_Learning_Projects/Customer Churn Prediction (1)
/scaler.pkl", "rb"))
loaded_model = pickle.load(open("Machine_Learning_Projects/Customer Churn Prediction (1)
/model.pkl", "rb"))

st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the predict button for getting a prediction")

st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)

tenure = st.number_input("Enter Tenure", min_value= 0, max_value=130, value= 10)

monthlycharge = st.number_input("Enter Monthly Charge", min_value=30, max_value=150)

gender = st.selectbox("Enter the Gender",["Male","Female"])

st.divider()

predictbutton = st.button("Predict!")

if predictbutton:
    gender_selected = 1 if gender == "Female" else 0

    X = [age, gender_selected, tenure, monthlycharge]

    X1 = np.array(X)

    X_array = loaded_scaler.transform([X1])

    prediction = loaded_model.predict(X_array)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.write(f"Predicted: {predicted}")

else:
    st.write("please enter the values and use predict button")    




