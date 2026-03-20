import streamlit as st
import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


model = joblib.load("w_rf_model.pkl")
columns = joblib.load("w_rf_columns.pkl")

st.title("UCS Prediction Model")
st.success("Model Accuracy nearly 86% 😊")
st.info("Enter the details to predict the maximum UCS of soil stabilization")

# Inputs
days = st.slider("Curing Days", 7, 100, 7)
percent = st.slider("Precursor Percent (%)", 5, 20, 8)
days_x_percent = days * (percent/100)  

precursor = st.selectbox("Precursor", [
    "CDW", "CDW+Slag", "Slag", "WA+FA"
])

# One-hot encoding (MATCH TRAINING EXACTLY)
precursor_dict = {
    "CDW": [1, 0, 0, 0],
    "CDW+Slag": [0, 1, 0, 0],
    "Slag": [0, 0, 1, 0],
    "WA+FA": [0, 0, 0, 1]
}


input_df = pd.DataFrame([[
    days,
    percent,
    days_x_percent,
    *precursor_dict[precursor]
]], columns=columns)


# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    
    st.success(f"Predicted Max UCS: {prediction[0]:.2f}")
    display_df = pd.DataFrame({
    "Days": [days],
    "Percent": [f"{percent}%"],
    "Precursor": [precursor],
    "Max UCS": [f"{prediction[0]:.2f} kPa"]
    })
    
    st.write("Output DataFrame:")
    st.write(display_df)