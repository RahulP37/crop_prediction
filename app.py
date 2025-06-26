import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Set Streamlit page config (MUST be first Streamlit call)
st.set_page_config(page_title="Crop Recommendation", layout="centered")

# Load model and preprocessing tools
model = load_model('crop_recommendation_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

# App title
st.title("🌱 Crop Recommendation System")
st.write("Enter the following parameters to get a recommended crop:")

# Input form
with st.form("input_form"):
    n = st.number_input("Nitrogen (N)", min_value=0.0)
    p = st.number_input("Phosphorus (P)", min_value=0.0)
    k = st.number_input("Potassium (K)", min_value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0)
    ph = st.number_input("pH", min_value=0.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    try:
        input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        crop_index = np.argmax(prediction)
        crop_label = encoder.categories_[0][crop_index]
        st.success(f"✅ Recommended Crop: **{crop_label}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
