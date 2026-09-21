import streamlit as st
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Delivery Delay Predictor", layout="centered")

st.title("🚚 Delivery Delay Prediction App")
st.write("Enter the delivery details below to predict if there will be a delay.")

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load("logi.sav")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model (logi.sav): {e}")
    st.stop()

# Create input form for the 11 features
with st.form("prediction_form"):
    st.subheader("Feature Inputs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        delivery_distance = st.number_input("Delivery Distance (miles)", min_value=0.0, value=15.0)
        traffic_congestion = st.slider("Traffic Congestion (1-5)", min_value=1, max_value=5, value=3)
        weather_condition = st.slider("Weather Condition (1-5)", min_value=1, max_value=5, value=2)
        delivery_slot = st.slider("Delivery Slot (e.g. 1-4)", min_value=1, max_value=4, value=2)
        driver_experience = st.number_input("Driver Experience (years)", min_value=0.0, value=5.0)
        num_stops = st.number_input("Number of Stops", min_value=0, value=2)

    with col2:
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, value=3)
        road_condition_score = st.slider("Road Condition Score (1-5)", min_value=1, max_value=5, value=4)
        package_weight = st.number_input("Package Weight (lbs)", min_value=0.0, value=10.0)
        fuel_efficiency = st.number_input("Fuel Efficiency (mpg)", min_value=0.0, value=15.0)
        warehouse_processing_time = st.number_input("Warehouse Processing Time (mins)", min_value=0, value=45)

    submit_button = st.form_submit_button("Predict Delay")

if submit_button:
    # Format features for prediction
    features = np.array([[
        delivery_distance, traffic_congestion, weather_condition,
        delivery_slot, driver_experience, num_stops, vehicle_age,
        road_condition_score, package_weight, fuel_efficiency,
        warehouse_processing_time
    ]])
    
    # Generate predictions
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    st.write("---")
    if prediction == 1:
        st.error(f"🚨 **Delayed!** (Probability of delay: {probability[1]:.2%})")
    else:
        st.success(f"✅ **On Time!** (Probability of being on time: {probability[0]:.2%})")
