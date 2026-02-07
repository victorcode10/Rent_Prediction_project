"""
Nigerian House Rent Price Prediction - Streamlit Web App (Dark Theme)
A user-friendly interface with black/dark mode to predict house rent prices in Nigeria
"""

import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model
import os

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Nigerian House Rent Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM DARK THEME ====================
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    /* Sidebar background */
    .sidebar .sidebar-content {
        background-color: #1c1f26;
        color: #ffffff;
    }
    /* Buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px 30px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #333333 0%, #111111 100%);
        padding: 30px;
        border-radius: 15px;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }
    /* Feature cards */
    .feature-card {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin: 10px 0;
        color: #ffffff;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f5f7fa;
    }
    /* Footer */
    .stMarkdown div {
        color: #b0b3b8;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_prediction_model():
    """Load the trained model (cached for performance)"""
    try:
        model = load_model('models/rent_model')
        return model
    except:
        st.error(" Model not found! Please run train.py first to generate the model.")
        st.stop()

model = load_prediction_model()

# ==================== HEADER ====================
st.markdown("<h1 style='text-align: center;'>🏠 Nigerian House Rent Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #b0b3b8;'>Predict monthly rent prices across Nigeria with AI</h4>", unsafe_allow_html=True)
st.markdown("---")

# ==================== SIDEBAR ====================
st.sidebar.header("🔧 Property Features")
st.sidebar.markdown("Fill in the property details below:")

city = st.sidebar.selectbox(" City", options=['Lagos', 'Abuja', 'Port Harcourt', 'Ibadan', 'Benin', 'Ilorin'])
bedrooms = st.sidebar.slider("🛏️ Number of Bedrooms", 1, 5, 2)
bathrooms = st.sidebar.slider("🚿 Number of Bathrooms", 1, 4, 2)
furnished = st.sidebar.radio("🛋️ Furnished?", options=['Yes', 'No'])
power_supply_hours = st.sidebar.slider("⚡ Power Supply (hours/day)", 0, 24, 12, step=4)
security_level = st.sidebar.select_slider("🔒 Security Level", options=['Low', 'Medium', 'High'], value='Medium')
estate_type = st.sidebar.radio("🏘️ Estate Type", options=['Estate', 'Non-Estate'])
proximity_to_road = st.sidebar.radio("🛣️ Proximity to Main Road", options=['Near', 'Far'])
parking_space = st.sidebar.radio("🚗 Parking Space Available?", options=['Yes', 'No'])

st.sidebar.markdown("---")
st.sidebar.info(" Adjust the values above and click 'Predict Rent' to get the estimated monthly rent.")

# ==================== MAIN AREA ====================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("###  Property Summary")
    st.markdown(f"""
    <div class="feature-card">
        <strong>Location:</strong> {city}<br>
        <strong>Bedrooms:</strong> {bedrooms} | <strong>Bathrooms:</strong> {bathrooms}<br>
        <strong>Furnished:</strong> {furnished}<br>
        <strong>Power Supply:</strong> {power_supply_hours} hours/day<br>
        <strong>Security:</strong> {security_level}<br>
        <strong>Estate Type:</strong> {estate_type}<br>
        <strong>Road Proximity:</strong> {proximity_to_road}<br>
        <strong>Parking:</strong> {parking_space}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("###  Rent Prediction")
    if st.button(" Predict Monthly Rent", use_container_width=True):
        input_data = pd.DataFrame({
            'city': [city],
            'number_of_bedrooms': [bedrooms],
            'number_of_bathrooms': [bathrooms],
            'furnished': [furnished],
            'power_supply_hours': [power_supply_hours],
            'security_level': [security_level],
            'estate_type': [estate_type],
            'proximity_to_road': [proximity_to_road],
            'parking_space': [parking_space]
        })
        with st.spinner(' Calculating rent price...'):
            prediction = predict_model(model, data=input_data)
            predicted_rent = prediction['prediction_label'].iloc[0]
        
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="margin: 0;">Predicted Monthly Rent</h2>
            <h1 style="margin: 10px 0; font-size: 48px;">₦{predicted_rent:,.0f}</h1>
            <p style="margin: 0; font-size: 16px;">per month</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success(" Prediction completed successfully!")
        st.markdown("---")
        st.markdown("###  Insights")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Annual Cost", f"₦{predicted_rent * 12:,.0f}")
        with col_b:
            st.metric("Per Bedroom", f"₦{predicted_rent / bedrooms:,.0f}")
        with col_c:
            st.metric("Daily Cost", f"₦{predicted_rent / 30:,.0f}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #b0b3b8;'>
    <p>🎓 Built by Nigerian Data Science Students | Powered by PyCaret & Streamlit</p>
    <p> Model trained on 3500+ Nigerian property listings</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### About This App")
st.sidebar.info("""
This AI-powered tool predicts house rent prices across major Nigerian cities using machine learning.

**Technology Stack:**
-  PyCaret for ML
-  Streamlit for UI
-  Python 3.10

**Created for learning purposes**
""")
