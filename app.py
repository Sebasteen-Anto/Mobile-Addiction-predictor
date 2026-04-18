import streamlit as st
import pickle
import numpy as np

# Configure web page
st.set_page_config(
    page_title="Digital Wellness Dashboard", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a professional, clinical, dark-mode dashboard
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: #0f111a;
        color: #e2e8f0;
    }
    
    /* Header typography */
    h1, h2, h3, h4, span, p {
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
        color: #f8fafc !important;
    }
    
    /* Primary button restyling */
    div.stButton > button:first-child {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 0rem;
        font-size: 1.05rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        width: 100%;
        margin-top: 1.5rem;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5);
        background: #2563eb;
    }
    
    /* Input fields styling */
    .stNumberInput > div > div > input {
        border-radius: 6px !important;
        background-color: #1a1d2d !important;
        color: #f8fafc !important;
        border: 1px solid #2d334a !important;
        padding-left: 1rem !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }
    
    /* Custom divider */
    hr {
        border-color: #2d334a !important;
        margin: 2.5rem 0 !important;
    }
    
    .metric-card {
        background: #1a1d2d;
        border: 1px solid #2d334a;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #3b82f6;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title Container
st.markdown("""
<div style='text-align: center; padding-bottom: 2.5rem; padding-top: 1rem;'>
    <h1 style='font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: -0.02em;'>
        Digital Wellness Analytics
    </h1>
    <p style='color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 0 auto; line-height: 1.6;'>
        Analyze your daily digital consumption patterns and receive an AI-driven behavioral assessment.
    </p>
</div>
""", unsafe_allow_html=True)

# Load the trained model
try:
    with open('addiction_model.pkl', 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        encoder = data['encoder']
except Exception as e:
    st.error("Prediction model not found. Please train the model first.")
    st.stop()

st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 1.5rem;'>Data Input</h3>", unsafe_allow_html=True)

# User Inputs with professional labels
col1, col2 = st.columns(2, gap="large")
with col1:
    daily_screen_time = st.number_input("Total Daily Screen Time (Hrs)", min_value=0.0, max_value=24.0, value=5.0, step=0.5, format="%.1f")
    social_media_hours = st.number_input("Social Media Usage (Hrs)", min_value=0.0, max_value=24.0, value=2.0, step=0.5, format="%.1f")
with col2:
    gaming_hours = st.number_input("Mobile Gaming (Hrs)", min_value=0.0, max_value=24.0, value=1.0, step=0.5, format="%.1f")
    work_study_hours = st.number_input("Productivity / Work (Hrs)", min_value=0.0, max_value=24.0, value=4.0, step=0.5, format="%.1f")

st.markdown("---")

# Prediction button and logic
if st.button("Generate Diagnostic Report"):
    with st.spinner('Running AI behavioral analysis...'):
        # Format input for testing
        input_data = np.array([[daily_screen_time, social_media_hours, gaming_hours, work_study_hours]])
        
        # Predict
        prediction_encoded = model.predict(input_data)
        prediction_label = encoder.inverse_transform(prediction_encoded)[0]
        
        # Color coding logic
        if "High" in prediction_label or "Severe" in prediction_label:
            border_color, text_color = "#ef4444", "#fecaca" 
            status = "Elevated Risk"
            recommendation = "High screen time detected. We recommend implementing scheduled downtime and tracking individual app usage to reduce digital dependency."
        elif "Moderate" in prediction_label:
            border_color, text_color = "#eab308", "#fef08a"
            status = "Moderate Attention"
            recommendation = "Your usage is moderate. Maintaining a balance between digital consumption and offline activities is advised."
        else:
            border_color, text_color = "#10b981", "#a7f3d0"
            status = "Healthy Baseline"
            recommendation = "Your digital habits appear well-balanced. Continue maintaining productive screen time patterns."
            
        st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 1.5rem;'>Diagnostic Report</h3>", unsafe_allow_html=True)
            
        # Display professional result dashboard
        st.markdown(f"""
        <div style='background: #1a1d2d; border-left: 5px solid {border_color}; border-radius: 6px; padding: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem;'>
            <div style='color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Assessment Result</div>
            <h2 style='color: {text_color}; margin-top: 0; margin-bottom: 0.5rem; font-size: 2.2rem;'>{prediction_label}</h2>
            <div style='display: inline-block; padding: 0.25rem 0.75rem; background: {border_color}20; color: {border_color}; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; margin-bottom: 1.5rem;'>{status}</div>
            <p style='color: #cbd5e1; font-size: 1.05rem; line-height: 1.6; margin: 0;'>{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display breakdown metrics
        st.markdown("<h4 style='font-size: 1.2rem; margin-bottom: 1rem; color: #94a3b8 !important;'>Usage Breakdown</h4>", unsafe_allow_html=True)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{social_media_hours}h</div>
                <div class='metric-label'>Social Media</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{gaming_hours}h</div>
                <div class='metric-label'>Gaming</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{work_study_hours}h</div>
                <div class='metric-label'>Productivity</div>
            </div>
            """, unsafe_allow_html=True)

