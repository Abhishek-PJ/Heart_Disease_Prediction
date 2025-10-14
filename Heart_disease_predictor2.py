import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# Load the saved models and preprocessing objects
with open(r"C:\Users\abhia\Files\Heart_Attack_Predictor\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(r"C:\Users\abhia\Files\Heart_Attack_Predictor\numerical_imputer.pkl", "rb") as f:
    numerical_imputer = pickle.load(f)
with open(r"C:\Users\abhia\Files\Heart_Attack_Predictor\random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open(r"C:\Users\abhia\Files\Heart_Attack_Predictor\knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)
with open(r"C:\Users\abhia\Files\Heart_Attack_Predictor\decision_tree_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

# Define feature names
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Set Page Config with Title and Icon
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# Custom CSS with visibility fixes - ensuring text colors contrast with backgrounds
st.markdown("""
    <style>
        /* Main Application Styling */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8eaed 100%);
            font-family: 'Helvetica Neue', Arial, sans-serif;
        }
        
        /* Text Visibility Fixes */
        h1, h2, h3, h4, h5, h6, p, span, label, div {
            color: #2c3e50 !important; /* Dark text for light backgrounds */
        }
        
        /* Header Styling */
        h1 {
            color: #2c3e50 !important;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e74c3c;
        }
        
        h2, h3, h4 {
            color: #2c3e50 !important;
            font-weight: 600;
        }
        
        /* Subtitle Styling */
        .subtitle {
            text-align: center;
            color: #34495e !important;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #2c3e50;
        }
        
        [data-testid="stSidebar"] h2 {
            color: #ecf0f1 !important;
            font-size: 1.5rem;
            border-bottom: 1px solid #e74c3c;
            padding-bottom: 0.7rem;
            margin-bottom: 1.5rem;
        }
        
        /* Force sidebar labels to be white */
        [data-testid="stSidebar"] .stRadio label, 
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stExpander div,
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span {
            color: #ecf0f1 !important;
        }
        
        /* Force sidebar dropdown text to be dark (for visibility) */
        [data-testid="stSidebar"] select,
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] div[role="button"] {
            color: #2c3e50 !important;
        }
        
        /* Card Styling for Results */
        .card {
            background-color: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            border-left: 4px solid #3498db;
        }
        
        .card * {
            color: #2c3e50 !important;
        }
        
        .risk-high {
            border-left: 4px solid #e74c3c;
        }
        
        .risk-low {
            border-left: 4px solid #2ecc71;
        }
        
        /* Button Styling */
        div.stButton > button {
            background-color: #e74c3c;
            color: white !important;
            font-weight: 600;
            border: none;
            border-radius: 6px;
            padding: 0.6rem 1.2rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
        }
        
        div.stButton > button:hover {
            background-color: #c0392b;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transform: translateY(-2px);
        }
        
        /* Progress Bar Styling */
        .stProgress > div > div > div {
            background-color: #e74c3c;
        }
        
        /* Risk visualization styling */
        .risk-meter {
            height: 8px;
            background-color: #eee;
            border-radius: 4px;
            margin-bottom: 5px;
            overflow: hidden;
        }
        
        .risk-meter-fill {
            height: 100%;
            border-radius: 4px;
        }
        
        .risk-text {
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        /* Risk assessment colors */
        .risk-low-text {
            color: #2ecc71 !important;
        }
        
        .risk-medium-text {
            color: #f39c12 !important;
        }
        
        .risk-high-text {
            color: #e74c3c !important;
        }
        
        /* Model info cards */
        .model-card {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            border-left: 3px solid #3498db;
        }
        
        .model-card * {
            color: #2c3e50 !important;
        }
        
        /* Footer styling */
        footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e0e0e0;
            text-align: center;
        }
        
        footer p {
            color: #7f8c8d !important;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# App Title and Description
st.markdown("<h1>Heart Disease Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-powered clinical decision support tool</p>', unsafe_allow_html=True)

# Main Content Area
col1, col2 = st.columns([1, 2])

# Sidebar for user input
with st.sidebar:
    st.markdown("<h2>Patient Information</h2>", unsafe_allow_html=True)
    
    # Create collapsible sections for better organization
    with st.expander("Demographics", expanded=True):
        age = st.slider("Age", 20, 100, 50)
        sex = st.radio("Sex", options=["Male", "Female"])
        sex_numeric = 1 if sex == "Male" else 0
    
    with st.expander("Clinical Measurements", expanded=True):
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 250)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        fbs_numeric = 1 if fbs == "Yes" else 0
        thalach = st.slider("Maximum Heart Rate", 70, 220, 150)
    
    with st.expander("Cardiac Assessment", expanded=True):
        cp_options = {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-anginal Pain",
            3: "Asymptomatic"
        }
        cp = st.selectbox("Chest Pain Type", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])
        
        restecg_options = {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }
        restecg = st.selectbox("Resting ECG", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])
        
        exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
        exang_numeric = 1 if exang == "Yes" else 0
        
        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0, 0.1)
        
        slope_options = {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])
    
    with st.expander("Advanced Indicators", expanded=True):
        ca = st.slider("Number of Major Vessels (0-4)", 0, 4, 1)
        
        thal_options = {
            0: "Normal",
            1: "Fixed Defect",
            2: "Reversible Defect",
            3: "Unknown"
        }
        thal = st.selectbox("Thalassemia", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])

# Create dataframe from inputs
user_data = {
    "age": age,
    "sex": sex_numeric,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs_numeric,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang_numeric,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}
user_input = pd.DataFrame(user_data, index=[0])

# Main content area
with col1:
    st.markdown("<h3>Patient Overview</h3>", unsafe_allow_html=True)
    
    # Create a neat patient summary with explicit background
    st.markdown("""
    <div style="background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; border-left: 4px solid #3498db;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <div><strong style="color: #2c3e50;">Age:</strong> <span style="color: #2c3e50;">{}</span></div>
            <div><strong style="color: #2c3e50;">Sex:</strong> <span style="color: #2c3e50;">{}</span></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <div><strong style="color: #2c3e50;">BP:</strong> <span style="color: #2c3e50;">{} mmHg</span></div>
            <div><strong style="color: #2c3e50;">Cholesterol:</strong> <span style="color: #2c3e50;">{} mg/dl</span></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <div><strong style="color: #2c3e50;">Max HR:</strong> <span style="color: #2c3e50;">{} bpm</span></div>
            <div><strong style="color: #2c3e50;">Chest Pain:</strong> <span style="color: #2c3e50;">{}</span></div>
        </div>
    </div>
    """.format(age, sex, trestbps, chol, thalach, cp_options[cp]), unsafe_allow_html=True)
    
    # Risk Factors Section
    st.markdown("<h3>Key Risk Indicators</h3>", unsafe_allow_html=True)
    
    # Function to determine risk level
    def risk_level(value, feature):
        if feature == "chol":
            if value < 200:
                return "Low", "2ecc71"
            elif value < 240:
                return "Moderate", "f39c12"
            else:
                return "High", "e74c3c"
        elif feature == "trestbps":
            if value < 120:
                return "Normal", "2ecc71"
            elif value < 130:
                return "Elevated", "3498db"
            elif value < 140:
                return "Stage 1", "f39c12"
            else:
                return "Stage 2", "e74c3c"
        elif feature == "age":
            if value < 45:
                return "Lower", "2ecc71"
            elif value < 65:
                return "Moderate", "f39c12"
            else:
                return "Higher", "e74c3c"
    
    # Display risk meters
    col_risk1, col_risk2, col_risk3 = st.columns(3)
    
    with col_risk1:
        chol_level, chol_color = risk_level(chol, "chol")
        st.markdown(f"""
        <div style="text-align: center;">
            <p style="margin-bottom: 5px; color: #2c3e50;"><strong>Cholesterol</strong></p>
            <div class="risk-meter">
                <div class="risk-meter-fill" style="width: {min(100, (chol-100)/5)}%; background-color: #{chol_color};"></div>
            </div>
            <p class="risk-text" style="color: #{chol_color};">{chol_level} ({chol} mg/dl)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_risk2:
        bp_level, bp_color = risk_level(trestbps, "trestbps")
        st.markdown(f"""
        <div style="text-align: center;">
            <p style="margin-bottom: 5px; color: #2c3e50;"><strong>Blood Pressure</strong></p>
            <div class="risk-meter">
                <div class="risk-meter-fill" style="width: {min(100, (trestbps-80)/1.2)}%; background-color: #{bp_color};"></div>
            </div>
            <p class="risk-text" style="color: #{bp_color};">{bp_level} ({trestbps} mmHg)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_risk3:
        age_level, age_color = risk_level(age, "age")
        st.markdown(f"""
        <div style="text-align: center;">
            <p style="margin-bottom: 5px; color: #2c3e50;"><strong>Age Risk</strong></p>
            <div class="risk-meter">
                <div class="risk-meter-fill" style="width: {min(100, age)}%; background-color: #{age_color};"></div>
            </div>
            <p class="risk-text" style="color: #{age_color};">{age_level} ({age} years)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional Risk Factors with explicit styling
    st.markdown("""
    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin: 20px 0;">
        <strong style="color: #2c3e50;">Additional Factors:</strong>
    """, unsafe_allow_html=True)
    
    risk_factors = []
    if sex == "Male":
        risk_factors.append("Male gender")
    if fbs == "Yes":
        risk_factors.append("Elevated fasting blood sugar")
    if exang == "Yes":
        risk_factors.append("Exercise-induced angina")
    if cp != 0:
        risk_factors.append(f"{cp_options[cp]}")
    if ca > 0:
        risk_factors.append(f"{ca} major vessels colored by fluoroscopy")
    
    if risk_factors:
        for factor in risk_factors:
            st.markdown(f'<p style="color: #2c3e50; margin: 5px 0 5px 15px;">• {factor}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: #2c3e50;">No additional significant risk factors identified.</p>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction area
with col2:
    st.markdown("<h3>Heart Disease Risk Assessment</h3>", unsafe_allow_html=True)
    
    # Preprocess user input for prediction
    user_input_features = user_input[feature_names]
    user_input_imputed = numerical_imputer.transform(user_input_features)
    user_input_scaled = scaler.transform(user_input_imputed)
    
    # Prediction Button with high visibility
    if st.button("Generate Risk Assessment", key="predict_button"):
        with st.spinner("Analyzing patient data..."):
            # Add a progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Get Predictions
            rf_prediction = rf_model.predict(user_input_scaled)[0]
            rf_proba = rf_model.predict_proba(user_input_scaled)[0][1]
            
            knn_prediction = knn_model.predict(user_input_scaled)[0]
            knn_proba = knn_model.predict_proba(user_input_scaled)[0][1]
            
            dt_prediction = dt_model.predict(user_input_scaled)[0]
            dt_proba = dt_model.predict_proba(user_input_scaled)[0][1]
            
            # Calculate ensemble risk score (average probability)
            ensemble_risk = (rf_proba + knn_proba + dt_proba) / 3
            
            # Show ensemble risk score with gauge visualization
            st.markdown("<h3>Overall Risk Assessment</h3>", unsafe_allow_html=True)
            
            # Create gauge chart for risk score with high visibility
            risk_level = "Low" if ensemble_risk < 0.4 else "Moderate" if ensemble_risk < 0.7 else "High"
            risk_color = "#2ecc71" if risk_level == "Low" else "#f39c12" if risk_level == "Moderate" else "#e74c3c"
            
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="background-color: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h2 style="color: {risk_color}; margin-bottom: 10px;">{risk_level} Risk</h2>
                    <div style="height: 20px; background-color: #eee; border-radius: 10px; margin: 20px 0; position: relative;">
                        <div style="width: {ensemble_risk * 100}%; height: 100%; background: linear-gradient(90deg, #2ecc71, #f39c12, #e74c3c); border-radius: 10px;"></div>
                        <div style="width: 10px; height: 30px; background-color: #2c3e50; border-radius: 5px; position: relative; top: -25px; left: calc({ensemble_risk * 100}% - 5px);"></div>
                    </div>
                    <p style="font-size: 1.5rem; font-weight: bold; color: #2c3e50;">{ensemble_risk:.1%}</p>
                    <p style="color: #2c3e50;">Probability of Heart Disease</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show individual model predictions
            st.markdown("<h4>Model Predictions</h4>", unsafe_allow_html=True)
            
            col_rf, col_knn, col_dt = st.columns(3)
            
            # Function to display prediction card with explicit colors
            def show_model_prediction(col, model_name, prediction, probability, icon):
                risk_class = "risk-high" if prediction == 1 else "risk-low"
                result_text = "Positive" if prediction == 1 else "Negative"
                result_color = "#e74c3c" if prediction == 1 else "#2ecc71"
                
                with col:
                    col.markdown(f"""
                    <div style="background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
                        margin-bottom: 15px; border-left: 4px solid {result_color}; height: 100%;">
                        <div style="text-align: center; margin-bottom: 10px;">
                            <span style="font-size: 1.5rem;">{icon}</span>
                            <h4 style="margin: 5px 0; color: #2c3e50;">{model_name}</h4>
                        </div>
                        <div style="text-align: center;">
                            <p style="font-size: 1.2rem; font-weight: bold; color: {result_color};">{result_text}</p>
                            <p style="font-size: 0.9rem; color: #2c3e50;">Confidence: {probability:.1%}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display each model's prediction
            show_model_prediction(col_rf, "Random Forest", rf_prediction, rf_proba, "🌲")
            show_model_prediction(col_knn, "K-Nearest Neighbors", knn_prediction, knn_proba, "🔗")
            show_model_prediction(col_dt, "Decision Tree", dt_prediction, dt_proba, "🌳")
            
            # Key findings and recommendations
            st.markdown("<h4>Key Findings & Recommendations</h4>", unsafe_allow_html=True)
            
            # Generate personalized recommendations based on risk factors
            recommendations = []
            if chol > 200:
                recommendations.append("Monitor cholesterol levels and consider dietary changes")
            if trestbps >= 130:
                recommendations.append("Follow up on blood pressure management")
            if age > 65:
                recommendations.append("Regular cardiac check-ups recommended for your age group")
            if exang == "Yes":
                recommendations.append("Further evaluation of exercise-induced angina advised")
            if ca > 0:
                recommendations.append("Follow up with cardiologist regarding vessel health")
            
            # Add general recommendations
            recommendations.append("Maintain a heart-healthy diet rich in fruits and vegetables")
            recommendations.append("Regular physical activity appropriate for your condition")
            
            # Display findings with explicit styling
            st.markdown("""
            <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin: 20px 0;">
            """, unsafe_allow_html=True)
            
            # Display findings based on risk score
            if ensemble_risk > 0.7:
                st.markdown("""
                <p style="color: #2c3e50;"><strong>Clinical Assessment:</strong> This patient shows multiple risk factors indicating a high probability of coronary artery disease. Immediate follow-up with a cardiologist is strongly recommended.</p>
                """, unsafe_allow_html=True)
            elif ensemble_risk > 0.4:
                st.markdown("""
                <p style="color: #2c3e50;"><strong>Clinical Assessment:</strong> This patient presents with moderate risk factors for coronary artery disease. Additional diagnostic testing may be warranted.</p>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <p style="color: #2c3e50;"><strong>Clinical Assessment:</strong> This patient currently shows a lower risk profile for coronary artery disease. Preventative measures and routine monitoring are advised.</p>
                """, unsafe_allow_html=True)
            
            # Display recommendations
            st.markdown('<p style="color: #2c3e50;"><strong>Recommendations:</strong></p>', unsafe_allow_html=True)
            for rec in recommendations[:5]:  # Limit to top 5 recommendations
                st.markdown(f'<p style="color: #2c3e50; margin: 5px 0 5px 15px;">• {rec}</p>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display disclaimer
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 20px;">
                <p style="font-size: 0.8rem; color: #2c3e50;"><strong>Disclaimer:</strong> This tool is designed to assist healthcare professionals and is not a replacement for clinical judgment. The predictions should be considered as supporting information for healthcare decisions, not as a definitive diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Display informational content when no prediction has been made
        st.markdown("""
        <div style="background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h4 style="color: #2c3e50;">About This Tool</h4>
            <p style="color: #2c3e50;">This clinical decision support system uses machine learning algorithms trained on extensive cardiac patient data to assess heart disease risk.</p>
            <p style="color: #2c3e50;">Enter patient information in the sidebar and click "Generate Risk Assessment" to receive a detailed analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # About the models
        st.markdown("<h4>Model Information</h4>", unsafe_allow_html=True)
        
        model_info = [
            {
                "name": "Random Forest",
                "icon": "🌲",
                "description": "Uses an ensemble of decision trees to provide robust predictions with high accuracy."
            },
            {
                "name": "K-Nearest Neighbors",
                "icon": "🔗",
                "description": "Classifies patients by comparing them to similar cases from clinical data."
            },
            {
                "name": "Decision Tree",
                "icon": "🌳",
                "description": "Uses logical decision paths similar to clinical decision trees for transparent predictions."
            }
        ]
        
        for model in model_info:
            st.markdown(f"""
            <div style="background-color: #f9f9f9; border-radius: 8px; padding: 15px; margin-bottom: 15px; border-left: 3px solid #3498db;">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 1.5rem; margin-right: 10px;">{model["icon"]}</span>
                    <div>
                        <h5 style="margin: 0; color: #2c3e50;">{model["name"]}</h5>
                        <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: #2c3e50;">{model["description"]}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Footer with explicit styling
st.markdown("""
<div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0; text-align: center;">
    <p style="color: #7f8c8d; font-size: 0.9rem;">© 2025 Heart Disease Risk Assessment Tool</p>
    <p style="color: #7f8c8d; font-size: 0.9rem;">Clinical decision support powered by machine learning algorithms</p>
</div>
""", unsafe_allow_html=True)