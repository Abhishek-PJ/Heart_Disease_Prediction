import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import requests
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #d33682;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #268bd2;
        margin-bottom: 1rem;
    }
    .result-positive {
        background-color: #ffcccb;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        color: #d33682;
        border: 2px solid #d33682;
    }
    .result-negative {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        color: #2aa198;
        border: 2px solid #2aa198;
    }
    .stButton>button {
        background-color: #268bd2;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1e6ea7;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #268bd2;
        margin-bottom: 1rem;
    }
    .sidebar-content {
        padding: 20px 10px;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #586e75;
        font-style: italic;
    }
    .heart-icon {
        font-size: 3rem;
        color: #d33682;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-table {
        margin-top: 20px;
        margin-bottom: 30px;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .expander-header {
        font-weight: bold;
        color: #268bd2;
    }
</style>
""", unsafe_allow_html=True)

# Function to create a simple heart icon using matplotlib
def get_heart_icon():
    fig, ax = plt.subplots(figsize=(3, 3))
    
    # Draw a heart shape
    t = np.linspace(0, 2*np.pi, 100)
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    
    # Scale and center
    x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1
    y = (y - np.min(y)) / (np.max(y) - np.min(y)) * 2 - 1
    
    ax.plot(x, y, color='#d33682', linewidth=3)
    ax.fill(x, y, color='#d33682', alpha=0.3)
    
    # Remove axes
    ax.axis('off')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Convert plot to image
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', transparent=True)
    plt.close(fig)
    return buf

# Function to create a feature description table
def create_feature_table():
    # Create a DataFrame with feature descriptions
    features_df = pd.DataFrame({
        'Feature': [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ],
        'Description': [
            'Age in years',
            'Sex (0 = Female, 1 = Male)',
            'Chest Pain Type (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal, 3 = Asymptomatic)',
            'Resting Blood Pressure (mm Hg)',
            'Serum Cholesterol (mg/dL)',
            'Fasting Blood Sugar > 120 mg/dL (1 = Yes, 0 = No)',
            'Resting ECG (0 = Normal, 1 = ST-T Abnormality, 2 = Left Ventricular Hypertrophy)',
            'Maximum Heart Rate Achieved',
            'Exercise Induced Angina (1 = Yes, 0 = No)',
            'ST Depression Induced by Exercise',
            'Slope of Peak Exercise ST Segment (0 = Upsloping, 1 = Flat, 2 = Downsloping)',
            'Number of Major Blood Vessels (0-3)',
            'Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)',
            'Heart Disease (1 = Disease, 0 = No Disease) (This is the label/output)'
        ]
    })
    
    # Style the DataFrame
    styled_df = features_df.style.set_properties(**{
        'background-color': '#f8f9fa',
        'border': '1px solid #ddd',
        'padding': '10px'
    })
    
    styled_df = styled_df.set_properties(subset=['Feature'], **{
        'font-weight': 'bold',
        'background-color': '#e9ecef'
    })
    
    return features_df

# Load the trained model
@st.cache_resource
def load_model():
    model_path = r"C:\Users\abhia\OneDrive\Desktop\Heart_Attack_Predictor\trained_model_of_rf_heart.sav"
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)

# Load dataset
@st.cache_data
def load_data():
    data_path = "heart.csv"
    return pd.read_csv(data_path)

try:
    model = load_model()
    df = load_data()
    feature_names = df.columns[:-1]  # Exclude target column
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    # Display heart icon using matplotlib
    heart_buf = get_heart_icon()
    st.image(heart_buf, caption="Heart Health Monitor", width=150)
    
    st.markdown("### About This App")
    st.markdown("""
    This application uses machine learning to predict the likelihood of heart disease based on various health parameters.
    
    The model was trained on clinical data and can help identify potential risk factors.
    """)
    
    st.markdown("### How to Use")
    st.markdown("""
    1. Enter your health parameters in the form
    2. Click the 'Predict Heart Disease' button
    3. Review your results and consult with a healthcare professional
    """)
    
    # Add feature description table in an expander
    with st.expander("📋 Feature Descriptions", expanded=False):
        st.markdown('<p class="expander-header">Understanding the Parameters</p>', unsafe_allow_html=True)
        
        # Option 1: Display the feature table as an image
        try:
            # Try to load the image from the URL
            response = requests.get("https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image-JnjBP0YaKCSdFjqrRWemVlBWHtsaOk.png")
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Feature Descriptions", use_column_width=True)
        except Exception as e:
            # Fallback to the table if image loading fails
            st.dataframe(create_feature_table(), hide_index=True)
    
    st.markdown('<div class="disclaimer">Disclaimer: This app is for educational purposes only and should not replace professional medical advice.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction Tool</h1>', unsafe_allow_html=True)

st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("""
This tool analyzes your health parameters to assess your risk of heart disease. 
Fill in the form below with your health information to get a prediction.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Add feature description table in the main content
with st.expander("📋 Click here to see all feature descriptions", expanded=False):
    st.markdown('<div class="feature-table">', unsafe_allow_html=True)
    try:
        # Try to load the image from the URL
        response = requests.get("https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image-JnjBP0YaKCSdFjqrRWemVlBWHtsaOk.png")
        img = Image.open(BytesIO(response.content))
        st.image(img, caption="Feature Descriptions", use_column_width=True)
    except Exception as e:
        # Fallback to the table if image loading fails
        st.dataframe(create_feature_table(), hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Dictionary to provide descriptions for known features
feature_descriptions = {
    "age": "Age (in years)",
    "sex": "Sex (0 = Female, 1 = Male)",
    "cp": "Chest Pain Type (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal, 3 = Asymptomatic)",
    "trestbps": "Resting Blood Pressure (in mm Hg)",
    "chol": "Serum Cholesterol Level (in mg/dL)",
    "fbs": "Fasting Blood Sugar > 120 mg/dL (1 = Yes, 0 = No)",
    "restecg": "Resting ECG Results (0 = Normal, 1 = ST-T Abnormality, 2 = Left Ventricular Hypertrophy)",
    "thalach": "Maximum Heart Rate Achieved",
    "exang": "Exercise Induced Angina (1 = Yes, 0 = No)",
    "oldpeak": "ST Depression Induced by Exercise", 
    "slope": "Slope of Peak Exercise ST Segment (0 = Upsloping, 1 = Flat, 2 = Downsloping)",
    "ca": "Number of Major Blood Vessels (0-3)",
    "thal": "Thalassemia Type (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)"
}

# User input fields
def get_user_input():
    st.markdown('<h2 class="sub-header">Patient Information</h2>', unsafe_allow_html=True)
    
    user_input = []
    
    # Create 3 columns for form layout
    col1, col2, col3 = st.columns(3)
    
    # Distribute form fields across columns
    columns = [col1, col2, col3]
    col_idx = 0
    
    for feature in feature_names:
        with columns[col_idx % 3]:
            if feature in feature_descriptions:
                if df[feature].dtype in ['int64', 'float64']:
                    min_val, max_val = int(df[feature].min()), int(df[feature].max())
                    value = int(df[feature].median())  # Default to median value
                    
                    # Special handling for certain fields
                    if feature == "age":
                        user_input.append(st.slider(feature_descriptions[feature], min_value=min_val, max_value=max_val, value=value))
                    elif feature == "sex":
                        user_input.append(st.radio(feature_descriptions[feature], options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male"))
                    elif feature == "cp":
                        options = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal", 3: "Asymptomatic"}
                        user_input.append(st.selectbox(feature_descriptions[feature], options=list(options.keys()), format_func=lambda x: options[x]))
                    elif feature == "fbs" or feature == "exang":
                        user_input.append(st.radio(feature_descriptions[feature], options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"))
                    else:
                        user_input.append(st.number_input(feature_descriptions[feature], min_value=min_val, max_value=max_val, value=value))
                else:
                    unique_values = sorted(df[feature].unique())
                    user_input.append(st.selectbox(feature_descriptions[feature], options=unique_values))
            col_idx += 1
    
    return np.array([list(map(float, user_input))])  # Ensure numerical input

# Progress bar for visual feedback
def show_progress():
    progress_bar = st.progress(0)
    for i in range(100):
        # Update progress bar
        progress_bar.progress(i + 1)
        # Simulate a delay
        import time
        time.sleep(0.01)
    progress_bar.empty()

# Collect user input
features = get_user_input()

# Create columns for button and results
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Prediction button
    if st.button("Predict Heart Disease"):
        show_progress()
        
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)[0]
        
        st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
        
        if prediction[0] == 1:
            st.markdown(f'<div class="result-positive">High Risk: Positive for Heart Disease<br>Confidence: {prediction_proba[1]:.2%}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-negative">Low Risk: Negative for Heart Disease<br>Confidence: {prediction_proba[0]:.2%}</div>', unsafe_allow_html=True)
        
        # Display a gauge chart for visualization
        st.markdown("### Risk Assessment")
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Create a gauge-like visualization
        risk_score = prediction_proba[1]
        colors = ['#d4edda', '#fff3cd', '#f8d7da']
        
        # Create the gauge
        ax.barh(0, 1, color='#f8f9fa', height=0.3)
        ax.barh(0, 0.33, color=colors[0], height=0.3)
        ax.barh(0, 0.33, left=0.33, color=colors[1], height=0.3)
        ax.barh(0, 0.34, left=0.66, color=colors[2], height=0.3)
        
        # Add a marker for the risk score
        ax.scatter(risk_score, 0, color='#073642', s=300, zorder=5, marker='v')
        
        # Add labels
        ax.text(0.16, -0.2, 'Low Risk', ha='center', va='center', fontsize=10)
        ax.text(0.5, -0.2, 'Moderate Risk', ha='center', va='center', fontsize=10)
        ax.text(0.83, -0.2, 'High Risk', ha='center', va='center', fontsize=10)
        ax.text(risk_score, 0.2, f'{risk_score:.2%}', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        
        st.pyplot(fig)
        
        # Display important factors
        st.markdown("### Key Factors Influencing Prediction")
        
        # Get feature importances if available in the model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Display top 5 features
            top_features = [(feature_names[i], importances[i]) for i in indices[:5]]
            
            # Create a bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=[feature_descriptions.get(f, f) for f, _ in top_features], 
                        y=[i for _, i in top_features], 
                        palette='viridis', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylabel('Importance')
            ax.set_title('Top 5 Important Features')
            
            st.pyplot(fig)
            
            # Display recommendations
            st.markdown("### Recommendations")
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            Based on your results, consider the following:
            
            1. **Consult a healthcare professional** for a comprehensive evaluation
            2. **Monitor your blood pressure and cholesterol** regularly
            3. **Maintain a heart-healthy lifestyle** with regular exercise and a balanced diet
            4. **Avoid smoking and limit alcohol consumption**
            5. **Manage stress** through relaxation techniques and adequate sleep
            """)
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #586e75; padding: 10px;">', unsafe_allow_html=True)
st.markdown("© 2023 Heart Disease Predictor | Developed for healthcare professionals")
st.markdown('</div>', unsafe_allow_html=True)