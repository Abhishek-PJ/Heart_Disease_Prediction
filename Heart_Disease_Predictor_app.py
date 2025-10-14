import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open(r'C:\Users\abhia\OneDrive\Desktop\Heart_Attack_Predictor\trained_model.sav', 'rb'))

def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return 'The person has a diseased heart' if prediction[0] == 0 else 'The person has a healthy heart'

# Streamlit UI
def main():
    st.set_page_config(page_title='Heart Disease Predictor', layout='centered')
    
    # Header
    st.markdown('<h1 style="color:#ff4d4d; text-align:center;">❤️ Heart Disease Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("User Input Parameters")
    st.sidebar.markdown("Provide the details below")
    
    # Input fields
    age = st.sidebar.slider("Age", 18, 100, 40)
    gender = st.sidebar.radio("Gender", ["Male", "Female"])
    chest_pain_type = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], index=2)
    resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.sidebar.number_input("Serum Cholesterol Level (mg/dL)", 100, 600, 200)
    fasting_blood_sugar = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dL", ["False", "True"])
    rest_ecg = st.sidebar.selectbox("Resting ECG Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"], index=0)
    max_heart_rate = st.sidebar.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exercise_angina = st.sidebar.radio("Exercise-Induced Angina", ["No", "Yes"])
    st_depression = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, step=0.1)
    st_slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"], index=1)
    num_major_vessels = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 1)
    thalassemia = st.sidebar.selectbox("Thalassemia Type", ["Normal", "Fixed Defect", "Reversible Defect"], index=2)
    
    # Convert categorical inputs to numerical
    gender = 1 if gender == "Male" else 0
    chest_pain_type = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_type)
    fasting_blood_sugar = 1 if fasting_blood_sugar == "True" else 0
    rest_ecg = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(rest_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)
    thalassemia = ["Normal", "Fixed Defect", "Reversible Defect"].index(thalassemia) + 1

    # Predict button
    if st.sidebar.button("Predict Heart Health", use_container_width=True):
        result = heart_disease_prediction([age, gender, chest_pain_type, resting_bp, cholesterol, fasting_blood_sugar, 
                                           rest_ecg, max_heart_rate, exercise_angina, st_depression, st_slope, 
                                           num_major_vessels, thalassemia])
        
        st.markdown("---")
        if "diseased" in result:
            st.error(f"⚠️ {result}")
        else:
            st.success(f"✅ {result}")
        st.markdown("---")
        
    # Footer
    st.markdown("<p style='text-align:center;color:gray;'>Developed with ❤️ by Abhia</p>", unsafe_allow_html=True)
    
if __name__ == '__main__':
    main()