# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 18:36:17 2025

@author: abhia
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open(r'C:\Users\abhia\OneDrive\Desktop\Heart_Attack_Predictor\trained_model.sav', 'rb'))


# creating a function for Prediction

def heart_disease_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person has healthy heart'
    else:
      return 'The person has diseased heart'
  
    
  
def main():
    
    
    # giving a title
    st.title('Heart Disease Predictor Web App')
    
    
    # getting the input data from the user
  
    
    age = st.text_input('Enter age')
    Gender = st.text_input('Gender')
    cp = st.text_input('CP value')
    trestbps = st.text_input('trestbps value')
    cholestrol = st.text_input('cholestrol Level')
    fbs = st.text_input('fbs value')
    restecg = st.text_input('restecg value')
    thalach = st.text_input('thalach value')
    exang = st.text_input('exang value')
    oldpeak = st.text_input('oldpeak value')
    slope = st.text_input('slope value')
    ca = st.text_input('ca value')
    thal = st.text_input('thal value')

    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = heart_disease_prediction([age, Gender, cp, trestbps, cholestrol, fbs, restecg, thalach,exang,oldpeak,slope,ca,thal])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    