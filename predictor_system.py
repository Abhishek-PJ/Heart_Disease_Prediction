# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import pickle

# Loading the saved model
# Use raw string (r'') or double backslashes to avoid escape character issues
loaded_model = pickle.load(open(r'C:\Users\abhia\OneDrive\Desktop\Heart_Attack_Predictor\trained_model.sav', 'rb'))

# Input data (replace with actual values)
input_data = (68, 1, 0, 144, 193, 1, 1, 141, 0, 3.4, 1, 2, 3)

# Changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make a prediction
prediction = loaded_model.predict(input_data_reshaped)  # Use loaded_model, not model
print(prediction)

# Interpret the prediction
if prediction[0] == 0:
    print('No disease')
else:
    print('Diseased Heart')