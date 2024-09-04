import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
# import pickle
import joblib

# # Load the pre-trained model
# with open('K-Nearest Neighborsmodel.pkl', 'rb') as file:
#     model = pickle.load(file)


model = joblib.load('K-Nearest Neighborsmodel.pkl')  # Ensure this is the model, not data

# Load and display accuracy
with open('accuracy.txt', 'r') as file:
    accuracy = file.read()

st.title("Model Accuracy and Real-Time Prediction new title")
st.write(f"Model {accuracy}")

# User input for real-time prediction
st.header("Real-Time Prediction")

# Load the test data
test_data = pd.read_csv('mobile_price_range_data.csv')

# Assuming the last column is the target
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Assume the model expects the same input features as X_test
input_data = []
for col in X_test.columns:
    input_value = st.number_input(f"Input for {col}", value=0.0)
    input_data.append(input_value)

# Convert to DataFrame for prediction
input_df = pd.DataFrame([input_data], columns=X_test.columns)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"Prediction: {prediction[0]}")

# Plot accuracy
st.header("Accuracy Plot")
st.bar_chart([float(accuracy.split(': ')[1])])
