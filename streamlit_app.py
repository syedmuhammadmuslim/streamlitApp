import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load and display accuracy
with open('accuracy.txt', 'r') as file:
    accuracy = file.read()

st.title("Model Accuracy and Real-Time Prediction")
st.write(f"Model Accuracy: {accuracy}")

# User input for real-time prediction
st.header("Real-Time Prediction")

# Assume the model expects the same input features as X_test
input_data = []
for col in range(len(X_test.columns)):
    input_value = st.number_input(f"Input for feature {col + 1}", value=0.0)
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
