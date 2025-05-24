from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('trained_data/student_pass_model.pkl')
scaler = joblib.load('trained_data/scaler.pkl')

@app.route('/')
def index():
    return "Welcome to the Student Performance Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the POST request
    data = request.json

    # Ensure all required fields are present in the input
    required_columns = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Attendance (%)', 'Stress_Level (1-10)']
    
    if not all(key in data for key in required_columns):
        return jsonify({'error': 'Invalid input data. Please provide all required fields.'}), 400

    # Prepare the input data
    input_data = pd.DataFrame([data])

    # Standardize the input data using the scaler
    input_data_scaled = scaler.transform(input_data)

    # Make prediction using the trained model
    prediction = model.predict(input_data_scaled)

    # Return the prediction result: 'pass' or 'fail'
    result = 'pass' if prediction[0] == 1 else 'fail'

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
