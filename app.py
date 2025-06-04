from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model and scaler with error handling
try:
    model = joblib.load('trained_data/student_pass_model.pkl')
    scaler = joblib.load('trained_data/scaler.pkl')
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.route('/')
def index():
    return "Welcome to the Student Performance Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            return jsonify({'error': 'Model or scaler not properly loaded'}), 500

        # Get the input data from the POST request
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Ensure all required fields are present in the input
        required_columns = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Attendance (%)', 'Stress_Level (1-10)']
        
        missing_fields = [field for field in required_columns if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400

        # Convert to numeric values and validate
        try:
            study_hours = float(data['Study_Hours_per_Week'])
            sleep_hours = float(data['Sleep_Hours_per_Night'])
            attendance = float(data['Attendance (%)'])
            stress_level = float(data['Stress_Level (1-10)'])
        except (ValueError, TypeError):
            return jsonify({'error': 'All input values must be numeric'}), 400

        # Realistic validation ranges
        if not (0 <= study_hours <= 80):
            return jsonify({'error': 'Study hours must be between 0 and 80 per week'}), 400
        if not (0 <= sleep_hours <= 24):
            return jsonify({'error': 'Sleep hours must be between 0 and 24 per night'}), 400
        if not (0 <= attendance <= 100):
            return jsonify({'error': 'Attendance must be between 0 and 100 percent'}), 400
        if not (1 <= stress_level <= 10):
            return jsonify({'error': 'Stress level must be between 1 and 10'}), 400

        # Additional realistic checks
        warnings = []
        if study_hours > 40:
            warnings.append("Very high study hours - this might indicate burnout risk")
        if sleep_hours < 6:
            warnings.append("Insufficient sleep may negatively impact performance")
        if attendance < 75:
            warnings.append("Low attendance is a strong predictor of poor performance")

        # Create input DataFrame with exact column names as used in training
        input_data = pd.DataFrame([{
            'Study_Hours_per_Week': study_hours,
            'Sleep_Hours_per_Night': sleep_hours,
            'Attendance (%)': attendance,
            'Stress_Level (1-10)': stress_level
        }])

        logger.info(f"Input data: {input_data.iloc[0].to_dict()}")

        # Ensure column order matches training data
        expected_columns = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Attendance (%)', 'Stress_Level (1-10)']
        input_data = input_data[expected_columns]

        # Standardize the input data using the scaler
        input_data_scaled = scaler.transform(input_data)
        logger.info(f"Scaled input: {input_data_scaled}")

        # Get prediction probabilities using the trained model
        prediction_probabilities = model.predict_proba(input_data_scaled)
        logger.info(f"Prediction probabilities: {prediction_probabilities}")

        # Calculate the probability of passing (assuming class 1 is pass)
        # Check if model has 2 classes
        if len(prediction_probabilities[0]) != 2:
            return jsonify({'error': 'Model output format unexpected'}), 500

        pass_probability = prediction_probabilities[0][1] * 100

        # More nuanced result determination
        if pass_probability >= 70:
            result = 'pass'
            confidence = 'high'
        elif pass_probability >= 50:
            result = 'pass'
            confidence = 'moderate'
        elif pass_probability >= 30:
            result = 'fail'
            confidence = 'moderate'
        else:
            result = 'fail'
            confidence = 'high'

        # Create response
        response = {
            'result': result,
            'percentage': f"{pass_probability:.2f}%",
            'confidence': confidence,
            'interpretation': get_interpretation(study_hours, sleep_hours, attendance, stress_level, pass_probability)
        }

        if warnings:
            response['warnings'] = warnings

        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

def get_interpretation(study_hours, sleep_hours, attendance, stress_level, pass_prob):
    """Provide interpretation of the prediction based on input factors"""
    interpretation = []
    
    # Study hours analysis
    if study_hours < 10:
        interpretation.append("Low study hours may be insufficient for good performance")
    elif study_hours > 35:
        interpretation.append("Very high study hours - consider if this is sustainable")
    else:
        interpretation.append("Study hours are in a reasonable range")
    
    # Sleep analysis
    if sleep_hours < 6:
        interpretation.append("Insufficient sleep can significantly impact academic performance")
    elif sleep_hours > 9:
        interpretation.append("Adequate sleep supports good academic performance")
    
    # Attendance analysis
    if attendance < 80:
        interpretation.append("Low attendance is strongly correlated with poor performance")
    elif attendance > 90:
        interpretation.append("Excellent attendance supports academic success")
    
    # Stress analysis
    if stress_level > 7:
        interpretation.append("High stress levels can negatively impact performance")
    elif stress_level < 3:
        interpretation.append("Very low stress might indicate lack of academic challenge")
    
    return " | ".join(interpretation)

@app.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint to get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Try to get model information
        model_type = type(model).__name__
        
        info = {
            'model_type': model_type,
            'feature_names': ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Attendance (%)', 'Stress_Level (1-10)'],
            'status': 'loaded'
        }
        
        # If it's a sklearn model, try to get more info
        if hasattr(model, 'n_features_in_'):
            info['n_features'] = model.n_features_in_
        if hasattr(model, 'classes_'):
            info['classes'] = model.classes_.tolist()
            
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)