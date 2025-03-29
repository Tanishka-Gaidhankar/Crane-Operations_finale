import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RandomForest
import pickle
from flask import Flask, jsonify
from flask_cors import CORS
import time
import threading

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your trained RandomForest model
# Assuming you've already trained and saved your model
with open('randomforest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Global variable to store the latest predictions
latest_predictions = {
    "overallHealth": 87,
    "brakeWear": 72,
    "cableTension": 65,
    "loadCapacity": 92,
    "timestamp": time.time()
}

# Function to generate synthetic data for predictions
def generate_synthetic_data():
    # Replace this with your actual synthetic data generation logic
    # This is just a simple example
    data = {
        'temperature': np.random.uniform(20, 35),
        'vibration': np.random.uniform(0.1, 2.5),
        'load': np.random.uniform(500, 2000),
        'operation_hours': np.random.uniform(0, 24),
        'pressure': np.random.uniform(100, 300),
        'humidity': np.random.uniform(30, 90),
        'noise_level': np.random.uniform(60, 100)
    }
    return pd.DataFrame([data])

# Function to run predictions in the background
def background_prediction():
    global latest_predictions
    
    while True:
        try:
            # Generate synthetic data
            synthetic_data = generate_synthetic_data()
            
            # Make predictions using your Random Forest model
            # This is a simplified example - adjust based on your model's outputs
           # Make predictions using your Random Forest model
# For classification models that output class probabilities
            predictions_proba = model.predict_proba(synthetic_data)

# Extract probabilities of the positive class (index 1) for each target
# Assuming model outputs probabilities for each target in order
            health_prob = predictions_proba[0][0][1]  # First target, positive class
            brake_prob = predictions_proba[1][0][1]   # Second target, positive class
            cable_prob = predictions_proba[2][0][1]   # Third target, positive class
            load_prob = predictions_proba[3][0][1]    # Fourth target, positive class

# Convert probabilities to dashboard metrics
            health_score = int(health_prob * 100)
            brake_wear = int(100 - (brake_prob * 30))
            cable_tension = int(100 - (cable_prob * 35))
            load_capacity = int(100 - (load_prob * 8))
            # Update the latest predictions
            latest_predictions = {
                "overallHealth": health_score,
                "brakeWear": brake_wear,
                "cableTension": cable_tension,
                "loadCapacity": load_capacity,
                "timestamp": time.time()
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
        
        # Sleep for a few seconds before the next prediction
        time.sleep(5)

# API endpoint to get the latest predictions
@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    return jsonify(latest_predictions)

# Start the background prediction thread
prediction_thread = threading.Thread(target=background_prediction, daemon=True)
prediction_thread.start()

if __name__ == '__main__':
    app.run(debug=True, port=5000)