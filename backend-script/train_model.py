# Save this as train_model.py and run it
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Create synthetic training data
n_samples = 1000
features = ['temperature', 'vibration', 'load', 'operation_hours', 'pressure', 'humidity', 'noise_level']
X = pd.DataFrame({
    'temperature': np.random.uniform(20, 35, n_samples),
    'vibration': np.random.uniform(0.1, 2.5, n_samples),
    'load': np.random.uniform(500, 2000, n_samples),
    'operation_hours': np.random.uniform(0, 24, n_samples),
    'pressure': np.random.uniform(100, 300, n_samples),
    'humidity': np.random.uniform(30, 90, n_samples),
    'noise_level': np.random.uniform(60, 100, n_samples)
})

# Create target classes (health status, brake status, etc.)
y = np.column_stack([
    np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8]),  # health status
    np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),  # brake wear
    np.random.choice([0, 1], size=n_samples, p=[0.35, 0.65]),  # cable tension
    np.random.choice([0, 1], size=n_samples, p=[0.1, 0.9])   # load capacity
])

# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save the model
with open('randomforest_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as randomforest_model.pkl")