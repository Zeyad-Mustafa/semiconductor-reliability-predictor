# src/data_generator.py
import numpy as np
import pandas as pd

class SemiconductorDataGenerator:
    def __init__(self, n_samples=1000, n_devices=5):
        self.n_samples = n_samples
        self.n_devices = n_devices
    
    def generate_data(self):
        """Generate simulated semiconductor sensor data"""
        data = []
        
        for device_id in range(self.n_devices):
            base_temp = 50 + np.random.normal(0, 5)
            base_voltage = 3.3 + np.random.normal(0, 0.1)
            base_current = 0.5 + np.random.normal(0, 0.05)
            
            temp_drift = np.random.uniform(0.001, 0.005)
            voltage_drift = np.random.uniform(-0.0001, 0.0001)
            current_drift = np.random.uniform(0.0001, 0.001)
            failure_time = np.random.uniform(800, 1200)
            
            for t in range(self.n_samples):
                temp = base_temp + t * temp_drift + np.random.normal(0, 1)
                voltage = base_voltage + t * voltage_drift + np.random.normal(0, 0.05)
                current = base_current + t * current_drift + np.random.normal(0, 0.01)
                
                if t > failure_time * 0.8:
                    accel_factor = (t - failure_time * 0.8) / (failure_time * 0.2)
                    temp += accel_factor * 10
                    current += accel_factor * 0.2
                    voltage -= accel_factor * 0.3
                
                failed = 1 if t > failure_time else 0
                
                data.append({
                    'device_id': device_id,
                    'time': t,
                    'temperature': temp,
                    'voltage': voltage,
                    'current': current,
                    'leakage': current * 0.1 + np.random.normal(0, 0.001),
                    'failed': failed
                })
        
        return pd.DataFrame(data)

# ================================

# src/model.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class ReliabilityPredictor:
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        
    def prepare_sequences(self, data, features=['temperature', 'voltage', 'current', 'leakage']):
        """Prepare sequences for LSTM training"""
        X, y = [], []
        
        for device_id in data['device_id'].unique():
            device_data = data[data['device_id'] == device_id]
            values = device_data[features].values
            targets = device_data['failed'].values
            
            if len(values) > 0:
                scaled_values = self.scaler.fit_transform(values)
                
                for i in range(len(scaled_values) - self.sequence_length):
                    X.append(scaled_values[i:i + self.sequence_length])
                    y.append(targets[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self, n_features):
        """Build LSTM model for failure prediction"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, n_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, X_train, y_train, epochs=20):
        """Train the reliability prediction model"""
        self.model = self.build_model(X_train.shape[2])
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        return history
    
    def predict(self, X):
        """Predict failure probability"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)

# ================================

# src/utils.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_sensor_data(data, sensors=['temperature', 'voltage', 'current', 'leakage']):
    """Plot sensor data over time for all devices"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for i, sensor in enumerate(sensors):
        ax = axes[i//2, i%2]
        for j, device_id in enumerate(data['device_id'].unique()):
            device_data = data[data['device_id'] == device_id]
            ax.plot(device_data['time'], device_data[sensor], 
                   alpha=0.7, label=f'Device {device_id}', color=colors[j % len(colors)])
        ax.set_title(f'{sensor.title()} over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel(sensor.title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def calculate_reliability_metrics(data):
    """Calculate reliability metrics for each device"""
    device_reliability = []
    
    for device_id in data['device_id'].unique():
        device_data = data[data['device_id'] == device_id]
        first_failure = device_data[device_data['failed'] == 1]
        
        if len(first_failure) > 0:
            ttf = first_failure['time'].iloc[0]
        else:
            ttf = data['time'].max()
        
        device_reliability.append({
            'device_id': device_id,
            'time_to_failure': ttf,
            'total_samples': len(device_data),
            'failed_samples': device_data['failed'].sum()
        })
    
    return pd.DataFrame(device_reliability)

def plot_predictions(y_true, y_pred, title="Prediction Results"):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(10, 6))
    
    # Sample for visualization if too many points
    if len(y_true) > 1000:
        indices = np.random.choice(len(y_true), 1000, replace=False)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred
    
    plt.scatter(range(len(y_true_sample)), y_true_sample, 
               alpha=0.6, s=20, label='Actual', color='red')
    plt.scatter(range(len(y_pred_sample)), y_pred_sample, 
               alpha=0.6, s=20, label='Predicted', color='blue')
    plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Threshold')
    plt.xlabel('Sample Index')
    plt.ylabel('Failure Probability')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ================================

