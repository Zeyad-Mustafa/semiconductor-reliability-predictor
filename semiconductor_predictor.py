# semiconductor_predictor.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class SemiconductorDataGenerator:
    def __init__(self, n_samples=1000, n_devices=5):
        self.n_samples = n_samples
        self.n_devices = n_devices
    
    def generate_data(self):
        """Generate simulated semiconductor sensor data"""
        data = []
        
        for device_id in range(self.n_devices):
            # Base parameters for each device
            base_temp = 50 + np.random.normal(0, 5)
            base_voltage = 3.3 + np.random.normal(0, 0.1)
            base_current = 0.5 + np.random.normal(0, 0.05)
            
            # Degradation parameters
            temp_drift = np.random.uniform(0.001, 0.005)  # Temperature increase over time
            voltage_drift = np.random.uniform(-0.0001, 0.0001)
            current_drift = np.random.uniform(0.0001, 0.001)
            
            # Failure time (when device fails)
            failure_time = np.random.uniform(800, 1200)
            
            for t in range(self.n_samples):
                # Normal degradation
                temp = base_temp + t * temp_drift + np.random.normal(0, 1)
                voltage = base_voltage + t * voltage_drift + np.random.normal(0, 0.05)
                current = base_current + t * current_drift + np.random.normal(0, 0.01)
                
                # Add failure acceleration near failure time
                if t > failure_time * 0.8:
                    accel_factor = (t - failure_time * 0.8) / (failure_time * 0.2)
                    temp += accel_factor * 10
                    current += accel_factor * 0.2
                    voltage -= accel_factor * 0.3
                
                # Binary failure status
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
            
            # Scale features
            if len(values) > 0:
                scaled_values = self.scaler.fit_transform(values)
                
                # Create sequences
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

def run_analysis():
    """Main analysis function"""
    st.title("🔬 Semiconductor Reliability Predictor")
    
    # Generate data
    st.header("Data Generation")
    if st.button("Generate Sensor Data"):
        with st.spinner("Generating semiconductor sensor data..."):
            generator = SemiconductorDataGenerator(n_samples=800, n_devices=3)
            data = generator.generate_data()
            st.session_state['data'] = data
            st.success("Data generated successfully!")
    
    if 'data' not in st.session_state:
        st.info("Click 'Generate Sensor Data' to start")
        return
    
    data = st.session_state['data']
    
    # Display data overview
    st.header("Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(data))
    with col2:
        st.metric("Devices", data['device_id'].nunique())
    with col3:
        st.metric("Failed Samples", data['failed'].sum())
    
    # Visualizations
    st.header("Sensor Data Visualization")
    
    # Time series plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sensors = ['temperature', 'voltage', 'current', 'leakage']
    
    for i, sensor in enumerate(sensors):
        ax = axes[i//2, i%2]
        for device_id in data['device_id'].unique():
            device_data = data[data['device_id'] == device_id]
            ax.plot(device_data['time'], device_data[sensor], alpha=0.7, label=f'Device {device_id}')
        ax.set_title(f'{sensor.title()} over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel(sensor.title())
        ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Model Training
    st.header("Model Training")
    if st.button("Train Reliability Model"):
        with st.spinner("Training LSTM model..."):
            predictor = ReliabilityPredictor(sequence_length=30)
            
            # Prepare data
            X, y = predictor.prepare_sequences(data)
            
            # Train model
            history = predictor.train(X, y, epochs=10)
            
            # Store in session
            st.session_state['predictor'] = predictor
            st.session_state['X'] = X
            st.session_state['y'] = y
            
            st.success("Model trained successfully!")
            
            # Show training history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.plot(history.history['loss'], label='Training Loss')
            ax1.plot(history.history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.legend()
            
            ax2.plot(history.history['accuracy'], label='Training Accuracy')
            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Predictions
    if 'predictor' in st.session_state:
        st.header("Failure Predictions")
        
        predictor = st.session_state['predictor']
        X = st.session_state['X']
        y = st.session_state['y']
        
        # Make predictions
        predictions = predictor.predict(X)
        
        # Show prediction results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction Accuracy", f"{((predictions.flatten() > 0.5) == y).mean():.3f}")
        with col2:
            st.metric("Mean Absolute Error", f"{mean_absolute_error(y, predictions):.3f}")
        
        # Prediction visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(range(len(y)), y, alpha=0.5, label='Actual Failures', s=20)
        ax.scatter(range(len(predictions)), predictions, alpha=0.7, label='Predicted Probability', s=20)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Failure Probability')
        ax.set_title('Actual vs Predicted Failures')
        ax.legend()
        st.pyplot(fig)
        
        # Reliability metrics
        st.header("Reliability Metrics")
        
        # Calculate time to failure for each device
        device_reliability = []
        for device_id in data['device_id'].unique():
            device_data = data[data['device_id'] == device_id]
            first_failure = device_data[device_data['failed'] == 1]
            if len(first_failure) > 0:
                ttf = first_failure['time'].iloc[0]
            else:
                ttf = data['time'].max()
            device_reliability.append({'device_id': device_id, 'time_to_failure': ttf})
        
        reliability_df = pd.DataFrame(device_reliability)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(reliability_df['device_id'], reliability_df['time_to_failure'])
        ax.set_xlabel('Device ID')
        ax.set_ylabel('Time to Failure')
        ax.set_title('Device Reliability Comparison')
        st.pyplot(fig)

if __name__ == "__main__":
    # For Streamlit
    run_analysis()
    
    # For standalone execution
    if not hasattr(st, '_is_running_with_streamlit'):
        # Generate data
        generator = SemiconductorDataGenerator(n_samples=500, n_devices=2)
        data = generator.generate_data()
        
        # Train model
        predictor = ReliabilityPredictor(sequence_length=30)
        X, y = predictor.prepare_sequences(data)
        history = predictor.train(X, y, epochs=5)
        
        # Make predictions
        predictions = predictor.predict(X)
        accuracy = ((predictions.flatten() > 0.5) == y).mean()
        
        print(f"Dataset shape: {data.shape}")
        print(f"Model accuracy: {accuracy:.3f}")
        print(f"Failed samples: {data['failed'].sum()}")
        
        # Plot results
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(data.groupby('time')['temperature'].mean())
        plt.title('Average Temperature Over Time')
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(y)), y, alpha=0.5, label='Actual')
        plt.scatter(range(len(predictions)), predictions, alpha=0.7, label='Predicted')
        plt.title('Failure Predictions')
        plt.legend()
        plt.show()