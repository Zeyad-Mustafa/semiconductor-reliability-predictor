#  Semiconductor Reliability Predictor

A machine learning system that predicts semiconductor device failures using time-series sensor data and LSTM neural networks.

##  Quick Start

```bash
# Clone and setup
git clone https://github.com/Zeyad-Mustafa/semiconductor-reliability-predictor.git
cd semiconductor-reliability-predictor

# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run src/main.py

# Or run standalone analysis
python src/main.py
```

##  Project Structure

```
semiconductor-reliability-predictor/
├── src/
│   ├── main.py                 # Main application
│   ├── data_generator.py       # Sensor data simulation
│   ├── model.py               # LSTM reliability model
│   └── utils.py               # Helper functions
├── notebooks/
│   └── analysis.ipynb         # Jupyter analysis notebook
├── data/
│   └── sample_data.csv        # Generated sample data
├── requirements.txt           # Dependencies
├── setup.py                  # Installation setup
└── README.md                 # This file
```

##  Features

- **Time-Series Modeling**: LSTM networks for failure prediction
- **Physics-Informed**: Realistic degradation patterns (electromigration, BTI)
- **Interactive Dashboard**: Real-time visualization with Streamlit
- **Anomaly Detection**: Early warning system for device failures
- **Multiple Sensors**: Temperature, voltage, current, leakage monitoring

##  Tech Stack

- **Python 3.8+**
- **TensorFlow/Keras** - Deep learning models
- **Streamlit** - Interactive dashboard
- **Pandas/NumPy** - Data processing
- **Matplotlib** - Visualization
- **Scikit-learn** - ML utilities

##  Model Performance

- **Accuracy**: ~85-90% failure prediction
- **Early Detection**: Predicts failures 50-100 time steps ahead
- **Multiple Devices**: Handles 5+ concurrent device monitoring

##  Configuration

Edit parameters in `src/main.py`:
```python
# Data generation
N_SAMPLES = 1000      # Time steps per device
N_DEVICES = 5         # Number of devices to simulate

# Model parameters
SEQUENCE_LENGTH = 50  # LSTM input sequence length
EPOCHS = 20          # Training epochs
```

##  Usage Examples

### Generate Data
```python
from src.data_generator import SemiconductorDataGenerator
generator = SemiconductorDataGenerator(n_samples=800, n_devices=3)
data = generator.generate_data()
```

### Train Model
```python
from src.model import ReliabilityPredictor
predictor = ReliabilityPredictor(sequence_length=30)
X, y = predictor.prepare_sequences(data)
predictor.train(X, y, epochs=20)
```

### Make Predictions
```python
predictions = predictor.predict(X)
failure_probability = predictions[-1][0]  # Latest prediction
```

##  Sensor Data

The system monitors these key parameters:
- **Temperature**: Thermal stress indicator
- **Voltage**: Power supply stability
- **Current**: Device load and efficiency
- **Leakage**: Insulation degradation

##  Performance Tips

- Use GPU acceleration: `pip install tensorflow-gpu`
- Increase sequence length for better accuracy
- Add more devices for robust training
- Tune LSTM architecture for your specific use case



##  License

MIT License - see LICENSE file for details


