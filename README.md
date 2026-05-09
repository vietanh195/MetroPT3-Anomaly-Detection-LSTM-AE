# Metro AirFlow Anomaly Detection (Sentinel SCADA)

Welcome to the **Metro AirFlow Anomaly Detection** project. This system provides real-time anomaly detection for air compressor telemetry data using an **LSTM Autoencoder** model. The project features a robust FastAPI backend for AI inference and a modern, real-time React frontend (Sentinel SCADA) for system monitoring and Root Cause Analysis (XAI).

## Features

- **Real-Time Data Processing**: Simulates streaming data from the MetroPT3 dataset over WebSockets to represent live train air compressor telemetry.
- **AI Anomaly Detection**: Uses a PyTorch-based LSTM Autoencoder to calculate Mean Absolute Error (MAE) and detect anomalous patterns based on sliding window data.
- **Explainable AI (XAI)**: When an anomaly is detected, the system calculates the reconstruction error weights for each feature, automatically highlighting the root cause component.
- **Sentinel SCADA Dashboard**:
  - Dynamic graphs built with **Recharts**.
  - Futuristic, neon-styled intuitive UI for industrial monitoring.
  - Interactive Threshold controls and immediate Warning/Alarm states.

---

## Project Structure

```text
Metro_AirFlow_Anomaly_Detection/
│
├── backend/                       # Python FastAPI Backend
│   ├── main.py                    # API endpoints & WebSocket handling
│   ├── buffer.py                  # Sliding window calculation logic for features
│   ├── simulator.py               # Live data simulator using CSV data
│   └── requirements.txt*          # Python package dependencies
│
├── frontend/                      # React (Vite) Frontend
│   ├── src/                       # React components & CSS
│   │   ├── App.jsx                # Main Dashboard logic
│   │   ├── App.css                # Scada & Grid UI Styles
│   │   └── index.css              # Global themes
│   ├── package.json               # Node application dependencies
│   └── vite.config.js             # Vite configuration
│
├── *.py                           # Machine Learning & Utils
│   ├── train_model.py             # Script to train the LSTM Autoencoder
│   ├── test_model.py              # Script to test and evaluate the model
│   ├── preprocessing.py           # Data cleaning and feature engineering preparation
│   └── inspect_files.py           # Helper script to inspect raw files
│
├── model/                         # Saved Model Artifacts
│   ├── best_lstm_autoencoder.pth  # Trained PyTorch weights
│   └── minmax_scaler.pkl          # Scikit-learn scaler for data normalization
│
└── MetroPT3(AirCompressor).csv    # Raw Time-Series Dataset
```

---

## Setup & Installation

### 1. Backend (FastAPI / Machine Learning)

Make sure you have Python 3.9+ installed.

1. **Navigate to root directory** (or create a virtual environment here):

   ```bash
   cd Metro_AirFlow_Anomaly_Detection
   python -m venv venv
   source venv/bin/activate       # On Linux/Mac
   venv\Scripts\activate          # On Windows
   ```

2. **Install Python dependencies** (FastAPI, PyTorch, Pandas, Scikit-learn, etc.):

   ```bash
   pip install fastapi uvicorn pandas numpy torch scikit-learn websockets
   ```

   _(Note: Adjust the package installation depending on your specific requirements and CUDA environment)._

3. **Start the API Server**:

   ```bash
   cd backend
   python -m uvicorn main:app --reload
   ```

   The API provides WebSocket live telemetry at `ws://localhost:8000/ws/metrics` and REST points for controlling Demo Scenarios.

### 2. Frontend (React + Vite)

Make sure you have Node.js (v16+) installed.

1. **Navigate to the frontend directory**:

   ```bash
   cd Metro_AirFlow_Anomaly_Detection/frontend
   ```

2. **Install node dependencies**:

   ```bash
   npm install
   ```

3. **Run the development server**:
   ```bash
   npm run dev
   ```
   The dashboard will be available at `http://localhost:5173`.

---

## How to Use Demo Scenarios

Once both servers are running, access the user interface.
You can switch the simulation mode from the dropdown menu in the top right corner of the **SENTINEL SCADA** dashboard:

- **DEMO: NORMAL**: Feeds nominal data from the dataset. MAE will remain within the optimal range.
- **DEMO: WARNING**: Intentionally alters some metrics slightly. MAE begins to raise towards limits, triggering the yellow warning interface.
- **DEMO: ALARM**: Actively injects critical system failure metrics. MAE violently spikes past the threshold, changing the interface to red, highlighting the RCA feature graph, and generating a system-wide popup warning.

---

## Training the Model Customizations

If you wish to retrain the model, modify hyperparameters in `train_model.py` and run:

```bash
python train_model.py
```

This script relies on `MetroPT3(AirCompressor).csv`. Ensure the dataset exists locally before initiating training. It will save `best_lstm_autoencoder.pth` once completed.

## Authors

**University Thesis (ĐATN) Project** - Developed for tracking and analyzing AirFlow System Anomalies in Metro Networks.
