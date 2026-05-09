import pandas as pd
import torch
import joblib

print("--- Checking CSV ---")
try:
    df = pd.read_csv("MetroPT3(AirCompressor).csv", nrows=5)
    print("Columns:", list(df.columns))
except Exception as e:
    print("CSV Error:", e)

print("\n--- Checking Model ---")
try:
    model = torch.load("best_lstm_autoencoder.pth", map_location='cpu')
    if isinstance(model, dict) and "state_dict" in model:
        print("It is a state_dict checkpoint.")
    elif isinstance(model, dict):
        print("It might be a standalone state_dict dictionary. Keys:", list(model.keys())[:5])
    else:
        print("It is a full model object. Type:", type(model))
except Exception as e:
    print("Model Error:", e)

print("\n--- Checking Scaler ---")
try:
    scaler = joblib.load("minmax_scaler.pkl")
    print("Scaler type:", type(scaler))
    print("Scaler features out:", scaler.n_features_in_)
except Exception as e:
    print("Scaler Error:", e)
