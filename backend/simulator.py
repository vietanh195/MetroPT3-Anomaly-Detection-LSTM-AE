import pandas as pd
import numpy as np
from typing import Dict, Any
from buffer import RealtimeBuffer
from ai_engine import engine

class DataSimulator:
    def __init__(self, csv_path="../MetroPT3(AirCompressor).csv"):
        self.csv_path = csv_path
        print(f"Loading partial data from {csv_path}...")
        try:
            # Tải một lượng lớn data để cover hết các kịch bản
            self.df = pd.read_csv(csv_path, nrows=950000)
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='mixed', dayfirst=True)
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)
            print("Data loaded successfully.")
        except Exception as e:
            print("Error loading data:", e)
            self.df = pd.DataFrame()

        self.scenario = "Normal"
        self.current_idx = 0
        self.buffer = RealtimeBuffer(window_size=60, rolling_window=30)
        
        # Mapping indices for different scenarios
        # Ground Truth Alarm: 2020-04-18 00:00:00 -> 23:59:00
        # Data frequency is ~6 rows/min, meaning 1 hour = 360 rows.
        
        if not self.df.empty:
            # Find approximate indices
            # Normal: March 1st Data 
            normal_mask = self.df['timestamp'] >= pd.to_datetime('2020-02-28 00:00:00')
            caution_mask = self.df['timestamp'] >= pd.to_datetime('2020-05-23 09:42:00')
            warning_mask = self.df['timestamp'] >= pd.to_datetime('2020-04-17 23:33:00')
            
            idx_normal = normal_mask.idxmax() if normal_mask.any() else 0
            idx_caution = caution_mask.idxmax() if caution_mask.any() else 200000
            idx_warning = warning_mask.idxmax() if warning_mask.any() else 250000
        else:
            idx_normal, idx_caution, idx_warning = 0, 10000, 20000

        self.scenarios = {
            "Normal": {"start": idx_normal, "end": idx_normal + 10000},
            "Caution": {"start": idx_caution, "end": idx_caution + 10000},
            "Warning": {"start": idx_warning, "end": idx_warning + 10000}
        }
        
        self.last_ai_result = {
            "mae": 0.0,
            "status": "Green",
            "idle": False,
            "rca": []
        }
        
        self.set_scenario("Normal")

    def set_scenario(self, scenario_name: str) -> bool:
        if scenario_name in self.scenarios:
            self.scenario = scenario_name
            self.current_idx = self.scenarios[scenario_name]["start"]
            print(f"Switched scenario to {scenario_name}")
            self.warmup_buffer()
            return True
        return False

    def warmup_buffer(self):
        """Khởi động Buffer bằng cách tua nhanh 90 phút dữ liệu trước current_idx"""
        print("Warming up buffer (loading last 90 minutes)...")
        if self.df.empty: return
        self.buffer = RealtimeBuffer(window_size=60, rolling_window=30)
        
        # 90 phút * 6 dòng/phút = 540 rows
        start_warmup = max(0, self.current_idx - 540)
        if start_warmup < self.current_idx:
            warmup_chunk = self.df.iloc[start_warmup:self.current_idx]
            for _, row in warmup_chunk.iterrows():
                row_dict = row.to_dict()
                feat_window = self.buffer.push_raw_record(row_dict)
                if feat_window is not None:
                    # Chạy 1 lần Inference mồi
                    op_state = 1 if row_dict.get('Motor_current', 0) > 1.0 else 0
                    self.last_ai_result = engine.predict(feat_window, operating_state=op_state)
        print("Warmup complete. Ready for real-time.")

    def get_current_data(self) -> Dict[str, Any]:
        if self.df.empty: return {}

        if self.current_idx >= self.scenarios[self.scenario]["end"]:
            self.current_idx = self.scenarios[self.scenario]["start"]
            self.warmup_buffer()

        row = self.df.iloc[self.current_idx].to_dict()
        self.current_idx += 1
        
        # Chèn vào Buffer
        feat_window = self.buffer.push_raw_record(row)
        
        # Nếu đủ 1 phút, nhận được cửa sổ 60 features -> update AI
        if feat_window is not None:
            op_state = 1 if row.get('Motor_current', 0) > 1.0 else 0
            ai_result = engine.predict(feat_window, operating_state=op_state)
            self.last_ai_result = ai_result
            
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, pd.Timestamp):
                clean_row[k] = str(v)
            elif isinstance(v, (np.floating, np.integer)):
                clean_row[k] = float(v) if isinstance(v, np.floating) else int(v)
            else:
                clean_row[k] = v

        # Calculate time to next row
        time_to_next = 1.0
        try:
            if self.current_idx < self.scenarios[self.scenario]["end"] and self.current_idx < len(self.df):
                next_t = self.df.iloc[self.current_idx]['timestamp']
                curr_t = row['timestamp']
                delta = (next_t - curr_t).total_seconds()
                time_to_next = float(max(0.1, delta))  # At least 0.1s to avoid infinite loop locking
        except Exception:
            pass

        return {
            "raw_data": clean_row,
            "ai_inference": self.last_ai_result,
            "time_to_next_seconds": time_to_next
        }

simulator = DataSimulator()
