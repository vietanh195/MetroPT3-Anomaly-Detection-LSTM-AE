import torch
import torch.nn as nn
import joblib
import numpy as np
from collections import deque

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[16, 8], dropout=0.5):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.LSTM(prev_dim, h_dim, batch_first=True))
            prev_dim = h_dim
        self.encoder = nn.ModuleList(encoder_layers)
        
        self.latent_dim = hidden_dims[-1]
        
        decoder_layers = []
        prev_dim = self.latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.LSTM(prev_dim, h_dim, batch_first=True))
            prev_dim = h_dim
        self.decoder = nn.ModuleList(decoder_layers)
        
        self.output_layer = nn.Linear(hidden_dims[0], input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        seq_len = x.size(1)
        out = x
        for lstm in self.encoder:
            out, _ = lstm(out)
            out = self.dropout(out)
        latent = out[:, -1, :]
        repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)
        out = repeated
        for lstm in self.decoder:
            out, _ = lstm(out)
            out = self.dropout(out)
        out = self.output_layer(out)
        return out

class AIEngine:
    def __init__(self, model_path='../model/best_lstm_autoencoder.pth', scaler_path='../model/minmax_scaler.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"AI Engine loading on {self.device}...")
        
        # 1. Khởi tạo Scaler
        try:
            self.scaler = joblib.load(scaler_path)
            self.feature_names = getattr(self.scaler, 'feature_names_in_', [f'feat_{i}' for i in range(25)])
            print("Scaler loaded.")
        except Exception as e:
            print(f"Warning: Scaler not found ({e}), using dummy.")
            self.scaler = None
            self.feature_names = [f'feat_{i}' for i in range(25)]

        # 2. Khởi tạo Trọng số Loss (Khớp với train_model.py)
        self.weights = np.ones(25, dtype=np.float32)
        important_keywords = ['operating_state', 'MPG', 'DV_eletric']
        for i, name in enumerate(self.feature_names):
            if any(kw in name for kw in important_keywords):
                self.weights[i] = 1.5

        # 3. Khởi tạo Ngưỡng Động (Adaptive Threshold)
        # Giữ lịch sử MAE của 24h (1440 phút)
        self.mae_history = deque(maxlen=1440)
        self.base_threshold = 0.12 # Ngưỡng sàn dự phòng (Nên cập nhật bằng percentile 99.8 của X_val)

        # 4. Tải Model
        try:
            self.model = LSTMAutoencoder(input_dim=25, hidden_dims=[16, 8], dropout=0.5)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("PyTorch Model loaded successfully.")
        except Exception as e:
            print(f"Warning: PyTorch model not found ({e}).")
            self.model = None

    def predict(self, window_data_25_features, operating_state=1, k=4.1):
        """
        k: Hệ số điều chỉnh độ nhạy (Frontend có thể truyền vào)
        """
        if self.model is None or self.scaler is None:
            return self._mock_predict()

        # Operational Context Masking (Lọc nhiễu khi máy nghỉ)
        if operating_state == 0:
            return {
                "mae_absolute": 0.0,
                "mae_percentage": 0.0,
                "threshold": self.base_threshold,
                "status": "Green",
                "idle": True,
                "rca": []
            }

        try:
            # Scale & Predict
            scaled_data = self.scaler.transform(window_data_25_features)
            tensor_data = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                reconstructed = self.model(tensor_data)
            
            # Tính Weighted MAE
            mae_per_feature = torch.abs(tensor_data[0] - reconstructed[0]).mean(dim=0).cpu().numpy()
            weighted_mae_per_feature = mae_per_feature * self.weights
            total_mae = float(weighted_mae_per_feature.mean())
            
            # Lưu lịch sử để tính Threshold
            self.mae_history.append(total_mae)
            
            # Tính Adaptive Threshold (Dựa vào dữ liệu quá khứ)
            if len(self.mae_history) > 30: # Ít nhất 30 phút mới bắt đầu tính động
                hist_arr = np.array(list(self.mae_history)[:-1]) # Không tính điểm hiện tại vào nền
                rolling_mean = np.mean(hist_arr)
                rolling_std = np.std(hist_arr)
                adaptive_threshold = rolling_mean + k * rolling_std
            else:
                adaptive_threshold = self.base_threshold
                
            final_threshold = max(adaptive_threshold, self.base_threshold)
            
            # Tính phần trăm MAE so với Threshold
            mae_percentage = (total_mae / final_threshold) * 100
            
            # Phân loại trạng thái (Đèn giao thông)
            status = "Green"
            if mae_percentage >= 100.0:  # Vượt Threshold -> Đỏ
                status = "Red"
            elif mae_percentage >= 85.0: # Đạt 85% Threshold -> Vàng
                status = "Yellow"
                
            # Root Cause Analysis (XAI)
            rca = []
            if status in ["Red", "Yellow"]:
                # Lấy Top 3 Features đóng góp lỗi lớn nhất
                top_indices = np.argsort(weighted_mae_per_feature)[::-1][:3]
                rca = [{"feature": self.feature_names[i], "error": float(weighted_mae_per_feature[i])} for i in top_indices]
                
            return {
                "mae_absolute": float(total_mae),
                "mae_percentage": round(mae_percentage, 2),
                "threshold": float(final_threshold),
                "status": status,
                "idle": False,
                "rca": rca
            }
        except Exception as e:
            print("Inference error:", e)
            return self._mock_predict()
            
    def _mock_predict(self):
        import random
        return {
            "mae_absolute": random.uniform(0.05, 0.12),
            "mae_percentage": random.uniform(20.0, 50.0),
            "threshold": 0.2,
            "status": "Green",
            "idle": False,
            "rca": []
        }

# Singleton instance
engine = AIEngine()
